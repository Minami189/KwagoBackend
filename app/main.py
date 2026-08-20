import os
from typing import Any, Dict, List
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app import scanner
from app.schemas import (
    ScanRequest,
    ScanVerdictResponse,
    QuotaResponse,
    SmsScanRequest,
    SmsScanResponse,
    CnnAnalysisResult,
    UrlAnalysisResult,
)

sms_classifier = None

app = FastAPI(
    title="KwagoBackend URL Threat Scanner",
    description="FastAPI backend for analyzing URLs using the VirusTotal API with a custom weighted final verdict algorithm.",
    version="1.0.0",
)

# Enable CORS for cross-origin requests from web/mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_key = os.getenv("KWAGO_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KWAGO_API_KEY environment variable is not configured on the server."
        )
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized API key."
        )



@app.on_event("startup")
def startup_event():
    global sms_classifier
    try:
        sms_classifier = scanner.SMSClassifier()
        print("SMS Classifier successfully loaded!")
    except Exception as e:
        print(f"Warning: SMS Classifier could not be loaded: {e}")


# --- Routes ---

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint serving basic API metadata and docs linkage.
    """
    return {
        "name": "KwagoBackend URL Threat Scanner API",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint to verify that the service is running and healthy.
    """
    return {
        "status": "healthy",
        "classifier_loaded": sms_classifier is not None
    }



@app.post("/scan", response_model=ScanVerdictResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def scan_and_calculate_verdict(request: ScanRequest):
    """
    Submit a URL for VirusTotal scanning, fetch the engine analysis results,
    and calculate a customized weighted final threat verdict.
    """
    url_to_scan = request.url.strip()
    if not url_to_scan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL input field cannot be empty."
        )

    try:
        analysis_results = None

        # 1. Compute URL ID and check for existing report
        url_id = scanner.get_url_id(url_to_scan)
        report_response = await scanner.get_url_report(url_id)

        if report_response["status_code"] == 200:
            # Report exists, extract results
            analysis_results = scanner.extract_analysis_results(report_response["json"])

        # 2. Fallback to submitting new scan if no cached report was found
        if not analysis_results:
            scan_response = await scanner.scan_url(url_to_scan)
            
            # Handle API error response structures
            if "error" in scan_response:
                error_details = scan_response["error"].get("message", "Unknown error from VirusTotal.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"VirusTotal URL scan request failed: {error_details}"
                )

            scan_id = scan_response.get("data", {}).get("id")
            if not scan_id:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Unable to obtain a scan ID from the VirusTotal submission response."
                )

            # Retrieve scan results (polls until completed)
            analysis_response = await scanner.get_scan_results(scan_id)
            
            if "error" in analysis_response:
                error_details = analysis_response["error"].get("message", "Unknown error from VirusTotal.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"VirusTotal analysis retrieval failed: {error_details}"
                )

            analysis_results = scanner.extract_analysis_results(analysis_response)

        if not analysis_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analysis engine results found in the VirusTotal response."
            )

        # 3. Calculate weighted verdict
        source_weights = scanner.load_source_weights()
        verdict, normalized_score, total_weight, contributions = scanner.compute_weighted_verdict(
            analysis_results, source_weights
        )

        # Sum weighted scores to format explanation
        total_weighted_score = sum(
            scanner.normalize_category(str(engine_data.get("category", "undetected")).lower())
            * source_weights.get(engine_name, scanner.DEFAULT_SOURCE_WEIGHT)
            for engine_name, engine_data in analysis_results.items()
        )

        explanation = scanner.format_verdict_explanation(
            verdict,
            normalized_score,
            total_weighted_score,
            total_weight,
            contributions,
        )

        return ScanVerdictResponse(
            url=url_to_scan,
            verdict=verdict,
            normalized_score=normalized_score,
            total_weight=total_weight,
            explanation=explanation,
            contributions=contributions,
            raw_results=analysis_results,
        )

    except HTTPException:
        # Re-raise known HTTP exceptions directly
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the scan: {str(e)}"
        )


@app.get("/quota", response_model=QuotaResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def check_quota():
    """
    Retrieve current VirusTotal API usage and quotas.
    """
    try:
        quota_data = await scanner.check_api_quota()
        return QuotaResponse(
            endpoint=quota_data["endpoint"],
            quota_info=quota_data["response"],
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check API quota: {str(e)}"
        )


@app.post("/scan-sms", response_model=SmsScanResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def scan_sms_message(request: SmsScanRequest):
    """
    Scan an SMS message and return both CNN-BiGRU model scoring and VirusTotal URL scoring.
    The Android client will use these scores for client-side ensemble decision making.
    """
    message_to_scan = request.message.strip()
    if not message_to_scan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The SMS message input field cannot be empty."
        )

    if sms_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The SMS classifier model or tokenizer is not available on the server."
        )

    try:
        # 1. CNN-BiGRU Deep Learning Model Scoring
        cnn_probability = sms_classifier.predict(message_to_scan)
        cnn_verdict = "spam" if cnn_probability >= 0.5 else "benign"
        cnn_res = CnnAnalysisResult(
            score=cnn_probability,
            verdict=cnn_verdict
        )

        # 2. VirusTotal URL Threat Scoring (if has_url is True and a URL is provided)
        if request.has_url and request.extracted_url:
            primary_url = request.extracted_url.strip()
            url_res_dict = await scanner.analyze_sms_url(primary_url)
            url_res = UrlAnalysisResult(**url_res_dict)
        else:
            url_res = UrlAnalysisResult(
                has_url=False,
                extracted_url=None,
                score=None,
                verdict=None,
                total_weight=None,
                explanation="No URL requested for scanning or no URL provided.",
                contributions=[]
            )


        return SmsScanResponse(
            message=message_to_scan,
            cnn_analysis=cnn_res,
            url_analysis=url_res,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while scanning the SMS message: {str(e)}"
        )


if __name__ == "__main__":
    # Allow running directly via python -m app.main or python app/main.py from workspace root
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)