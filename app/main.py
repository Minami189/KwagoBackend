from typing import Any, Dict, List
import uvicorn
from fastapi import FastAPI, HTTPException, status

from app import scanner
from app.schemas import (
    ScanRequest,
    ScanVerdictResponse,
    QuotaResponse,
    SmsScanRequest,
    SmsScanResponse,
)

sms_classifier = None

app = FastAPI(
    title="KwagoBackend URL Threat Scanner",
    description="FastAPI backend for analyzing URLs using the VirusTotal API with a custom weighted final verdict algorithm.",
    version="1.0.0",
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


@app.post("/scan", response_model=ScanVerdictResponse, status_code=status.HTTP_200_OK)
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


@app.get("/quota", response_model=QuotaResponse, status_code=status.HTTP_200_OK)
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


@app.post("/scan-sms", response_model=SmsScanResponse, status_code=status.HTTP_200_OK)
async def scan_sms_message(request: SmsScanRequest):
    """
    Scan a full SMS message using the trained CNN-BiGRU model and return a threat/spam verdict.
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
        probability = sms_classifier.predict(message_to_scan)
        verdict = "spam" if probability >= 0.5 else "benign"
        return SmsScanResponse(
            message=message_to_scan,
            verdict=verdict,
            probability=probability,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while scanning the SMS message: {str(e)}"
        )


if __name__ == "__main__":
    # Allow running directly via python -m app.main or python app/main.py from workspace root
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)