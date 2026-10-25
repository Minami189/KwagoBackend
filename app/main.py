import os
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, Query
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
    UrlReputationSyncResponse,
    MisclassificationReportRequest,
    MisclassificationReportResponse,
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
async def startup_event():
    global sms_classifier
    try:
        sms_classifier = scanner.SMSClassifier()
        print("SMS Classifier successfully loaded!")
    except Exception as e:
        print(f"Warning: SMS Classifier could not be loaded: {e}")
    
    # Open database connection pool
    await scanner.init_db()



@app.on_event("shutdown")
async def shutdown_event():
    # Close database connection pool
    await scanner.close_db()



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



@app.post("/scan", response_model=UrlAnalysisResult, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
@app.post("/scan-url", response_model=UrlAnalysisResult, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def scan_and_calculate_verdict(request: ScanRequest):
    """
    Submit a URL link for threat scanning and calculate a customized weighted threat verdict.
    Utilizes dual-layer caching (in-memory -> Supabase database) to prevent redundant VirusTotal API calls.
    """
    url_to_scan = request.url.strip()
    if not url_to_scan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL input field cannot be empty."
        )

    try:
        # 1. Check dual-layer caching layer (in-memory -> Supabase database)
        cached_scan = await scanner.lookup_cached_url(url_to_scan)
        if cached_scan:
            return UrlAnalysisResult(**cached_scan)

        # 2. Cache miss: Run live VirusTotal query
        url_res_dict = await scanner.analyze_sms_url(url_to_scan)

        # 3. Store in database global cache (linked to CACHE_SMS)
        await scanner.save_url_scan_to_db(
            url=url_to_scan,
            sms_id="CACHE_SMS",
            is_malicious=1 if url_res_dict.get("verdict") == "malicious" else 0,
            scan_result=url_res_dict
        )

        return UrlAnalysisResult(**url_res_dict)

    except HTTPException:
        # Re-raise known HTTP exceptions directly
        raise
    except Exception as e:
        # Wrap unexpected failures in a 500 internal server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred while processing the URL scan: {str(e)}"
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


@app.get("/url-reputations", response_model=UrlReputationSyncResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def get_url_reputations(since_timestamp: Optional[int] = Query(None, description="Unix timestamp in milliseconds to filter records created/updated after this time.")):
    """
    Retrieve cached URL threat reputation records for client/VPN synchronization.
    Supports incremental sync via the optional since_timestamp query parameter (in milliseconds).
    """
    try:
        sync_data = await scanner.get_url_reputations(since_timestamp)
        return UrlReputationSyncResponse(**sync_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve URL reputations: {str(e)}"
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

    global sms_classifier
    if sms_classifier is None:
        try:
            sms_classifier = scanner.SMSClassifier()
        except Exception as e:
            print(f"Warning: Lazy-loading SMS Classifier failed: {e}")

    if sms_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The SMS classifier model or tokenizer is not available on the server."
        )

    try:
        # Resolve all candidate web URLs from extracted_urls, urls, and extracted_url (filtering out emails)
        candidate_urls = scanner.resolve_candidate_urls(
            has_url=request.has_url,
            extracted_url=request.extracted_url,
            extracted_urls=request.extracted_urls,
            urls=request.urls,
        )
        has_url_flag = bool(request.has_url and candidate_urls)

        # 1. CNN-BiGRU Deep Learning Model Scoring
        cnn_probability = sms_classifier.predict(message_to_scan)
        cnn_verdict, cnn_explanation = scanner.generate_cnn_explanation(
            message=message_to_scan,
            cnn_score=float(cnn_probability),
            has_url=has_url_flag
        )
        
        cnn_res = CnnAnalysisResult(
            score=float(cnn_probability),
            verdict=cnn_verdict,
            explanation=cnn_explanation
        )

        ml_confidence_val = request.ml_confidence or 0.0

        # 2. VirusTotal URL Threat Scoring with dual-layer caching (in-memory + database) across all URLs
        url_res_dict = {}
        per_url_scans: list[tuple[str, dict, bool]] = []  # (url_str, scan_dict, is_cache_hit)

        if has_url_flag:
            for target_url in candidate_urls:
                cached_scan = await scanner.lookup_cached_url(target_url)
                if cached_scan:
                    scan_item = dict(cached_scan)
                    scan_item["extracted_url"] = target_url
                    per_url_scans.append((target_url, scan_item, True))
                else:
                    scan_item = await scanner.analyze_sms_url(target_url)
                    per_url_scans.append((target_url, scan_item, False))

            url_res_dict = scanner.aggregate_multi_url_results([item[1] for item in per_url_scans])
            url_res = UrlAnalysisResult(**url_res_dict)
        else:
            url_res = UrlAnalysisResult(
                has_url=False,
                extracted_url=None,
                extracted_urls=[],
                score=None,
                verdict=None,
                total_weight=None,
                explanation="No URL requested for scanning or no URL provided.",
                contributions=[],
                individual_results=[]
            )

        # 3. Synthesize Overall Verdict, Score, and Executive Summary Explanation
        # (Malicious URLs automatically trigger higher URL weighting and escalation)
        overall_verdict, overall_score, overall_explanation = scanner.generate_overall_summary(
            message=message_to_scan,
            ml_confidence=float(ml_confidence_val),
            cnn_score=float(cnn_probability),
            cnn_explanation=cnn_explanation,
            url_analysis=url_res_dict if has_url_flag else {}
        )

        # 4. Only save SMS logs and classification results if user allowed saving AND overall_score reaches suspicious threshold (>= 0.50)
        saved_sms_id = None
        if request.allow_save and overall_score >= 0.50:
            # Save SMS to public.sms_message and retrieve key
            saved_sms_id = await scanner.save_sms_message_to_db(request.sender, message_to_scan, 0)
            if saved_sms_id:
                # Save CNN, local ML, URL, and Ensemble predictions to public.analysis_result
                norm_ml_prediction = scanner.normalize_ml_prediction(request.ml_prediction, ml_confidence_val)
                extracted_url_score = url_res_dict.get("score") if (has_url_flag and url_res_dict) else None

                await scanner.save_analysis_result_to_db(
                    sms_id=saved_sms_id,
                    ml_prediction=norm_ml_prediction,
                    ml_confidence=float(ml_confidence_val),
                    dl_prediction=cnn_verdict,
                    dl_confidence=float(cnn_probability),
                    url_score=float(extracted_url_score) if extracted_url_score is not None else None,
                    ensemble_score=float(overall_score),
                    ensemble_verdict=str(overall_verdict).lower()
                )

                # Store EACH URL separately in Supabase (public.url and public.url_analysis) linked to saved_sms_id
                if has_url_flag and per_url_scans:
                    for target_url, single_scan_dict, _ in per_url_scans:
                        is_mal = 1 if (
                            single_scan_dict.get("verdict") == "malicious"
                            or (single_scan_dict.get("score") is not None and float(single_scan_dict.get("score")) >= 0.45)
                        ) else 0
                        await scanner.save_url_scan_to_db(
                            url=target_url,
                            sms_id=saved_sms_id,
                            is_malicious=is_mal,
                            scan_result=single_scan_dict
                        )

        # For any URL that was newly scanned (cache miss) and NOT saved under a user sms_id, save separately under CACHE_SMS
        if has_url_flag and per_url_scans and not saved_sms_id:
            for target_url, single_scan_dict, was_cache_hit in per_url_scans:
                if not was_cache_hit:
                    is_mal = 1 if (
                        single_scan_dict.get("verdict") == "malicious"
                        or (single_scan_dict.get("score") is not None and float(single_scan_dict.get("score")) >= 0.45)
                    ) else 0
                    await scanner.save_url_scan_to_db(
                        url=target_url,
                        sms_id="CACHE_SMS",
                        is_malicious=is_mal,
                        scan_result=single_scan_dict
                    )

        # 5. Automatically log to NTC report table if auto_report is enabled AND overall_score reaches suspicious/harmful threshold (>= 0.50)
        if request.auto_report and overall_score >= 0.50:
            await scanner.save_ntc_report_to_db(
                message=message_to_scan,
                url_analysis=url_res_dict if has_url_flag else {},
                ml_score=float(ml_confidence_val),
                dl_score=float(cnn_probability),
                final_score=float(overall_score),
                verdict=str(overall_verdict),
                sender=request.sender or "UNKNOWN",
                status="pending"
            )

        return SmsScanResponse(
            message=message_to_scan,
            overall_verdict=overall_verdict,
            overall_score=overall_score,
            overall_explanation=overall_explanation,
            cnn_analysis=cnn_res,
            url_analysis=url_res,
         )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while scanning the SMS message: {str(e)}"
        )


@app.post(
    "/report-misclassification",
    response_model=MisclassificationReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report SMS Classification Error",
    description="Submit user feedback when a message is misclassified (False Positive or False Negative) for model improvement and dataset review.",
    dependencies=[Depends(verify_api_key)],
)
async def report_misclassification(request: MisclassificationReportRequest):
    try:
        resolved_urls = scanner.resolve_candidate_urls(
            has_url=request.has_url,
            extracted_url=request.extracted_url,
            extracted_urls=request.extracted_urls,
            urls=request.urls,
        )
        extracted_url_str = ", ".join(resolved_urls) if resolved_urls else request.extracted_url

        report_id = await scanner.save_misclassification_report_to_db(
            message=request.message,
            original_verdict=request.original_verdict,
            original_score=request.original_score,
            user_verdict=request.user_verdict,
            original_ml_score=request.original_ml_score,
            original_dl_score=request.original_dl_score,
            sender=request.sender,
            has_url=bool(request.has_url and resolved_urls),
            extracted_url=extracted_url_str,
            report_type=request.report_type,
            user_comment=request.user_comment,
            app_version=request.app_version,
            device_id=request.device_id,
        )

        if not report_id:
            import uuid
            report_id = str(uuid.uuid4())

        return MisclassificationReportResponse(
            status="success",
            report_id=report_id,
            message="Misclassification report received successfully. Thank you for your feedback!"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while saving the misclassification report: {str(e)}"
        )


if __name__ == "__main__":
    # Allow running directly via python -m app.main or python app/main.py from workspace root
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)