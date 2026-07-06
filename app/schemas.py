from typing import Any, Dict, List
from pydantic import BaseModel, Field

class ScanRequest(BaseModel):
    url: str = Field(
        ...,
        description="The URL or string to scan with VirusTotal.",
        json_schema_extra={"example": "http://malicious-web-example.com"}
    )


class ScanVerdictResponse(BaseModel):
    url: str = Field(..., description="The URL that was analyzed.")
    verdict: str = Field(..., description="The calculated verdict (malicious, suspicious, or benign).")
    normalized_score: float = Field(..., description="The normalized threat score ranging from 0.0 (safe) to 1.0 (dangerous).")
    total_weight: float = Field(..., description="The sum of all scanning engine weights that were evaluated.")
    explanation: str = Field(..., description="Human-readable breakdown explanation of the calculation.")
    contributions: List[str] = Field(..., description="The individual contributions of each scanned engine.")
    raw_results: Dict[str, Any] = Field(..., description="The raw analysis results from each VirusTotal engine.")


class QuotaResponse(BaseModel):
    endpoint: str = Field(..., description="The specific VirusTotal usage/quota endpoint that responded.")
    quota_info: Dict[str, Any] = Field(..., description="The detailed response payload representing API limits and usage.")


class SmsScanRequest(BaseModel):
    message: str = Field(
        ...,
        description="The full SMS message text to scan for threat/spam classification.",
        json_schema_extra={"example": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com"}
    )


class SmsScanResponse(BaseModel):
    message: str = Field(..., description="The original SMS message analyzed.")
    verdict: str = Field(..., description="The calculated threat verdict ('spam' or 'benign').")
    probability: float = Field(..., description="The raw threat probability score from the CNN-BiGRU model (0.0 to 1.0).")

