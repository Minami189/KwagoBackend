from typing import Any, Dict, List, Optional
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
    has_url: bool = Field(
        ...,
        description="Indicates whether the SMS contains a URL extracted by the client.",
        json_schema_extra={"example": False}
    )
    extracted_url: Optional[str] = Field(
        None,
        description="The URL extracted from the message by the client, if any.",
        json_schema_extra={"example": None}
    )
    allow_save: bool = Field(
        False,
        description="Whether the user allowed auto-saving of the SMS content.",
        json_schema_extra={"example": False}
    )
    sender: Optional[str] = Field(
        None,
        description="The sender's phone number.",
        json_schema_extra={"example": "+1234567890"}
    )





class CnnAnalysisResult(BaseModel):
    score: float = Field(..., description="CNN-BiGRU deep learning model threat/spam probability score (0.0 to 1.0).")
    verdict: str = Field(..., description="CNN model verdict ('spam' or 'benign').")


class UrlAnalysisResult(BaseModel):
    has_url: bool = Field(..., description="Whether an HTTP/HTTPS URL was detected in the message.")
    extracted_url: Optional[str] = Field(None, description="The URL extracted from the message, if any.")
    score: Optional[float] = Field(None, description="VirusTotal normalized threat score (0.0 safe to 1.0 dangerous), or null if no URL.")
    verdict: Optional[str] = Field(None, description="VirusTotal verdict ('malicious', 'suspicious', 'benign'), or null if no URL.")
    total_weight: Optional[float] = Field(None, description="Total weight of evaluated VirusTotal engines, or null if no URL.")
    explanation: Optional[str] = Field(None, description="Human-readable breakdown of the URL scoring calculation.")
    contributions: Optional[List[str]] = Field(None, description="Individual VirusTotal engine contributions.")


class SmsScanResponse(BaseModel):
    message: str = Field(..., description="The original SMS message analyzed.")
    cnn_analysis: CnnAnalysisResult = Field(..., description="Scoring results from the CNN-BiGRU deep learning model.")
    url_analysis: UrlAnalysisResult = Field(..., description="Scoring results from the VirusTotal URL threat scanner.")


