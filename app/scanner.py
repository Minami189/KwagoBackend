import asyncio
import base64
import collections
import csv
import os
import pickle
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from supabase import create_client, Client
from cachetools import TTLCache

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
if SUPABASE_URL and SUPABASE_URL.endswith("/rest/v1/"):
    SUPABASE_URL = SUPABASE_URL[:-9]

SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

supabase: Client = None

if SUPABASE_URL and SUPABASE_SECRET_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        print("Supabase client initialized successfully!")
    except Exception as e:
        print(f"Warning: Supabase client initialization failed: {e}")



# In-memory TTL cache for URL scan results (1000 items capacity, 1 hour lifespan)
memory_cache = TTLCache(maxsize=1000, ttl=3600)


VIRUS_TOTAL_API_KEY = os.getenv("VIRUS_TOTAL_API_KEY")
WEIGHT_FILE = Path(__file__).parent / "source_weights.csv"
DEFAULT_SOURCE_WEIGHT = 0.3
CATEGORY_SCORES = {
    "malicious": 1.0,
    "suspicious": 0.75,
    "undetected": 0.0,
    "harmless": 0.0,
    "type-unsupported": 0.1,
    "timeout": 0.05,
}


def load_source_weights() -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not WEIGHT_FILE.exists():
        return weights

    with WEIGHT_FILE.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = row.get("source")
            weight = row.get("weight")
            if source and weight:
                try:
                    weights[source.strip()] = float(weight)
                except ValueError:
                    continue
    return weights


def normalize_category(category: str) -> float:
    return CATEGORY_SCORES.get(category.lower(), 0.0)


def compute_weighted_verdict(
    analysis_results: Dict[str, Any], weights: Dict[str, float]
) -> Tuple[str, float, float, List[str]]:
    total_weighted_score = 0.0
    total_weight = 0.0
    contribution_lines: List[str] = []
    has_trusted_malicious = False
    malicious_count = 0
    suspicious_count = 0

    for engine_name, engine_data in analysis_results.items():
        category = str(engine_data.get("category", "undetected")).lower()
        score = normalize_category(category)
        weight = weights.get(engine_name, DEFAULT_SOURCE_WEIGHT)
        total_weighted_score += score * weight
        total_weight += weight

        if category == "malicious":
            malicious_count += 1
            if weight >= 0.5:
                has_trusted_malicious = True
        elif category == "suspicious":
            suspicious_count += 1

        if category in ["malicious", "suspicious"]:
            contribution_lines.append(
                f"{engine_name} ({category})"
            )

    if total_weight == 0:
        raise ValueError("No weighted engines were available to compute a verdict.")

    normalized_score = total_weighted_score / total_weight

    # Harsher Security Boost Rules based on vendor consensus and trusted vendors:
    if malicious_count >= 15:
        normalized_score = max(normalized_score, 1.00)
    elif malicious_count >= 10:
        normalized_score = max(normalized_score, 0.95)
    elif malicious_count >= 5:
        normalized_score = max(normalized_score, 0.85)
    elif malicious_count >= 3:
        normalized_score = max(normalized_score, 0.75)
    elif has_trusted_malicious or malicious_count >= 2:
        normalized_score = max(normalized_score, 0.65)
    elif malicious_count >= 1:
        normalized_score = max(normalized_score, 0.45)
    elif suspicious_count >= 2:
        normalized_score = max(normalized_score, 0.40)
    elif suspicious_count >= 1:
        normalized_score = max(normalized_score, 0.25)

    # Stricter Verdict Thresholds
    if normalized_score >= 0.45:
        verdict = "malicious"
    elif normalized_score >= 0.20:
        verdict = "suspicious"
    else:
        verdict = "benign"

    return verdict, round(normalized_score, 4), total_weight, contribution_lines




def format_verdict_explanation(
    verdict: str,
    normalized_score: float,
    total_weighted_score: float,
    total_weight: float,
    contributions: List[str],
) -> str:
    if verdict == "benign":
        return "This URL is clean. No security engines flagged it as malicious."

    flagged_engines = ", ".join(contributions)
    if flagged_engines:
        return f"This URL is flagged as {verdict} (threat score: {normalized_score:.2f}). Detected by: {flagged_engines}."
    else:
        return f"This URL is flagged as {verdict} (threat score: {normalized_score:.2f})."


def generate_cnn_explanation(message: str, cnn_score: float, has_url: bool = True) -> Tuple[str, str]:
    """
    Analyzes message keywords and CNN model probability to output a precise, human-readable explanation sentence.
    Thresholds: Safe (< 0.50), Suspicious/Spam (0.50 - 0.85), Harmful (>= 0.85).
    If has_url is False, explanation will never mention web links.
    """
    import re
    msg_lower = message.lower()
    
    # Specific keyword pattern categories
    reward_bonus_pattern = r"\b(bonus|deposit|promo|promos|gift\s*card|free|prize|reward|congrats|congratulations|won|winner|claim|cashback|earn|discount|php|\$\d+|\b\d+\s*pesos)\b"
    urgency_security_pattern = r"\b(suspended|locked|security\s*breach|unauthorized|compromised|deactivated|freeze|fraud|urgent\s*action|immediate\s*verification|immediately\s*verify)\b"
    action_download_pattern = r"\b(downloading\s*the\s*app|download\s*app|install\s*app|app\s*store|apk|click\s*here|click\s*link|register\s*now|sign\s*up)\b"
    link_pattern = r"http[s]?://"

    reward_match = re.search(reward_bonus_pattern, msg_lower)
    urgency_match = re.search(urgency_security_pattern, msg_lower)
    action_match = re.search(action_download_pattern, msg_lower)
    link_match = re.search(link_pattern, msg_lower)

    if cnn_score >= 0.50:
        verdict = "harmful" if cnn_score >= 0.85 else "spam"
        if reward_match and action_match:
            explanation = "Promotes unsolicited monetary bonuses and app download incentives common in smishing scams."
        elif reward_match:
            explanation = "Promotes unsolicited monetary bonuses, deposit rewards, or financial incentives."
        elif urgency_match:
            explanation = "Uses urgent account suspension or security breach warnings demanding immediate verification."
        elif action_match:
            explanation = "Demands unsolicited app downloads or external registrations."
        elif link_match and has_url:
            explanation = "Contains urgency cues combined with unverified web links."
        else:
            explanation = "Exhibits linguistic patterns common in unsolicited spam or smishing messages."
    else:
        verdict = "benign"
        explanation = "Message text displays standard conversational language with no smishing indicators."

    return verdict, explanation


def compute_ensemble_score(
    ml_confidence: float,
    cnn_score: float,
    url_score: Optional[float]
) -> float:
    """
    Computes the ensemble classification score:
    - Case 3A (URL present & completed): 50% DL / 25% URL / 25% ML
    - Case 1B & 2B (No URL or pending URL scan): 66.7% DL (2/3) + 33.3% ML (1/3)
    """
    ml_val = ml_confidence / 100.0 if ml_confidence > 1.0 else ml_confidence
    dl_val = cnn_score / 100.0 if cnn_score > 1.0 else cnn_score

    if url_score is not None:
        u_val = url_score / 100.0 if url_score > 1.0 else url_score
        score = (0.50 * dl_val) + (0.25 * u_val) + (0.25 * ml_val)
    else:
        # Re-normalize 66.7% DL (2/3) + 33.3% ML (1/3)
        score = ((2.0 / 3.0) * dl_val) + ((1.0 / 3.0) * ml_val)

    return round(score, 4)


def generate_overall_summary(
    message: str,
    ml_confidence: float,
    cnn_score: float,
    cnn_explanation: str,
    url_analysis: dict,
) -> Tuple[str, float, str]:
    """
    Synthesizes a deep multi-layer overall threat verdict, overall threat score, and executive summary explanation
    comparing findings across the Local ML layer, Deep Learning layer, and URL Threat Scanner layer.
    Verdict Thresholds:
    - Safe: Below 50% (< 0.50)
    - Suspicious: 50% to 85% (0.50 - 0.85)
    - Harmful: Above 85% (>= 0.85)
    """
    import re
    msg_lower = message.lower()
    
    has_url = url_analysis.get("has_url", False)
    url_verdict = url_analysis.get("verdict")
    # Only treat url_score as active if has_url is True, score is not None, AND verdict is NOT pending
    url_score = url_analysis.get("score") if (has_url and url_analysis.get("score") is not None and url_verdict != "pending") else None
    extracted_url = url_analysis.get("extracted_url")
    url_contributions = url_analysis.get("contributions") or []

    # Calculate exact ensemble score (re-normalizes over ML & DL if URL is None or pending)
    final_score = compute_ensemble_score(ml_confidence, cnn_score, url_score)

    ml_pct = int(round(ml_confidence * 100 if ml_confidence <= 1.0 else ml_confidence))
    dl_pct = int(round(cnn_score * 100 if cnn_score <= 1.0 else cnn_score))

    # Normalize cnn_explanation lower for embedding
    cnn_reason_lower = cnn_explanation[0].lower() + cnn_explanation[1:] if cnn_explanation else "contains suspicious patterns"
    if cnn_reason_lower.endswith("."):
        cnn_reason_lower = cnn_reason_lower[:-1]

    # Label layers
    ml_label = "Harmful" if ml_pct >= 85 else ("Suspicious" if ml_pct >= 50 else "Safe")

    # 1. Overall Verdict Determination
    if final_score >= 0.85 or url_verdict == "malicious":
        overall_verdict = "Harmful"
        if url_verdict == "malicious" and final_score < 0.85:
            final_score = max(final_score, 0.85)
    elif final_score >= 0.50 or url_verdict == "suspicious":
        overall_verdict = "Suspicious"
        if url_verdict == "suspicious" and final_score < 0.50:
            final_score = max(final_score, 0.50)
    else:
        overall_verdict = "Safe"

    explanation_parts = []

    # --- CATEGORY 1: SAFE OVERALL VERDICT ---
    if overall_verdict == "Safe":
        if has_url and url_verdict in ["benign", "clean"]:
            if ml_pct >= 50 and dl_pct < 50:
                explanation_parts.append(
                    f"Although the local ML layer marked this as {ml_label.lower()}, the Deep Learning layer evaluated the message text as safe and the embedded web link was verified clean, making the overall message Safe."
                )
            elif ml_pct < 50 and dl_pct >= 50:
                explanation_parts.append(
                    f"Although the Deep Learning layer detected potential smishing cues ({cnn_reason_lower}), the local ML layer marked it as safe and the embedded web link was verified clean, making the overall message Safe."
                )
            elif ml_pct >= 50 and dl_pct >= 50:
                explanation_parts.append(
                    "Although message text exhibits smishing cues, the overall message is verified as Safe because the embedded web link was verified clean."
                )
            else:
                explanation_parts.append(
                    "Both local ML and Deep Learning layers verified this message as safe, and the embedded web link was confirmed clean."
                )
        else:
            # No URL or URL pending
            if ml_pct >= 50 and dl_pct < 50:
                explanation_parts.append(
                    f"Although the local ML layer marked this as {ml_label.lower()}, the Deep Learning layer evaluated the message text as safe, so the overall message is verified as Safe."
                )
            elif ml_pct < 50 and dl_pct >= 50:
                explanation_parts.append(
                    f"Although the Deep Learning layer detected potential smishing cues ({cnn_reason_lower}), the local ML layer marked it as safe, so the overall message is verified as Safe."
                )
            else:
                explanation_parts.append(
                    "Both local ML and Deep Learning layers verified this message as safe, showing no smishing indicators."
                )

    # --- CATEGORY 2: SUSPICIOUS OVERALL VERDICT ---
    elif overall_verdict == "Suspicious":
        if ml_pct >= 50 and dl_pct < 50:
            if has_url and url_verdict in ["benign", "clean"]:
                explanation_parts.append(
                    f"Although the Deep Learning layer evaluated the text as safe and the web link was clean, the local ML layer flagged this as {ml_label.lower()}, keeping the overall risk level at Suspicious."
                )
            else:
                explanation_parts.append(
                    f"Although the Deep Learning model evaluated the message text as safe, the local ML layer flagged this as {ml_label.lower()}, resulting in an overall Suspicious classification."
                )
        elif ml_pct < 50 and dl_pct >= 50:
            if has_url and url_verdict in ["benign", "clean"]:
                explanation_parts.append(
                    f"Though the local ML layer marked this as safe and the embedded web link was clean, the Deep Learning layer detected smishing risk because it {cnn_reason_lower}, classifying the overall message as Suspicious."
                )
            else:
                explanation_parts.append(
                    f"Though the local ML layer marked this as safe, the Deep Learning layer detected smishing risk because it {cnn_reason_lower}, classifying the overall message as Suspicious."
                )
        elif ml_pct >= 50 and dl_pct >= 50:
            explanation_parts.append(
                f"Both classification layers indicated smishing risk as the message {cnn_reason_lower}, resulting in an overall Suspicious verdict."
            )
        else: # ml_pct < 50 and dl_pct < 50, but URL is suspicious
            explanation_parts.append(
                f"Both local ML and Deep Learning layers verified the message text as safe, but the embedded web link ({extracted_url}) has an unverified or suspicious reputation, making the overall verdict Suspicious."
            )

    # --- CATEGORY 3: HARMFUL OVERALL VERDICT ---
    else: # overall_verdict == "Harmful"
        if url_verdict == "malicious":
            flagged_by = f" (detected by {', '.join(url_contributions[:2])})" if url_contributions else ""
            if ml_pct < 50 and dl_pct < 50:
                explanation_parts.append(
                    f"Both local ML and Deep Learning layers verified the message text as safe, but the embedded web link ({extracted_url}) was confirmed as a high-risk malicious phishing site{flagged_by}."
                )
            elif ml_pct < 50 and dl_pct >= 50:
                explanation_parts.append(
                    f"Though the local ML layer marked this as safe, the Deep Learning layer detected smishing risk because it {cnn_reason_lower}. Furthermore, the embedded web link ({extracted_url}) was confirmed as a high-risk malicious phishing site{flagged_by}."
                )
            elif ml_pct >= 50 and dl_pct < 50:
                explanation_parts.append(
                    f"Although the local ML layer flagged this as {ml_label.lower()} and the Deep Learning model evaluated text as safe, the embedded web link ({extracted_url}) was confirmed as a high-risk malicious phishing site{flagged_by}."
                )
            else:
                explanation_parts.append(
                    f"Both the local ML and Deep Learning layers confirmed High Risk because the message {cnn_reason_lower}. Furthermore, the embedded web link ({extracted_url}) was confirmed as a high-risk malicious phishing site{flagged_by}."
                )
        else:
            # Text driven Harmful
            if ml_pct < 50 and dl_pct >= 85:
                explanation_parts.append(
                    f"Though the local ML layer marked this as safe, the Deep Learning layer detected High Risk because it {cnn_reason_lower}."
                )
            elif ml_pct >= 85 and dl_pct < 50:
                explanation_parts.append(
                    f"Although the local ML layer flagged High Risk, the Deep Learning model evaluated the message text as safe, but high local ML risk keeps the overall verdict as Harmful."
                )
            elif ml_pct >= 85 and dl_pct >= 85:
                explanation_parts.append(
                    f"Both the local ML and Deep Learning layers confirmed High Risk because the message {cnn_reason_lower}."
                )
            else:
                explanation_parts.append(
                    f"Both classification layers indicated high smishing risk as the message {cnn_reason_lower}."
                )

    # Append pending URL caution warning if URL is present but pending
    if has_url and extracted_url and url_verdict == "pending":
        explanation_parts.append(
            f"Exercise caution: this message contains a web link ({extracted_url}) that has not been verified by online threat intelligence yet, so its safety cannot be guaranteed."
        )

    overall_explanation = " ".join(explanation_parts)

    return overall_verdict, final_score, overall_explanation

    return overall_verdict, final_score, overall_explanation








# Async VT scanners

def get_url_id(url: str) -> str:
    """
    Computes a base64-encoded URL representation without padding,
    as required by the VirusTotal API v3.
    """
    return base64.urlsafe_b64encode(url.encode()).decode().strip("=")


async def get_url_report(url_id: str) -> dict:
    """
    Retrieves the analysis report for a URL directly using its base64 ID.
    Returns a dict containing 'status_code' and the response 'json'.
    """
    if not VIRUS_TOTAL_API_KEY:
        raise RuntimeError("VIRUS_TOTAL_API_KEY is not set in .env")

    headers = {
        "x-apikey": VIRUS_TOTAL_API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
        ) as response:
            result = await response.json()
            return {"status_code": response.status, "json": result}


async def scan_url(url: str) -> dict:
    print(f"Scanning input: {url}")
    if not VIRUS_TOTAL_API_KEY:
        raise RuntimeError("VIRUS_TOTAL_API_KEY is not set in .env")

    headers = {
        "x-apikey": VIRUS_TOTAL_API_KEY,
    }
    data = {"url": url}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data=data,
        ) as response:
            result = await response.json()
            print("Scan result:", result)
            return result


async def get_scan_results(scan_id: str, max_retries: int = 15, delay: int = 2) -> dict:
    if not VIRUS_TOTAL_API_KEY:
        raise RuntimeError("VIRUS_TOTAL_API_KEY is not set in .env")

    headers = {
        "x-apikey": VIRUS_TOTAL_API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            async with session.get(
                f"https://www.virustotal.com/api/v3/analyses/{scan_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    result = await response.json()
                    print(f"Failed to fetch analysis on attempt {attempt+1}: {result}")
                    return result

                result = await response.json()
                attributes = result.get("data", {}).get("attributes", {})
                status = attributes.get("status")

                print(f"Analysis status on attempt {attempt+1}: {status}")

                if status == "completed":
                    return result

                await asyncio.sleep(delay)

        print(f"Polling completed without status='completed'. Returning latest response.")
        return result


def extract_analysis_results(response_json: Dict[str, Any]) -> Dict[str, Any]:
    data = response_json.get("data", {})
    attributes = data.get("attributes", {})
    results = attributes.get("results") or attributes.get("last_analysis_results")
    if isinstance(results, dict):
        return results

    if isinstance(attributes.get("last_analysis_results"), dict):
        return attributes.get("last_analysis_results")

    return {}


async def check_api_quota() -> Dict[str, Any]:
    if not VIRUS_TOTAL_API_KEY:
        raise RuntimeError("VIRUS_TOTAL_API_KEY is not set in .env")

    headers = {
        "x-apikey": VIRUS_TOTAL_API_KEY,
    }
    endpoints = [
        "https://www.virustotal.com/api/v3/usage",
        "https://www.virustotal.com/api/v3/quotas",
        "https://www.virustotal.com/api/v3/users/me",
    ]

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            async with session.get(endpoint, headers=headers) as response:
                result = await response.json()
                if response.status == 200 and result:
                    print(f"Quota response from {endpoint}:", result)
                    return {"endpoint": endpoint, "response": result}

    raise RuntimeError("Unable to fetch VirusTotal quota information from any known endpoint.")


def format_quota_response(quota_data: Dict[str, Any]) -> str:
    endpoint = quota_data.get("endpoint")
    response = quota_data.get("response", {})
    data = response.get("data", {})
    attributes = data.get("attributes", {}) or response.get("attributes", {}) or {}

    lines = [f"Quota information from {endpoint}"]
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    return f"Quota response from {endpoint}: {response}"


# --- SMS CNN-BiGRU Classifier ---

ONNX_MODEL_PATH = Path(__file__).parent.parent / "models" / "cnn_bigru_model.onnx"
TOKENIZER_PATH = Path(__file__).parent.parent / "models" / "tokenizer_b.pkl"


class MockTokenizer:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    self.__dict__.update(item)


class InspectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "OrderedDict":
            return collections.OrderedDict
        if name == "defaultdict":
            return collections.defaultdict
        return MockTokenizer


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\+?63[\d\*]{9,10}', '', text)
    text = re.sub(r'\b0\d{10}\b', '', text)
    text = re.sub(r'\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{4}\b', '', text)
    emoji_pattern = re.compile(
        u'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        u'\U000024C2-\U0001F251]+',
        flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\s.,!?'\-]", '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class SMSClassifier:
    def __init__(self):
        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model file not found at {ONNX_MODEL_PATH}")
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer pickle file not found at {TOKENIZER_PATH}")

        print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        self.input_name = self.session.get_inputs()[0].name
        self.max_len = 150

        print(f"Loading Tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, "rb") as f:
            tok_obj = InspectUnpickler(f).load()
            self.word_index = getattr(tok_obj, "word_index", {})
            self.num_words = getattr(tok_obj, "num_words", 10000)
            self.oov_token = getattr(tok_obj, "oov_token", "<OOV>")
            self.oov_index = self.word_index.get(self.oov_token, 1)

    def preprocess(self, text: str) -> np.ndarray:
        cleaned = clean_text(text)
        words = cleaned.split()

        sequence = []
        for w in words:
            idx = self.word_index.get(w)
            if idx is not None and (self.num_words is None or idx < self.num_words):
                sequence.append(idx)
            else:
                if self.oov_index is not None:
                    sequence.append(self.oov_index)

        # Pad sequence (post-padding, post-truncating)
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [0] * (self.max_len - len(sequence))

        return np.array([sequence], dtype=np.float32)

    def predict(self, text: str) -> float:
        input_data = self.preprocess(text)
        raw_outputs = self.session.run(None, {self.input_name: input_data})
        prob = float(raw_outputs[0][0][0])
        return prob


def extract_urls(text: str) -> List[str]:
    """
    Extracts HTTP, HTTPS, or WWW URLs from text.
    Strips trailing punctuation such as periods, commas, or parentheses.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    pattern = r'(?:https?://|www\.)[^\s]+'
    raw_urls = re.findall(pattern, text)
    cleaned_urls = []
    for u in raw_urls:
        u = u.rstrip(".,;!?)>]\'")
        if u.startswith("www."):
            u = "http://" + u
        if u and u not in cleaned_urls:
            cleaned_urls.append(u)
    return cleaned_urls


async def analyze_sms_url(url: str) -> Dict[str, Any]:
    """
    Performs VirusTotal weighted threat scoring on a URL extracted from an SMS.
    Returns a dictionary matching the UrlAnalysisResult schema structure.
    Catches exceptions gracefully to prevent API failure if VT lookup fails.
    """
    try:
        analysis_results = None

        # 1. Compute URL ID and check for existing report
        url_id = get_url_id(url)
        report_response = await get_url_report(url_id)

        if report_response.get("status_code") == 200:
            analysis_results = extract_analysis_results(report_response.get("json", {}))

        # 2. Fallback to submitting new scan if no cached report was found
        if not analysis_results:
            scan_response = await scan_url(url)
            if "error" not in scan_response:
                scan_id = scan_response.get("data", {}).get("id")
                if scan_id:
                    analysis_response = await get_scan_results(scan_id)
                    if "error" not in analysis_response:
                        analysis_results = extract_analysis_results(analysis_response)

        if not analysis_results:
            return {
                "has_url": True,
                "extracted_url": url,
                "score": None,
                "verdict": "pending",
                "total_weight": None,
                "explanation": "URL security scan is pending verification.",
                "contributions": [],
            }

        # 3. Calculate weighted verdict
        source_weights = load_source_weights()
        verdict, normalized_score, total_weight, contributions = compute_weighted_verdict(
            analysis_results, source_weights
        )

        total_weighted_score = sum(
            normalize_category(str(engine_data.get("category", "undetected")).lower())
            * source_weights.get(engine_name, DEFAULT_SOURCE_WEIGHT)
            for engine_name, engine_data in analysis_results.items()
        )

        explanation = format_verdict_explanation(
            verdict,
            normalized_score,
            total_weighted_score,
            total_weight,
            contributions,
        )

        return {
            "has_url": True,
            "extracted_url": url,
            "score": normalized_score,
            "verdict": verdict,
            "total_weight": total_weight,
            "explanation": explanation,
            "contributions": contributions,
        }

    except Exception as e:
        print(f"Error analyzing SMS URL ({url}): {e}")
        return {
            "has_url": True,
            "extracted_url": url,
            "score": None,
            "verdict": None,
            "total_weight": None,
            "explanation": f"Unable to analyze URL via VirusTotal: {str(e)}",
            "contributions": [],
        }


async def init_db():
    """
    Verify Supabase connection on startup.
    """
    if supabase:
        try:
            # Query the system cache anchor to verify connection
            supabase.table("sms_message").select("sms_id").eq("sms_id", "CACHE_SMS").execute()
            print("Supabase HTTP connection: Verified successfully!")
        except Exception as e:
            print(f"Warning: Supabase HTTP connection verification failed: {e}")
    else:
        print("Warning: Supabase client is not initialized.")


async def close_db():
    """
    No active database connection pool to close (using HTTP client).
    """
    pass


def recalculate_cached_url_score(scan_data: dict) -> dict:
    """
    Ensures cached scan results reflect the latest harsher security boost rules based on vendor consensus.
    """
    if not isinstance(scan_data, dict):
        return scan_data

    contributions = scan_data.get("contributions") or []
    malicious_count = sum(1 for c in contributions if "(malicious)" in str(c).lower())
    suspicious_count = sum(1 for c in contributions if "(suspicious)" in str(c).lower())
    current_score = scan_data.get("score") or 0.0

    boosted_score = current_score
    if malicious_count >= 15:
        boosted_score = max(boosted_score, 1.00)
    elif malicious_count >= 10:
        boosted_score = max(boosted_score, 0.95)
    elif malicious_count >= 5:
        boosted_score = max(boosted_score, 0.85)
    elif malicious_count >= 3:
        boosted_score = max(boosted_score, 0.75)
    elif malicious_count >= 2:
        boosted_score = max(boosted_score, 0.65)
    elif malicious_count >= 1:
        boosted_score = max(boosted_score, 0.45)
    elif suspicious_count >= 2:
        boosted_score = max(boosted_score, 0.40)
    elif suspicious_count >= 1:
        boosted_score = max(boosted_score, 0.25)

    if round(boosted_score, 4) != round(current_score, 4):
        scan_data["score"] = round(boosted_score, 4)
        if boosted_score >= 0.45:
            scan_data["verdict"] = "malicious"
        elif boosted_score >= 0.20:
            scan_data["verdict"] = "suspicious"
        verdict = scan_data.get("verdict", "malicious")
        flagged_engines = ", ".join(contributions)
        if flagged_engines:
            scan_data["explanation"] = f"This URL is flagged as {verdict} (threat score: {scan_data['score']:.2f}). Detected by: {flagged_engines}."

    return scan_data


async def lookup_cached_url(url: str) -> dict | None:
    """
    Check the in-memory cache and then the Supabase database for a fresh cached scan of the URL.
    Returns the scan results dict if a fresh match (< 7 days old) is found, otherwise None.
    """
    # 1. Check in-memory cache
    if url in memory_cache:
        print(f"Memory cache hit for URL: {url}")
        return recalculate_cached_url_score(memory_cache[url])

    # 2. Check Supabase database cache (under CACHE_SMS)
    if not supabase:
        return None

    try:
        # Retrieve scan result joined with the url table
        response = supabase.table("url_analysis") \
            .select("is_malicious, scan_result, created_at, url!inner(full_url)") \
            .eq("url.full_url", url) \
            .eq("sms_id", "CACHE_SMS") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
            
        if response.data:
            record = response.data[0]
            created_at_str = record.get("created_at")
            if created_at_str:
                from datetime import datetime, timezone, timedelta
                created_time = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                if datetime.now(timezone.utc) - created_time < timedelta(days=7):
                    scan_data = json.loads(record["scan_result"])
                    scan_data = recalculate_cached_url_score(scan_data)
                    # Store updated score in memory cache
                    memory_cache[url] = scan_data
                    print(f"Database cache hit for URL: {url}")
                    return scan_data

    except Exception as e:
        print(f"Database cache lookup error for {url}: {e}")
    return None


async def save_url_scan_to_db(url: str, sms_id: str, is_malicious: int, scan_result: dict):
    """
    Insert a scan result into public.url_analysis associated with the given sms_id.
    Saves the unique URL components inside public.url table if it doesn't exist yet.
    """
    if not supabase:
        return
    try:
        import uuid
        from urllib.parse import urlparse
        
        # 1. Check if the URL already exists in public.url
        res = supabase.table("url").select("url_id").eq("full_url", url).execute()
        url_db_id = None
        
        if res.data:
            url_db_id = res.data[0]["url_id"]
        else:
            # Insert new URL entry
            url_db_id = str(uuid.uuid4())
            parsed = urlparse(url)
            scheme = parsed.scheme
            host = parsed.hostname or parsed.netloc or ""
            path = parsed.path
            if path == "/":
                path = ""
            query = parsed.query
            
            # Execute insert. In case of concurrent inserts, we handle unique constraint violation gracefully.
            try:
                supabase.table("url").insert({
                    "url_id": url_db_id,
                    "full_url": url,
                    "scheme": scheme,
                    "host": host,
                    "path": path,
                    "query": query
                }).execute()
            except Exception as inner_e:
                print(f"Concurrent URL insert detected, re-fetching URL ID: {inner_e}")
                res_retry = supabase.table("url").select("url_id").eq("full_url", url).execute()
                if res_retry.data:
                    url_db_id = res_retry.data[0]["url_id"]
        
        if not url_db_id:
            raise ValueError("Unable to determine or insert url_id.")
            
        # 2. Insert the analysis record linked to this URL ID
        analysis_id = str(uuid.uuid4())
        scan_result_str = json.dumps(scan_result)
        
        supabase.table("url_analysis").insert({
            "analysis_id": analysis_id,
            "sms_id": sms_id,
            "url_id": url_db_id,
            "is_malicious": is_malicious,
            "scan_result": scan_result_str
        }).execute()
    except Exception as e:
        print(f"Failed to save URL scan to db for sms_id {sms_id}: {e}")




async def save_sms_message_to_db(sender_number: str, message_content: str, is_processed: int) -> str | None:
    """
    Insert a new user SMS message into public.sms_message and return the auto-generated sms_id.
    """
    if not supabase:
        return None
    try:
        response = supabase.table("sms_message").insert({
            "sender_number": sender_number or "UNKNOWN",
            "message_content": message_content,
            "is_processed": is_processed
        }).execute()
        if response.data:
            return str(response.data[0]["sms_id"])
    except Exception as e:
        print(f"Failed to save SMS message to database: {e}")
    return None


async def save_analysis_result_to_db(sms_id: str, ml_prediction: str, ml_confidence: float, dl_prediction: str, dl_confidence: float):
    """
    Insert the CNN prediction and confidence levels into public.analysis_result linked to the given sms_id.
    """
    if not supabase:
        return
    try:
        import uuid
        analysis_id = str(uuid.uuid4())
        supabase.table("analysis_result").insert({
            "analysis_id": analysis_id,
            "sms_id": sms_id,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "dl_prediction": dl_prediction,
            "dl_confidence": dl_confidence
        }).execute()
    except Exception as e:
        print(f"Failed to save analysis result for sms_id {sms_id}: {e}")


async def save_ntc_report_to_db(
    message: str,
    url_analysis: dict,
    ml_score: float,
    dl_score: float,
    final_score: float,
    verdict: str,
    sender: str,
    status: str = "pending"
) -> str | None:
    """
    Insert a smishing incident report into public.ntc_report for NTC regulatory compliance.
    """
    if not supabase:
        return None
    try:
        import uuid
        report_id = str(uuid.uuid4())
        record = {
            "id": report_id,
            "message": message,
            "url_analysis": url_analysis if url_analysis else {},
            "ml_score": float(ml_score),
            "dl_score": float(dl_score),
            "final_score": float(final_score),
            "verdict": str(verdict),
            "sender": sender or "UNKNOWN",
            "status": status
        }
        res = supabase.table("ntc_report").insert(record).execute()
        if res.data:
            print(f"NTC report successfully logged to DB with id {report_id}")
            return report_id
    except Exception as e:
        print(f"Failed to save NTC report to database: {e}")
    return None


async def save_misclassification_report_to_db(
    message: str,
    original_verdict: str,
    original_score: float,
    user_verdict: str,
    sender: str | None = None,
    has_url: bool = False,
    extracted_url: str | None = None,
    report_type: str | None = None,
    user_comment: str | None = None,
    app_version: str | None = None,
    device_id: str | None = None,
    status: str = "pending_review"
) -> str | None:
    """
    Insert a user-submitted misclassification report into public.misclassification_reports.
    Auto-derives report_type ('false_positive' vs 'false_negative') if not explicitly provided.
    """
    if not supabase:
        return None
    try:
        import uuid
        report_id = str(uuid.uuid4())

        # Auto-derive report_type if omitted
        if not report_type:
            user_clean = (user_verdict or "").strip().lower()
            orig_clean = (original_verdict or "").strip().lower()
            if user_clean in ["safe", "benign", "legitimate"]:
                report_type = "false_positive"
            elif user_clean in ["harmful", "suspicious", "smishing", "spam", "malicious"]:
                report_type = "false_negative"
            elif orig_clean in ["safe", "benign"]:
                report_type = "false_negative"
            else:
                report_type = "false_positive"

        record = {
            "id": report_id,
            "sender": sender or "UNKNOWN",
            "message": message,
            "has_url": bool(has_url),
            "extracted_url": extracted_url,
            "original_verdict": str(original_verdict),
            "original_score": float(original_score),
            "user_verdict": str(user_verdict),
            "report_type": str(report_type),
            "user_comment": user_comment,
            "app_version": app_version,
            "device_id": device_id,
            "status": status
        }
        res = supabase.table("misclassification_reports").insert(record).execute()
        if res.data:
            print(f"Misclassification report successfully logged to DB with id {report_id}")
            return report_id
    except Exception as e:
        print(f"Failed to save misclassification report to database: {e}")
    return None


async def prune_expired_records_db():
    """
    Worker task to delete SMS messages and linked logs older than 7 days (168 hours).
    """
    if not supabase:
        return
    try:
        # Attempt to run SQL pruning via RPC function if defined
        supabase.rpc("prune_expired_records").execute()
        print("Database cache pruning RPC: Success.")
    except Exception as e:
        print(f"Database pruning RPC failed: {e}. Falling back to client-side pruning...")
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            supabase.table("sms_message") \
                .delete() \
                .lt("received_timestamp", cutoff) \
                .neq("sms_id", "CACHE_SMS") \
                .execute()
            print("Database cache fallback pruning: Success.")
        except Exception as ex:
            print(f"Fallback database pruning failed: {ex}")


async def get_url_reputations(since_timestamp_ms: Optional[int] = None) -> dict:
    """
    Fetch all cached URL threat reputations from Supabase (under CACHE_SMS anchor),
    optionally filtering entries created after since_timestamp_ms (milliseconds).
    """
    from datetime import datetime, timezone
    
    if not supabase:
        return {
            "total_records": 0,
            "last_synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "urls": []
        }

    try:
        query = supabase.table("url_analysis") \
            .select("scan_result, created_at, url!inner(full_url, host)") \
            .eq("sms_id", "CACHE_SMS")

        if since_timestamp_ms is not None and since_timestamp_ms > 0:
            dt = datetime.fromtimestamp(since_timestamp_ms / 1000.0, tz=timezone.utc)
            iso_str = dt.isoformat()
            query = query.gte("created_at", iso_str)

        response = query.execute()
        urls_list = []
        if response.data:
            for row in response.data:
                scan_data = json.loads(row.get("scan_result") or "{}")
                full_url = scan_data.get("extracted_url") or row.get("url", {}).get("full_url", "")
                host = row.get("url", {}).get("host", "")
                if not host and full_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(full_url)
                    host = parsed.hostname or parsed.netloc or ""

                verdict_val = scan_data.get("verdict")
                if not verdict_val:
                    verdict_val = "benign"

                score_val = scan_data.get("score")
                score_float = float(score_val) if score_val is not None else 0.0

                weight_val = scan_data.get("total_weight")
                weight_float = float(weight_val) if weight_val is not None else 0.0

                explanation_val = scan_data.get("explanation") or ""
                contributions_val = scan_data.get("contributions") or []

                urls_list.append({
                    "extracted_url": full_url,
                    "normalized_host": host,
                    "verdict": str(verdict_val),
                    "score": score_float,
                    "total_weight": weight_float,
                    "explanation": str(explanation_val),
                    "contributions": list(contributions_val)
                })

        last_synced = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "total_records": len(urls_list),
            "last_synced_at": last_synced,
            "urls": urls_list
        }
    except Exception as e:
        print(f"Failed to fetch url reputations from DB: {e}")
        return {
            "total_records": 0,
            "last_synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "urls": []
        }






