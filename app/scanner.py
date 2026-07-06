import asyncio
import base64
import collections
import csv
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

load_dotenv()

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

    for engine_name, engine_data in analysis_results.items():
        category = str(engine_data.get("category", "undetected")).lower()
        score = normalize_category(category)
        weight = weights.get(engine_name, DEFAULT_SOURCE_WEIGHT)
        total_weighted_score += score * weight
        total_weight += weight
        contribution_lines.append(
            f"{engine_name}: category={category}, score={score}, weight={weight}"
        )

    if total_weight == 0:
        raise ValueError("No weighted engines were available to compute a verdict.")

    normalized_score = total_weighted_score / total_weight

    if normalized_score >= 0.65:
        verdict = "malicious"
    elif normalized_score >= 0.35:
        verdict = "suspicious"
    else:
        verdict = "benign"

    return verdict, normalized_score, total_weight, contribution_lines


def format_verdict_explanation(
    verdict: str,
    normalized_score: float,
    total_weighted_score: float,
    total_weight: float,
    contributions: List[str],
) -> str:
    expression = (
        f"weighted_score = {total_weighted_score:.4f} / {total_weight:.4f} = "
        f"{normalized_score:.4f}"
    )
    explanation_lines = [
        "Final verdict explanation:",
        expression,
        f"Final verdict threshold result: {verdict}",
        "Engine contributions:",
    ]
    explanation_lines.extend(contributions)
    return "\n".join(explanation_lines)


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

