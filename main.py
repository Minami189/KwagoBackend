import asyncio
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
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


async def get_scan_results(
    scan_id: str, max_retries: int = 10, wait_seconds: int = 3
) -> dict:
    if not VIRUS_TOTAL_API_KEY:
        raise RuntimeError("VIRUS_TOTAL_API_KEY is not set in .env")

    headers = {
        "x-apikey": VIRUS_TOTAL_API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_retries + 1):
            async with session.get(
                f"https://www.virustotal.com/api/v3/analyses/{scan_id}",
                headers=headers,
            ) as response:
                result = await response.json()

            data = result.get("data", {})
            attributes = data.get("attributes", {})
            status = attributes.get("status")
            if status == "completed":
                print(f"Analysis completed after {attempt} attempt(s).")
                print("Scan results:", result)
                return result

            if attempt == max_retries:
                print(
                    f"Analysis did not complete after {max_retries} attempts;"
                    f" returning the latest available response."
                )
                print("Scan results:", result)
                return result

            print(
                f"Analysis status: {status}. Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_retries}..."
            )
            await asyncio.sleep(wait_seconds)


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


async def main() -> None:
    url_input = input(
        "Enter the URL or string to scan, or type 'quota' to check API quota: "
    ).strip()
    if url_input.lower() == "quota":
        quota_data = await check_api_quota()
        print("\n=== API Quota ===")
        print(format_quota_response(quota_data))
        return

    try:
        quota_data = await check_api_quota()
        print("\n=== API Quota ===")
        print(format_quota_response(quota_data))
    except RuntimeError as error:
        print(f"\n=== API Quota ===\nUnable to fetch quota: {error}")

    scan_result = await scan_url(url_input)
    scan_id = scan_result.get("data", {}).get("id")
    if not scan_id:
        print("Unable to obtain a scan id from VirusTotal response.")
        return

    analysis_response = await get_scan_results(scan_id)
    analysis_results = extract_analysis_results(analysis_response)
    if not analysis_results:
        print("No analysis engine results were found in the VirusTotal response.")
        return

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

    print("\n=== Final Verdict ===")
    print(explanation)


if __name__ == "__main__":
    asyncio.run(main())