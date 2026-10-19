"""
KwagoBackend - VirusTotal URL Scanner for Threshold Testing Dataset
------------------------------------------------------------------
Extracts links from each SMS in `sms_cleaned.csv`, scans them using
the VirusTotal API v3 with the weighted reputation scoring algorithm,
and appends a `url_scan_result` column.

Features:
- Rate limit (4 requests/minute) auto-pacing and retry loop on 429 / issues.
- Persistent JSON caching (vt_url_cache.json) to eliminate duplicate calls.
- Full multi-link aggregation (malicious dominance) for multi-link SMS.
- Incremental checkpointing: safe to pause/resume anytime without losing progress.
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv

# Resolve project root and load environment
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app import scanner

VT_API_KEY = os.getenv("VIRUS_TOTAL_API_KEY")
URL_PATTERN = re.compile(r'(?:https?://|www\.)[^\s]+', re.IGNORECASE)


def extract_and_clean_urls(text: str) -> List[str]:
    """
    Extracts all URLs from message text, normalizes them, and deduplicates.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_matches = URL_PATTERN.findall(text)
    cleaned_urls = []
    seen = set()

    for u in raw_matches:
        cleaned = u.strip().rstrip(".,;!?)>]'\"")
        if not cleaned:
            continue
        # Ensure scheme for VT API and consistency
        if cleaned.lower().startswith("www."):
            cleaned = "http://" + cleaned
        elif not cleaned.lower().startswith("http://") and not cleaned.lower().startswith("https://"):
            cleaned = "http://" + cleaned

        norm_key = cleaned.lower()
        if norm_key not in seen:
            seen.add(norm_key)
            cleaned_urls.append(cleaned)

    return cleaned_urls


def get_url_id(url: str) -> str:
    """Computes base64-encoded URL representation without padding for VT API v3."""
    return base64.urlsafe_b64encode(url.encode()).decode().strip("=")


class VirusTotalBatchScanner:
    """
    Handles robust VirusTotal scanning with persistent caching and retry loops.
    """

    def __init__(
        self,
        api_key: str,
        cache_file: Path,
        min_request_interval: float = 15.2,
        retry_delay: float = 20.0,
    ):
        self.api_key = api_key
        self.cache_file = cache_file
        self.min_request_interval = min_request_interval
        self.retry_delay = retry_delay
        self.last_request_time = 0.0
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.source_weights = scanner.load_source_weights()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[*] Loaded {len(data)} cached URL scans from {self.cache_file.name}")
                    return data
            except Exception as e:
                print(f"[!] Warning loading cache file: {e}. Starting fresh cache.")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[!] Warning saving cache file: {e}")

    async def _rate_limit_throttle(self):
        """Ensures at least min_request_interval seconds between consecutive VT API queries."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_sec = self.min_request_interval - elapsed
            await asyncio.sleep(wait_sec)
        self.last_request_time = time.time()

    async def _vt_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Executes an HTTP request to VirusTotal with continuous retry on 429 or transient errors.
        """
        headers = kwargs.pop("headers", {})
        headers["x-apikey"] = self.api_key

        while True:
            await self._rate_limit_throttle()
            try:
                async with session.request(method, url, headers=headers, **kwargs) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except Exception:
                        data = {}

                    # Rate limited by VirusTotal
                    if status == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") == "QuotaExceededError"):
                        msg = data.get("error", {}).get("message", "Quota / minute limit exceeded")
                        print(f"    [VT RATE LIMIT 429] {msg}. Retrying in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                        continue

                    # Server-side transient issues (500, 502, 503, 504)
                    if status in (500, 502, 503, 504):
                        print(f"    [VT SERVER ERROR {status}]. Retrying in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                        continue

                    return status, data

            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                print(f"    [NETWORK ERROR] {net_err}. Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
            except Exception as unk_err:
                print(f"    [UNEXPECTED ERROR] {unk_err}. Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)

    async def scan_single_url(self, session: aiohttp.ClientSession, target_url: str) -> Dict[str, Any]:
        """
        Scans a single URL or retrieves it from cache, computing the weighted threat verdict.
        """
        norm_key = target_url.strip().lower()
        if norm_key in self.cache:
            return self.cache[norm_key]

        print(f"  [*] Scanning URL with VirusTotal: {target_url}")
        url_id = get_url_id(target_url)

        # 1. Check if report already exists in VirusTotal
        status, report_data = await self._vt_request(
            session, "GET", f"https://www.virustotal.com/api/v3/urls/{url_id}"
        )

        analysis_results = None
        if status == 200:
            analysis_results = scanner.extract_analysis_results(report_data)

        # 2. If report was not found (404), submit URL for live analysis
        if not analysis_results:
            post_status, post_data = await self._vt_request(
                session, "POST", "https://www.virustotal.com/api/v3/urls", data={"url": target_url}
            )
            scan_id = post_data.get("data", {}).get("id") if isinstance(post_data, dict) else None

            if scan_id:
                # Poll analysis until completion or results are ready
                max_polls = 15
                for poll_i in range(max_polls):
                    poll_status, poll_data = await self._vt_request(
                        session, "GET", f"https://www.virustotal.com/api/v3/analyses/{scan_id}"
                    )
                    if poll_status == 200:
                        attributes = poll_data.get("data", {}).get("attributes", {})
                        if attributes.get("status") == "completed" or attributes.get("results"):
                            analysis_results = scanner.extract_analysis_results(poll_data)
                            break
                    await asyncio.sleep(3)

        # 3. Calculate weighted score
        if analysis_results:
            verdict, normalized_score, total_weight, contributions = scanner.compute_weighted_verdict(
                analysis_results, self.source_weights
            )
            result = {
                "has_url": True,
                "extracted_url": target_url,
                "score": float(normalized_score),
                "verdict": verdict,
                "total_weight": float(total_weight),
                "contributions": contributions,
            }
        else:
            # Fallback if VT produced no analysis results
            result = {
                "has_url": True,
                "extracted_url": target_url,
                "score": 0.0,
                "verdict": "undetected",
                "total_weight": 0.0,
                "contributions": [],
            }

        # Cache result permanently
        self.cache[norm_key] = result
        self._save_cache()
        print(f"    --> Result for {target_url}: score={result['score']}, verdict={result['verdict']}")
        return result

    async def scan_sms_message(self, session: aiohttp.ClientSession, text: str) -> Optional[float]:
        """
        Extracts links from message, scans them, and returns the aggregated weighted score.
        If no link is present, returns None (to be left empty in the CSV).
        """
        urls = extract_and_clean_urls(text)
        if not urls:
            return None

        per_url_scans = []
        for u in urls:
            res = await self.scan_single_url(session, u)
            per_url_scans.append(res)

        aggregated = scanner.aggregate_multi_url_results(per_url_scans)
        final_score = aggregated.get("score")
        if final_score is not None:
            return round(float(final_score), 4)
        return 0.0


async def process_dataset(
    input_csv: Path,
    output_csv: Path,
    cache_file: Path,
    backup: bool = True,
):
    if not VT_API_KEY:
        print("[!] ERROR: VIRUS_TOTAL_API_KEY is not configured in .env file.")
        sys.exit(1)

    if not input_csv.exists():
        print(f"[!] ERROR: Input dataset not found at {input_csv}")
        sys.exit(1)

    print(f"[*] Reading dataset from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"[*] Total rows in dataset: {len(rows)}")

    # Add url_scan_result column if not present
    target_field = "url_scan_result"
    if target_field not in fieldnames:
        fieldnames.append(target_field)

    # Backup original input CSV if requested
    if backup and input_csv == output_csv:
        backup_path = input_csv.with_suffix(".csv.bak")
        if not backup_path.exists():
            import shutil
            shutil.copyfile(input_csv, backup_path)
            print(f"[*] Created backup copy at: {backup_path.name}")

    # Inspect messages needing scan
    msgs_with_urls = 0
    unique_urls_to_scan = set()
    for r in rows:
        urls = extract_and_clean_urls(r.get("Message", ""))
        if urls:
            msgs_with_urls += 1
            for u in urls:
                unique_urls_to_scan.add(u.lower())

    print(f"[*] Messages with URLs: {msgs_with_urls} / {len(rows)}")
    print(f"[*] Unique URLs to evaluate: {len(unique_urls_to_scan)}")

    vt_scanner = VirusTotalBatchScanner(
        api_key=VT_API_KEY,
        cache_file=cache_file,
        min_request_interval=15.2,
        retry_delay=20.0,
    )

    already_cached = sum(1 for u in unique_urls_to_scan if u in vt_scanner.cache)
    print(f"[*] Unique URLs already in cache: {already_cached} / {len(unique_urls_to_scan)}")
    remaining = len(unique_urls_to_scan) - already_cached
    est_minutes = round((remaining * 15.2) / 60, 1)
    print(f"[*] URLs requiring live VT scan: {remaining} (~{est_minutes} minutes at 4 req/min)")

    def write_current_progress(current_rows: List[Dict[str, Any]]):
        for attempt in range(5):
            try:
                with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(current_rows)
                return
            except PermissionError:
                time.sleep(1.0)
            except Exception as e:
                print(f"[!] Warning writing progress to {output_csv.name}: {e}")
                time.sleep(1.0)

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for idx, row in enumerate(rows, 1):
            msg_text = row.get("Message", "")
            urls = extract_and_clean_urls(msg_text)

            # If message has no link, leave empty string
            if not urls:
                row[target_field] = ""
            else:
                existing_val = row.get(target_field)
                # If row was already scanned and valid, keep it
                if existing_val is not None and str(existing_val).strip() != "":
                    try:
                        row[target_field] = float(existing_val)
                    except ValueError:
                        pass
                else:
                    score = await vt_scanner.scan_sms_message(session, msg_text)
                    row[target_field] = score if score is not None else ""

            # Checkpoint progress to disk every 25 rows or whenever a URL is scanned
            if urls or idx % 25 == 0 or idx == len(rows):
                write_current_progress(rows)
                if idx % 100 == 0 or urls:
                    print(f"[{idx}/{len(rows)}] Rows processed. (Target: {output_csv.name})")

    # Final write
    write_current_progress(rows)
    print(f"\n[+] Successfully generated CSV with '{target_field}' column!")
    print(f"    Output saved to: {output_csv.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan URLs in SMS dataset with VirusTotal and append weighted scan results."
    )
    default_dir = PROJECT_ROOT / "Threshold testing"
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "sms_cleaned.csv",
        help="Path to input SMS CSV (default: Threshold testing/sms_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "sms_cleaned.csv",
        help="Path to output SMS CSV (default: Threshold testing/sms_cleaned.csv)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_dir / "vt_url_cache.json",
        help="Path to persistent JSON cache (default: Threshold testing/vt_url_cache.json)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup creation (.csv.bak) when overwriting input.",
    )

    args = parser.parse_args()
    asyncio.run(
        process_dataset(
            input_csv=args.input,
            output_csv=args.output,
            cache_file=args.cache,
            backup=not args.no_backup,
        )
    )


if __name__ == "__main__":
    main()
