"""
KwagoBackend - Retraining Dataset Exporter
-----------------------------------------
Extracts encrypted misclassification reports and SMS messages from Supabase,
decrypts them in-memory using DB_ENCRYPTION_KEY, and exports them into a clean CSV
ready for retraining the Deep Learning (CNN-BiGRU) and Local ML models.

Usage:
    python scripts/export_retraining_data.py [--output output_path.csv] [--status verified|all]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from supabase import create_client
from app.crypto_utils import decrypt_text


def export_data(output_file: str, status_filter: str = "all"):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url or not supabase_key:
        print("[!] Error: SUPABASE_URL and SUPABASE_SECRET_KEY must be set in .env")
        sys.exit(1)

    print(f"[*] Connecting to Supabase at {supabase_url}...")
    supabase = create_client(supabase_url, supabase_key)

    print(f"[*] Fetching misclassification reports (filter: status='{status_filter}')...")
    query = supabase.table("misclassification_reports").select(
        "id, sender, message, user_verdict, original_verdict, report_type, has_url, extracted_url, created_at, status"
    )

    if status_filter != "all":
        query = query.eq("status", status_filter)

    response = query.execute()
    rows = response.data or []
    print(f"    - Found {len(rows)} report(s).")

    decrypted_records = []
    for r in rows:
        decrypted_msg = decrypt_text(r.get("message"))
        decrypted_sender = decrypt_text(r.get("sender"))
        user_verdict = (r.get("user_verdict") or "").strip().lower()

        # Map verdict to binary classification (1 = Phishing/Harmful, 0 = Benign/Safe)
        label = 1 if user_verdict in ["harmful", "phishing", "malicious", "smishing"] else 0

        decrypted_records.append({
            "id": r.get("id"),
            "sender": decrypted_sender,
            "message": decrypted_msg,
            "classification": "Phishing" if label == 1 else "Benign",
            "label": label,
            "original_verdict": r.get("original_verdict"),
            "user_verdict": r.get("user_verdict"),
            "report_type": r.get("report_type"),
            "has_url": r.get("has_url"),
            "extracted_url": r.get("extracted_url"),
            "status": r.get("status"),
            "created_at": r.get("created_at")
        })

    out_path = Path(output_file)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        if decrypted_records:
            writer = csv.DictWriter(f, fieldnames=list(decrypted_records[0].keys()))
            writer.writeheader()
            writer.writerows(decrypted_records)
        else:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "sender", "message", "classification", "label", "original_verdict",
                "user_verdict", "report_type", "has_url", "extracted_url", "status", "created_at"
            ])
            writer.writeheader()

    print(f"[+] Successfully exported and decrypted {len(decrypted_records)} samples to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and decrypt training data from Supabase.")
    parser.add_argument("--output", "-o", default="retraining_data.csv", help="Path to save the output CSV")
    parser.add_argument("--status", "-s", default="all", choices=["all", "verified", "pending_review"], help="Filter by status")
    args = parser.parse_args()

    export_data(args.output, args.status)
