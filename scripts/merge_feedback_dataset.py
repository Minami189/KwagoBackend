"""
KwagoBackend - Dataset Merge & Curation Tool for Retraining
------------------------------------------------------------
Merges verified feedback from `retraining_data.csv` (exported from Supabase)
into the primary training dataset `Threshold testing/sms_cleaned.csv`.

Deduplicates text to ensure no exact duplicates are introduced,
and maintains proper column formatting for the retraining notebooks.

Usage:
    python scripts/merge_feedback_dataset.py [--input retraining_data.csv] [--base "Threshold testing/sms_cleaned.csv"] [--output "Threshold testing/sms_cleaned.csv"]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def merge_datasets(input_csv: str, base_csv: str, output_csv: str):
    input_path = Path(input_csv)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    base_path = Path(base_csv)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path

    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not input_path.exists():
        print(f"[!] Error: Feedback data file not found at {input_path.resolve()}")
        return

    if not base_path.exists():
        print(f"[!] Error: Base dataset file not found at {base_path.resolve()}")
        return

    # 1. Read Base Dataset
    print(f"[*] Reading base dataset from {base_path.name}...")
    existing_messages = set()
    base_rows = []

    with base_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or [
            "SIM Network", "Date and Time", "Sender", "Message", "Language", "Classification", "url_scan_result"
        ]
        for row in reader:
            msg = row.get("Message", "").strip()
            if msg:
                existing_messages.add(msg.lower())
            base_rows.append(row)

    print(f"    - Existing samples: {len(base_rows):,}")

    # 2. Read Feedback Data
    print(f"[*] Reading exported feedback from {input_path.name}...")
    feedback_rows = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        feedback_rows = list(reader)

    print(f"    - Retrieved feedback samples: {len(feedback_rows):,}")

    # 3. Filter and Format New Records
    new_added = 0
    duplicates_skipped = 0

    for fb in feedback_rows:
        msg = fb.get("message", "").strip()
        if not msg:
            continue

        if msg.lower() in existing_messages:
            duplicates_skipped += 1
            continue

        existing_messages.add(msg.lower())

        classification = fb.get("classification") or ("Phishing" if fb.get("label") == "1" else "Benign")
        sender = fb.get("sender") or "Feedback"
        date_str = fb.get("created_at") or datetime.now().strftime("%d %B %Y, %H:%M")

        new_row = {
            "SIM Network": "",
            "Date and Time": date_str,
            "Sender": sender,
            "Message": msg,
            "Language": "English",
            "Classification": classification,
            "url_scan_result": "1.0" if classification == "Phishing" and fb.get("has_url") == "True" else "0.0"
        }
        base_rows.append(new_row)
        new_added += 1

    # 4. Save Merged Dataset
    print(f"[*] Saving merged dataset with {len(base_rows):,} total records to {output_path.name}...")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)

    print("\n" + "=" * 50)
    print(" DATASET MERGE SUMMARY")
    print("=" * 50)
    print(f" [+] New samples added:       {new_added:,}")
    print(f" [-] Duplicates skipped:      {duplicates_skipped:,}")
    print(f" [=] Total updated dataset:   {len(base_rows):,} samples")
    print(f" [*] Saved to:                {output_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge decrypted feedback into training dataset.")
    parser.add_argument("--input", "-i", default="retraining_data.csv", help="Decrypted feedback CSV path")
    parser.add_argument("--base", "-b", default="Threshold testing/sms_cleaned.csv", help="Original base dataset path")
    parser.add_argument("--output", "-o", default="Threshold testing/sms_cleaned.csv", help="Output destination")
    args = parser.parse_args()

    merge_datasets(args.input, args.base, args.output)
