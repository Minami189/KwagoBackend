# Maintenance & Retraining Scripts

This directory contains standalone Python utilities for database maintenance, data export, and dataset curation.

---

## 1. `export_retraining_data.py`
Connects to Supabase, queries user misclassification reports, decrypts the messages in memory using `DB_ENCRYPTION_KEY`, and exports them to a clean CSV.

### Usage:
```bash
# Export all feedback records
python scripts/export_retraining_data.py --output retraining_data.csv --status all

# Export only verified feedback records
python scripts/export_retraining_data.py --output verified_data.csv --status verified
```

---

## 2. `merge_feedback_dataset.py`
Merges the exported feedback CSV into the main training dataset (`Threshold testing/sms_cleaned.csv`). Automatically removes duplicates and formats columns for the training notebooks.

### Usage:
```bash
python scripts/merge_feedback_dataset.py --input retraining_data.csv --base "Threshold testing/sms_cleaned.csv"
```
