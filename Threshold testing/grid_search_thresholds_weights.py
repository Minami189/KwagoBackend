"""
KwagoBackend - Exhaustive Grid Sweep: All Thresholds, All Cases, All Ensemble Weights
-------------------------------------------------------------------------------------
Evaluates performance across:
1. Real VirusTotal URL Threat Scores (loaded from `sms_cleaned.csv` `url_scan_result` column).
2. Diverse 3-Layer Ensemble Weights testing different URL weight circumstances:
   - w_URL from 0.05 to 0.70 (5% to 70% URL influence)
   - Balanced, DL-dominant, and ML-dominant splits
   - Both Pure Linear Ensemble and Malicious Dominance Override
3. Rule-Based URL Override Architecture:
   - URL threat score is NOT part of the linear ensemble weight.
   - Message text is evaluated by 2-layer (DL + ML) text ensemble.
   - Rule-based URL trigger: If a message has a URL and its VirusTotal threat score >= theta_url,
     automatically classify the entire message as Harmful (Phishing), testing theta_url from 0.20 to 1.00.
4. Standalone Benchmarks: Pure DL, Pure ML, 2-Layer Text Ensemble.

Exports complete results to: all_thresholds_cases_weights_results.csv
"""

import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Ensure project root and testing dir are in Python module search path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TESTING_DIR = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = TESTING_DIR / "RFXGB Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import onnxruntime as ort
import predict_interactive as pi
from app import scanner


def load_dataset(csv_path: str = None) -> Tuple[List[Dict[str, str]], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads dataset from sms_cleaned.csv including the real VirusTotal `url_scan_result` column.
    """
    path = Path(csv_path) if csv_path else (TESTING_DIR / "sms_cleaned.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path.resolve()}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    texts = [r.get("Message", "") for r in rows]
    y_true = np.array([
        1 if r.get("Classification", "").strip().lower() == "phishing" else 0
        for r in rows
    ], dtype=int)

    url_scores = np.zeros(len(rows), dtype=np.float32)
    has_url = np.zeros(len(rows), dtype=bool)

    for i, r in enumerate(rows):
        val = r.get("url_scan_result", "")
        if val is not None and str(val).strip() != "":
            try:
                url_scores[i] = float(val)
                has_url[i] = True
            except ValueError:
                has_url[i] = False
        else:
            has_url[i] = False

    print(f"[*] Loaded dataset: {len(rows):,} total records")
    print(f"    - Phishing count: {int(np.sum(y_true == 1)):,} ({np.mean(y_true == 1)*100:.1f}%)")
    print(f"    - Benign count:   {int(np.sum(y_true == 0)):,} ({np.mean(y_true == 0)*100:.1f}%)")
    print(f"    - Messages with URL (real VT scan): {int(np.sum(has_url)):,} ({np.mean(has_url)*100:.1f}%)")
    print(f"    - Text-only messages (no URL):     {int(np.sum(~has_url)):,} ({np.mean(~has_url)*100:.1f}%)")

    return rows, texts, y_true, has_url, url_scores


def generate_dl_scores(texts: List[str], classifier: scanner.SMSClassifier) -> np.ndarray:
    print("\n[*] Generating Deep Learning (CNN-BiGRU) scores via ONNX Runtime...")
    t0 = time.time()
    sequences = []
    for t in texts:
        cleaned = scanner.clean_text(t)
        words = cleaned.split()
        seq = [
            classifier.word_index.get(w, classifier.oov_index)
            for w in words
            if (classifier.num_words is None or
                classifier.word_index.get(w, classifier.oov_index) < classifier.num_words)
        ]
        if len(seq) > classifier.max_len:
            seq = seq[:classifier.max_len]
        else:
            seq = seq + [0] * (classifier.max_len - len(seq))
        sequences.append(seq)

    input_arr = np.array(sequences, dtype=np.float32)
    raw_outputs = classifier.session.run(None, {classifier.input_name: input_arr})
    dl_scores = raw_outputs[0].flatten()
    print(f"    DL inference finished in {time.time() - t0:.2f}s ({len(texts)} samples)")
    return dl_scores


def generate_real_ml_scores(texts: List[str]) -> np.ndarray:
    print("\n[*] Generating Authentic Local ML (RF + XGBoost) scores via ONNX Runtime...")
    t0 = time.time()

    scaler_path = str(MODEL_DIR / "scaler.onnx")
    tfidf_path = str(MODEL_DIR / "tfidf.onnx")
    rf_path = str(MODEL_DIR / "rf_model.onnx")
    xgb_path = str(MODEL_DIR / "xgboost_model.onnx")

    scaler_sess = ort.InferenceSession(scaler_path)
    tfidf_sess = ort.InferenceSession(tfidf_path)
    rf_sess = ort.InferenceSession(rf_path)
    xgb_sess = ort.InferenceSession(xgb_path)

    batch_size = 500
    all_combined = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        cleaned = [pi.clean_text(t) for t in batch]
        prep = [pi.preprocess_text(c) for c in cleaned]
        raw_num = np.array([pi.extract_numerical_features(t) for t in batch], dtype=np.float32)

        scaled_num = scaler_sess.run(None, {"num_input": raw_num})[0]
        text_arr = np.array([[p] for p in prep], dtype=object)
        text_tfidf = tfidf_sess.run(None, {"text_input": text_arr})[0]

        comb = np.hstack([text_tfidf, scaled_num]).astype(np.float32)
        all_combined.append(comb)

    X_all = np.vstack(all_combined)

    rf_raw = rf_sess.run(None, {"float_input": X_all})[1][:, 1]
    rf_probs = pi.sigmoid(rf_raw)
    xgb_probs = xgb_sess.run(None, {"float_input": X_all})[1][:, 1]

    ml_scores = (0.75 * rf_probs) + (0.25 * xgb_probs)
    print(f"    Local ML inference finished in {time.time() - t0:.2f}s ({len(texts)} samples)")
    return ml_scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    youden_j = recall - fpr
    cost_index = (fp * 1.0 + fn * 5.0) / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "youden_j": round(youden_j, 4),
        "cost_index": round(cost_index, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_samples": total
    }


def run_exhaustive_sweep():
    output_csv = str(TESTING_DIR / "all_thresholds_cases_weights_results.csv")

    rows, texts, y_true, has_url, url_scores = load_dataset()
    classifier = scanner.SMSClassifier()

    dl_scores = generate_dl_scores(texts, classifier)
    ml_scores = generate_real_ml_scores(texts)

    thresholds = [round(th, 2) for th in np.arange(0.10, 0.95, 0.05)]

    # 2-Layer Weight Configurations (w_DL, w_ML)
    two_layer_weights = [
        (1.00, 0.00, "100% DL (Pure DL)"),
        (0.90, 0.10, "90% DL / 10% ML"),
        (0.80, 0.20, "80% DL / 20% ML"),
        (0.75, 0.25, "75% DL / 25% ML"),
        (round(2.0/3.0, 4), round(1.0/3.0, 4), "66.7% DL / 33.3% ML (Previous Baseline)"),
        (0.60, 0.40, "60% DL / 40% ML"),
        (0.50, 0.50, "50% DL / 50% ML (Balanced Baseline)"),
        (0.40, 0.60, "40% DL / 60% ML"),
        (round(1.0/3.0, 4), round(2.0/3.0, 4), "33.3% DL / 66.7% ML"),
        (0.20, 0.80, "20% DL / 80% ML"),
        (0.10, 0.90, "10% DL / 90% ML"),
        (0.00, 1.00, "100% ML (Pure ML)"),
    ]

    # Expanded 3-Layer Weight Configurations (w_DL, w_ML, w_URL) testing diverse URL weights
    three_layer_weights = [
        # --- Light URL Weights (5% to 15%) ---
        (0.475, 0.475, 0.05, "47.5% DL / 47.5% ML / 5% URL (Very Light URL)"),
        (0.636, 0.314, 0.05, "63.6% DL / 31.4% ML / 5% URL (DL Heavy, 5% URL)"),
        (0.314, 0.636, 0.05, "31.4% DL / 63.6% ML / 5% URL (ML Heavy, 5% URL)"),
        (0.45, 0.45, 0.10, "45% DL / 45% ML / 10% URL"),
        (0.60, 0.30, 0.10, "60% DL / 30% ML / 10% URL"),
        (0.30, 0.60, 0.10, "30% DL / 60% ML / 10% URL"),
        (0.425, 0.425, 0.15, "42.5% DL / 42.5% ML / 15% URL"),
        (0.70, 0.15, 0.15, "70% DL / 15% ML / 15% URL"),
        (0.15, 0.70, 0.15, "15% DL / 70% ML / 15% URL"),

        # --- Moderate URL Weights (20% to 30%) ---
        (0.40, 0.40, 0.20, "40% DL / 40% ML / 20% URL"),
        (0.60, 0.20, 0.20, "60% DL / 20% ML / 20% URL"),
        (0.20, 0.60, 0.20, "20% DL / 60% ML / 20% URL"),
        (0.25, 0.50, 0.25, "25% DL / 50% ML / 25% URL (Production Current Baseline)"),
        (0.50, 0.25, 0.25, "50% DL / 25% ML / 25% URL (Previous Baseline)"),
        (0.375, 0.375, 0.25, "37.5% DL / 37.5% ML / 25% URL (Balanced Text)"),
        (0.35, 0.35, 0.30, "35% DL / 35% ML / 30% URL"),
        (0.40, 0.30, 0.30, "40% DL / 30% ML / 30% URL"),
        (0.60, 0.10, 0.30, "60% DL / 10% ML / 30% URL"),
        (round(1.0/3.0, 4), round(1.0/3.0, 4), round(1.0/3.0, 4), "33.3% DL / 33.3% ML / 33.3% URL (Equal 3-Way)"),

        # --- Substantial URL Weights (35% to 50%) ---
        (0.50, 0.15, 0.35, "50% DL / 15% ML / 35% URL"),
        (0.30, 0.30, 0.40, "30% DL / 30% ML / 40% URL"),
        (0.40, 0.20, 0.40, "40% DL / 20% ML / 40% URL (Heavy URL)"),
        (0.20, 0.40, 0.40, "20% DL / 40% ML / 40% URL (ML Heavy, 40% URL)"),
        (0.25, 0.25, 0.50, "25% DL / 25% ML / 50% URL (Dominant URL)"),
        (0.35, 0.15, 0.50, "35% DL / 15% ML / 50% URL"),
        (0.15, 0.35, 0.50, "15% DL / 35% ML / 50% URL"),

        # --- High URL Weights (60% to 70%) ---
        (0.20, 0.20, 0.60, "20% DL / 20% ML / 60% URL"),
        (0.15, 0.15, 0.70, "15% DL / 15% ML / 70% URL"),
    ]

    # Rule-Based URL Threat Thresholds (theta_url) to test
    url_rule_thresholds = [
        0.20,  # VT Suspicious boundary
        0.25,  # Single suspicious vendor
        0.30,
        0.35,
        0.40,  # Single trusted vendor boost floor
        0.45,  # VT Malicious boundary / Single malicious vendor
        0.50,
        0.60,
        0.65,  # Multi-malicious boost floor
        0.75,  # 3+ malicious vendors
        0.85,  # 5+ malicious vendors
        0.95,  # 10+ malicious vendors
        1.00,  # 15+ malicious vendors / full consensus
    ]

    records = []
    print(f"\n[*] Running exhaustive grid sweep across all thresholds, weights, and URL decision models...")

    # =========================================================================
    # CASE 1: Text-Only Messages (N = 3,989)
    # =========================================================================
    mask_c1 = ~has_url
    y_c1 = y_true[mask_c1]
    dl_c1 = dl_scores[mask_c1]
    ml_c1 = ml_scores[mask_c1]

    for w_dl, w_ml, w_label in two_layer_weights:
        ensemble_score = (w_dl * dl_c1) + (w_ml * ml_c1)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c1, y_pred)
            records.append({
                "case_id": "Case 1",
                "case_name": "Text Only (No URL)",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": 0.0,
                "url_rule_threshold": "",
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # CASE 2: URL Subset - Evaluated Solely with Text Models (N = 485)
    # =========================================================================
    mask_c2 = has_url
    y_c2 = y_true[mask_c2]
    dl_c2 = dl_scores[mask_c2]
    ml_c2 = ml_scores[mask_c2]
    url_c2 = url_scores[mask_c2]

    for w_dl, w_ml, w_label in two_layer_weights:
        ensemble_score = (w_dl * dl_c2) + (w_ml * ml_c2)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c2, y_pred)
            records.append({
                "case_id": "Case 2",
                "case_name": "URL Subset (Text Models Only)",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": 0.0,
                "url_rule_threshold": "",
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # FULL DATASET: REAL VIRUSTOTAL 3-LAYER LINEAR ENSEMBLE (N = 4,474)
    # URL score directly weighted in 3-layer decision without manual overrides
    # =========================================================================
    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        full_ensemble = np.zeros(len(texts), dtype=np.float32)
        w_text_tot = w_dl + w_ml
        for i in range(len(texts)):
            if not has_url[i]:
                full_ensemble[i] = (w_dl * dl_scores[i] + w_ml * ml_scores[i]) / w_text_tot
            else:
                full_ensemble[i] = (w_dl * dl_scores[i]) + (w_ml * ml_scores[i]) + (w_url * url_scores[i])

        for th in thresholds:
            y_pred = (full_ensemble >= th).astype(int)
            m = compute_metrics(y_true, y_pred)
            records.append({
                "case_id": "Real VT 3-Layer Linear",
                "case_name": "Real VT 3-Layer Linear Ensemble",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "url_rule_threshold": "",
                "weight_label": f"Linear | {w_label}",
                "threshold": th,
                **m
            })

    # =========================================================================
    # FULL DATASET: REAL VIRUSTOTAL 3-LAYER ENSEMBLE WITH MALICIOUS OVERRIDE (N = 4,474)
    # If real VT score >= 0.45, final ensemble score is floored at 0.85
    # =========================================================================
    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        full_ensemble = np.zeros(len(texts), dtype=np.float32)
        w_text_tot = w_dl + w_ml
        for i in range(len(texts)):
            if not has_url[i]:
                full_ensemble[i] = (w_dl * dl_scores[i] + w_ml * ml_scores[i]) / w_text_tot
            else:
                s_lin = (w_dl * dl_scores[i]) + (w_ml * ml_scores[i]) + (w_url * url_scores[i])
                if url_scores[i] >= 0.45:
                    full_ensemble[i] = max(s_lin, 0.85)
                else:
                    full_ensemble[i] = s_lin

        for th in thresholds:
            y_pred = (full_ensemble >= th).astype(int)
            m = compute_metrics(y_true, y_pred)
            records.append({
                "case_id": "Real VT 3-Layer Override",
                "case_name": "Real VT 3-Layer with Malicious Override",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "url_rule_threshold": 0.45,
                "weight_label": f"Override (>=0.45->0.85) | {w_label}",
                "threshold": th,
                **m
            })

    # =========================================================================
    # FULL DATASET: RULE-BASED URL OVERRIDE ARCHITECTURE (N = 4,474)
    # URL is NOT part of the linear ensemble weights.
    # Text is evaluated via 2-layer ensemble: (w_dl * DL + w_ml * ML).
    # If message has URL and URL threat score >= theta_url -> PREDICT HARMFUL (1).
    # Else -> PREDICT (Text Ensemble >= threshold).
    # =========================================================================
    rule_base_text_configs = [
        (0.50, 0.50, "50% DL / 50% ML Text Base"),
        (round(2.0/3.0, 4), round(1.0/3.0, 4), "66.7% DL / 33.3% ML Text Base"),
        (round(1.0/3.0, 4), round(2.0/3.0, 4), "33.3% DL / 66.7% ML Text Base"),
        (0.75, 0.25, "75% DL / 25% ML Text Base"),
        (0.25, 0.75, "25% DL / 75% ML Text Base"),
        (1.00, 0.00, "100% DL Text Base"),
        (0.00, 1.00, "100% ML Text Base"),
    ]

    for w_dl, w_ml, t_label in rule_base_text_configs:
        text_score = (w_dl * dl_scores) + (w_ml * ml_scores)

        for theta_url in url_rule_thresholds:
            url_triggered = (has_url & (url_scores >= theta_url))

            for th in thresholds:
                # Rule: Automatic Harmful if URL trigger fires, otherwise follow text threshold
                y_pred = (url_triggered | (text_score >= th)).astype(int)
                m = compute_metrics(y_true, y_pred)
                records.append({
                    "case_id": "Rule-Based URL Override",
                    "case_name": f"Rule-Based Override (URL >= {theta_url:.2f} -> Harmful)",
                    "weights_dl": w_dl,
                    "weights_ml": w_ml,
                    "weights_url": 0.0,
                    "url_rule_threshold": theta_url,
                    "weight_label": f"Rule (URL>={theta_url:.2f}) + [{t_label}]",
                    "threshold": th,
                    **m
                })

    # =========================================================================
    # STANDALONE MODELS (FULL DATASET, N = 4,474)
    # =========================================================================
    for th in thresholds:
        y_pred = (dl_scores >= th).astype(int)
        m = compute_metrics(y_true, y_pred)
        records.append({
            "case_id": "Pure DL Standalone",
            "case_name": "100% DL Standalone (CNN-BiGRU)",
            "weights_dl": 1.0,
            "weights_ml": 0.0,
            "weights_url": 0.0,
            "url_rule_threshold": "",
            "weight_label": "100% DL (Pure DL)",
            "threshold": th,
            **m
        })

    for th in thresholds:
        y_pred = (ml_scores >= th).astype(int)
        m = compute_metrics(y_true, y_pred)
        records.append({
            "case_id": "Pure ML Standalone",
            "case_name": "100% ML Standalone (RF + XGBoost)",
            "weights_dl": 0.0,
            "weights_ml": 1.0,
            "weights_url": 0.0,
            "url_rule_threshold": "",
            "weight_label": "100% ML (Pure ML)",
            "threshold": th,
            **m
        })

    # Write all records to CSV
    output_path = Path(output_csv)
    fieldnames = [
        "case_id", "case_name", "weight_label", "weights_dl", "weights_ml", "weights_url",
        "url_rule_threshold", "threshold", "accuracy", "precision", "recall", "specificity",
        "f1", "youden_j", "fpr", "fnr", "cost_index", "tp", "tn", "fp", "fn", "total_samples"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[OK] Grid sweep complete! {len(records):,} total configurations computed and saved to:")
    print(f"     {output_path.resolve()}\n")

    return records


def print_leaderboard(records: List[Dict[str, Any]]):
    print("=" * 105)
    print(" LEADERBOARD: TOP-PERFORMING CONFIGURATIONS BY ARCHITECTURE & CASE")
    print("=" * 105)

    unique_cases = list(dict.fromkeys(r["case_id"] for r in records))

    for cid in unique_cases:
        case_records = [r for r in records if r["case_id"] == cid]
        cname = case_records[0]["case_name"]
        total_n = case_records[0]["total_samples"]

        best_acc = max(case_records, key=lambda x: (x["accuracy"], x["f1"]))
        best_f1 = max(case_records, key=lambda x: (x["f1"], x["youden_j"]))
        best_j = max(case_records, key=lambda x: (x["youden_j"], x["f1"]))
        best_cost = min(case_records, key=lambda x: x["cost_index"])

        print(f"\n[{cid}] (N = {total_n:,})")
        print(f"  * BEST ACCURACY:")
        print(f"    - Weight / Rule: {best_acc['weight_label']}")
        print(f"    - Threshold:     {best_acc['threshold']:.2f}")
        print(f"    - Metrics:       Acc = {best_acc['accuracy']*100:.2f}% | F1 = {best_acc['f1']:.4f} | Prec = {best_acc['precision']*100:.2f}% | Recall = {best_acc['recall']*100:.2f}%")
        print(f"    - Errors:        FP = {best_acc['fp']} | FN = {best_acc['fn']}")

        print(f"  * BEST F1 HARMONIC:")
        print(f"    - Weight / Rule: {best_f1['weight_label']}")
        print(f"    - Threshold:     {best_f1['threshold']:.2f}")
        print(f"    - Metrics:       F1 = {best_f1['f1']:.4f} | Acc = {best_f1['accuracy']*100:.2f}% | Youden J = {best_f1['youden_j']:.4f}")

        print(f"  * LOWEST COST INDEX (Missing Phishing is 5x False Alarm):")
        print(f"    - Weight / Rule: {best_cost['weight_label']}")
        print(f"    - Threshold:     {best_cost['threshold']:.2f}")
        print(f"    - Cost Index:    {best_cost['cost_index']:.4f} (FP = {best_cost['fp']}, FN = {best_cost['fn']})")


def print_url_architecture_comparison(records: List[Dict[str, Any]]):
    print("\n" + "=" * 115)
    print(" COMPREHENSIVE ARCHITECTURE BENCHMARK: LINEAR WEIGHTED vs. RULE-BASED OVERRIDE vs. STANDALONE")
    print("=" * 115)

    pure_dl = [r for r in records if r["case_id"] == "Pure DL Standalone"]
    pure_ml = [r for r in records if r["case_id"] == "Pure ML Standalone"]
    linear_vt = [r for r in records if r["case_id"] == "Real VT 3-Layer Linear"]
    override_vt = [r for r in records if r["case_id"] == "Real VT 3-Layer Override"]
    rule_override = [r for r in records if r["case_id"] == "Rule-Based URL Override"]

    def peak(recs):
        return max(recs, key=lambda x: (x["f1"], x["accuracy"]))

    header = f"{'Architecture':<40} | {'Optimal Config / Rule':<38} | {'Th':<5} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'FP':<4} | {'FN':<4}"
    print(header)
    print("-" * len(header))

    models = [
        ("Pure DL Standalone (CNN-BiGRU)", peak(pure_dl)),
        ("Pure ML Standalone (RF + XGB)", peak(pure_ml)),
        ("Real VT 3-Layer Linear Ensemble", peak(linear_vt)),
        ("Real VT 3-Layer with Malicious Override", peak(override_vt)),
        ("Rule-Based URL Override (Decoupled)", peak(rule_override)),
    ]

    for name, p in models:
        lbl = p["weight_label"][:38]
        th = f"{p['threshold']:.2f}"
        acc = f"{p['accuracy']*100:.2f}%"
        prec = f"{p['precision']*100:.2f}%"
        rec = f"{p['recall']*100:.2f}%"
        f1 = f"{p['f1']:.4f}"
        print(f"{name:<40} | {lbl:<38} | {th:<5} | {acc:<7} | {prec:<7} | {rec:<7} | {f1:<7} | {p['fp']:<4} | {p['fn']:<4}")

    print("\n" + "-" * 115)
    print(" RULE-BASED URL OVERRIDE: PERFORMANCE ACROSS DIFFERENT URL THREAT THRESHOLDS (theta_url)")
    print(" (With 50% DL / 50% ML text ensemble base at its peak decision threshold)")
    print("-" * 115)

    base_rule_recs = [r for r in rule_override if "50% DL / 50% ML" in r["weight_label"]]
    unique_thetas = sorted(list(set(r["url_rule_threshold"] for r in base_rule_recs)))

    sub_header = f"{'URL Threshold (theta_url)':<26} | {'Peak Text Th':<13} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<9} | {'F1':<7} | {'FP':<4} | {'FN':<4} | {'Notes'}"
    print(sub_header)
    print("-" * 115)

    for th_u in unique_thetas:
        sub = [r for r in base_rule_recs if abs(r["url_rule_threshold"] - th_u) < 1e-4]
        best_sub = max(sub, key=lambda x: (x["f1"], x["accuracy"]))
        note = ""
        if th_u == 0.20:
            note = "Catches all VT suspicious & above"
        elif th_u == 0.45:
            note = "VT Malicious cutoff / Single vendor detection"
        elif th_u == 0.65:
            note = "Multi-vendor consensus / high-confidence"
        elif th_u == 0.85:
            note = "Severe consensus (5+ vendors)"
        elif th_u == 1.00:
            note = "15+ vendors consensus"

        print(f"URL >= {th_u:<20.2f} | {best_sub['threshold']:<13.2f} | {best_sub['accuracy']*100:<8.2f}% | {best_sub['precision']*100:<9.2f}% | {best_sub['recall']*100:<8.2f}% | {best_sub['f1']:<7.4f} | {best_sub['fp']:<4} | {best_sub['fn']:<4} | {note}")


def print_url_weight_circumstances(records: List[Dict[str, Any]]):
    print("\n" + "=" * 115)
    print(" CIRCUMSTANCE ANALYSIS: VARYING THE URL ENSEMBLE WEIGHT (w_URL) IN 3-LAYER ENSEMBLE")
    print(" (Evaluated with Balanced Text 50/50 split across different w_URL values)")
    print("=" * 115)

    linear_recs = [r for r in records if r["case_id"] == "Real VT 3-Layer Linear"]
    header = f"{'w_URL (URL Weight)':<22} | {'w_DL':<6} | {'w_ML':<6} | {'Peak Th':<8} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<9} | {'F1':<7} | {'FP':<4} | {'FN':<4}"
    print(header)
    print("-" * len(header))

    # Pick representative URL weights
    target_url_weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.3333, 0.40, 0.50, 0.60, 0.70]
    for w_u in target_url_weights:
        # Filter for roughly equal DL and ML weights
        matches = [
            r for r in linear_recs
            if abs(r["weights_url"] - w_u) < 0.015 and abs(r["weights_dl"] - r["weights_ml"]) < 0.05
        ]
        if matches:
            best = max(matches, key=lambda x: (x["f1"], x["accuracy"]))
            print(f"{w_u*100:<5.1f}% ({w_u:.2f}){'':<9} | {best['weights_dl']:<6.3f} | {best['weights_ml']:<6.3f} | {best['threshold']:<8.2f} | {best['accuracy']*100:<8.2f}% | {best['precision']*100:<9.2f}% | {best['recall']*100:<8.2f}% | {best['f1']:<7.4f} | {best['fp']:<4} | {best['fn']:<4}")


if __name__ == "__main__":
    records = run_exhaustive_sweep()
    print_leaderboard(records)
    print_url_architecture_comparison(records)
    print_url_weight_circumstances(records)
