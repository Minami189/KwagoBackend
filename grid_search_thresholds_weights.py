"""
KwagoBackend - Exhaustive Grid Sweep: All Thresholds, All Cases, All Ensemble Weights
-------------------------------------------------------------------------------------
Evaluates performance across:
- All Thresholds: 0.10 to 0.90 (step 0.05)
- All Cases: Case 1 (Text-only), Case 2 (URL Pending), Case 3A (Clean URL),
             Case 3B (Suspicious URL), Case 3C (Malicious URL), and Full Dataset.
- All Weight Configurations:
    - 2-Layer weights (DL : ML): 1.0:0.0, 0.9:0.1, 0.8:0.2, 0.75:0.25, 0.67:0.33 (Baseline),
                                0.6:0.4, 0.5:0.5, 0.4:0.6, 0.33:0.67, 0.2:0.8, 0.1:0.9, 0.0:1.0
    - 3-Layer weights (DL : ML : URL): 0.50:0.25:0.25 (Baseline), 0.60:0.20:0.20,
                                       0.40:0.30:0.30, 0.33:0.33:0.33, 0.70:0.15:0.15,
                                       0.40:0.20:0.40, 0.50:0.15:0.35, 0.60:0.10:0.30

Exports complete results to: all_thresholds_cases_weights_results.csv
"""

import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from app import scanner


def load_dataset(csv_path: str = "sms_cleaned.csv"):
    path = Path(csv_path)
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

    url_pattern = re.compile(r'(?:https?://|www\.)[^\s]+', re.IGNORECASE)
    has_url = np.array([bool(url_pattern.search(t)) for t in texts], dtype=bool)

    return rows, texts, y_true, has_url


def generate_dl_scores(texts: List[str], classifier: scanner.SMSClassifier) -> np.ndarray:
    print("[*] Generating Deep Learning (CNN-BiGRU) scores via ONNX Runtime...")
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


def generate_out_of_fold_ml_scores(texts: List[str], y: np.ndarray) -> np.ndarray:
    print("[*] Generating Out-Of-Fold Local ML (TF-IDF + Classifier) scores via 5-Fold Stratified CV...")
    t0 = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ml_scores = np.zeros(len(y), dtype=np.float32)

    for train_idx, val_idx in skf.split(texts, y):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_tr = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)

        clf = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced")
        clf.fit(X_tr, y[train_idx])
        ml_scores[val_idx] = clf.predict_proba(X_val)[:, 1]

    print(f"    Local ML CV finished in {time.time() - t0:.2f}s ({len(texts)} samples)")
    return ml_scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    # Cost index: Asymmetric fraud loss where missing phishing (FN) is 5x worse than False Alarm (FP)
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
    csv_file = "sms_cleaned.csv"
    output_csv = "all_thresholds_cases_weights_results.csv"

    rows, texts, y_true, has_url = load_dataset(csv_file)
    classifier = scanner.SMSClassifier()

    dl_scores = generate_dl_scores(texts, classifier)
    ml_scores = generate_out_of_fold_ml_scores(texts, y_true)

    # Thresholds: 0.10 to 0.90 with step 0.05
    thresholds = [round(th, 2) for th in np.arange(0.10, 0.95, 0.05)]

    # 2-Layer Weight Configurations (w_DL, w_ML)
    two_layer_weights = [
        (1.00, 0.00, "100% DL (Pure DL)"),
        (0.90, 0.10, "90% DL / 10% ML"),
        (0.80, 0.20, "80% DL / 20% ML"),
        (0.75, 0.25, "75% DL / 25% ML"),
        (round(2.0/3.0, 4), round(1.0/3.0, 4), "66.7% DL / 33.3% ML (Current Backend Baseline)"),
        (0.60, 0.40, "60% DL / 40% ML"),
        (0.50, 0.50, "50% DL / 50% ML (Equal Weight)"),
        (0.40, 0.60, "40% DL / 60% ML"),
        (round(1.0/3.0, 4), round(2.0/3.0, 4), "33.3% DL / 66.7% ML"),
        (0.20, 0.80, "20% DL / 80% ML"),
        (0.10, 0.90, "10% DL / 90% ML"),
        (0.00, 1.00, "100% ML (Pure ML)"),
    ]

    # 3-Layer Weight Configurations (w_DL, w_ML, w_URL)
    three_layer_weights = [
        (0.50, 0.25, 0.25, "50% DL / 25% ML / 25% URL (Current Backend Baseline)"),
        (0.60, 0.20, 0.20, "60% DL / 20% ML / 20% URL"),
        (0.40, 0.30, 0.30, "40% DL / 30% ML / 30% URL"),
        (round(1.0/3.0, 4), round(1.0/3.0, 4), round(1.0/3.0, 4), "33.3% DL / 33.3% ML / 33.3% URL (Equal)"),
        (0.70, 0.15, 0.15, "70% DL / 15% ML / 15% URL"),
        (0.40, 0.20, 0.40, "40% DL / 20% ML / 40% URL (Heavy URL)"),
        (0.50, 0.15, 0.35, "50% DL / 15% ML / 35% URL"),
        (0.60, 0.10, 0.30, "60% DL / 10% ML / 30% URL"),
    ]

    records = []
    print(f"\n[*] Running exhaustive grid sweep across all thresholds ({len(thresholds)}), cases (6), and weights...")

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
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # CASE 2: URL Present, but Scan Pending / Timeout (N = 485)
    # =========================================================================
    mask_c2 = has_url
    y_c2 = y_true[mask_c2]
    dl_c2 = dl_scores[mask_c2]
    ml_c2 = ml_scores[mask_c2]

    for w_dl, w_ml, w_label in two_layer_weights:
        ensemble_score = (w_dl * dl_c2) + (w_ml * ml_c2)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c2, y_pred)
            records.append({
                "case_id": "Case 2",
                "case_name": "URL Pending / Timeout",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": 0.0,
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # CASE 3A: URL Scan Completed - Clean URL (S_URL = 0.00) (N = 485)
    # =========================================================================
    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        # S_URL = 0.00
        ensemble_score = (w_dl * dl_c2) + (w_ml * ml_c2) + (w_url * 0.00)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c2, y_pred)
            records.append({
                "case_id": "Case 3A",
                "case_name": "Clean URL (S_URL = 0.00)",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # CASE 3B: URL Scan Completed - Suspicious URL (S_URL = 0.30, floor 0.50) (N = 485)
    # =========================================================================
    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        # URL score = 0.30 with backend Suspicious floor of max(score, 0.50)
        raw_score = (w_dl * dl_c2) + (w_ml * ml_c2) + (w_url * 0.30)
        ensemble_score = np.maximum(raw_score, 0.50)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c2, y_pred)
            records.append({
                "case_id": "Case 3B",
                "case_name": "Suspicious URL (S_URL = 0.30, floor 0.50)",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # CASE 3C: URL Scan Completed - Malicious URL (S_URL = 0.85, override 0.85) (N = 485)
    # =========================================================================
    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        # Malicious URL override: max(score, 0.85)
        raw_score = (w_dl * dl_c2) + (w_ml * ml_c2) + (w_url * 0.85)
        ensemble_score = np.maximum(raw_score, 0.85)
        for th in thresholds:
            y_pred = (ensemble_score >= th).astype(int)
            m = compute_metrics(y_c2, y_pred)
            records.append({
                "case_id": "Case 3C",
                "case_name": "Malicious URL (S_URL = 0.85, override 0.85)",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # =========================================================================
    # FULL DATASET OVERALL SIMULATION (N = 4,474)
    # Combines Case 1 (text-only) with realistic URL simulation on has_url
    # =========================================================================
    # On has_url samples, ground truth benign gets URL=0.0 (Clean), phishing gets URL=0.85 (Malicious)
    simulated_url_scores = np.zeros(len(texts), dtype=np.float32)
    for i in range(len(texts)):
        if has_url[i]:
            simulated_url_scores[i] = 0.85 if y_true[i] == 1 else 0.00

    for w_dl, w_ml, w_url, w_label in three_layer_weights:
        full_ensemble = np.zeros(len(texts), dtype=np.float32)
        for i in range(len(texts)):
            if not has_url[i]:
                # 2-layer re-normalization
                w_tot = w_dl + w_ml
                full_ensemble[i] = (w_dl * dl_scores[i] + w_ml * ml_scores[i]) / w_tot
            else:
                # 3-layer with malicious override if simulated_url_scores >= 0.45
                s_lin = (w_dl * dl_scores[i]) + (w_ml * ml_scores[i]) + (w_url * simulated_url_scores[i])
                if simulated_url_scores[i] >= 0.45:
                    full_ensemble[i] = max(s_lin, 0.85)
                else:
                    full_ensemble[i] = s_lin

        for th in thresholds:
            y_pred = (full_ensemble >= th).astype(int)
            m = compute_metrics(y_true, y_pred)
            records.append({
                "case_id": "Full Population",
                "case_name": "Full Dataset Realistic Ensemble",
                "weights_dl": w_dl,
                "weights_ml": w_ml,
                "weights_url": w_url,
                "weight_label": w_label,
                "threshold": th,
                **m
            })

    # Write all records to CSV
    output_path = Path(output_csv)
    fieldnames = [
        "case_id", "case_name", "weight_label", "weights_dl", "weights_ml", "weights_url",
        "threshold", "accuracy", "precision", "recall", "specificity", "f1", "youden_j",
        "fpr", "fnr", "cost_index", "tp", "tn", "fp", "fn", "total_samples"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[OK] Grid sweep complete! {len(records):,} total configurations computed and saved to:")
    print(f"     {output_path.resolve()}\n")

    return records


def print_leaderboard(records: List[Dict[str, Any]]):
    print("=" * 90)
    print(" LEADERBOARD: TOP-PERFORMING CONFIGURATIONS BY CASE")
    print("=" * 90)

    unique_cases = list(dict.fromkeys(r["case_id"] for r in records))

    for cid in unique_cases:
        case_records = [r for r in records if r["case_id"] == cid]
        cname = case_records[0]["case_name"]
        total_n = case_records[0]["total_samples"]

        # Best Youden J (balanced sensitivity + specificity)
        best_j = max(case_records, key=lambda x: (x["youden_j"], x["f1"]))
        # Best F1
        best_f1 = max(case_records, key=lambda x: (x["f1"], x["youden_j"]))
        # Best High-Precision (Threshold >= 0.80)
        high_prec_pool = [r for r in case_records if r["threshold"] >= 0.80]
        best_hp = max(high_prec_pool, key=lambda x: x["precision"]) if high_prec_pool else None

        print(f"\n[{cid}: {cname}] (N = {total_n:,})")
        print(f"  * BEST BALANCED (Youden's J):")
        print(f"    - Weight:    {best_j['weight_label']}")
        print(f"    - Threshold: {best_j['threshold']:.2f}")
        print(f"    - Metrics:   Youden J = {best_j['youden_j']:.4f} | Recall = {best_j['recall']*100:.2f}% | Prec = {best_j['precision']*100:.2f}% | F1 = {best_j['f1']:.4f}")
        print(f"    - Errors:    FP = {best_j['fp']} | FN = {best_j['fn']}")

        print(f"  * BEST F1 HARMONIC:")
        print(f"    - Weight:    {best_f1['weight_label']}")
        print(f"    - Threshold: {best_f1['threshold']:.2f}")
        print(f"    - Metrics:   F1 = {best_f1['f1']:.4f} | Recall = {best_f1['recall']*100:.2f}% | Prec = {best_f1['precision']*100:.2f}%")

        if best_hp:
            print(f"  * BEST HIGH-PRECISION (Strict Harmful Cutoff >= 0.80):")
            print(f"    - Weight:    {best_hp['weight_label']}")
            print(f"    - Threshold: {best_hp['threshold']:.2f}")
            print(f"    - Metrics:   Precision = {best_hp['precision']*100:.2f}% | Recall = {best_hp['recall']*100:.2f}% | FP = {best_hp['fp']}")


if __name__ == "__main__":
    records = run_exhaustive_sweep()
    print_leaderboard(records)
