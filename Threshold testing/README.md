# Smishing Detection Threshold Testing & Benchmarking Suite

This directory contains empirical benchmarking scripts, exhaustive grid searches, mathematical derivations, and evaluation datasets used to determine the optimal decision thresholds and ensemble weights for KwagoBackend.

---

## Directory Structure

```text
Threshold testing/
├── README.md                                # Master directory guide (this file)
├── optimal_threshold_calculation.md         # Mathematical framework and statistical derivations
├── sms_cleaned.csv                          # Evaluation dataset (4,474 real SMS with VT scan results)
├── vt_url_cache.json                        # Persistent local cache of VirusTotal API query results
│
├── Benchmarking Scripts:
│   ├── calculate_optimal_thresholds.py      # Evaluates Youden's J, F1, and Case 1-3 boundaries
│   ├── grid_search_thresholds_weights.py    # Exhaustive sweep across all weights & override architectures
│   └── generate_scanned_sms_csv.py          # Batch URL scanner appending VT scores to dataset
│
├── Benchmark Results & Reports:
│   ├── all_cases_and_weights_results.txt    # Text executive report of multi-layer simulations
│   └── all_thresholds_cases_weights_results.csv # Complete grid search metrics (Precision, Recall, F1)
│
└── RFXGB Model/                             # Local Mobile ML model assets (RF / XGBoost ONNX & notebooks)
    ├── kwago_PreProcessing.ipynb
    ├── kwago_Training_RFXGB.ipynb
    ├── rf_model.onnx
    ├── xgboost_model.onnx
    └── predict_interactive.py
```

---

## Script Overviews & Execution

### 1. `calculate_optimal_thresholds.py`
Evaluates Youden's $J$ statistic, harmonic F1 curves, and cost-sensitive loss curves on the ONNX CNN-BiGRU model using batch vectorized inference across `sms_cleaned.csv`.

```powershell
python "Threshold testing/calculate_optimal_thresholds.py"
```

### 2. `grid_search_thresholds_weights.py`
Runs an exhaustive sweep evaluating:
- Standalone DL vs. Standalone ML vs. 2-Layer Text Ensemble
- Pure Linear 3-Layer Ensemble ($w_{\text{URL}}$ from 5% to 70%)
- Malicious Dominance Override
- Rule-Based URL Override ($\theta_{\text{URL}}$ from 0.20 to 1.00)

```powershell
python "Threshold testing/grid_search_thresholds_weights.py"
```

### 3. `generate_scanned_sms_csv.py`
Extracts all embedded URLs from the dataset, queries VirusTotal API v3 with rate limiting (4 requests/min) and caching, and populates the `url_scan_result` column in `sms_cleaned.csv`.

```powershell
python "Threshold testing/generate_scanned_sms_csv.py"
```

---

## Key Benchmark Takeaways
* **Offline (Local ML — `50% RF / 50% XGBoost`)**: Uses thresholds of `Safe < 0.40`, `Suspicious: 0.40 <= S < 0.70`, and `Harmful: S >= 0.70` to maximize offline recall while keeping false alarms minimal.
* **Online (No URL — `33.3% DL / 66.7% ML`)**: Evaluates text-only messages (and pending URL scans) with thresholds of `Safe < 0.50`, `Suspicious: 0.50 <= S < 0.75`, and `Harmful: S >= 0.75`.
* **Online (With URL — `20% DL / 60% ML / 20% URL`)**: Combines deep learning, local ML, and VirusTotal URL reputation at `Safe < 0.50`, `Suspicious: 0.50 <= S < 0.75`, and `Harmful: S >= 0.75` ($F_1 = 0.9594$, Recall $= 97.84\%$ at `0.50`; Precision $= 99.22\%$, FPR $= 0.39\%$ at `0.75`).
* **Malicious URL Override**: Recovers **$100\%$ ($31/31$)** of evasive phishing attacks that hid behind harmless-sounding text.
