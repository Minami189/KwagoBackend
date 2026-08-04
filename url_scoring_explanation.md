# VirusTotal URL Threat Scoring Algorithm Explanation

This document explains the mathematical formula and programmatic steps used to calculate the threat score and final verdict of an analyzed URL in the backend service.

---

## 1. Code Location

The threat scoring calculation is defined in:
* **File:** [app/scanner.py](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py)
* **Function:** [compute_weighted_verdict(analysis_results, weights)](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py#L53-L80)

---

## 2. Core Scoring Steps

### Step 1: Mapping Engine Verdicts to Numeric Scores
When VirusTotal returns scanning results from its engines, each engine reports a category. We map these categories to numerical threat levels (from `0.0` safe to `1.0` dangerous) using the `CATEGORY_SCORES` lookup table in [app/scanner.py](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py#L21-L28):

| Engine Verdict Category | Numeric Threat Score | Notes |
| :--- | :--- | :--- |
| `malicious` | **`1.0`** | Explicitly flagged as dangerous |
| `suspicious` | **`0.75`** | Flagged as highly suspicious |
| `type-unsupported` | **`0.10`** | Minor penalty for unsupported scanning types |
| `timeout` | **`0.05`** | Minor penalty for scans that timed out |
| `harmless` | **`0.00`** | Explicitly marked safe |
| `undetected` | **`0.00`** | Default mapping for engines that did not flag |

---

### Step 2: Applying Vendor Weights
We load reputational weights for each security vendor from [app/source_weights.csv](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/source_weights.csv).
* High-confidence, enterprise-grade vendors (e.g., Kaspersky, Microsoft, Sophos, Forcepoint) have weights ranging from **`0.75`** to **`0.95`**.
* Unrecognized or generic engines default to a baseline weight of **`0.3`** (`DEFAULT_SOURCE_WEIGHT`).

---

### Step 3: Weighted Average Calculation
We calculate the initial normalized threat score by multiplying each engine's category score by its weight, summing the results, and dividing by the sum of all weights:

$$\text{Initial Normalized Score} = \frac{\sum (\text{category\_score}_i \times \text{weight}_i)}{\sum \text{weight}_i}$$

---

### Step 4: Single Trusted Engine Boost (Ensemble Override)
To prevent a single correct detection from a high-confidence vendor from being mathematically diluted to `0.0` by 80+ passive or un-updated engines, the code checks for high-confidence detections:

```python
if category == "malicious" and weight >= 0.5:
    has_trusted_malicious = True
```

If `has_trusted_malicious` is `True`, the code enforces a score floor of **`0.40`**:

```python
if has_trusted_malicious:
    normalized_score = max(normalized_score, 0.40)
```

---

### Step 5: Applying Threat Verdict Thresholds
Finally, the normalized score is mapped to a verdict category returned to the client:

| Calculated Score Range | Verdict Category | Android Client Handling |
| :--- | :--- | :--- |
| **`score >= 0.65`** | **`malicious`** | Direct threat detected by multiple engines. |
| **`0.35 <= score < 0.65`** | **`suspicious`** | Flagged by at least one high-confidence engine. |
| **`score < 0.35`** | **`benign`** | No high-confidence engines flagged the URL. |
