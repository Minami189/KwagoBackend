# KwagoBackend URL & SMS Threat Scanner

FastAPI backend for multi-layer SMS smishing classification, deep learning NLP analysis (CNN-BiGRU), and VirusTotal weighted URL threat scanning.

---

## 1. System Architecture & Decision Matrix

KwagoBackend combines three distinct threat detection layers to determine an overall classification score ($S$) and synthesize human-readable executive explanations for mobile users:

1. **Local ML Layer (Random Forest + XGBoost)**: Mobile client pre-scan confidence score (`Safe`: $< 40\%$, `Suspicious`: $40\% - 70\%$, `Harmful`: $\ge 70\%$).
2. **Server Deep Learning Layer (CNN-BiGRU ONNX Model)**: NLP analysis extracting linguistic urgency and scam phrasing (`Safe`: $< 50\%$, `Suspicious`: $50\% - 75\%$, `Harmful`: $\ge 75\%$).
3. **URL Threat Scanner Layer (VirusTotal API)**: Weighted threat reputation analysis over security vendor engines.

---

### Ensemble Decision Logic Matrix

| Case Scenario | Active Layers & Weights | Ensemble Formula |
| :--- | :--- | :--- |
| **Case 1: No Web Link (or Only Email Address)** | 66.7% Local ML + 33.3% CNN DL | $S = (0.667 \times \text{ML}) + (0.333 \times \text{DL})$ |
| **Case 2: Web Link Present, but Scan Pending / Unavailable** | 66.7% Local ML + 33.3% CNN DL *(+ Caution Warning Appended)* | $S = (0.667 \times \text{ML}) + (0.333 \times \text{DL})$ |
| **Case 3A: Web Link(s) Present & Scan Completed (Clean / Suspicious)** | 60% Local ML + 20% CNN DL + 20% URL Scan | $S = (0.60 \times \text{ML}) + (0.20 \times \text{DL}) + (0.20 \times \text{URL})$ |
| **Case 3B: Any Web Link Confirmed Malicious ($S_{\text{URL}} \ge 0.45$)** | **50% URL Scan** + 35% Local ML + 15% CNN DL *(+ Escalation Floor $\ge 0.75$)* | $S = \max\big(0.75, (0.50 \times \text{URL}) + (0.35 \times \text{ML}) + (0.15 \times \text{DL})\big)$ |

---

### Classification Thresholds & Verdict Matrix

| Verdict | Probability Score Range ($S$) | Behavior | Status Badge / UI Color | System Actions |
| :--- | :--- | :--- | :--- | :--- |
| **Safe** | **Below 50% ($S < 0.50$)** | No threat or scam patterns detected. | Green (`#26CE6B`) | Allowed normally; bypassed from database logging. |
| **Suspicious** | **50% to 75% ($0.50 \le S < 0.75$)** | Unsolicited, promotional, or high-urgency content. | Orange (`#FFF07048`) | Caution alert shown; logged to DB if `allow_save = true`. |
| **Harmful** | **Above 75% ($S \ge 0.75$)** | High risk SMS scam / credential phishing. | Red (`#FF4D55`) | Threat warning generated; logged to DB if `allow_save = true`. |

> Detailed mathematical derivations, tipping points, and dataset evaluations on `sms_cleaned.csv` are documented in [`Threshold testing/optimal_threshold_calculation.md`](Threshold%20testing/optimal_threshold_calculation.md).

---

### Multi-Layer Explanation Synthesis Matrix

The backend explicitly compares layer verdicts (Local ML, Server DL, URL Threat Scanner) to explain **WHY** the final decision was reached:

* **ML Risk vs DL Safe (Overall Safe)**:  
  > *"Although the local ML layer marked this as suspicious, the Deep Learning layer evaluated the message text as safe, so the overall message is verified as Safe."* (Appends web link verified clean if URL present).
* **DL Risk vs ML Safe (Overall Safe)**:  
  > *"Although the Deep Learning layer detected potential smishing cues, the local ML layer marked it as safe, so the overall message is verified as Safe."* (Appends web link verified clean if URL present).
* **ML Risk vs DL Safe (Overall Suspicious)**:  
  > *"Although the Deep Learning model evaluated the message text as safe, the local ML layer flagged this as suspicious, resulting in an overall Suspicious classification."*
* **DL Risk vs ML Safe (Overall Suspicious)**:  
  > *"Though the local ML layer marked this as safe, the Deep Learning layer detected smishing risk because it {reason}, classifying the overall message as Suspicious."*
* **Consensus Risk (Overall Suspicious / Harmful)**:  
  > *"Both classification layers indicated smishing risk as the message {reason}, resulting in an overall Suspicious verdict."*
* **Malicious URL Elevation (Overall Harmful)**:  
  > *"Although message text appeared lower risk, the overall message is Harmful because the embedded web link ({url}) was confirmed as a high-risk malicious phishing site."*
* **Pending URL Caution Warning**:  
  > *"Exercise caution: this message contains a web link ({url}) that has not been verified by online threat intelligence yet, so its safety cannot be guaranteed."*

---

## 2. VirusTotal URL Threat Scanning Mechanics

The URL threat analysis layer evaluates embedded links in real time to catch active credential phishing, banking trojans, and malware distributions that evasion-crafted SMS text might otherwise mask.

---

### A. Dual-Layer Caching Architecture

To respect VirusTotal API quotas and achieve sub-second response times for mobile users, the backend implements a **dual-layer caching pipeline**:

`````text
Incoming URL
     │
     ▼
[ 1. In-Memory Cache (RAM) ] ── (Hit) ──► Re-evaluate Boost Tiers ──► Return (< 1ms)
     │ (Miss)
     ▼
[ 2. Supabase Global Cache ] ── (Hit) ──► Store in RAM ──► Return (~50ms)
     │ (Miss)
     ▼
[ 3. Live VirusTotal API v3 ] ──► Store in Supabase & RAM ──► Return (5-12s)
```

1. **In-Memory Cache (`memory_cache`)**:
   * Stored in process memory for zero-latency lookups on recently scanned URLs.
2. **Supabase Database Cache (`public.url_analysis` joined with `public.url`)**:
   * Anchored globally under `sms_id = 'CACHE_SMS'`.
   * **Permanent Retention (No Automatic Deletion)**: Analyzed URLs and threat reputation entries are preserved permanently to continuously expand the threat intelligence dataset and avoid redundant VirusTotal quota consumption.
   * **Latest Scan Prioritization**: Database lookups query with `.order("created_at", desc=True).limit(1)` to ensure newly updated threat definitions take priority.
3. **Dynamic Score Recalculation ([`recalculate_cached_url_score`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py))**:
   * Whenever a cached scan is retrieved (from RAM or Supabase), its raw engine detections are re-evaluated against the latest consensus boost rules dynamically. This guarantees that updated security policies apply instantly without needing to purge or invalidate existing cached rows.

---

### B. Live VirusTotal API v3 Pipeline

When a URL misses the cache, the backend processes it asynchronously using `aiohttp`:

1. **Base64 URL Identifier ([`get_url_id`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py))**:
   * Standardizes the URL into a URL-safe Base64 string without trailing padding (`=`), as required by VirusTotal API v3:
     ```python
     url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
     ```
2. **Instant Report Retrieval (`GET /api/v3/urls/{url_id}`)**:
   * Checks whether VirusTotal already has an existing analysis report from other global scanners. If HTTP 200 is returned, the engine extracts `last_analysis_results` immediately.
3. **On-Demand Scan Submission & Polling (`POST /api/v3/urls` + `GET /api/v3/analyses/{scan_id}`)**:
   * If no pre-existing report is found, the URL is submitted for on-demand analysis.
   * The backend polls `GET /api/v3/analyses/{scan_id}` with exponential backoff (up to 15 attempts, 2-second interval) until `status == "completed"`.
4. **Non-Blocking Pending Fallback**:
   * If polling takes too long or VirusTotal is under heavy queue load, the endpoint returns `"verdict": "pending"` with `"score": null` instead of failing or timing out. This allows the SMS ensemble to dynamically rebalance weights equally across the text ML ($50\%$) and DL ($50\%$) layers without blocking the mobile user.

---

### C. Weighted Threat Scoring Algorithm

Rather than counting vendor detections as equal votes, KwagoBackend normalizes engine categories and weights vendors by enterprise credibility:

#### 1. Engine Category Normalization
Each antivirus vendor returns a verdict category that maps to a baseline threat value:

| Category | Normalized Value | Description |
| :--- | :--- | :--- |
| **`malicious`** | `1.0` | Confirmed malware, phishing, or scam domain |
| **`phishing`** | `0.8` | Credential harvesting / brand impersonation |
| **`suspicious`** | `0.5` | Suspicious redirection, newly registered domain |
| **`type-unsupported`** | `0.1` | Specialized protocol / uncommon TLD |
| **`timeout`** | `0.05` | Scanner request timed out |
| **`harmless` / `undetected`** | `0.0` | Clean domain / no threat identified |

#### 2. Security Vendor Weights ([`source_weights.csv`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/source_weights.csv))
Reputable enterprise security engines have higher voting weights. Unlisted engines default to a weight of `0.15`:

| Vendor Engine | Weight | Notes / Specialization |
| :--- | :--- | :--- |
| **VirusTotal Baseline** | `1.00` | Aggregator baseline |
| **Microsoft** | `0.95` | Global telemetry and Defender threat intelligence |
| **Kaspersky** | `0.90` | High accuracy, aggressive heuristic detection |
| **CrowdStrike** | `0.90` | Enterprise EDR intelligence |
| **ESET / TrendMicro / Forcepoint / Google Safe Browsing** | `0.85` | Specialized web protection & URL categorization |
| **Bitdefender / Symantec** | `0.80` | High-reputation enterprise antivirus |
| **McAfee / Sophos** | `0.75` | Enterprise endpoint intelligence |
| **Avast / Avira** | `0.70` | Consumer threat intelligence |
| **Malwarebytes / Tencent / Fortinet** | `0.60 - 0.65` | APAC coverage & firewall intelligence |
| **Default (Unlisted Engines)** | `0.15` | Baseline weight for secondary engines |

#### 3. Base Normalized Score Formula
$$\text{Normalized Score} = \frac{\sum_{i=1}^{N} \left(\text{Category Score}_i \times \text{Weight}_i\right)}{\sum_{i=1}^{N} \text{Weight}_i}$$

---

### D. Vendor Consensus Boost Rules (Dilution Protection)

In standard weighted averages, a URL flagged by 20+ vendors can have its threat score diluted down to `~0.50` if 50+ secondary engines report `undetected`. KwagoBackend enforces **Vendor Consensus Boost Tiers** to eliminate false-confidence dilution:

| Flagged Vendors Condition | Boosted Threat Score ($S_{\text{URL}}$) | Applied URL Verdict |
| :--- | :--- | :--- |
| **$\ge 15$ Malicious Vendors** | **`1.00`** *(Maximum Threat)* | `malicious` |
| **$\ge 10$ Malicious Vendors** | **`0.95`** | `malicious` |
| **$\ge 5$ Malicious Vendors** | **`0.85`** | `malicious` |
| **$\ge 3$ Malicious Vendors** | **`0.75`** | `malicious` |
| **Trusted Vendor (Weight $\ge 0.5$) OR $\ge 2$ Malicious** | **`0.65`** | `malicious` |
| **$1$ Malicious Vendor** | **`0.45`** | `malicious` |
| **$\ge 2$ Suspicious Vendors** | **`0.40`** | `suspicious` |
| **$1$ Suspicious Vendor** | **`0.25`** | `suspicious` |

---

### E. URL Verdict Thresholds

| Final URL Threat Score ($S_{\text{URL}}$) | Verdict | Risk Assessment |
| :--- | :--- | :--- |
| **$S_{\text{URL}} \ge 0.45$** | **`malicious`** | Confirmed phishing, malware, or credential harvesting link. |
| **$0.20 \le S_{\text{URL}} < 0.45$** | **`suspicious`** | Domain exhibits deceptive characteristics or single-engine warning. |
| **$S_{\text{URL}} < 0.20$** | **`benign`** | Verified clean across all security vendor engines. |
| **Uncompleted / Timeout** | **`pending`** | Scan pending completion; safety cannot be guaranteed. |

---

### F. Multi-URL Handling, Malicious Link Dominance & Separate Database Storage

SMS messages often contain multiple web links (e.g., primary phishing link + backup mirror, or a malicious link paired with a benign decoy link) as well as email addresses:

1. **Multi-URL Resolution & Email Filtering ([`resolve_candidate_urls`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py))**:
   * Accepts `extracted_urls: [...]`, `urls: [...]`, and `extracted_url` in the request payload, deduplicating links while normalizing scheme-less domains (such as `www.promo-bonus.ph` $\rightarrow$ `http://www.promo-bonus.ph` for VirusTotal and host parsing).
   * Automatically ignores email addresses (e.g., `support@gcash-help.com`, `hr@company-recruitment.com`). If a message contains only an email address (`has_url: false`, `urls: []`), it is evaluated strictly via the text ensemble (50% ML / 50% DL).
2. **Malicious Link Dominance ([`aggregate_multi_url_results`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py) & [`compute_ensemble_score`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py))**:
   * **Anti-Dilution Across Multiple Links**: If **even a single URL** in a multi-link message is flagged as `malicious` ($S_{\text{URL}} \ge 0.45$), the aggregated URL verdict is immediately set to `malicious` and the highest-scoring malicious URL dominates the URL threat score ($90\%$ weight on the peak malicious link, floored at its full threat score so benign links in the same SMS cannot dilute it).
   * **Elevated Ensemble Weighting**: When any URL is confirmed malicious, the overall SMS ensemble shifts weight heavily toward the URL scanner (**50% URL / 35% ML / 15% DL** instead of the standard 25% URL / 50% ML / 25% DL) and enforces a minimum **`Harmful`** escalation floor ($\ge 0.85$).
3. **Separate Supabase Persistence per URL ([`save_url_scan_to_db`](file:///c:/Users/Soon1/OneDrive/Desktop/Model-API%20testing/app/scanner.py))**:
   * Every URL in the message is saved **separately** in Supabase:
     * Each unique link is parsed (`scheme`, `host`, `path`, `query`) and stored as its own row in `public.url`.
     * Each link's individual VirusTotal scan result and `is_malicious` flag are stored as a separate record in `public.url_analysis` linked to the message's `sms_id` (when `allow_save = true` and `overall_score >= 0.65`) or under `CACHE_SMS` for global threat caching.
4. **Clean URL Mitigation & Pending URL Caution**:
   * **Clean URL Mitigation**: If all links in the message are verified **Clean** ($S_{\text{URL}} < 0.20$) and the weighted score is $< 0.65$, the message is verified as **`Safe`**.
   * **Pending URL Caution**: If URL scans are `pending`, the ensemble calculates over the text layers ($50\%$ ML / $50\%$ DL) and appends a caution notice.

---

### G. Client Offline Threat Shield Synchronization

Every newly evaluated URL is stored in Supabase with host normalization (`public.url` and `public.url_analysis`). Mobile clients and local VPN filters can incrementally synchronize these cached threat definitions:

* **Endpoint**: `GET /url-reputations?since_timestamp=<epoch_millis>`
* Returns verified malicious and suspicious URLs, hostnames, and threat scores so mobile clients can block them offline or at the DNS level.

---

## 3. Folder Structure

```text
KwagoBackend/
|-- app/                  (FastAPI backend application)
|   |-- __init__.py
|   |-- main.py            (FastAPI application & API routes)
|   |-- scanner.py         (Ensemble scoring, ONNX inference, VirusTotal scanner, Supabase logging)
|   |-- crypto_utils.py    (Fernet AES-256 database encryption & PII anonymization)
|   |-- schemas.py         (Pydantic request & response schemas)
|   \-- source_weights.csv (Vendor reputation weights for VirusTotal threat scoring)
|-- database/             (PostgreSQL database schema migrations for Supabase)
|   |-- misclassification_reports_table.sql
|   |-- ntc_report_table.sql
|   \-- README.md
|-- docs/                 (Technical specifications & scoring formulas)
|   |-- url_scoring_explanation.md
|   |-- weighted_verdict_plan.md
|   \-- README.md
|-- models/               (Server NLP deep learning ONNX models & notebooks)
|   |-- cnn_bigru_model.onnx
|   |-- tokenizer_b.pkl
|   \-- kwago_Training_CNNBiGRU.ipynb
|-- scripts/              (Maintenance & feedback retraining utilities)
|   |-- export_retraining_data.py
|   |-- merge_feedback_dataset.py
|   \-- README.md
|-- Threshold testing/    (Benchmarking, grid search, and dataset simulations)
|   |-- calculate_optimal_thresholds.py
|   |-- grid_search_thresholds_weights.py
|   |-- generate_scanned_sms_csv.py
|   |-- optimal_threshold_calculation.md
|   |-- sms_cleaned.csv
|   \-- README.md
|-- .env                  (Local environment variables - gitignored)
|-- .gitignore
|-- .vercelignore
|-- README.md
|-- requirements.txt
\-- vercel.json
```

---
## 4. Setup & Environment Configuration

### 1. Create & Activate Virtual Environment
```bash
python -m venv .venv
```
* **PowerShell**: `.venv\Scripts\Activate.ps1`
* **Command Prompt**: `.venv\Scripts\activate.bat`
* **macOS/Linux**: `source .venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
Create a `.env` file in the project root:
```env
VIRUS_TOTAL_API_KEY=your_virustotal_api_key
KWAGO_API_KEY=your_custom_auth_api_key_for_this_backend
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SECRET_KEY=your_supabase_secret_key
```

---

## 5. Running the Server

Start the application with Uvicorn:
```bash
python -m app.main
```
or:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

---

## 6. API Reference

All protected endpoints require the HTTP Authorization Header:
`Authorization: Bearer <KWAGO_API_KEY>`

---

### A. Health Check (`GET /health`)

Verify service status and classifier model availability.

* **Endpoint:** `GET /health`
* **Response Body:**
  ```json
  {
    "status": "healthy",
    "classifier_loaded": true
  }
  ```

---

### B. SMS Smishing Scan (`POST /scan-sms`)

Scans an SMS message text, evaluates local ML confidence, runs server CNN-BiGRU inference, and performs VirusTotal URL lookups across one or more web links. Each extracted URL is cached and stored separately in Supabase (`public.url` and `public.url_analysis`), and if any single URL is malicious, it is weighted dominantly in the final verdict.

* **Endpoint:** `POST /scan-sms`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`

#### Example 1: Message with Multiple URLs
For a message containing multiple web links:
> `"URGENT: Verify your account at https://bdo-security-update.com/login and check backup mirror http://bdo-mirror.ph/verify"`

```json
{
  "message": "URGENT: Verify your account at https://bdo-security-update.com/login and check backup mirror http://bdo-mirror.ph/verify",
  "has_url": true,
  "extracted_url": "https://bdo-security-update.com/login",
  "extracted_urls": [
    "https://bdo-security-update.com/login",
    "http://bdo-mirror.ph/verify"
  ],
  "urls": [
    "https://bdo-security-update.com/login",
    "http://bdo-mirror.ph/verify"
  ],
  "sender": "+63917***4567",
  "ml_prediction": "smishing",
  "ml_confidence": 0.88,
  "allow_save": false,
  "auto_report": false
}
```

#### Example 2: Message with Both an Email and Multiple URLs
For a message containing both an email address and multiple web links (`support@gcash-help.com` is recognized as an email and excluded from the URLs array):
> `"Contact support@gcash-help.com or visit https://gcash-claim.net/ and www.promo-bonus.ph for details"`

```json
{
  "message": "Contact support@gcash-help.com or visit https://gcash-claim.net/ and www.promo-bonus.ph for details",
  "has_url": true,
  "extracted_url": "https://gcash-claim.net/",
  "extracted_urls": [
    "https://gcash-claim.net/",
    "www.promo-bonus.ph"
  ],
  "urls": [
    "https://gcash-claim.net/",
    "www.promo-bonus.ph"
  ],
  "sender": "GCashPromo",
  "ml_prediction": "suspicious",
  "ml_confidence": 0.74,
  "allow_save": false,
  "auto_report": false
}
```

#### Example 3: Message with Only an Email (No Web Links)
For a message containing only an email address:
> `"Please forward your latest bank statement to hr@company-recruitment.com for processing."`

```json
{
  "message": "Please forward your latest bank statement to hr@company-recruitment.com for processing.",
  "has_url": false,
  "extracted_url": null,
  "extracted_urls": [],
  "urls": [],
  "sender": "Recruiter",
  "ml_prediction": "benign",
  "ml_confidence": 0.12,
  "allow_save": false,
  "auto_report": false
}
```

* **Response Body (`SmsScanResponse`):**
  ```json
  {
    "message": "URGENT: Verify your account at https://bdo-security-update.com/login and check backup mirror http://bdo-mirror.ph/verify",
    "overall_verdict": "Harmful",
    "overall_score": 0.9105,
    "overall_explanation": "Both the local ML and Deep Learning layers confirmed High Risk because the message uses urgent security alerts or account suspension warnings to pressure immediate action. Furthermore, the embedded web link (https://bdo-security-update.com/login) was confirmed as a high-risk malicious phishing site.",
    "cnn_analysis": {
      "score": 0.95,
      "verdict": "harmful",
      "explanation": "Uses urgent security alerts or account suspension warnings to pressure immediate action."
    },
    "url_analysis": {
      "has_url": true,
      "extracted_url": "https://bdo-security-update.com/login",
      "extracted_urls": [
        "https://bdo-security-update.com/login",
        "http://bdo-mirror.ph/verify"
      ],
      "score": 0.90,
      "verdict": "malicious",
      "total_weight": 14.85,
      "explanation": "Scanned 2 URLs; 1 flagged as malicious (https://bdo-security-update.com/login). Malicious link prioritized with dominant threat score 0.90. This URL is flagged as malicious (threat score: 0.90). Detected by: Fortinet (malicious).",
      "contributions": [
        "Fortinet (malicious)"
      ],
      "individual_results": [
        {
          "has_url": true,
          "extracted_url": "https://bdo-security-update.com/login",
          "score": 0.90,
          "verdict": "malicious",
          "total_weight": 14.85,
          "explanation": "This URL is flagged as malicious (threat score: 0.90). Detected by: Fortinet (malicious).",
          "contributions": ["Fortinet (malicious)"]
        },
        {
          "has_url": true,
          "extracted_url": "http://bdo-mirror.ph/verify",
          "score": 0.0,
          "verdict": "clean",
          "total_weight": 14.85,
          "explanation": "This URL appears clean (threat score: 0.00). No security vendors flagged this URL.",
          "contributions": []
        }
      ]
    }
  }
  ```

---

### C. Direct URL Threat Scan (`POST /scan` or `POST /scan-url`)

Performs a weighted VirusTotal threat analysis on a URL. Utilizes in-memory and database dual-layer caching.

* **Endpoint:** `POST /scan` or `POST /scan-url`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`
* **Request Body:**
  ```json
  {
    "url": "http://suspicious-site.com/login"
  }
  ```

* **Response Body (`UrlAnalysisResult`):**
  ```json
  {
    "has_url": true,
    "extracted_url": "http://suspicious-site.com/login",
    "score": 0.85,
    "verdict": "malicious",
    "total_weight": 14.85,
    "explanation": "This URL is flagged as malicious (threat score: 0.85). Detected by: Kaspersky (malicious).",
    "contributions": [
      "Kaspersky (malicious)"
    ]
  }
  ```

---

### D. URL Reputation Client Sync (`GET /url-reputations`)

Retrieves cached URL threat reputation records for local mobile client or VPN synchronization. Supports incremental sync via timestamp query parameter.

* **Endpoint:** `GET /url-reputations`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`
* **Query Parameters:**
  * `since_timestamp` *(optional)*: Unix epoch timestamp in milliseconds (e.g. `1725292800000`).
* **Response Body (`UrlReputationSyncResponse`):**
  ```json
  {
    "total_records": 1,
    "last_synced_at": "2026-09-08T19:00:00Z",
    "urls": [
      {
        "extracted_url": "http://fake-claim.com",
        "normalized_host": "fake-claim.com",
        "verdict": "malicious",
        "score": 0.90,
        "total_weight": 14.85,
        "explanation": "This URL is flagged as malicious (threat score: 0.90). Detected by: Fortinet (malicious).",
        "contributions": [
          "Fortinet (malicious)"
        ]
      }
    ]
  }
  ```

---

### E. VirusTotal Quota Check (`GET /quota`)

Retrieves API usage limits and remaining VirusTotal quotas.

* **Endpoint:** `GET /quota`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`
* **Response Body:**
  ```json
  {
    "endpoint": "https://www.virustotal.com/api/v3/users/me",
    "quota_info": { ... }
  }
  ```

---

### F. Misclassification Report (`POST /report-misclassification`)

Submits user feedback when an SMS was misclassified (e.g. False Positive or False Negative) for dataset curation and model retraining.

* **Endpoint:** `POST /report-misclassification`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`
* **Request Body (`MisclassificationReportRequest`):**
  ```json
  {
    "message": "Dear customer, your BDO OTP is 492810. Do not share this with anyone.",
    "sender": "BDO",
    "has_url": false,
    "extracted_url": null,
    "original_verdict": "Harmful",
    "original_score": 0.85,
    "original_ml_score": 0.80,
    "original_dl_score": null,
    "user_verdict": "Safe",
    "report_type": "false_positive",
    "user_comment": "Official bank OTP message falsely flagged as harmful.",
    "app_version": "1.2.0",
    "device_id": "anon-device-12345"
  }
  ```
* **Response Body (`MisclassificationReportResponse`):**
  ```json
  {
    "status": "success",
    "report_id": "7f8b9c2a-1122-3344-5566-778899aabbcc",
    "message": "Misclassification report received successfully. Thank you for your feedback!"
  }
  ```


---

## 6. Data Privacy, Encryption & Model Retraining Workflow

To comply with data privacy standards (e.g. Philippine Data Privacy Act of 2012) while continuously improving model accuracy, KwagoBackend implements **Application-Level Symmetric Encryption (ALE)** and **PII Anonymization** before storing any SMS content in the database.

### A. Privacy & Encryption Pipeline

`````text
Incoming User SMS / Report
         │
         ▼
[ 1. PII Anonymization ] ──► Masks 4-8 digit OTPs (<OTP>) & Mobile numbers (<PHONE>)
         │
         ▼
[ 2. Fernet AES Encryption ] ──► Encrypts using DB_ENCRYPTION_KEY (.env)
         │
         ▼
[ 3. Supabase Database ] ──► Stores unreadable ciphertext tokens (gAAAAABn...)
```

* **Zero Plaintext at Rest:** The database never contains readable user SMS messages or phone numbers.
* **Key Isolation:** The encryption key resides exclusively in the server `.env`, never in database tables.
* **Scrubbed Training Signals:** Replacing phone numbers and OTPs ensures the AI learns smishing syntax and urgency cues rather than memorizing private phone numbers or OTP codes.

---

### B. End-to-End Retraining Cycle

When sufficient user misclassification reports or flagged samples have accumulated in Supabase:

#### Step 1: Export and Decrypt Data
Run the automated exporter to pull and decrypt reports in-memory:
```bash
python scripts/export_retraining_data.py --output retraining_data.csv --status all
```

#### Step 2: Merge into Main Dataset
Deduplicate and merge the newly labeled feedback samples into the training dataset:
```bash
python scripts/merge_feedback_dataset.py --input retraining_data.csv --base "Threshold testing/sms_cleaned.csv"
```

#### Step 3: Retrain and Export Updated ONNX
1. Open [`models/kwago_Training_CNNBiGRU.ipynb`](models/kwago_Training_CNNBiGRU.ipynb) or [`models/kwago_Training_RFXGB.ipynb`](models/kwago_Training_RFXGB.ipynb).
2. Re-run training with the enriched dataset.
3. Export the updated model to [`models/cnn_bigru_model.onnx`](models/cnn_bigru_model.onnx) and update tokenizer weights.
4. Restart FastAPI backend—it will automatically load the newly trained ONNX model on startup.
