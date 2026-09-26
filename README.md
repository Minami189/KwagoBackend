# KwagoBackend URL & SMS Threat Scanner

FastAPI backend for multi-layer SMS smishing classification, deep learning NLP analysis (CNN-BiGRU), and VirusTotal weighted URL threat scanning.

---

## 1. System Architecture & Decision Matrix

KwagoBackend combines three distinct threat detection layers to determine an overall classification score ($S$) and synthesize human-readable executive explanations for mobile users:

1. **Local ML Layer (Random Forest + XGBoost)**: Mobile client pre-scan confidence score (`Safe`: $< 50\%$, `Suspicious`: $50\% - 85\%$, `Harmful`: $\ge 85\%$).
2. **Server Deep Learning Layer (CNN-BiGRU ONNX Model)**: NLP analysis extracting linguistic urgency and scam phrasing (`Safe`: $< 50\%$, `Suspicious`: $50\% - 85\%$, `Harmful`: $\ge 85\%$).
3. **URL Threat Scanner Layer (VirusTotal API)**: Weighted threat reputation analysis over security vendor engines.

---

### Ensemble Decision Logic Matrix

| Case Scenario | Active Layers & Weights | Ensemble Formula |
| :--- | :--- | :--- |
| **Case 1: No Web Link** | 66.7% CNN DL + 33.3% Local ML | $S = \left(\frac{2}{3} \times \text{DL}\right) + \left(\frac{1}{3} \times \text{ML}\right)$ |
| **Case 2: Web Link Present, but Scan Pending / Unavailable** | 66.7% CNN DL + 33.3% Local ML *(+ Caution Warning Appended)* | $S = \left(\frac{2}{3} \times \text{DL}\right) + \left(\frac{1}{3} \times \text{ML}\right)$ |
| **Case 3: Web Link Present & Scan Completed** | 50% CNN DL + 25% URL Scan + 25% Local ML | $S = (0.50 \times \text{DL}) + (0.25 \times \text{URL}) + (0.25 \times \text{ML})$ |

---

### Classification Thresholds & Verdict Matrix

| Verdict | Probability Score Range ($S$) | Behavior | Status Badge / UI Color | System Actions |
| :--- | :--- | :--- | :--- | :--- |
| **Safe** | **Below 50% ($S < 0.50$)** | No threat or scam patterns detected. | Green (`#26CE6B`) | Allowed normally; bypassed from database logging. |
| **Suspicious** | **50% to 85% ($0.50 \le S < 0.85$)** | Unsolicited, promotional, or high-urgency content. | Orange (`#FFF07048`) | Caution alert shown; logged to DB if `allow_save = true`. |
| **Harmful** | **Above 85% ($S \ge 0.85$)** | High risk SMS scam / credential phishing. | Red (`#FF4D55`) | Threat warning generated; logged to DB if `allow_save = true`. |

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

```text
Incoming URL
     │
     ▼
[ 1. In-Memory Cache (RAM) ] ── (Hit) ──► Re-evaluate Boost Tiers ──► Return (< 1ms)
     │ (Miss)
     ▼
[ 2. Supabase Global Cache ] ── (Hit: < 7 Days) ──► Store in RAM ──► Return (~50ms)
     │ (Miss / Stale >= 7 Days)
     ▼
[ 3. Live VirusTotal API v3 ] ──► Store in Supabase & RAM ──► Return (5-12s)
```

1. **In-Memory Cache (`memory_cache`)**:
   * Stored in process memory for zero-latency lookups on recently scanned URLs.
2. **Supabase Database Cache (`public.url_analysis` joined with `public.url`)**:
   * Anchored globally under `sms_id = 'CACHE_SMS'`.
   * **7-Day Freshness Window**: Cached records older than 7 days (`timedelta(days=7)`) expire and trigger a fresh scan.
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
   * If polling takes too long or VirusTotal is under heavy queue load, the endpoint returns `"verdict": "pending"` with `"score": null` instead of failing or timing out. This allows the SMS ensemble to dynamically rebalance weights across the text DL ($66.7\%$) and ML ($33.3\%$) layers without blocking the mobile user.

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

### F. Multi-Layer Ensemble Integration & Clean Link Mitigation

The URL analysis score directly influences the overall SMS smishing classification:

1. **Standard Completed Scan (Case 3A)**:
   $$S = (0.50 \times \text{DL}) + (0.25 \times \text{URL}) + (0.25 \times \text{ML})$$
2. **Clean URL Mitigation**:
   * If message text exhibits smishing cues (e.g. promotional wording or urgency where $\text{DL} = 0.55$, $\text{ML} = 0.40$), but the link is verified **Clean** ($S_{\text{URL}} = 0.0$), the combined score drops below $0.50$ (**`Safe`**).
   * The explanation synthesizes this explicitly:
     > *"Although message text exhibits smishing cues, the overall message is verified as Safe because the embedded web link was verified clean."*
3. **Malicious URL Escalation**:
   * If the URL is confirmed `malicious` ($S_{\text{URL}} \ge 0.85$), the overall message escalates directly to **`Harmful`**, citing the specific flagging antivirus vendors (e.g. *"detected by Fortinet, Symantec"*).
4. **Pending URL Caution**:
   * If the URL is `pending`, the ensemble safely recalculates over text layers ($66.7\%$ DL / $33.3\%$ ML) and appends:
     > *"Exercise caution: this message contains a web link ({url}) that has not been verified by online threat intelligence yet, so its safety cannot be guaranteed."*

---

### G. Client Offline Threat Shield Synchronization

Every newly evaluated URL is stored in Supabase with host normalization (`public.url` and `public.url_analysis`). Mobile clients and local VPN filters can incrementally synchronize these cached threat definitions:

* **Endpoint**: `GET /url-reputations?since_timestamp=<epoch_millis>`
* Returns verified malicious and suspicious URLs, hostnames, and threat scores so mobile clients can block them offline or at the DNS level.

---

## 3. Folder Structure

```text
KwagoBackend/
├── app/
│   ├── __init__.py
│   ├── main.py            (FastAPI application & API routes)
│   ├── scanner.py         (Ensemble scoring, ONNX model inference, VirusTotal scanner, Supabase logging)
│   ├── schemas.py         (Pydantic request & response schemas)
│   └── source_weights.csv (Engine weights for VirusTotal threat scoring)
├── models/
│   ├── cnn_bigru_model.onnx (CNN-BiGRU ONNX model file)
│   └── tokenizer_b.pkl    (Pickle tokenizer object)
├── .env                   (Local environment variables - gitignored)
├── .gitignore
├── README.md
├── requirements.txt
└── walkthrough.md
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

Scans an SMS message text, evaluates local ML confidence, runs server CNN-BiGRU inference, and performs VirusTotal URL lookup if a web link is included.

* **Endpoint:** `POST /scan-sms`
* **Headers:** `Authorization: Bearer <KWAGO_API_KEY>`
* **Request Body:**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com",
    "has_url": true,
    "extracted_url": "http://fake-claim.com",
    "allow_save": false,
    "sender": "+639123456789",
    "ml_prediction": "smishing",
    "ml_confidence": 0.85,
    "auto_report": true
  }
  ```

* **Response Body (`SmsScanResponse`):**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com",
    "overall_verdict": "Harmful",
    "overall_score": 0.9125,
    "overall_explanation": "Both the local ML and Deep Learning layers confirmed High Risk because the message promotes unsolicited monetary bonuses, deposit rewards, or financial incentives. Furthermore, the embedded web link (http://fake-claim.com) was confirmed as a high-risk malicious phishing site.",
    "cnn_analysis": {
      "score": 0.95,
      "verdict": "harmful",
      "explanation": "Promotes unsolicited monetary bonuses, deposit rewards, or financial incentives."
    },
    "url_analysis": {
      "has_url": true,
      "extracted_url": "http://fake-claim.com",
      "score": 0.90,
      "verdict": "malicious",
      "total_weight": 14.85,
      "explanation": "This URL is flagged as malicious (threat score: 0.90). Detected by: Fortinet (malicious).",
      "contributions": [
        "Fortinet (malicious)"
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