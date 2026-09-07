# KwagoBackend URL & SMS Threat Scanner

FastAPI backend for multi-layer SMS smishing classification, deep learning NLP analysis (CNN-BiGRU), and VirusTotal weighted URL threat scanning.

---

## 1. System Architecture & Decision Matrix

KwagoBackend combines three distinct threat detection layers to determine an overall classification score ($S$) and synthesize human-readable executive explanations for mobile users:

1. **Local ML Layer (Random Forest + XGBoost)**: Mobile client pre-scan confidence score.
2. **Server Deep Learning Layer (CNN-BiGRU ONNX Model)**: NLP analysis extracting linguistic urgency and scam phrasing.
3. **URL Threat Scanner Layer (VirusTotal API)**: Weighted threat reputation analysis over security vendor engines.

---

### Ensemble Decision Logic Matrix

| Case Scenario | Active Layers & Weights | Ensemble Formula |
| :--- | :--- | :--- |
| **Case 1: No Web Link** | 66.7% CNN DL + 33.3% Local ML | $S = \left(\frac{2}{3} \times \text{DL}\right) + \left(\frac{1}{3} \times \text{ML}\right)$ |
| **Case 2: Web Link Present, but Scan Pending / Unavailable** | 66.7% CNN DL + 33.3% Local ML *(+ Caution Warning Appended)* | $S = \left(\frac{2}{3} \times \text{DL}\right) + \left(\frac{1}{3} \times \text{ML}\right)$ |
| **Case 3: Web Link Present & Scan Completed** | 50% CNN DL + 25% URL Scan + 25% Local ML | $S = (0.50 \times \text{DL}) + (0.25 \times \text{URL}) + (0.25 \times \text{ML})$ |

---

### Classification Thresholds

| Ensemble Score ($S$) | Overall Verdict | Risk Level | System Behavior |
| :--- | :--- | :--- | :--- |
| **$S \ge 0.85$** | **Harmful** | High Risk | Threat warning generated; logged to database if `allow_save = true`. |
| **$0.70 \le S < 0.85$** | **Suspicious** | Medium Risk | Caution alert shown; logged to database if `allow_save = true`. |
| **$S < 0.70$** | **Safe** | Low Risk | Allowed normally; bypassed from database logging. |

---

### Explanation Synthesis Rules

* **Clean URL Mitigation**: If text analysis flags risk (ML/DL $\ge 0.70$) but an embedded web link is verified clean (`0.0`), driving $S < 0.70$, the backend outputs:  
  > *"Although message text exhibits smishing cues, the overall message is verified as Safe because the embedded web link was verified clean."*
* **Pending URL Caution Warning**: When a web link has not completed scanning, the backend appends:  
  > *"Exercise caution: this message contains a web link ({url}) that has not been verified by online threat intelligence yet, so its safety cannot be guaranteed."*

---

## 2. Folder Structure

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

## 3. Setup & Environment Configuration

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

## 4. Running the Server

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

## 5. API Reference

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
    "ml_confidence": 0.85
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
    "last_synced_at": "2026-09-07T21:50:00Z",
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