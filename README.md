# KwagoBackend

FastAPI backend for threat scanning and analyzing URLs using the VirusTotal API with a custom weighted final verdict algorithm.

## Folder Structure

```text
Minami189/KwagoBackend
├── app/
│   ├── __init__.py
│   ├── main.py            (FastAPI routing & application startup)
│   ├── schemas.py         (Pydantic request & response models)
│   ├── scanner.py         (VirusTotal scanning & weighted score computation logic)
│   └── source_weights.csv (Weights of different VirusTotal engines)
├── .env                   (Local environment configuration - gitignored)
├── .gitignore
├── README.md
├── requirements.txt
└── weighted_verdict_plan.md
```

## Setup

1. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```env
   VIRUS_TOTAL_API_KEY=your_virustotal_api_key
   ```

5. **Exiting the Virtual Environment:**
   To deactivate the environment when you are done, run:
   ```bash
   deactivate
   ```

## Usage

Start the development server using uvicorn from the root directory:
```bash
python -m app.main
```
or:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. You can test the endpoints interactively at:
* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

### Exposing the Backend (For Mobile Devices & External Access)

#### Option 1: Expose to Local Wi-Fi Network (Same Wi-Fi)
To allow physical Android devices on the same Wi-Fi network to access the server:
1. Start Uvicorn listening on all network interfaces:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Find your computer's IPv4 address (`ipconfig` on Windows or `ifconfig` on macOS/Linux).
3. Set your Android app's `API_URL` to `http://<YOUR_IPV4_ADDRESS>:8000/`.

#### Option 2: Expose to Public Internet (via Tunnels)
To get a public HTTPS URL accessible from anywhere:
* **Using ngrok:**
  ```bash
  ngrok http 8000
  ```
* **Using localtunnel:**
  ```bash
  npx localtunnel --port 8000
  ```
Set your Android app's `API_URL` to the generated HTTPS forwarding URL.

## API Reference

### Health Check

Verify that the service is running and checking classifier model loading status.

* **Endpoint:** `GET /health`
* **Example Request using `curl`:**
  ```bash
  curl -X GET "http://localhost:8000/health"
  ```
* **Example Response:**
  ```json
  {
    "status": "healthy",
    "classifier_loaded": true
  }
  ```

### Scan URL

Submit a URL for threat analysis and custom weighted scoring.

* **Endpoint:** `POST /scan`
* **Content-Type:** `application/json`
* **Request Body:**
  ```json
  {
    "url": "http://malicious-web-example.com"
  }
  ```

* **Example Response:**
  ```json
  {
    "url": "http://malicious-web-example.com",
    "verdict": "suspicious",
    "normalized_score": 0.456,
    "total_weight": 14.85,
    "explanation": "Final verdict explanation:\nweighted_score = ...",
    "contributions": [
      "Microsoft: category=clean, score=0.0, weight=0.95",
      "Kaspersky: category=malicious, score=1.0, weight=0.9"
    ],
    "raw_results": { ... }
  }
  ```

### Check Quota

Retrieve the current VirusTotal usage limits and API quotas.

* **Endpoint:** `GET /quota`
* **Example Request using `curl`:**
  ```bash
  curl -X GET "http://localhost:8000/quota"
  ```

### Scan SMS Message

Scan an SMS message and receive both the **CNN-BiGRU deep learning model score** and the **VirusTotal URL threat score**. The Android client uses both sub-scores to make its final client-side ensemble decision.

* **Endpoint:** `POST /scan-sms`
* **Content-Type:** `application/json`
* **Request Body:**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com",
    "has_url": true,
    "extracted_url": "http://fake-claim.com"
  }
  ```

* **Example Request using `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/scan-sms" \
       -H "Content-Type: application/json" \
       -d "{\"message\": \"CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com\", \"has_url\": true, \"extracted_url\": \"http://fake-claim.com\"}"
  ```


* **Example Response (SMS with URL):**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com",
    "cnn_analysis": {
      "score": 0.6763,
      "verdict": "spam"
    },
    "url_analysis": {
      "has_url": true,
      "extracted_url": "http://fake-claim.com",
      "score": 0.8500,
      "verdict": "malicious",
      "total_weight": 14.85,
      "explanation": "Final verdict explanation:\nweighted_score = 12.6225 / 14.8500 = 0.8500\nFinal verdict threshold result: malicious\nEngine contributions:\n...",
      "contributions": [
        "Microsoft: category=clean, score=0.0, weight=0.95",
        "Kaspersky: category=malicious, score=1.0, weight=0.9"
      ]
    }
  }
  ```

* **Example Response (SMS without URL):**
  ```json
  {
    "message": "Hey mom, I will be home for dinner around 7pm.",
    "cnn_analysis": {
      "score": 0.0018,
      "verdict": "benign"
    },
    "url_analysis": {
      "has_url": false,
      "extracted_url": null,
      "score": null,
      "verdict": null,
      "total_weight": null,
      "explanation": "No URL found in the SMS message.",
      "contributions": []
    }
  }
  ```

#### Response Fields Explanation (For Android Client Ensemble Decision)

* `message` (`string`): The original raw SMS text analyzed.
* `cnn_analysis` (`object`):
  * `score` (`float`, `0.0` - `1.0`): The raw threat/spam probability score computed by the CNN-BiGRU deep learning model.
  * `verdict` (`string`): CNN model classification (`"spam"` if `score >= 0.5`, else `"benign"`).
* `url_analysis` (`object`):
  * `has_url` (`boolean`): Indicates whether an HTTP/HTTPS/WWW URL was extracted from the SMS.
  * `extracted_url` (`string|null`): The primary URL extracted from the SMS message.
  * `score` (`float|null`, `0.0` - `1.0`): VirusTotal weighted threat score (`0.0` safe to `1.0` dangerous), or `null` if no URL is present.
  * `verdict` (`string|null`): VirusTotal URL threat classification (`"malicious"`, `"suspicious"`, `"benign"`, or `null`).
  * `total_weight` (`float|null`): Sum of weights of VirusTotal scanning engines evaluated.
  * `explanation` (`string|null`): Detailed calculation formula breakdown.
  * `contributions` (`array`): Individual engine scanning scores and assigned weights.