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

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```env
   VIRUS_TOTAL_API_KEY=your_virustotal_api_key
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

## API Reference

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

Scan an SMS message using the trained CNN-BiGRU deep learning model to detect spam/phishing threats.

* **Endpoint:** `POST /scan-sms`
* **Content-Type:** `application/json`
* **Request Body:**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com"
  }
  ```

* **Example Request using `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/scan-sms" \
       -H "Content-Type: application/json" \
       -d "{\"message\": \"CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com\"}"
  ```

* **Example Response:**
  ```json
  {
    "message": "CONGRATS! You won a $1000 gift card. Claim now at http://fake-claim.com",
    "verdict": "spam",
    "probability": 0.676306
  }
  ```