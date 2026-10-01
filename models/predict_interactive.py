import os
import re
import json
import numpy as np
import onnxruntime as ort
import requests

# Set working directory to script location to resolve model paths correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load weights configuration if exists
rf_weight = 0.75
xgb_weight = 0.25
local_weight = 0.50
cnn_weight = 0.50
suspicious_threshold = 0.50
smishing_threshold = 0.80

if os.path.exists("ml_layer_weights.json"):
    try:
        with open("ml_layer_weights.json", "r") as f:
            weights = json.load(f)
            rf_weight = float(weights.get("rf_weight", rf_weight))
            xgb_weight = float(weights.get("xgb_weight", xgb_weight))
            local_weight = float(weights.get("local_weight", local_weight))
            cnn_weight = float(weights.get("cnn_weight", cnn_weight))
            suspicious_threshold = float(weights.get("suspicious_threshold", suspicious_threshold))
            smishing_threshold = float(weights.get("smishing_threshold", smishing_threshold))
    except Exception as e:
        print(f"Warning: Could not load ml_layer_weights.json: {e}. Using defaults.")

# Constants mirroring LocalClassifier.kt
STOPWORDS = {
    "kung", "any", "shouldn't", "naman", "para", "sila", "by", "did", "they're", "under", "mo", "it'd", "alin", "isn", "because", "pa", "d", "couldn", "to", "your", "it's", "himself", "was", "her", "nang", "some", "siya", "kasi", "own", "has", "sino", "he'll", "are", "being", "you", "for", "between", "itself", "it'll", "ourselves", "mga", "ma", "not", "again", "now", "shan", "nila", "at", "out", "she'll", "have", "m", "more", "pala", "you'll", "above", "on", "shouldn", "their", "mightn", "dito", "din", "yours", "should", "you'd", "is", "into", "ll", "through", "them", "ko", "were", "no", "having", "our", "be", "myself", "re", "pero", "and", "nor", "yourself", "will", "she", "wouldn", "all", "ka", "iyon", "he", "theirs", "aren't", "once", "same", "weren't", "me", "how", "we've", "hadn't", "ang", "wala", "needn", "had", "during", "haven", "am", "couldn't", "why", "themselves", "lang", "i'm", "we're", "just", "that'll", "a", "saan", "na", "yung", "up", "they've", "ain", "natin", "rin", "yourselves", "ours", "namin", "who", "off", "kami", "opo", "hindi", "where", "as", "o", "such", "didn't", "against", "t", "s", "few", "herself", "he's", "before", "wasn", "niya", "when", "so", "doesn't", "you're", "may", "if", "haven't", "mustn", "or", "shan't", "then", "they'll", "raw", "aren", "bakit", "mightn't", "i'd", "hasn't", "we", "do", "i'll", "my", "daw", "can", "from", "doesn", "ba", "you've", "po", "weren", "tayo", "but", "other", "hasn", "below", "won", "most", "after", "each", "does", "the", "she'd", "he'd", "don't", "wasn't", "don", "didn", "ng", "that", "doing", "we'd", "i've", "whom", "won't", "i", "wouldn't", "him", "than", "its", "there", "both", "in", "what", "talaga", "until", "we'll", "ano", "here", "down", "about", "y", "too", "they'd", "should've", "of", "doon", "hadn", "been", "ay", "hers", "very", "mustn't", "with", "they", "nga", "an", "this", "ho", "ve", "she's", "further", "his", "these", "sa", "those", "isn't", "needn't", "ito", "while", "only", "which", "it"
}

PH_BANKS = ["bdo", "bpi", "metrobank", "landbank", "rcbc", "unionbank", "eastwest", "psbank", "chinabank", "security bank", "pnb", "gcash", "maya", "paymaya", "gotyme", "seabank", "tonik"]
PH_TELCOS = ["smart", "globe", "tnt", "sun", "dito", "gomo", "tm"]
PH_URGENCY = ["agad", "ngayon", "mawala", "deadline", "huling araw", "expir", "panalo", "manalo", "libreng", "libre", "premyo", "reward", "kunin", "i-click", "i-verify", "i-update", "i-confirm", "mag-claim", "i-redeem", "i-activate", "mag-log", "mag-login"]
URL_SHORTENERS = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "short.link", "rb.gy", "cutt.ly", "tiny.cc", "is.gd"]
CTA_PHRASES = ["click here", "verify now", "claim your", "act now", "limited time", "expires today", "call now", "text now", "reply now", "visit now", "click link", "tap here", "open now", "log in now", "sign in now", "update now", "confirm now", "validate now", "redeem now"]

# Preprocessing Pipeline matching LocalClassifier.kt
def clean_text(text):
    t = text
    t = re.sub(r'[\n\r\t]+', ' ', t)
    t = re.sub(r'https?://\S+|www\.\S+', ' URL ', t)
    t = re.sub(r'\+?63[\d\*]{9,10}', '', t)
    t = re.sub(r'\b0\d{10}\b', '', t)
    t = re.sub(r'\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{4}\b', '', t)
    emoji_regex = re.compile(
        u'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0\U000024C2-\U0001F251]+',
        flags=re.UNICODE)
    t = emoji_regex.sub('', t)
    t = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\s.,!?'\-]", '', t)
    t = t.lower()
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_text(text):
    if not text.strip():
        return ''
    t = text
    t = re.sub(r'(?<!\w)\d+(?!\w)', '', t)
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)

def count_occurrences(text, sub):
    count = 0
    idx = 0
    while True:
        idx = text.find(sub, idx)
        if idx != -1:
            count += 1
            idx += len(sub)
        else:
            break
    return count

def extract_numerical_features(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    url_present = 1.0 if urls else 0.0
    url_count = float(len(urls))
    has_shortener = 0.0
    has_https = 0.0
    domain_length = 0.0
    subdomain_count = 0.0
    has_ip = 0.0
    path_depth = 0.0
    url_special_chars = 0.0
    if urls:
        url = urls[0]
        has_shortener = 1.0 if any(s in url.lower() for s in URL_SHORTENERS) else 0.0
        has_https = 1.0 if url.startswith('https') else 0.0
        domain_match = re.search(r'https?://([^/]+)', url)
        domain = domain_match.group(1) if domain_match else ''
        domain_length = float(len(domain))
        subdomain_count = float(max(0, domain.count('.') - 1))
        has_ip = 1.0 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0.0
        path = url.replace(domain_match.group(0) if domain_match else '', '')
        path_depth = float(len([p for p in path.split('/') if p]))
        url_special_chars = float(len(re.findall(r'[-_~%@]', url)))
    
    char_count = float(len(text))
    word_count = float(len([w for w in text.split() if w]))
    punct_count = float(len(re.findall(r'[.,!?]', text)))
    digit_density = round(len([c for c in text if c.isdigit()]) / len(text), 4) if text else 0.0
    upper_ratio = round(len([c for c in text if c.isupper()]) / len(text), 4) if text else 0.0
    
    text_lower = text.lower()
    has_cta = 1.0 if any(p in text_lower for p in CTA_PHRASES) else 0.0
    cta_count = float(sum(count_occurrences(text_lower, p) for p in CTA_PHRASES))
    
    has_ph_bank = 1.0 if any(b in text_lower for b in PH_BANKS) else 0.0
    has_ph_telco = 1.0 if any(te in text_lower for te in PH_TELCOS) else 0.0
    has_ph_urgency = 1.0 if any(u in text_lower for u in PH_URGENCY) else 0.0
    
    return [
        url_present, url_count, has_shortener, has_https, domain_length, subdomain_count, has_ip, path_depth, url_special_chars,
        char_count, word_count, punct_count, digit_density, upper_ratio,
        has_cta, cta_count,
        has_ph_bank, has_ph_telco, has_ph_urgency
    ]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Load ONNX models
print("Loading ONNX sessions...")
try:
    scaler_session = ort.InferenceSession("scaler.onnx")
    tfidf_session = ort.InferenceSession("tfidf.onnx")
    rf_session = ort.InferenceSession("rf_model.onnx")
    xgb_session = ort.InferenceSession("xgboost_model.onnx")
    print("Models loaded successfully!\n")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def classify_interactive():
    global rf_weight, xgb_weight, local_weight, cnn_weight, suspicious_threshold, smishing_threshold
    
    # ANSI escape colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print("=" * 60)
    print(f"{CYAN}Kwago Hybrid Inference CLI Predictor{RESET}")
    print("=" * 60)
    
    # Weight modification step
    print(f"Current Weights Configuration:")
    print(f"  - Random Forest Weight (Local): {rf_weight}")
    print(f"  - XGBoost Weight (Local):       {xgb_weight}")
    print(f"  - Local Pipeline Weight (Hybrid): {local_weight}")
    print(f"  - CNN Remote API Weight (Hybrid): {cnn_weight}")
    print(f"  - Thresholds: Suspicious: {suspicious_threshold}, Smishing: {smishing_threshold}")
    
    modify = input("\nDo you want to modify these weights for this session? (y/N): ").strip().lower()
    if modify == 'y':
        try:
            rf_weight = float(input(f"Enter Random Forest weight (current {rf_weight}): ") or rf_weight)
            xgb_weight = float(input(f"Enter XGBoost weight (current {xgb_weight}): ") or xgb_weight)
            local_weight = float(input(f"Enter Local hybrid weight (current {local_weight}): ") or local_weight)
            cnn_weight = float(input(f"Enter CNN hybrid weight (current {cnn_weight}): ") or cnn_weight)
            suspicious_threshold = float(input(f"Enter Suspicious threshold (current {suspicious_threshold}): ") or suspicious_threshold)
            smishing_threshold = float(input(f"Enter Smishing threshold (current {smishing_threshold}): ") or smishing_threshold)
            
            # Normalize local weights if needed
            total_local = rf_weight + xgb_weight
            if total_local != 1.0 and total_local > 0:
                print(f"\nNote: RF and XGB weights sum to {total_local}. Normalizing them...")
                rf_weight /= total_local
                xgb_weight /= total_local
                print(f"  -> Normalized RF weight: {rf_weight:.2f}")
                print(f"  -> Normalized XGB weight: {xgb_weight:.2f}")
        except ValueError:
            print("Invalid input. Proceeding with existing weights.")

    # Main interactive loop
    while True:
        print("\n" + "-" * 50)
        message = input("Enter SMS message to classify (or type 'exit' to quit): ").strip()
        if not message or message.lower() == 'exit':
            break
            
        include_cnn = input("Include Remote CNN API prediction? (y/N): ").strip().lower() == 'y'
        
        # Preprocessing
        cleaned = clean_text(message)
        prep = preprocess_text(cleaned)
        raw_num = np.array(extract_numerical_features(message), dtype=np.float32).reshape(1, -1)
        
        # Scale structured features
        scaled_num = scaler_session.run(None, {"num_input": raw_num})[0]
        
        # Text tfidf
        text_tfidf = tfidf_session.run(None, {"text_input": np.array([[prep]], dtype=object)})[0]
        
        # Concatenate features
        combined_features = np.hstack([text_tfidf, scaled_num]).astype(np.float32)
        
        # Inference
        rf_raw = rf_session.run(None, {"float_input": combined_features})[1][0][1]
        rf_prob = sigmoid(rf_raw)
        
        xgb_prob = xgb_session.run(None, {"float_input": combined_features})[1][0][1]
        
        # Ensembled Local Score
        local_prob = rf_weight * rf_prob + xgb_weight * xgb_prob
        
        final_prob = local_prob
        cnn_prob = None
        cnn_verdict = "N/A"
        
        if include_cnn:
            print("Querying remote CNN API...")
            try:
                # 6 seconds timeout mirroring SmishingDetector.kt
                res = requests.post("https://kwagobackend.onrender.com/scan-sms", json={"message": message}, timeout=6)
                if res.status_code == 200:
                    res_json = res.json()
                    cnn_prob = float(res_json.get("probability", 0.0))
                    cnn_verdict = res_json.get("verdict", "Unknown")
                    final_prob = local_weight * local_prob + cnn_weight * cnn_prob
                    print(f"  -> CNN API Response: Probability={cnn_prob:.4f}, Verdict={cnn_verdict}")
                else:
                    print(f"  -> API error (status code {res.status_code}). Falling back to local prediction.")
            except Exception as e:
                print(f"  -> Connection to CNN API failed: {e}. Falling back to local prediction.")
        
        # Calculate final verdict
        if final_prob >= smishing_threshold:
            verdict = "SMISHING"
            color = RED
        elif final_prob >= suspicious_threshold:
            verdict = "SUSPICIOUS"
            color = YELLOW
        else:
            verdict = "SAFE"
            color = GREEN
            
        # Display Results
        print("\n" + "=" * 40)
        print(f"CLASSIFICATION REPORT:")
        print(f"=" * 40)
        print(f"Message: \"{message}\"")
        print(f"Cleaned: \"{cleaned}\"")
        print(f"Prep:    \"{prep}\"")
        print("-" * 40)
        print(f"Individual Model Predictions:")
        print(f"  - Random Forest Prob (scaled):  {rf_prob:.4f} (raw logit: {rf_raw:.4f})")
        print(f"  - XGBoost Prob:                 {xgb_prob:.4f}")
        print(f"  - Ensembled Local Score ({int(xgb_weight*100)}-{int(rf_weight*100)}): {local_prob:.4f}")
        if include_cnn:
            print(f"  - Remote CNN API Score:         {cnn_prob if cnn_prob is not None else 'N/A'}")
        print("-" * 40)
        print(f"Final Probability Score:          {CYAN}{final_prob:.4f}{RESET}")
        print(f"Final Verdict:                    {color}{verdict}{RESET}")
        print("=" * 40)

if __name__ == "__main__":
    try:
        classify_interactive()
    except KeyboardInterrupt:
        print("\nExiting CLI Predictor. Goodbye!")
