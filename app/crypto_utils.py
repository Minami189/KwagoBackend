import os
import re
from typing import Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

_RAW_KEY = os.getenv("DB_ENCRYPTION_KEY")
_cipher: Optional[Fernet] = None

if _RAW_KEY:
    try:
        _cipher = Fernet(_RAW_KEY.strip().encode("utf-8"))
    except Exception as e:
        print(f"Warning: Failed to initialize Fernet cipher with DB_ENCRYPTION_KEY: {e}")
        _cipher = None
else:
    print("Warning: DB_ENCRYPTION_KEY is not configured in .env. Text encryption will be bypassed.")


def anonymize_pii(text: str) -> str:
    """
    Masks personal identifiers (phone numbers and OTP digits) before encryption
    so that even decrypted training data remains compliant with privacy standards.
    """
    if not isinstance(text, str) or not text:
        return text
    # Mask 4 to 8 digit OTP numbers
    text = re.sub(r"\b\d{4,8}\b", "<OTP>", text)
    # Mask PH phone numbers (+639xxxxxxxxx or 09xxxxxxxxx)
    text = re.sub(r"(\+?639\d{9}|09\d{9})", "<PHONE>", text)
    return text


def encrypt_text(plaintext: Optional[str], scrub_pii: bool = False) -> Optional[str]:
    """
    Encrypts a string into a URL-safe Fernet ciphertext token.
    If no encryption key is set, falls back to the original string.
    """
    if plaintext is None or not isinstance(plaintext, str) or not plaintext.strip():
        return plaintext
    if not _cipher:
        return plaintext
    try:
        content = anonymize_pii(plaintext) if scrub_pii else plaintext
        return _cipher.encrypt(content.encode("utf-8")).decode("utf-8")
    except Exception as e:
        print(f"Error encrypting text: {e}")
        return plaintext


def decrypt_text(ciphertext: Optional[str]) -> Optional[str]:
    """
    Decrypts ciphertext back into the original plaintext.
    Falls back gracefully if the record was saved prior to encryption.
    """
    if ciphertext is None or not isinstance(ciphertext, str) or not ciphertext.strip():
        return ciphertext
    if not _cipher:
        return ciphertext
    try:
        return _cipher.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception:
        # Returns raw text if it was not encrypted (backwards compatibility)
        return ciphertext
