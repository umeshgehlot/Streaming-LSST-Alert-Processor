import base64
import hashlib
import hmac
import json
import os
import time

from fastapi import Header, HTTPException
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> bool:
        return False

from database import fetch_user_by_email, hash_password

load_dotenv()


AUTH_SECRET = os.environ.get("ASTRO_AUTH_SECRET", "local-dev-secret-change-me")
TOKEN_TTL_SECONDS = 60 * 60 * 8


def base64_url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def base64_url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def sign_payload(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_encoded = base64_url_encode(body)
    signature = hmac.new(AUTH_SECRET.encode("utf-8"), body_encoded.encode("utf-8"), hashlib.sha256).digest()
    signature_encoded = base64_url_encode(signature)
    return f"{body_encoded}.{signature_encoded}"


def parse_token(token: str) -> dict:
    try:
        body_encoded, signature_encoded = token.split(".", maxsplit=1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    expected_signature = hmac.new(AUTH_SECRET.encode("utf-8"), body_encoded.encode("utf-8"), hashlib.sha256).digest()
    provided_signature = base64_url_decode(signature_encoded)
    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    payload = json.loads(base64_url_decode(body_encoded).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


def authenticate_user(email: str, password: str) -> dict:
    user = fetch_user_by_email(email)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    computed = hash_password(password, user["salt"])
    if not hmac.compare_digest(computed, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


def create_access_token(user: dict) -> str:
    now = int(time.time())
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "name": user["full_name"],
        "role": user["role"],
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return sign_payload(payload)


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", maxsplit=1)[1]
    payload = parse_token(token)
    return {
        "id": payload["sub"],
        "email": payload["email"],
        "full_name": payload["name"],
        "role": payload["role"],
    }
