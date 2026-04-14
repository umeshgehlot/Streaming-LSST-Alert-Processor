import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

logger = logging.getLogger("astronomy_api")

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app_limiter = limiter

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Token handling
security = HTTPBearer()


class RateLimitConfig:
    """Rate limit configuration per endpoint"""
    LIMITS = {
        "login": "5/minute",
        "upload": "3/minute",
        "train": "10/hour",
        "detect": "60/minute",
        "default": "100/minute"
    }


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Check if token is expired
        exp = payload.get("exp")
        if exp is None or datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        logger.error(f"JWT verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


class TokenManager:
    """Manage JWT tokens with blacklist support"""

    def __init__(self):
        self.blacklist: Dict[str, datetime] = {}
        self.refresh_tokens: Dict[str, str] = {}

    def revoke_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            exp = payload.get("exp")
            if exp:
                self.blacklist[token] = datetime.fromtimestamp(exp, timezone.utc)
        except JWTError:
            pass

    def is_token_revoked(self, token: str) -> bool:
        """Check if token is revoked"""
        if token in self.blacklist:
            # Clean up expired tokens
            if datetime.now(timezone.utc) > self.blacklist[token]:
                del self.blacklist[token]
                return False
            return True
        return False

    def generate_refresh_token(self, user_id: str) -> str:
        """Generate refresh token"""
        refresh_token_data = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.now(timezone.utc) + timedelta(days=7)
        }
        refresh_token = jwt.encode(refresh_token_data, SECRET_KEY, algorithm=ALGORITHM)
        self.refresh_tokens[user_id] = refresh_token
        return refresh_token


# Global token manager
token_manager = TokenManager()


def get_current_user_active(credentials: HTTPAuthorizationCredentials = security) -> dict:
    """Get current active user with token validation"""
    token = credentials.credentials

    # Check if token is revoked
    if token_manager.is_token_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token using auth.py's parse_token for consistency
    from auth import parse_token
    payload = parse_token(token)

    return {
        "id": payload["sub"],
        "email": payload["email"],
        "full_name": payload["name"],
        "role": payload["role"],
    }


def get_rate_limit(endpoint: str) -> str:
    """Get rate limit for endpoint"""
    return RateLimitConfig.LIMITS.get(endpoint, RateLimitConfig.LIMITS["default"])


class SecurityHeaders:
    """Security headers middleware"""

    @staticmethod
    def get_headers() -> Dict[str, str]:
        """Get security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }


def validate_ip_address(request: Request) -> str:
    """Get and validate client IP address"""
    # Check for forwarded IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return get_remote_address(request)


def sanitize_input(input_str: str) -> str:
    """Basic input sanitization"""
    if not input_str:
        return ""

    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", '"', "'", "/"]
    sanitized = input_str
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")

    return sanitized.strip()