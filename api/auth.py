"""
Authentication Module - JWT-based authentication.

Provides:
- JWT token generation and validation
- API key authentication (for programmatic access)
- User management (simple in-memory for now, swap for DB later)
- Password hashing with bcrypt
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Annotated
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

class AuthSettings:
    """Authentication configuration."""
    
    # JWT Settings
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Key Settings
    API_KEY_HEADER: str = "X-API-Key"
    
    # Password Settings
    MIN_PASSWORD_LENGTH: int = 8


auth_settings = AuthSettings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# API Key header
api_key_header = APIKeyHeader(name=auth_settings.API_KEY_HEADER, auto_error=False)


# =============================================================================
# MODELS
# =============================================================================

class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    api_keys: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class UserInDB(User):
    """User with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Data encoded in JWT."""
    username: str
    exp: datetime
    type: str = "access"  # "access" or "refresh"


class UserCreate(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[str] = None
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response (no sensitive data)."""
    username: str
    email: Optional[str]
    full_name: Optional[str]
    created_at: datetime


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., description="Name/description for this API key")


class APIKeyResponse(BaseModel):
    """API key response (key shown only once)."""
    key: str
    name: str
    created_at: datetime


# =============================================================================
# USER STORE (In-Memory - Replace with DB in production)
# =============================================================================

class UserStore:
    """
    Simple in-memory user store.
    
    In production, replace with database queries.
    """
    
    _users: dict[str, UserInDB] = {}
    _api_keys: dict[str, str] = {}  # key -> username
    
    @classmethod
    def initialize_default_user(cls):
        """Create default admin user if no users exist."""
        if not cls._users:
            default_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "changeme123")
            cls.create_user(UserCreate(
                username="admin",
                password=default_password,
                email="admin@localhost",
                full_name="Administrator",
            ))
            logger.info("📝 Created default admin user (username: admin)")
    
    @classmethod
    def get_user(cls, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return cls._users.get(username)
    
    @classmethod
    def get_user_by_api_key(cls, api_key: str) -> Optional[UserInDB]:
        """Get user by API key."""
        username = cls._api_keys.get(api_key)
        if username:
            return cls._users.get(username)
        return None
    
    @classmethod
    def create_user(cls, user_data: UserCreate) -> UserInDB:
        """Create a new user."""
        if user_data.username in cls._users:
            raise ValueError(f"User {user_data.username} already exists")
        
        hashed_password = pwd_context.hash(user_data.password)[:72]
        user = UserInDB(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
        )
        cls._users[user.username] = user
        return user
    
    @classmethod
    def create_api_key(cls, username: str, name: str) -> str:
        """Create API key for user."""
        user = cls._users.get(username)
        if not user:
            raise ValueError(f"User {username} not found")
        
        # Generate secure API key
        api_key = f"dsa_{secrets.token_urlsafe(32)}"
        
        cls._api_keys[api_key] = username
        user.api_keys.append(api_key)
        
        return api_key
    
    @classmethod
    def revoke_api_key(cls, api_key: str) -> bool:
        """Revoke an API key."""
        username = cls._api_keys.pop(api_key, None)
        if username and username in cls._users:
            user = cls._users[username]
            if api_key in user.api_keys:
                user.api_keys.remove(api_key)
            return True
        return False


# =============================================================================
# PASSWORD & TOKEN UTILITIES
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        "sub": username,
        "exp": expire,
        "type": "access",
    }
    
    return jwt.encode(to_encode, auth_settings.SECRET_KEY, algorithm=auth_settings.ALGORITHM)


def create_refresh_token(username: str) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=auth_settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": username,
        "exp": expire,
        "type": "refresh",
    }
    
    return jwt.encode(to_encode, auth_settings.SECRET_KEY, algorithm=auth_settings.ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token,
            auth_settings.SECRET_KEY,
            algorithms=[auth_settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        exp = datetime.fromtimestamp(payload.get("exp", 0))
        token_type: str = payload.get("type", "access")
        
        if username is None:
            return None
        
        return TokenData(username=username, exp=exp, type=token_type)
        
    except JWTError:
        return None


# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = UserStore.get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user_optional(
    token: Annotated[Optional[str], Depends(oauth2_scheme)],
    api_key: Annotated[Optional[str], Security(api_key_header)],
) -> Optional[User]:
    """
    Get current user from JWT token or API key.
    Returns None if no valid authentication provided.
    """
    # Try API key first
    if api_key:
        user = UserStore.get_user_by_api_key(api_key)
        if user and not user.disabled:
            return user
    
    # Try JWT token
    if token:
        token_data = decode_token(token)
        if token_data and token_data.type == "access":
            user = UserStore.get_user(token_data.username)
            if user and not user.disabled:
                return user
    
    return None


async def get_current_user(
    user: Annotated[Optional[User], Depends(get_current_user_optional)]
) -> User:
    """
    Get current user - raises 401 if not authenticated.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Get current active (non-disabled) user."""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


# =============================================================================
# WEBSOCKET AUTHENTICATION
# =============================================================================

async def get_user_from_token(token: str) -> Optional[User]:
    """Authenticate user from token string (for WebSocket)."""
    if not token:
        return None
    
    # Check if it's an API key
    if token.startswith("dsa_"):
        user = UserStore.get_user_by_api_key(token)
        if user and not user.disabled:
            return user
    
    # Try as JWT
    token_data = decode_token(token)
    if token_data and token_data.type == "access":
        user = UserStore.get_user(token_data.username)
        if user and not user.disabled:
            return user
    
    return None


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_auth():
    """Initialize authentication system."""
    UserStore.initialize_default_user()
    logger.info("🔐 Authentication system initialized")