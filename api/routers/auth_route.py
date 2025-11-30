"""
Auth Router - Authentication endpoints.

Provides:
- POST /auth/token - Get JWT token (login)
- POST /auth/register - Register new user
- POST /auth/refresh - Refresh access token
- GET /auth/me - Get current user
- POST /auth/api-keys - Create API key
- DELETE /auth/api-keys/{key} - Revoke API key
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger

from auth import (
    Token,
    UserCreate,
    UserResponse,
    APIKeyCreate,
    APIKeyResponse,
    User,
    auth_settings,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_active_user,
    UserStore,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# TOKEN ENDPOINTS
# =============================================================================

@router.post(
    "/token",
    response_model=Token,
    summary="Login and get access token",
    description="Authenticate with username and password to receive JWT tokens."
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 compatible token endpoint.
    
    Use username and password to get access and refresh tokens.
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    access_token = create_access_token(user.username)
    refresh_token = create_refresh_token(user.username)
    
    logger.info(f"🔑 User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="Use refresh token to get a new access token."
)
async def refresh_token(refresh_token: str) -> Token:
    """Exchange refresh token for new access token."""
    token_data = decode_token(refresh_token)
    
    if not token_data or token_data.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user = UserStore.get_user(token_data.username)
    if not user or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )
    
    new_access_token = create_access_token(user.username)
    
    return Token(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# =============================================================================
# USER MANAGEMENT
# =============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account."
)
async def register(user_data: UserCreate) -> UserResponse:
    """Register a new user."""
    try:
        user = UserStore.create_user(user_data)
        logger.info(f"👤 New user registered: {user.username}")
        
        return UserResponse(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get the currently authenticated user's information."
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> UserResponse:
    """Get current user info."""
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
    )


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Generate a new API key for programmatic access. The key is only shown once!"
)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> APIKeyResponse:
    """
    Create a new API key.
    
    **Important**: The API key is only shown once! Store it securely.
    """
    try:
        api_key = UserStore.create_api_key(current_user.username, key_data.name)
        logger.info(f"🔑 API key created for user: {current_user.username}")
        
        return APIKeyResponse(
            key=api_key,
            name=key_data.name,
            created_at=current_user.created_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/api-keys/{api_key}",
    summary="Revoke API key",
    description="Revoke an API key. This action cannot be undone."
)
async def revoke_api_key(
    api_key: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """Revoke an API key."""
    # Verify the key belongs to this user
    if api_key not in current_user.api_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or doesn't belong to you",
        )
    
    success = UserStore.revoke_api_key(api_key)
    
    if success:
        logger.info(f"🔑 API key revoked for user: {current_user.username}")
        return {"message": "API key revoked"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )


@router.get(
    "/api-keys",
    summary="List API keys",
    description="List all API keys for the current user (keys are masked)."
)
async def list_api_keys(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> list:
    """List API keys (masked)."""
    return [
        {
            "key": f"{key[:8]}...{key[-4:]}",
            "full_key_prefix": key[:12],
        }
        for key in current_user.api_keys
    ]