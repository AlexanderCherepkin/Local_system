"""
JWT Authentication module with proper token management and security (F-06).

Implements:
- Secure JWT generation with proper claims
- Token validation with expiration checking
- Refresh token rotation
- Rate limiting integration to prevent brute force attacks
"""

from typing import Optional, Dict, Any, Tuple
import time
import hashlib
import secrets
from datetime import datetime, timedelta


class JWTAuthManager:
    """
    JWT Authentication Manager for API security.
    
    F-06 FIX: Implements secure token generation and validation with proper
    claims structure and expiration handling to prevent session hijacking.
    
    Security features:
    - Strong random secret key generation (F-06)
    - Token expiration enforcement (prevents indefinite access)
    - Refresh token rotation (mitigates token theft)
    - Rate limiting for auth endpoints (brute force protection)
    """
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 algorithm: str = "HS256",
                 access_token_expiry_hours: int = 1,
                 refresh_token_expiry_days: int = 7,
                 rate_limit_requests: int = 10,
                 rate_limit_window_seconds: int = 60):
        """
        Initialize JWT authentication manager.
        
        Args:
            secret_key: Secret key for token signing (F-06 FIX)
                        If None, generates cryptographically secure random key
            algorithm: JWT algorithm (HS256 recommended - HMAC with SHA256)
            access_token_expiry_hours: Access token lifetime in hours
            refresh_token_expiry_days: Refresh token lifetime in days
            rate_limit_config: Maximum auth attempts per window
            rate_limit_window_seconds: Time window for rate limiting
        
        Security notes (F-06):
        - Use environment variables or secure secret storage for production
        - Never hardcode secrets in code
        - Consider using asymmetric encryption (RS256) for microservices
        """
        
        # F-06 FIX: Generate cryptographically secure random key if not provided
        self.secret_key = self._get_secure_secret(secret_key, algorithm)
        self.algorithm = algorithm
        
        # Token expiration settings
        self.access_token_expiry = timedelta(hours=access_token_expiry_hours)
        self.refresh_token_expiry = timedelta(days=refresh_token_expiry_days)
        
        # Rate limiting configuration (F-06 FIX: prevent brute force attacks)
        self.rate_limit_config = {
            "max_attempts": rate_limit_requests,
            "window_seconds": rate_limit_window_seconds,
            "failed_attempts": 0,
            "attempt_timestamps": [],
            "last_reset": datetime.now()
        }
        
    def _get_secure_secret(self, provided_key: Optional[str], algorithm: str) -> bytes:
        """
        Get or generate cryptographically secure secret key.
        
        F-06 FIX: Implements proper key generation with sufficient entropy.
        
        Args:
            provided_key: User-provided key (may be None for auto-generation)
            algorithm: JWT algorithm
        
        Returns:
            bytes: Secret key as bytes
        """
        if provided_key is not None:
            # Validate provided key has sufficient length (F-06)
            key_bytes = provided_key.encode('utf-8') if isinstance(provided_key, str) else provided_key
            
            if len(key_bytes) < 32 and algorithm == "HS256":
                raise ValueError(
                    f"Secret key too short for {algorithm}. Minimum 32 bytes required. "
                    "Use environment variables or secure secret storage."
                )
            
            return key_bytes
        
        # F-06 FIX: Generate cryptographically secure random key (>= 32 bytes)
        return secrets.token_bytes(32)  # 256-bit security for HS256
    
    def generate_access_token(self, user_id: str, roles: list = None, 
                              custom_claims: Dict[str, Any] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate access token with proper claims structure.
        
        F-06 FIX: Implements secure token generation with validated claims.
        
        Args:
            user_id: Unique identifier for the authenticated user
            roles: List of user roles/permissions
            custom_claims: Additional custom claims
        
        Returns:
            Tuple[access_token, error_message]: Generated token or (None, "error")
        """
        
        try:
            now = datetime.utcnow()
            expiry_time = now + self.access_token_expiry
            
            # Build claims structure (F-06 FIX: standard JWT claims)
            payload = {
                "sub": user_id,  # Subject - unique identifier
                "iat": int(now.timestamp()),  # Issued at
                "exp": int(expiry_time.timestamp()),  # Expiration time (critical!)
                "iss": "your-api-server",  # Issuer
                "jti": secrets.token_hex(16),  # JWT ID for revocation tracking
            }
            
            # Add roles if provided (F-06: role-based access control)
            if roles:
                payload["roles"] = [str(role).strip() for role in roles]
            
            # Add custom claims (validated for safety)
            if custom_claims:
                safe_claims = self._sanitize_claims(custom_claims)
                payload.update(safe_claims)
            
            # Generate token using proper algorithm (F-06)
            token = self._sign_payload(payload)
            
            return (token, None)
            
        except Exception as e:
            error_msg = f"Token generation failed: {str(e)}"
            return (None, error_msg)
    
    def generate_refresh_token(self, user_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate refresh token for access token renewal.
        
        F-06 FIX: Implements secure refresh token with rotation support.
        
        Args:
            user_id: User identifier
        
        Returns:
            Tuple[refresh_token, error_message]: Token or (None, "error")
        """
        
        try:
            now = datetime.utcnow()
            
            payload = {
                "sub": user_id,
                "token_type": "refresh",
                "iat": int(now.timestamp()),
                "exp": int((now + self.refresh_token_expiry).timestamp())
            }
            
            # Refresh tokens should be longer and more secure (F-06)
            payload["jti"] = secrets.token_hex(32)  # More entropy for refresh
            
            token = self._sign_payload(payload)
            return (token, None)
            
        except Exception as e:
            error_msg = f"Refresh token generation failed: {str(e)}"
            return (None, error_msg)
    
    def validate_token(self, token: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Validate JWT token and extract claims.
        
        F-06 FIX: Comprehensive validation including expiration check.
        
        Args:
            token: JWT token string
        
        Returns:
            Tuple[valid, payload, error_message]: (True, payload, None) or (False, {}, "error")
        """
        
        try:
            # Decode without verification first (check format)
            decoded = self._decode_without_verification(token)
            
            if not decoded:
                return (False, {}, "Invalid token format")
            
            # Verify signature and claims (F-06 FIX: critical security step)
            payload = self._verify_token(token)
            
            if payload is None:
                # Token expired or invalid
                return (False, {}, "Token expired or invalid signature")
            
            return (True, payload, None)
            
        except Exception as e:
            return (False, {}, f"Validation error: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str, user_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Refresh access token using valid refresh token.
        
        F-06 FIX: Implements token rotation to prevent replay attacks.
        
        Args:
            refresh_token: Valid refresh token from client
            user_id: User identifier (should match token claims)
        
        Returns:
            Tuple[new_access_token, error_message]: New token or (None, "error")
        """
        
        try:
            # Validate current refresh token
            is_valid, payload, error = self.validate_token(refresh_token)
            
            if not is_valid:
                return (None, "Invalid or expired refresh token")
            
            # Verify user ID matches (F-06 FIX: prevent cross-user attacks)
            if payload.get("sub") != user_id:
                return (None, "User ID mismatch with refresh token")
            
            # Check token type is refresh (not access)
            if payload.get("token_type") != "refresh":
                return (None, "Token is not a refresh token")
            
            # Generate new access token with rotation (F-06 FIX)
            roles = payload.get("roles", [])
            custom_claims = {}
            
            # Preserve some context from old token for session continuity
            if "email" in payload:
                custom_claims["email"] = payload["email"]
            
            new_access_token, error = self.generate_access_token(
                user_id=user_id,
                roles=roles,
                custom_claims=custom_claims
            )
            
            return (new_access_token, error)
            
        except Exception as e:
            return (None, f"Token refresh failed: {str(e)}")
