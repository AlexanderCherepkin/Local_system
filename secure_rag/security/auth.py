"""JWT Token Authentication for Secure RAG."""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Set

try:
    from jose import JWTError, jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    logger = logging.getLogger("secure_rag.security.auth")

logger = logging.getLogger("secure_rag.security.auth")

JWT_SECRET_KEY = "secure-rag-jwt-secret-key"
ALGORITHM = "HS256"


@dataclass(frozen=True)
class TokenPayload:
    """Зашифрованные данные JWT-токена."""
    user_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    iat: int = 0
    exp: int = 0
    trusted: bool = True
    
    def is_expired(self) -> bool:
        if self.exp and time.time() > self.exp:
            return True
        return False


def decode_jwt_token(token: str, secret_key: Optional[str] = None) -> Optional[TokenPayload]:
    """
    Декодировать JWT токен и извлечь user_id + roles.
    
    Токен считается валидным если:
    1. Подпись корректна (fail closed)
    2. Токен не истёк
    3. Содержит required claims: user_id, roles
    """
    if not token or not isinstance(token, str):
        return None
    
    secret_key = secret_key or JWT_SECRET_KEY
    
    try:
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[ALGORITHM]
        )
    except JWTError as e:
        logger.error("JWT decode error: %s", str(e))
        return None
    
    if "user_id" not in payload or "roles" not in payload:
        logger.error("JWT missing required claims")
        return None
    
    try:
        decoded = TokenPayload(
            user_id=str(payload["user_id"]).strip(),
            roles=tuple(str(r).strip() for r in (payload.get("roles", []) or []) if str(r or "").strip()),
            iat=int(payload.get("iat", 0)),
            exp=int(payload.get("exp", 0)),
            trusted=bool(payload.get("trusted", True))
        )
    except (ValueError, TypeError) as e:
        logger.error("Invalid JWT payload: %s", str(e))
        return None
    
    if decoded.is_expired():
        logger.warning("JWT token expired for user_id=%s", decoded.user_id)
        return None
    
    return decoded


def extract_principal_from_request(
    request: Dict[str, Any],
    secret_key: Optional[str] = None,
) -> Optional[Principal]:
    """
    Извлечь Principal из HTTP-запроса через JWT токен.
    
    Токен в header Authorization или body под ключом 'token'.
    Формат токена: Bearer <jwt> или просто <jwt>
    """
    from security.acl import Principal
    
    token = None
    auth_header = request.get("headers", {}).get("authorization") or request.get("headers", {}).get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        else:
            token = auth_header
    
    body_token = request.get("body", {}).get("token") or request.get("body", {}).get("access_token")
    if not token and body_token:
        token = str(body_token)
    
    decoded_payload = decode_jwt_token(token, secret_key)
    if not decoded_payload:
        return None
    
    principal = Principal(
        user_id=decoded_payload.user_id,
        roles=decoded_payload.roles,
        trusted=decoded_payload.trusted
    )
    
    logger.info("JWT validated: user_id=%s, roles=%s", decoded_payload.user_id, decoded_payload.roles)
    return principal
