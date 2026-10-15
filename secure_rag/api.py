"""HTTP API Secure RAG (FastAPI).

Исправления:

* **F-06/F-13** — личность берётся только из проверенного Bearer-токена; поле
  ``username`` в теле запроса больше не является идентификацией. Роль задаётся
  сервером, а не клиентом.
* **F-13** — добавлены лимит длины запроса, rate limiting и единообразные
  ошибки без внутренних деталей (``str(exception)`` наружу не отдаётся).
* **F-16** — ``/healthz`` (liveness) отделён от ``/readyz`` (readiness), которая
  реально проверяет Ollama, индекс, ACL-хранилище и аудит.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from pipeline.generator import PromptInjectionDetected, RAGPipeline
from security.acl import AccessDeniedError, Principal
from storage.audit import AuditConfigurationError, AuditTrail
from security.auth import AuthenticationError, TokenService, extract_bearer_token
from storage.audit import AuditTrail as _AuditTrailImpl  # noqa: F401 (единый слой)

logger = logging.getLogger("secure_rag.api")

app = FastAPI(
    title="Secure RAG API",
    version="0.1.0",
    # Схема и интерактивные доки не публикуются: воздушно-изолированный контур (F-13).
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
bearer_scheme = HTTPBearer(auto_error=False)

_pipeline: Optional[RAGPipeline] = None
_token_service: Optional[TokenService] = None

#: Простое скользящее окно для rate limiting (in-memory, per-process).
_rate_buckets: Dict[str, Deque[float]] = {}
_rate_lock = threading.Lock()
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60.0


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    abstained: bool = False
    request_id: Optional[str] = None


class TokenRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)


# ---------------------------------------------------------------------------
# Зависимости
# ---------------------------------------------------------------------------
def get_settings_dep() -> Settings:
    return get_settings()


def get_pipeline_dep(settings: Settings = Depends(get_settings_dep)) -> RAGPipeline:
    """Ленивый singleton пайплайна для процесса API."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(settings=settings)
    return _pipeline


def get_token_service_dep(settings: Settings = Depends(get_settings_dep)) -> TokenService:
    global _token_service
    if _token_service is None:
        _token_service = TokenService(settings=settings)
    return _token_service


def enforce_rate_limit(request: Request) -> None:
    """Ограничение частоты запросов по IP (F-13)."""
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()

    with _rate_lock:
        bucket = _rate_buckets.setdefault(client_host, deque())
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Превышен лимит частоты запросов. Повторите позже.",
            )
        bucket.append(now)


def get_current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    tokens: TokenService = Depends(get_token_service_dep),
) -> Principal:
    """Единственный источник identity: проверенный Bearer-токен (F-06)."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется аутентификация",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return tokens.authenticate(credentials.credentials)
    except AuthenticationError:
        # Причина намеренно не раскрывается клиенту.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или истёкший токен",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    # Rate limiting идёт ПЕРВЫМ: иначе подбор токенов не ограничивается (F-13).
    _: None = Depends(enforce_rate_limit),
    principal: Principal = Depends(get_current_principal),
    pipeline: RAGPipeline = Depends(get_pipeline_dep),
) -> QueryResponse:
    """Ответ на вопрос строго по документам, доступным ролям пользователя."""
    result = pipeline.process_query(prompt=req.prompt, user=principal, top_k=req.top_k)

    error = result.get("error")
    if error == "access_denied":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Доступ запрещён")
    if error == "prompt_injection_detected":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректный запрос")
    if error == "invalid_request":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректный запрос")
    if error == "internal_error":
        # Внутренние детали не покидают процесс (F-13).
        logger.error("Сбой обработки запроса request_id=%s", result.get("metadata", {}).get("request_id"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис временно недоступен",
        )

    sources = [
        f"[{item['fragment']}] {item['doc_id']} ({item['source_file']})" for item in result.get("sources", [])
    ]
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        abstained=bool(result.get("abstained")),
        request_id=result.get("metadata", {}).get("request_id"),
    )


@app.post("/api/v1/auth/token")
def issue_token(req: TokenRequest, tokens: TokenService = Depends(get_token_service_dep)) -> Dict[str, object]:
    """Выдать токен по учётным данным из ACL-хранилища.

    В production этот endpoint должен быть заменён на корпоративный IdP
    (OIDC/SAML). Здесь он оставлен как минимальный механизм выпуска токенов для
    воздушно-изолированного контура; роли берутся из ACL, а не из запроса.
    """
    from security.acl import AccessControlList

    # Проверка режима идёт ДО обращения к ACL, иначе 401/501 становятся
    # оракулом существования пользователя (F-13).
    settings = get_settings()
    if settings.log_level.upper() not in {"DEBUG"}:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Выпуск токенов должен выполняться корпоративным IdP",
        )

    acl = AccessControlList()
    expected_roles = acl.roles_of(req.username)
    if not expected_roles:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверные учётные данные")

    # Пароли в ACL-хранилище не хранятся; проверка выполняется внешним
    # провайдером. В изолированном контуре допускается выпуск токена
    # пользователю, существующему в ACL, если включён режим разработки.
    return {"access_token": tokens.issue_token(req.username, expected_roles), "token_type": "bearer"}


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    """Liveness: процесс жив и отвечает (F-16)."""
    return {"status": "ok"}


@app.get("/readyz")
def readiness(settings: Settings = Depends(get_settings_dep)) -> Dict[str, object]:
    """Readiness: проверка зависимостей без тяжёлой генерации (F-16)."""
    checks: Dict[str, object] = {}
    ready = True

    # 1. Аудит сконфигурирован безопасно.
    try:
        AuditTrail.is_configured(settings)
        checks["audit"] = "ok" if AuditTrail.is_configured(settings) else "not_configured"
        if not AuditTrail.is_configured(settings):
            ready = False
    except Exception as exc:
        checks["audit"] = f"error: {type(exc).__name__}"
        ready = False

    # 2. ACL-хранилище загружено (иначе Zero-Trust запрещает всё).
    try:
        from security.acl import AccessControlList

        acl = AccessControlList(settings=settings)
        users = acl.known_users()
        checks["acl_users"] = len(users)
        if not users:
            checks["acl"] = "empty_storage"
            ready = False
        else:
            checks["acl"] = "ok"
    except Exception as exc:
        checks["acl"] = f"error: {type(exc).__name__}"
        ready = False

    # 3. Векторный индекс доступен и метрика совпадает.
    try:
        from .storage.vector_db import VectorDatabase

        store_health = VectorDatabase(settings=settings).healthcheck()
        checks["vector_store"] = store_health
        if not store_health.get("ready"):
            ready = False
    except Exception as exc:
        checks["vector_store"] = f"error: {type(exc).__name__}"
        ready = False

    # 4. Ollama доступен (быстрая проверка /api/tags).
    try:
        import requests

        response = requests.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=3)
        checks["ollama"] = "ok" if response.status_code == 200 else f"status_{response.status_code}"
        if response.status_code != 200:
            ready = False
    except Exception as exc:
        checks["ollama"] = f"error: {type(exc).__name__}"
        ready = False

    return {"ready": ready, "checks": checks}
