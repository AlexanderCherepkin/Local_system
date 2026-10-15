"""Единый аудит-слой Secure RAG (F-15).

Раньше существовало три несвязанные реализации: ``storage/vector_db.py`` писал
JSONL вручную и оставлял ``request_hash`` пустым, ``storage/audit_pipeline.py``
использовал HMAC, но при отсутствии ``AUDIT_SECRET`` молча брал фиксированный
``fallback-secret-replace-in-production``, а ``audit/doc_ids.py`` дублировал
схему. Здесь — один интерфейс с общими правилами:

* **fail-fast**: при отсутствии или заведомо небезопасном ``AUDIT_SECRET``
  аудит не стартует (``AuditConfigurationError``), а не подписывает события
  предсказуемым ключом;
* ``request_hash`` вычисляется через HMAC-SHA256 и **никогда** не содержит
  исходный запрос; сам запрос в аудит не пишется (только его длина и хэш);
* каждая запись несёт ``request_id``, principal, версии моделей, фактические
  ``doc_ids``, решение ACL, задержки этапов и статус;
* формат — JSONL с ротацией (архивация, а не удаление).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config.settings import INSECURE_AUDIT_SECRETS, Settings, get_settings

logger = logging.getLogger("secure_rag.storage.audit")

AUDIT_LOGGER_NAME = "secure_rag.audit_trail"
SCHEMA_VERSION = "1.0"


class AuditConfigurationError(RuntimeError):
    """Аудит не сконфигурирован безопасно (fail-fast)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditRecord:
    """Запись аудита.

    Реализована без pydantic намеренно: слой должен работать в окружении, где
    установлен только минимум зависимостей (air-gapped сборка).
    """

    FIELDS = (
        "schema_version",
        "request_id",
        "key_id",
        "timestamp",
        "user_id",
        "roles",
        "request_hash",
        "request_length",
        "stage",
        "doc_ids_attempted",
        "doc_ids_extracted",
        "doc_ids_denied",
        "acl_decision",
        "redaction_applied",
        "citations_valid",
        "model_embedding",
        "model_generation",
        "latency_embedding_ms",
        "latency_search_ms",
        "latency_generation_ms",
        "latency_total_ms",
        "status",
        "error_type",
    )

    def __init__(self, **values: Any) -> None:
        self.schema_version: str = str(values.get("schema_version", SCHEMA_VERSION))
        self.request_id: str = str(values.get("request_id") or uuid.uuid4())
        self.key_id: str = str(values.get("key_id", "key-v1"))
        self.timestamp: str = str(values.get("timestamp") or utc_now_iso())
        self.user_id: str = str(values.get("user_id", ""))
        self.roles: List[str] = list(values.get("roles") or [])
        self.request_hash: str = str(values.get("request_hash", ""))
        self.request_length: int = int(values.get("request_length", 0) or 0)
        self.stage: str = str(values.get("stage", "query"))
        self.doc_ids_attempted: List[str] = list(values.get("doc_ids_attempted") or [])
        self.doc_ids_extracted: List[str] = list(values.get("doc_ids_extracted") or [])
        self.doc_ids_denied: List[str] = list(values.get("doc_ids_denied") or [])
        self.acl_decision: str = str(values.get("acl_decision", "allow"))
        self.redaction_applied: bool = bool(values.get("redaction_applied", False))
        self.citations_valid: bool = bool(values.get("citations_valid", True))
        self.model_embedding: str = str(values.get("model_embedding", ""))
        self.model_generation: str = str(values.get("model_generation", ""))
        self.latency_embedding_ms: float = float(values.get("latency_embedding_ms", 0.0) or 0.0)
        self.latency_search_ms: float = float(values.get("latency_search_ms", 0.0) or 0.0)
        self.latency_generation_ms: float = float(values.get("latency_generation_ms", 0.0) or 0.0)
        self.latency_total_ms: float = float(values.get("latency_total_ms", 0.0) or 0.0)
        self.status: str = str(values.get("status", "success"))
        self.error_type: Optional[str] = values.get("error_type")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELDS}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.FIELDS})


class _JsonlFormatter(logging.Formatter):
    """Пишет готовую JSON-строку без изменений."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class AuditTrail:
    """Аудит-слой: подпись запросов HMAC и запись JSONL с ротацией."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        log_file: Optional[Path] = None,
        secret: Optional[str] = None,
        key_id: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.log_file = Path(log_file or self.settings.audit_log_file)
        self.key_id = key_id or self.settings.audit_key_id

        resolved_secret = secret if secret is not None else self.settings.audit_secret.get_secret_value()
        self._validate_secret(resolved_secret)
        self._secret = resolved_secret.encode("utf-8")

        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._logger = self._init_logger()

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_secret(secret: str) -> None:
        if not secret or secret in INSECURE_AUDIT_SECRETS:
            raise AuditConfigurationError(
                "AUDIT_SECRET не задан или равен небезопасному значению по умолчанию. "
                "Аудит с предсказуемым ключом недопустим: задайте AUDIT_SECRET в окружении."
            )

    @staticmethod
    def is_configured(settings: Optional[Settings] = None) -> bool:
        """Можно ли инициализировать аудит без исключения (используется healthcheck)."""
        settings = settings or get_settings()
        secret = settings.audit_secret.get_secret_value()
        return bool(secret) and secret not in INSECURE_AUDIT_SECRETS

    def _init_logger(self) -> logging.Logger:
        audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False  # аудит не должен утекать в общий stdout

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._archive_if_oversized()

        audit_logger.handlers.clear()
        handler = RotatingFileHandler(
            filename=str(self.log_file),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(_JsonlFormatter())
        audit_logger.addHandler(handler)
        return audit_logger

    def _archive_if_oversized(self) -> None:
        """Архивация вместо удаления: прежний лог сохраняется как ``*.bak``."""
        if not self.log_file.exists():
            return
        if self.log_file.stat().st_size <= self.max_bytes:
            return
        backup = self.log_file.with_name(f"{self.log_file.stem}_bak{self.log_file.suffix}")
        if backup.exists():
            backup.unlink()
        self.log_file.rename(backup)

    # ------------------------------------------------------------------
    # Подпись запроса
    # ------------------------------------------------------------------
    def compute_request_hash(self, user_id: str, query: str) -> str:
        """HMAC-SHA256 от ``user_id:query``. Сам запрос нигде не сохраняется."""
        message = f"{user_id}:{query}".encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    @staticmethod
    def stable_digest(value: str) -> str:
        """Детерминированный дайджест для идемпотентной индексации (не секрет)."""
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def write(self, record: AuditRecord) -> None:
        try:
            self._logger.info(record.to_json())
        except OSError as exc:
            # Аудит не должен ронять бизнес-логику, но сбой должен быть замечен.
            logger.error("Не удалось записать аудит-событие: %s", exc)

    def log_event(self, **values: Any) -> AuditRecord:
        record = AuditRecord(key_id=self.key_id, **values)
        self.write(record)
        return record

    # ------------------------------------------------------------------
    # Замер задержек
    # ------------------------------------------------------------------
    @staticmethod
    @contextmanager
    def stage_timer(metrics: Dict[str, float], stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            metrics[f"latency_{stage}_ms"] = round((time.perf_counter() - start) * 1000, 2)

    # ------------------------------------------------------------------
    # Чтение (для отчётов)
    # ------------------------------------------------------------------
    def read_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.log_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Пропуск повреждённой аудит-записи.")
        if limit is not None:
            return records[-limit:]
        return records

        return record.getMessage()


def generate_audit_report(
    settings: Optional[Settings] = None, log_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Сводный отчёт по аудит-журналу (CLI ``audit-report``).

    ``log_file`` позволяет отчитаться по произвольному журналу (используется
    тестами и при нескольких окружениях).
    """
    settings = settings or get_settings()
    if not AuditTrail.is_configured(settings):
        raise AuditConfigurationError(
            "Невозможно сформировать отчёт: AUDIT_SECRET не сконфигурирован безопасно."
        )

    trail = AuditTrail(settings=settings, log_file=log_file)
    records = trail.read_records()

    denied = [r for r in records if r.get("acl_decision") == "deny"]
    errors = [r for r in records if r.get("status") != "success"]
    redacted = [r for r in records if r.get("redaction_applied")]
    lat_total = sorted(float(r.get("latency_total_ms", 0.0) or 0.0) for r in records if r.get("latency_total_ms"))

    def _pct(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1))))
        return round(values[index], 2)

    return {
        "generated_at": utc_now_iso(),
        "audit_log_file": str(trail.log_file),
        "key_id": trail.key_id,
        "total_events": len(records),
        "acl_denied": len(denied),
        "errors": len(errors),
        "redactions_applied": len(redacted),
        "latency_total_ms": {
            "count": len(lat_total),
            "p50": _pct(lat_total, 50),
            "p95": _pct(lat_total, 95),
            "max": round(lat_total[-1], 2) if lat_total else 0.0,
        },
        "unique_users": sorted({str(r.get("user_id", "")) for r in records if r.get("user_id")}),
    }


__all__ = [
    "AuditTrail",
    "AuditRecord",
    "AuditConfigurationError",
    "generate_audit_report",
    "AUDIT_LOGGER_NAME",
    "SCHEMA_VERSION",
]
