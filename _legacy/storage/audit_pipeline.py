# audit_pipeline.py

import os
import time
import uuid
import hmac
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Схема записи аудита (Pydantic)
# ==========================================
class AuditRecord(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    key_id: str
    request_hash: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Документы
    doc_ids_attempted: List[str] = Field(default_factory=list)
    doc_ids_extracted: List[str] = Field(default_factory=list)
    doc_ids_deleted: List[str] = Field(default_factory=list)
    
    # Латентность по этапам (в миллисекундах)
    latency_embedding_ms: float = 0.0
    latency_search_ms: float = 0.0
    latency_generation_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Безопасность и метрики
    acl_denied: bool = False
    redaction_applied: bool = False
    error_message: Optional[str] = None


# ==========================================
# 2. Безопасный хэш запроса с HMAC
# ==========================================
AUDIT_SECRET = os.getenv("AUDIT_SECRET", "fallback-secret-replace-in-production").encode("utf-8")
AUDIT_KEY_ID = os.getenv("AUDIT_KEY_ID", "key-v1")

def compute_secure_request_hash(user_id: str, query: str) -> tuple[str, str]:
    """
    Вычисляет защищенный хэш запроса с использованием HMAC-SHA256.
    Возвращает кортеж: (hexdigest, key_id).
    """
    message = f"{user_id}:{query}".encode("utf-8")
    request_hash = hmac.new(AUDIT_SECRET, message, hashlib.sha256).hexdigest()
    return request_hash, AUDIT_KEY_ID


# ==========================================
# 4. Настройка ротации логов (JSONL)
# ==========================================
def init_audit_logger(log_file_path: str = "audit.jsonl", max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    logger = logging.getLogger("audit_trail")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return record.getMessage()

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

def write_audit_log(record: AuditRecord):
    logger = logging.getLogger("audit_trail")
    if not logger.handlers:
        init_audit_logger()
    logger.info(record.model_dump_json())


# ==========================================
# 3. Замер латентности и пайплайн запроса
# ==========================================
@contextmanager
def stage_timer(metrics_dict: dict, stage_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics_dict[f"latency_{stage_name}_ms"] = round(elapsed_ms, 2)


# Заглушки этапов пайплайна для демонстрации
def generate_embedding(query: str):
    time.sleep(0.03)  # имитация работы модели эмбеддингов
    return [0.1, 0.2, 0.3]

def vector_db_search(vector):
    time.sleep(0.05)  # имитация поиска в БД
    return [{"id": "doc_42"}, {"id": "doc_99"}]

def generate_response(query, search_results):
    time.sleep(0.10)  # имитация генерации ответа LLM
    return "Ответ сформирован успешно."


def process_query(user_id: str, query: str):
    request_id = str(uuid.uuid4())
    metrics = {}
    request_hash, key_id = compute_secure_request_hash(user_id, query)
    
    extracted_doc_ids = []
    acl_denied = False
    redaction_applied = False
    error_message = None
    
    total_start = time.perf_counter()
    try:
        # 1. Эмбеддинг
        with stage_timer(metrics, "embedding"):
            vector = generate_embedding(query)
            
        # 2. Поиск
        with stage_timer(metrics, "search"):
            search_results = vector_db_search(vector)
            extracted_doc_ids = [doc["id"] for doc in search_results]
            
        # 3. Генерация
        with stage_timer(metrics, "generation"):
            response = generate_response(query, search_results)
            
    except PermissionError as e:
        acl_denied = True
        error_message = str(e)
        raise
    except Exception as e:
        error_message = str(e)
        raise
    finally:
        total_latency = (time.perf_counter() - total_start) * 1000
        metrics["total_latency_ms"] = round(total_latency, 2)
        
        # Формирование и строгая валидация аудиторской записи
        record = AuditRecord(
            request_id=request_id,
            key_id=key_id,
            request_hash=request_hash,
            doc_ids_extracted=extracted_doc_ids,
            latency_embedding_ms=metrics.get("latency_embedding_ms", 0.0),
            latency_search_ms=metrics.get("latency_search_ms", 0.0),
            latency_generation_ms=metrics.get("latency_generation_ms", 0.0),
            total_latency_ms=metrics.get("total_latency_ms", 0.0),
            acl_denied=acl_denied,
            redaction_applied=redaction_applied,
            error_message=error_message
        )
        
        write_audit_log(record)


if __name__ == "__main__":
    # Пример тестирования работы
    init_audit_logger("audit.jsonl")
    print("Запуск тестового запроса...")
    process_query(user_id="user_123", query="Как настроить аудит безопасности?")
    print("Запрос обработан, запись добавлена в audit.jsonl")