"""Настройки Enterprise RAG-системы с Zero-Trust и Air-Gapped поддержкой."""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings:
    """Параметры безопасности (Zero-Trust Data Ingestion)."""
    
    PASSWORD_PATTERNS = [
        r'(?i)(password|pwd|pass)\s*[=:]\s*["\']?\S+["\']?',
        r'(?i)passwd\s*[=:]\s*["\']?\S+',
        r'(?i)secret\s*[=:]\s*["\']?\S+',
    ]
    
    API_KEY_PATTERNS = [
        r'api[_-]?key\s*[=:]\s*["\']?[\w\-]{16,}',
        r'bearer\s+[a-zA-Z0-9_-]+',
        r'token\s*[:=]\s*["\']?[a-zA-Z0-9_\-]{20,}',
    ]
    
    PII_PATTERNS = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Телефоны
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
        r'\b\d{17}\b',  # Chinese ID
    ]
    
    @classmethod
    def detect_sensitive_data(cls, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Обнаружить чувствительные данные в тексте."""
        findings = {
            "passwords": [],
            "api_keys": [],
            "pii_data": []
        }
        
        for pattern_name, patterns in cls.__dict__.items():
            if not pattern_name.endswith('_PATTERNS'):
                continue
                
            category = pattern_name.replace('_PATTERNS', '').lower()
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    findings[category].append({
                        "type": category,
                        "pattern_category": pattern_name,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "length": len(match.group(0)),
                        "suggestion": f"Remove {category} data before indexing"
                    })
        
        return findings


class AuditRecord(BaseModel):
    """Pydantic-модель для структуры записи аудита."""
    request_hash: Optional[str] = Field(default=None, description="Хэш запроса пользователя")
    user_id: Optional[str] = Field(default=None, description="ID пользователя")
    doc_ids: List[str] = Field(default_factory=list, description="Список doc_id извлеченных документов")
    latency_ms: Optional[float] = Field(default=None, description="Задержка в миллисекундах")
    stage: Optional[str] = Field(default=None, description="Этап пайплайна")
    allowed_roles: List[str] = Field(default_factory=list, description="Разрешенные роли")
    status: Optional[str] = Field(default=None, description="Успех/ошибка")


class AuditLoggingConfig(BaseSettings):
    """Объединенная конфигурация аудита, логирования и метрик."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    audit_log_file: Path = Field(
        default=Path(__file__).parent.parent / "audit_logs" / "enterprise_rag_audit.jsonl"
    )
    metrics_file: Path = Field(
        default=Path(__file__).parent.parent / "metrics" / "pipeline_metrics.jsonl"
    )
    max_bytes_per_file: int = Field(default=10 * 1024 * 1024, gt=0, description="Максимальный размер файла лога в байтах")
    backup_count: int = Field(default=5, ge=0, description="Количество бэкапов логов")
    audit_record_template: AuditRecord = Field(default_factory=AuditRecord)


class VectorDBConfig:
    """Настройки векторной базы данных."""
    VECTOR_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
    COLLECTION_NAME = "enterprise_documents"
    ALLOWED_METADATA_FIELDS = [
        "allowed_roles",
        "doc_id",
        "source_file",
        "category",
        "department"
    ]


class EmbeddingConfig:
    """Настройки модели эмбеддингов (Air-Gapped)."""
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    NORMALIZE_EMBEDDINGS = True
    MODEL_CACHE_DIR = Path(__file__).parent.parent / "models"


class GeneratorConfig:
    """Настройки LLM генератора (Ollama)."""
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    TEMPERATURE = 0.7


class ChunkingConfig:
    """Настройки разбиения текста."""
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


class SecurityConfig:
    """Параметры контроля доступа (ACL)."""
    ACL_ENABLED = True
    ROLE_HIERARCHY = {
        "admin": [],
        "hr_manager": ["hr_viewer", "hr_analyst"],
        "hr_analyst": ["hr_viewer"],
        "hr_viewer": [],
        "legal": [],
        "finance": [],
        "engineer": [],
        "viewer": []
    }


class Settings(BaseSettings):
    """Главная конфигурация системы на базе pydantic-settings с валидацией при старте."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Основные параметры среды из .env с fail-fast валидацией
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API endpoint")
    default_model: str = Field(default="llama3", description="Модель для генерации ответов")
    context_window: int = Field(default=4096, gt=0, description="Контекстное окно LLM")
    min_chunk_length: int = Field(default=100, ge=0, description="Минимальная длина чанка для индексации")
    default_allowed_roles: List[str] = Field(default_factory=lambda: ["viewer"], description="Роли по умолчанию")
    log_format_str: str = Field(default="ISO8601", description="Формат логирования")

    # Пути к данным и моделям
    data_dir: Path = Field(default=Path(__file__).parent.parent / "data")
    embeddings_db_path: Path = Field(default=VectorDBConfig.VECTOR_DB_PATH)
    chunks_file: Path = Field(default=Path(__file__).parent.parent / "data" / "chunks.json")
    
    # Параметры chunking
    chunk_size: int = Field(default=ChunkingConfig.CHUNK_SIZE)
    chunk_overlap: int = Field(default=ChunkingConfig.CHUNK_OVERLAP)
    
    # Настройки векторной БД
    vector_db_path: Path = Field(default=VectorDBConfig.VECTOR_DB_PATH)
    collection_name: str = Field(default=VectorDBConfig.COLLECTION_NAME)
    allowed_metadata_fields: List[str] = Field(default_factory=lambda: VectorDBConfig.ALLOWED_METADATA_FIELDS)
    
    # Параметры поиска
    top_k_results: int = Field(default=GeneratorConfig.TOP_K_RESULTS)
    similarity_threshold: float = Field(default=GeneratorConfig.SIMILARITY_THRESHOLD)
    temperature: float = Field(default=GeneratorConfig.TEMPERATURE)

    # Настройки эмбеддингов (интегрированы внутрь Settings)
    embedding_model: str = Field(default=EmbeddingConfig.EMBEDDING_MODEL, description="Модель эмбеддингов")
    embedding_dimension: int = Field(default=EmbeddingConfig.EMBEDDING_DIMENSION, description="Размерность эмбеддингов")
    normalize_embeddings: bool = Field(default=EmbeddingConfig.NORMALIZE_EMBEDDINGS, description="Нормализовать ли эмбеддинги")
    model_cache_dir: Path = Field(default=EmbeddingConfig.MODEL_CACHE_DIR, description="Путь к кэшу моделей")
    
    # Композиция подсистем
    security_settings: SecuritySettings = Field(default_factory=SecuritySettings)
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
    audit_logging: AuditLoggingConfig = Field(default_factory=AuditLoggingConfig)
    generator_config: GeneratorConfig = Field(default_factory=GeneratorConfig)

    @classmethod
    def get_settings(cls) -> "Settings":
        """Получить экземпляр настроек с валидацией при старте."""
        return cls()


def initialize_directories():
    """Инициализировать необходимые директории."""
    settings_instance = Settings.get_settings()
    
    dirs_to_create = [
        settings_instance.data_dir,
        settings_instance.vector_db_path,
        settings_instance.audit_logging.audit_log_file.parent,
        settings_instance.chunks_file.parent,
        settings_instance.model_cache_dir,
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


# Вызов инициализации при импорте модуля
initialize_directories()