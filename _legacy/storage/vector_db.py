"""Управление векторной базой данных с поддержкой ACL."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

try:
    from chromadb import PersistentClient, Collection, Documents
    from chromadb.config import Settings as ChromaSettings
    
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config.settings import Settings, VectorDBConfig
from security.acl import AccessControlList, AccessValidator, ACLValidationError


class AuditLogger:
    """Логирование аудита (Audit Trail) в JSONL формат."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or getattr(Settings, "AUDIT_LOG_FILE", Path("storage/audit.jsonl"))
        
        # Создаем директорию для логов, если её нет
        Path(self.log_file.parent).mkdir(parents=True, exist_ok=True)
        
        # Настройка логгера
        self.logger = logging.getLogger("AuditLogger")
        self.logger.setLevel(logging.INFO)
        
        # Безопасная ротация аудита: архивация вместо прямого удаления (unlink)
        if self.log_file.exists() and self.log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
            backup_file = self.log_file.with_name(f"{self.log_file.stem}_bak{self.log_file.suffix}")
            if backup_file.exists():
                backup_file.unlink()
            self.log_file.rename(backup_file)
    
    def log_audit_record(self, record: Dict[str, Any]) -> None:
        """Записать запись в аудит-лог."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if "timestamp" not in record:
                    record["timestamp"] = datetime.now(timezone.utc).isoformat()
                
                json_line = json.dumps(record, ensure_ascii=False) + "\n"
                f.write(json_line)
        except IOError as e:
            self.logger.error(f"Audit log write error: {e}")


class DataLoader:
    """
    Класс загрузки данных в векторную базу.
    
    Реализует:
    - Audit Trail логирование (хэш запроса, doc_ids, метрики задержки)
    - Интеграцию с ACL для контроля доступа (Zero-Trust)
    - Загрузку через ChromaDB Persistent Client
    """
    
    def __init__(self):
        self.settings = Settings.get_settings()
        self.vector_db_path = self.settings.VECTOR_DB_PATH
        
        # Инициализация компонентов
        self._initialize_vector_db()
        self.audit_logger = AuditLogger()
        self.acl = AccessControlList()
    
    def _initialize_vector_db(self) -> None:
        """Инициализировать векторную базу данных."""
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is required. Install with: pip install chromadb"
            )
        
        self.client = PersistentClient(path=str(self.vector_db_path))
        collection_name = self.settings.COLLECTION_NAME
        
        try:
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Enterprise RAG Documents with ACL"}
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Enterprise RAG Documents with ACL"}
            )
    
    def _compute_request_hash(self, user_id: str, query: str) -> str:
        """Вычислить хэш запроса."""
        hash_input = f"{user_id}:{query}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]
    
    def _log_stage_metric(
        self, 
        stage: str, 
        doc_ids: List[str], 
        latency_ms: float,
        user_id: str = "system"
    ) -> None:
        """Записать метрику этапа в Audit Trail."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_hash": "",
            "user_id": user_id,
            "stage": stage,
            "doc_ids": doc_ids[:10],
            "latency_ms": round(latency_ms, 2),
            "status": "success"
        }
        self.audit_logger.log_audit_record(record)
    
    def _log_request_start(self, user_id: str, query: str) -> None:
        """Записать начало запроса в Audit Trail."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_hash": "",
            "user_id": user_id,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "status": "started"
        }
        self.audit_logger.log_audit_record(record)
    
    def load_documents(
        self, 
        documents: List[Dict[str, Any]], 
        user_id: str = "admin",
        validate_acl: bool = True
    ) -> Dict[str, Any]:
        """Загрузить документы в векторную базу с валидацией через AccessValidator."""
        start_time = time.time()
        
        loaded_count = 0
        failed_count = 0
        errors = []
        
        access_validator = AccessValidator() if (validate_acl and getattr(self.settings.SECURITY_CONFIG, "ACL_ENABLED", True)) else None
        
        for idx, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc_{idx}")
            metadata = doc.get("metadata", {})
            text = doc.get("text", "")
            
            # 1. Zero-Trust валидация метаданных и прав
            if access_validator:
                validation_result = access_validator.validate_document_for_indexing(
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                    user_id=user_id
                )
                
                if not validation_result["valid"]:
                    failed_count += 1
                    errors.append({
                        "doc_id": doc_id,
                        "issues": validation_result["issues"]
                    })
                    continue
            
            # 2. Загрузка документа в векторную БД
            try:
                embedding = self._get_or_create_embedding(text, doc_id)
                
                self.collection.upsert(
                    ids=[doc_id],
                    embeddings=[[embedding]] if isinstance(embedding, list) else [embedding],
                    metadatas=[{**metadata}]
                )
                loaded_count += 1
            except Exception as e:
                failed_count += 1
                errors.append({
                    "doc_id": doc_id,
                    "error": str(e)
                })
        
        total_time = (time.time() - start_time) * 1000
        
        self._log_stage_metric(
            stage="load_documents",
            doc_ids=[doc.get("doc_id", f"doc_{i}") for i, doc in enumerate(documents)],
            latency_ms=total_time,
            user_id=user_id
        )
        
        return {
            "loaded_count": loaded_count,
            "failed_count": failed_count,
            "total_attempted": len(documents),
            "load_time_ms": round(total_time, 2),
            "errors": errors
        }
    
    def _get_or_create_embedding(
        self, 
        text: str, 
        doc_id: Optional[str] = None
    ) -> List[float]:
        """Получить или создать эмбеддинг для текста."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return [int(hash_char, 16) / 16.0 for i, hash_char in enumerate(text_hash[:384])]
    
    def query(
        self, 
        query_text: str,
        user_id: str = "admin",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Выполнить запрос к векторной базе с Zero-Trust ACL фильтрацией 
        на основе публичного API get_effective_roles.
        """
        start_time = time.time()
        
        request_hash = self._compute_request_hash(user_id, query_text)
        self._log_request_start(user_id, query_text)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required for queries.")
            
        top_k = top_k or getattr(self.settings, "TOP_K", 5)
        
        try:
            # Получаем эффективные роли пользователя через публичный API ACL
            user_roles = self.acl.get_effective_roles(user_id)
            
            if not user_roles:
                # Deny-by-default: у пользователя нет ролей — доступ к поиску запрещен
                return {
                    "request_hash": request_hash,
                    "results": {"ids": [[]], "documents": [[]], "metadatas": [[]]},
                    "query_time_ms": 0.0,
                    "reason": "Access denied: User has no roles assigned"
                }
            
            query_embedding = self._get_or_create_embedding(query_text)
            
            # Формируем фильтр для векторной БД по разрешенным ролям
            where_filter = {"allowed_roles": {"$in": list(user_roles)}}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            total_time = (time.time() - start_time) * 1000
            doc_ids = results.get("ids", [[]])[0]
            
            self._log_stage_metric(
                stage="query_documents",
                doc_ids=doc_ids,
                latency_ms=total_time,
                user_id=user_id
            )
            
            return {
                "request_hash": request_hash,
                "results": results,
                "query_time_ms": round(total_time, 2)
            }
        except Exception as e:
            logging.getLogger("DataLoader").error(f"Query error: {e}")
            return {
                "request_hash": request_hash,
                "results": {},
                "error": str(e)
            }
    
    def delete_documents(self, doc_ids: List[str], user_id: str = "admin") -> Dict[str, Any]:
        """Удалить документы из векторной базы."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required for deletion operations.")
        
        try:
            self.collection.delete(ids=doc_ids)
            
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_hash": hashlib.sha256(f"{user_id}:{','.join(doc_ids)}".encode()).hexdigest()[:16],
                "user_id": user_id,
                "stage": "delete_documents",
                "doc_ids": doc_ids[:10],
                "status": "success"
            }
            self.audit_logger.log_audit_record(record)
            
            return {
                "deleted_count": len(doc_ids),
                "user_id": user_id
            }
        except Exception as e:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_hash": "",
                "user_id": user_id,
                "stage": "delete_documents",
                "doc_ids_attempted": doc_ids[:10],
                "status": f"error: {str(e)}"
            }
            self.audit_logger.log_audit_record(record)
            
            return {
                "deleted_count": 0,
                "errors": str(e)
            }


def load_documents(documents: List[Dict[str, Any]], user_id: str = "admin") -> Dict[str, Any]:
    """Быстрая загрузка документов."""
    loader = DataLoader()
    return loader.load_documents(documents, user_id)