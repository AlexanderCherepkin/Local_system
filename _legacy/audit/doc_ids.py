from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

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