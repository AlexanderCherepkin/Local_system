"""Единые контракты данных Secure RAG.

Раньше в проекте было три несовместимых схемы чанка; канонический контракт
живёт здесь, а адаптеры допустимы только на границах модулей.
"""

from .schemas import (
    FIELD_ALLOWED_ROLES,
    FIELD_CHUNK_ID,
    FIELD_DOC_ID,
    SCHEMA_VERSION,
    AccessDecision,
    Chunk,
    ChunkMetadata,
    Document,
    RedactionFinding,
    RetrievedChunk,
    content_hash,
    make_chunk_id,
    normalize_chunk,
    utc_now_iso,
)

__all__ = [
    "SCHEMA_VERSION",
    "FIELD_ALLOWED_ROLES",
    "FIELD_CHUNK_ID",
    "FIELD_DOC_ID",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "RetrievedChunk",
    "RedactionFinding",
    "AccessDecision",
    "content_hash",
    "make_chunk_id",
    "normalize_chunk",
    "utc_now_iso",
]
