"""Пакет ingest: загрузка документов и нарезка на чанки."""

from .loader import (
    IngestionError,
    TextChunker,
    build_documents_from_path,
    ingest_documents,
)

__all__ = ["TextChunker", "ingest_documents", "build_documents_from_path", "IngestionError"]
