"""Сквозной ingest-путь Secure RAG (F-04).

Раньше ``main.py`` вызывал ``ingest_documents``, которой не существовало, а
``ingestion/chunker.py`` импортировал несуществующий ``create_chunks``.
Здесь реализован единственный обязательный путь:

    чтение файла -> нормализация -> redaction-gate -> ACL-валидация
    -> нарезка -> эмбеддинг -> upsert

Каждый этап fail-closed: документ без ``allowed_roles`` или с секретами класса
CRITICAL не индексируется.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from config.settings import Settings, get_settings
from core.schemas import Chunk, ChunkMetadata, Document
from pipeline.chunker import create_chunks
from security.acl import AccessControlList, AccessValidator, Principal
from security.redaction import process_text

logger = logging.getLogger("secure_rag.ingestion")

#: Расширения, которые разрешено индексировать.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".json", ".csv"}


class IngestionError(RuntimeError):
    """Ошибка ingest-пути."""


class TextChunker:
    """Обёртка нарезки, сохраняющая метаданные документа в каждом чанке."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def chunk_text(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        """Нарезать текст и вернуть чанки в каноническом контракте.

        Поле ``allowed_roles`` из метаданных документа обязательно попадает в
        метаданные каждого чанка (Zero-Trust на уровне индекса).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        metadata = dict(metadata or {})
        doc_id = str(metadata.get("doc_id") or "doc_default")

        raw_chunks = create_chunks(
            text=text,
            doc_id=doc_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            method="sliding",
            metadata=metadata,
        )

        chunks: List[Dict[str, Any]] = []
        total = len(raw_chunks)
        for raw in raw_chunks:
            chunk_meta = {**metadata, "doc_id": doc_id}
            chunk = Chunk(
                chunk_id=str(raw["chunk_id"]),
                doc_id=doc_id,
                chunk_index=int(raw["index"]),
                text=str(raw["text"]),
                start_pos=int(raw["start_pos"]),
                end_pos=int(raw["end_pos"]),
                total_chunks=total,
                strategy=str(raw.get("strategy", "size")),
                metadata=ChunkMetadata.from_dict(chunk_meta),
            )
            data = chunk.to_dict()
            data["word_count"] = chunk.word_count
            data["char_count"] = chunk.char_count
            chunks.append(data)
        return chunks


def read_document(path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Прочитать текстовый документ с явной поддержкой кодировок."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Неподдерживаемый тип файла: {path.suffix} ({path.name})")

    if path.stat().st_size > max_bytes:
        raise IngestionError(f"Файл слишком велик для индексации: {path.name}")

    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError(f"Не удалось декодировать файл: {path.name}")


def normalize_json(path: Path) -> List[Dict[str, Any]]:
    """Прочитать JSON/JSONL с документами.

    Поддерживаются форматы: список документов, ``{"documents": [...]}`` и JSONL.
    Каждый документ — ``{"doc_id": ..., "text": ..., "metadata": {...}}``.
    """
    raw = read_document(path)
    documents: List[Dict[str, Any]] = []

    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IngestionError(f"Некорректный JSON в {path.name}: {exc}") from exc
        documents = list(parsed) if isinstance(parsed, list) else [parsed]
    elif stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # JSONL fallback
            for line_no, line in enumerate(stripped.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise IngestionError(
                        f"Некорректная строка {line_no} в {path.name}: {exc}"
                    ) from exc
        else:
            documents = list(parsed.get("documents", [])) if isinstance(parsed, dict) else [parsed]
    else:
        raise IngestionError(f"Ожидался JSON в {path.name}")

    return [d for d in documents if isinstance(d, Mapping)]


def iter_source_files(path: Path) -> List[Path]:
    """Собрать файлы для индексации из файла или каталога."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _roles_from_metadata(metadata: Mapping[str, Any]) -> List[str]:
    roles = metadata.get("allowed_roles")
    if roles is None:
        return []
    if isinstance(roles, str):
        return [r.strip() for r in roles.split(",") if r.strip()]
    return [str(r).strip() for r in roles if str(r or "").strip()]


def build_documents_from_path(
    path: Path,
    allowed_roles: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Прочитать файлы и превратить их в документы для индексации.

    ``allowed_roles`` задаёт роли доступа по умолчанию для всех файлов из
    каталога. Документы без ``allowed_roles`` дальше будут отклонены ingest-gate:
    политика требует явного назначения прав.
    """
    default_roles = list(allowed_roles or [])
    documents: List[Dict[str, Any]] = []

    for file_path in iter_source_files(path):
        if file_path.suffix.lower() == ".json":
            for index, raw_doc in enumerate(normalize_json(file_path)):
                text = str(raw_doc.get("text") or raw_doc.get("content") or "")
                metadata = dict(raw_doc.get("metadata") or {})
                doc_id = str(raw_doc.get("doc_id") or f"{file_path.stem}_{index}")
                if not _roles_from_metadata(metadata) and default_roles:
                    metadata["allowed_roles"] = default_roles
                metadata.setdefault("source_file", file_path.name)
                documents.append({"doc_id": doc_id, "text": text, "metadata": metadata})
            continue

        text = read_document(file_path)
        metadata = {
            "doc_id": file_path.stem,
            "source_file": file_path.name,
            "allowed_roles": default_roles,
            "title": file_path.stem,
        }
        documents.append({"doc_id": file_path.stem, "text": text, "metadata": metadata})

    return documents


def ingest_documents(
    path: str,
    user_id: str = "admin",
    roles: Optional[Sequence[str]] = None,
    settings: Optional[Settings] = None,
    vector_db: Any = None,
    audit: Any = None,
) -> Dict[str, Any]:
    """Проиндексировать документы из файла или каталога.

    Возвращает сводку: сколько чанков проиндесировано, сколько отклонено и почему.
    """
    resolved_settings = settings or get_settings()
    source_path = Path(path)

    principal = Principal(user_id=user_id, roles=tuple(roles or (user_id,)))
    acl = AccessControlList(settings=resolved_settings)
    validator = AccessValidator(acl=acl)

    if vector_db is None:
        from storage.vector_db import VectorDatabase

        vector_db = VectorDatabase(settings=resolved_settings)

    default_roles = list(roles) if roles else None
    documents = build_documents_from_path(source_path, allowed_roles=default_roles)

    chunker = TextChunker(settings=resolved_settings)

    accepted_chunks: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for document in documents:
        doc_id = str(document.get("doc_id") or "")
        text = str(document.get("text") or "")
        metadata = dict(document.get("metadata") or {})

        # 1. Обязательный redaction-gate + ACL-проверка на уровне документа.
        verdict = validator.validate_document_for_indexing(
            doc_id=doc_id, text=text, metadata=metadata, user_id=principal.user_id
        )

        if not verdict["valid"]:
            rejected.append({"doc_id": doc_id, "issues": verdict["issues"]})
            continue

        # 2. Нарезаем ТОЛЬКО очищенный текст.
        chunks = chunker.chunk_text(verdict["cleaned_text"], metadata=metadata)
        if not chunks:
            rejected.append({"doc_id": doc_id, "issues": [{"type": "Empty_Document"}]})
            continue

        # 3. Отбрасываем слишком короткие чанки.
        for chunk in chunks:
            if len(chunk["text"].strip()) < resolved_settings.min_chunk_length and chunk.get("total_chunks", 1) > 1:
                continue
            accepted_chunks.append(chunk)

    result = vector_db.add_chunks(accepted_chunks, principal=principal, audit=audit)
    result.update(
        {
            "documents_total": len(documents),
            "documents_rejected": len(rejected),
            "rejected_documents": rejected,
            "source": str(source_path),
        }
    )

    logger.info(
        "Ingest завершён: документов=%d, чанков проиндексировано=%s, отклонено=%s",
        len(documents),
        result.get("indexed"),
        result.get("rejected"),
    )
    return result


__all__ = [
    "AccessValidator",
    "TextChunker",
    "IngestionError",
    "SUPPORTED_EXTENSIONS",
    "ingest_documents",
    "build_documents_from_path",
    "read_document",
    "normalize_json",
]

