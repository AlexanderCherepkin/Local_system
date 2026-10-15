"""Единые контракты данных Enterprise RAG-системы.

Назначение модуля
-----------------
До появления этого файла в проекте существовало ТРИ несовместимые схемы чанка:

* ``pipeline/chunker.py``          -> ``start_pos`` / ``end_pos``           (dict)
* ``utils/document_processor.py``  -> ``start_position`` / ``end_position`` (dataclass)
* ``utils/vector_search.py``       -> читал ``start_position`` через ``.get()``

В результате позиции чанков молча терялись, а передача ``DocumentChunk``
в векторный поиск падала с ``AttributeError: 'DocumentChunk' object has no
attribute 'get'``.

Здесь определён ЕДИНСТВЕННЫЙ канонический контракт. Правила использования:

1. Все модули создают объекты ``Chunk`` / ``Document``, а не произвольные dict.
2. Запись строгая: ``to_dict()`` всегда возвращает канонические имена полей.
3. Чтение терпимое: ``from_dict()`` / ``normalize_chunk()`` принимают устаревшие
   алиасы (``start_position``, ``document_id``), чтобы не ломать старые данные.

Реализовано на stdlib ``dataclasses`` намеренно: проект разворачивается в
изолированном контуре (air-gapped), поэтому новые runtime-зависимости не
добавляются.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = [
    "SCHEMA_VERSION",
    "FIELD_ALLOWED_ROLES",
    "FIELD_CHUNK_ID",
    "FIELD_DOC_ID",
    "FIELD_END_POS",
    "FIELD_START_POS",
    "utc_now_iso",
    "content_hash",
    "make_chunk_id",
    "normalize_chunk",
    "ChunkMetadata",
    "Chunk",
    "RetrievedChunk",
    "Document",
    "RedactionFinding",
    "AccessDecision",
    "AuditRecord",
]

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Канонические имена полей
# ---------------------------------------------------------------------------
FIELD_DOC_ID = "doc_id"
FIELD_CHUNK_ID = "chunk_id"
FIELD_ALLOWED_ROLES = "allowed_roles"
FIELD_START_POS = "start_pos"
FIELD_END_POS = "end_pos"

# Устаревшие алиасы -> каноническое имя. Используются ТОЛЬКО при чтении.
_LEGACY_ALIASES: Dict[str, str] = {
    "start_position": FIELD_START_POS,
    "end_position": FIELD_END_POS,
    "start": FIELD_START_POS,
    "end": FIELD_END_POS,
    "document_id": FIELD_DOC_ID,
    "parent_doc_id": FIELD_DOC_ID,
}

_DOC_ID_ALIASES = ("doc_id", "document_id", "parent_doc_id")


def utc_now_iso() -> str:
    """Текущее время в UTC в формате ISO-8601.

    ``datetime.utcnow()`` объявлен устаревшим начиная с Python 3.12, поэтому
    везде в проекте используется timezone-aware вариант.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def content_hash(text: str) -> str:
    """Детерминированный SHA-256 текста (для идемпотентной индексации)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Детерминированный и уникальный в пределах документа идентификатор чанка.

    Заменяет прежнюю конструкцию ``len([c for c in []]) + 1`` из
    ``ingestion/chunker.py``, которая всегда возвращала ``chunk_1`` и приводила
    к коллизии идентификаторов при upsert в векторную базу.
    """
    return f"{doc_id or 'doc'}::{int(chunk_index):05d}"


def _coerce_roles(value: Any) -> List[str]:
    """``allowed_roles`` может прийти списком, строкой или JSON-строкой."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return [raw]
            return [str(item) for item in parsed] if isinstance(parsed, list) else [raw]
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _resolve(data: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Прочитать поле из mapping с учётом устаревших алиасов."""
    if name in data:
        return data[name]
    for alias, canonical in _LEGACY_ALIASES.items():
        if canonical == name and alias in data:
            return data[alias]
    return default


def normalize_chunk(obj: Any) -> Dict[str, Any]:
    """Привести произвольное представление чанка к каноническому ``dict``.

    Поддерживаются: ``Chunk``, любой dataclass, ``Mapping`` и произвольный
    объект с атрибутами. Именно этот помощник устраняет падение
    ``AttributeError: 'DocumentChunk' object has no attribute 'get'``.
    """
    if obj is None:
        return {}

    if isinstance(obj, Chunk):
        return obj.to_dict()

    if isinstance(obj, Mapping):
        data: Dict[str, Any] = dict(obj)
    elif is_dataclass(obj) and not isinstance(obj, type):
        data = asdict(obj)
    elif hasattr(obj, "__dict__"):
        data = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        raise TypeError(f"Неподдерживаемый тип чанка: {type(obj).__name__}")

    # Метаданные могут лежать как во вложенном словаре, так и в плоском виде.
    meta = data.get("metadata")
    if isinstance(meta, ChunkMetadata):
        data["metadata"] = meta.to_dict()
    elif isinstance(meta, Mapping):
        data["metadata"] = dict(meta)
    elif is_dataclass(meta) and not isinstance(meta, type):
        data["metadata"] = asdict(meta)
    else:
        data["metadata"] = {}

    doc_id = ""
    for alias in _DOC_ID_ALIASES:
        if data.get(alias):
            doc_id = str(data[alias])
            break
    if not doc_id:
        doc_id = str(data["metadata"].get(FIELD_DOC_ID, "") or "")

    data[FIELD_DOC_ID] = doc_id
    data[FIELD_START_POS] = int(_resolve(data, FIELD_START_POS, 0) or 0)
    data[FIELD_END_POS] = int(_resolve(data, FIELD_END_POS, 0) or 0)
    data["text"] = str(data.get("text", "") or "")
    data.setdefault("chunk_index", 0)
    data.setdefault("strategy", "size")
    data.setdefault("total_chunks", 0)
    if not data.get(FIELD_CHUNK_ID):
        data[FIELD_CHUNK_ID] = make_chunk_id(doc_id, int(data.get("chunk_index", 0) or 0))
    return data


@dataclass
class ChunkMetadata:
    """Метаданные документа/чанка.

    Поле ``allowed_roles`` ОБЯЗАТЕЛЬНО для индексации (Zero-Trust, см.
    ``police.txt``). Пустой список означает «доступ запрещён всем», а не
    «доступ разрешён всем».
    """

    doc_id: str = ""
    source_file: str = ""
    title: str = ""
    department: str = ""
    allowed_roles: List[str] = field(default_factory=list)
    content_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN = ("doc_id", "source_file", "title", "department", "allowed_roles", "content_hash", "extra")

    def __post_init__(self) -> None:
        self.allowed_roles = _coerce_roles(self.allowed_roles)
        if self.extra is None:
            self.extra = {}

    @property
    def has_access_control(self) -> bool:
        """True, если задан непустой список разрешённых ролей."""
        return bool(self.allowed_roles)

    def to_dict(self) -> Dict[str, Any]:
        """Плоский словарь для внутреннего использования."""
        data = {
            FIELD_DOC_ID: self.doc_id,
            "source_file": self.source_file,
            "title": self.title,
            "department": self.department,
            FIELD_ALLOWED_ROLES: list(self.allowed_roles),
            "content_hash": self.content_hash,
        }
        data.update({k: v for k, v in self.extra.items() if k not in self._KNOWN})
        return data

    def to_vector_metadata(self) -> Dict[str, Any]:
        """Метаданные для векторной БД (ChromaDB принимает только скаляры).

        Списки сериализуются в JSON-строку; обратно их разбирает
        ``ChunkMetadata.from_dict``.
        """
        flat: Dict[str, Any] = {}
        for key, value in self.to_dict().items():
            if isinstance(value, (list, tuple, set)):
                flat[key] = json.dumps(list(value), ensure_ascii=False)
            elif value is None or isinstance(value, (str, int, float, bool)):
                flat[key] = value
            else:
                flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        return flat

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ChunkMetadata":
        data = dict(data or {})
        known = set(cls._KNOWN)
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            doc_id=str(_resolve(data, FIELD_DOC_ID, "") or ""),
            source_file=str(data.get("source_file", "") or ""),
            title=str(data.get("title", "") or ""),
            department=str(data.get("department", "") or ""),
            allowed_roles=_coerce_roles(data.get(FIELD_ALLOWED_ROLES)),
            content_hash=str(data.get("content_hash", "") or ""),
            extra=extra,
        )


@dataclass
class Chunk:
    """Канонический чанк текста.

    ``start_pos`` / ``end_pos`` — позиции в ИСХОДНОМ тексте документа, что
    позволяет восстанавливать фрагмент и строить цитаты в ответах LLM.
    """

    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    total_chunks: int = 0
    strategy: str = "size"
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = ChunkMetadata()
        elif not isinstance(self.metadata, ChunkMetadata):
            self.metadata = ChunkMetadata.from_dict(self.metadata)
        if not self.metadata.doc_id:
            self.metadata.doc_id = self.doc_id
        if not self.metadata.content_hash:
            self.metadata.content_hash = content_hash(self.text)
        if not self.chunk_id:
            self.chunk_id = make_chunk_id(self.doc_id, self.chunk_index)
        self.start_pos = int(self.start_pos or 0)
        self.end_pos = int(self.end_pos or 0)
        self.chunk_index = int(self.chunk_index or 0)

    @property
    def char_count(self) -> int:
        return len(self.text or "")

    @property
    def word_count(self) -> int:
        return len((self.text or "").split())

    def to_dict(self) -> Dict[str, Any]:
        """Строгая каноническая сериализация (без устаревших алиасов)."""
        return {
            FIELD_CHUNK_ID: self.chunk_id,
            FIELD_DOC_ID: self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "text": self.text,
            FIELD_START_POS: self.start_pos,
            FIELD_END_POS: self.end_pos,
            "strategy": self.strategy,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata.to_dict(),
            "embedding": self.embedding,
        }

    def to_vector_metadata(self) -> Dict[str, Any]:
        """Скалярные метаданные для ChromaDB."""
        meta = self.metadata.to_vector_metadata()
        meta.update(
            {
                FIELD_CHUNK_ID: self.chunk_id,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                FIELD_START_POS: self.start_pos,
                FIELD_END_POS: self.end_pos,
                "strategy": self.strategy,
                "char_count": self.char_count,
            }
        )
        return meta

    @classmethod
    def from_dict(cls, data: Any) -> "Chunk":
        data = normalize_chunk(data)
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, ChunkMetadata):
            metadata = ChunkMetadata.from_dict(metadata)
        return cls(
            chunk_id=str(data.get(FIELD_CHUNK_ID) or ""),
            doc_id=str(data.get(FIELD_DOC_ID) or ""),
            chunk_index=int(data.get("chunk_index", 0) or 0),
            text=str(data.get("text", "") or ""),
            start_pos=int(data.get(FIELD_START_POS, 0) or 0),
            end_pos=int(data.get(FIELD_END_POS, 0) or 0),
            total_chunks=int(data.get("total_chunks", 0) or 0),
            strategy=str(data.get("strategy", "size") or "size"),
            metadata=metadata,
            embedding=data.get("embedding"),
        )


@dataclass
class RetrievedChunk(Chunk):
    """Чанк, возвращённый поиском, с метриками близости.

    ``similarity`` — косинусная близость в диапазоне [-1, 1] (сравнивается с
    ``SIMILARITY_THRESHOLD``); ``distance`` — сырое расстояние из векторной БД.
    """

    similarity: float = 0.0
    distance: Optional[float] = None

    @property
    def score(self) -> float:
        """Единая мера релевантности (чем больше, тем лучше)."""
        return float(self.similarity)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"similarity": self.similarity, "score": self.score, "distance": self.distance})
        return data

    @classmethod
    def from_chunk(cls, chunk: Any, similarity: float = 0.0, distance: Optional[float] = None) -> "RetrievedChunk":
        """Собрать результат поиска из обычного чанка."""
        data = normalize_chunk(chunk)
        base = Chunk.from_dict(data)
        return cls(
            chunk_id=base.chunk_id,
            doc_id=base.doc_id,
            chunk_index=base.chunk_index,
            text=base.text,
            start_pos=base.start_pos,
            end_pos=base.end_pos,
            total_chunks=base.total_chunks,
            strategy=base.strategy,
            metadata=base.metadata,
            embedding=base.embedding,
            similarity=float(similarity),
            distance=distance,
        )


@dataclass
class Document:
    """Исходный документ до нарезки."""

    doc_id: str = ""
    text: str = ""
    title: str = ""
    source_file: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ChunkMetadata):
            self.metadata = ChunkMetadata.from_dict(self.metadata)
        if not self.metadata.doc_id:
            self.metadata.doc_id = self.doc_id
        if self.title and not self.metadata.title:
            self.metadata.title = self.title
        if self.source_file and not self.metadata.source_file:
            self.metadata.source_file = self.source_file

    @property
    def char_count(self) -> int:
        return len(self.text or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_DOC_ID: self.doc_id,
            "text": self.text,
            "title": self.title,
            "source_file": self.source_file,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        data = dict(data or {})
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, ChunkMetadata):
            metadata = ChunkMetadata.from_dict(metadata)
        doc_id = str(_resolve(data, FIELD_DOC_ID, "") or metadata.doc_id or "")
        return cls(
            doc_id=doc_id,
            text=str(data.get("text", "") or ""),
            title=str(data.get("title", "") or metadata.title or ""),
            source_file=str(data.get("source_file", "") or metadata.source_file or ""),
            metadata=metadata,
        )
