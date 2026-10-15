"""Общие фикстуры и заглушки для тестов Secure RAG.

Тесты намеренно не требуют ``chromadb``, ``sentence-transformers`` и Ollama:
внешние зависимости подменяются детерминированными стабами, поэтому логику
можно проверять в изолированном контуре.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

# Корень репозитория в sys.path (чтобы ``secure_rag`` импортировался без установки).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secure_rag.config.settings import Settings, set_settings  # noqa: E402
from secure_rag.security.acl import AccessControlList, Principal  # noqa: E402

AUDIT_SECRET = "pytest-secret-value-not-default"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Настройки, изолированные во временном каталоге."""
    instance = Settings(
        data_dir=tmp_path / "data",
        vector_db_path=tmp_path / "data" / "chroma",
        chunks_file=tmp_path / "data" / "chunks.json",
        model_cache_dir=tmp_path / "models",
        audit_log_file=tmp_path / "audit" / "audit.jsonl",
        audit_secret=AUDIT_SECRET,
        chunk_size=120,
        chunk_overlap=20,
        top_k_results=3,
        oversample_factor=2,
        max_top_k=10,
    )
    set_settings(instance)
    yield instance
    set_settings(None)


class FakeEmbedder:
    """Детерминированный эмбеддер фиксированной размерности для тестов."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension_value = dimension
        self.calls: List[str] = []

    def embed_text(self, text: str) -> List[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] / 255.0) for i in range(self.dimension_value)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def dimension(self) -> int:
        return self.dimension_value


class FakeCollection:
    """In-memory замена коллекции ChromaDB с ACL-фильтром и cosine-метрикой."""

    def __init__(self, metric: str = "cosine") -> None:
        self.metadata: Dict[str, Any] = {"hnsw:space": metric}
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.last_query: Optional[Dict[str, Any]] = None

    def upsert(self, ids, documents=None, embeddings=None, metadatas=None) -> None:
        ids = list(ids or [])
        documents = list(documents or [None] * len(ids))
        embeddings = list(embeddings or [None] * len(ids))
        metadatas = list(metadatas or [{} for _ in ids])
        for chunk_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self.rows[chunk_id] = {
                "id": chunk_id,
                "document": doc,
                "embedding": emb,
                "metadata": meta or {},
            }

    def delete(self, ids) -> None:
        for chunk_id in ids or []:
            self.rows.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.rows)

    @staticmethod
    def _matches(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
        if not where:
            return True
        for key, condition in where.items():
            value = meta.get(key)
            if isinstance(condition, dict):
                if "$in" in condition:
                    allowed = condition["$in"]
                    candidates = value if isinstance(value, list) else [value]
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            parsed = None
                        if isinstance(parsed, list):
                            candidates = parsed
                    if not any(c in allowed for c in candidates):
                        return False
                elif "$eq" in condition:
                    if value != condition["$eq"]:
                        return False
                else:
                    return False
            elif value != condition:
                return False
        return True

    def query(self, query_embeddings=None, n_results=5, where=None, include=None):
        self.last_query = {"where": where, "n_results": n_results}
        include = list(include or ["documents", "metadatas", "distances"])

        candidates = [row for row in self.rows.values() if self._matches(row["metadata"], where)]

        query_vector = (query_embeddings or [None])[0]
        scored: List[tuple] = []
        for row in candidates:
            distance = 0.0
            if query_vector and row["embedding"]:
                dot = sum(a * b for a, b in zip(query_vector, row["embedding"]))
                distance = max(0.0, 1.0 - dot)
            scored.append((distance, row))
        scored.sort(key=lambda item: item[0])

        ids: List[str] = []
        documents: List[Any] = []
        metadatas: List[Any] = []
        distances: List[float] = []
        for distance, row in scored[:n_results]:
            ids.append(row["id"])
            if "documents" in include:
                documents.append(row["document"])
            if "metadatas" in include:
                metadatas.append(row["metadata"])
            if "distances" in include:
                distances.append(distance)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }


class FakeClient:
    """In-memory замена ``PersistentClient``."""

    def __init__(self, metric: str = "cosine", preexisting_metric: Optional[str] = None) -> None:
        self.metric = metric
        self.collections: Dict[str, FakeCollection] = {}
        # Имитация уже существующего индекса с другой метрикой: ChromaDB
        # возвращает такую коллекцию как есть, игнорируя запрошенную метрику.
        self._preexisting_metric = preexisting_metric

    def get_or_create_collection(self, name: str, metadata=None):
        collection = self.collections.get(name)
        if collection is None:
            if self._preexisting_metric is not None:
                metric = self._preexisting_metric
            else:
                metric = (metadata or {}).get("hnsw:space", self.metric)
            collection = FakeCollection(metric=metric)
            self.collections[name] = collection
        return collection

    def create_collection(self, name: str, metadata=None):
        metric = self._preexisting_metric or (metadata or {}).get("hnsw:space", self.metric)
        collection = FakeCollection(metric=metric)
        self.collections[name] = collection
        return collection



class FakeGenerator:
    """Генератор без обращения к Ollama."""

    def __init__(self, answer: str = "Согласно документу [1].", settings=None) -> None:
        self.answer = answer
        self.settings = settings
        self.model_name = "fake-model"
        self.top_k_rag = getattr(settings, "top_k_results", 3)
        self.prompts: List[str] = []

    def generate_response(self, prompt, context, require_citations=True, **kwargs):
        from secure_rag.pipeline.generator import extract_citations, validate_citations

        self.prompts.append(prompt)
        window = min(len(context), self.top_k_rag)
        valid, invalid = validate_citations(self.answer, window)
        return {
            "success": valid,
            "answer": self.answer if valid else "",
            "citations": sorted(set(extract_citations(self.answer))),
            "abstained": not valid,
            "metadata": {"model_used": self.model_name, "invalid_citations": invalid},
        }


@pytest.fixture()
def acl(settings: Settings, tmp_path: Path) -> AccessControlList:
    """ACL на изолированном хранилище с известными пользователями."""
    storage = tmp_path / "acl_storage.json"
    storage.write_text(
        json.dumps(
            {
                "users": {
                    "admin": ["admin"],
                    "hr_user": ["hr"],
                    "eng_user": ["engineering"],
                    "noroles": [],
                },
                "roles": {
                    "admin": ["read_all"],
                    "hr": ["read_hr"],
                    "engineering": ["read_eng"],
                },
            }
        ),
        encoding="utf-8",
    )
    return AccessControlList(settings=settings, storage_path=storage)


@pytest.fixture()
def principal_hr() -> Principal:
    return Principal(user_id="hr_user", roles=("hr",))


@pytest.fixture()
def principal_eng() -> Principal:
    return Principal(user_id="eng_user", roles=("engineering",))


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def store(settings: Settings, acl, embedder: FakeEmbedder):
    """Векторное хранилище на in-memory клиенте (chromadb не требуется)."""
    from conftest import FakeClient
    from secure_rag.storage.vector_db import VectorDatabase

    return VectorDatabase(
        settings=settings,
        embedder=embedder,
        acl=acl,
        client=FakeClient(metric=settings.distance_metric),
    )


@pytest.fixture()
def fake_generator(settings: Settings) -> FakeGenerator:
    return FakeGenerator(settings=settings)


