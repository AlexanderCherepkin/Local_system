"""Тесты ingest-пути и нарезки (F-02, F-04, F-19).

Проверяется единственный допустимый путь индексации:
чтение файла -> нормализация -> redaction-gate -> ACL -> нарезка -> upsert,
а также контракт чанков (позиции, метаданные, ограничение размера).
"""

from __future__ import annotations

import json

import pytest

from secure_rag.ingestion.loader import (
    IngestionError,
    SUPPORTED_EXTENSIONS,
    TextChunker,
    build_documents_from_path,
    ingest_documents,
    iter_source_files,
    normalize_json,
    read_document,
)
from secure_rag.pipeline.chunker import (
    DocumentChunker,
    create_chunks,
    split_long_sentences,
)

LONG_TEXT = (
    "Первый раздел описывает порядок оформления отпуска. "
    "Второй раздел описывает расчёт больничного листа. "
    "Третий раздел описывает правила командировок и суточные. "
) * 6


class TestReadDocument:
    def test_reads_utf8(self, tmp_path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Текст документа.", encoding="utf-8")

        assert read_document(path) == "Текст документа."

    def test_unsupported_extension_rejected(self, tmp_path) -> None:
        path = tmp_path / "malware.exe"
        path.write_bytes(b"MZ")

        with pytest.raises(IngestionError):
            read_document(path)

    def test_oversized_file_rejected(self, tmp_path) -> None:
        path = tmp_path / "big.txt"
        path.write_text("x" * 2048, encoding="utf-8")

        with pytest.raises(IngestionError):
            read_document(path, max_bytes=1024)

    def test_iter_source_files_skips_unknown_types(self, tmp_path) -> None:
        (tmp_path / "ok.txt").write_text("a", encoding="utf-8")
        (tmp_path / "skip.bin").write_bytes(b"\x00")

        found = [p.name for p in iter_source_files(tmp_path)]

        assert found == ["ok.txt"]

    def test_iter_source_files_on_single_file(self, tmp_path) -> None:
        path = tmp_path / "single.md"
        path.write_text("# t", encoding="utf-8")

        assert iter_source_files(path) == [path]


class TestNormalizeJson:
    def test_plain_list(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(json.dumps([{"doc_id": "a", "text": "A"}]), encoding="utf-8")

        assert normalize_json(path) == [{"doc_id": "a", "text": "A"}]

    def test_wrapped_in_documents_key(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(json.dumps({"documents": [{"doc_id": "a", "text": "A"}]}), encoding="utf-8")

        assert len(normalize_json(path)) == 1

    def test_jsonl_fallback(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(
            '{"doc_id": "a", "text": "A"}\n{"doc_id": "b", "text": "B"}\n', encoding="utf-8"
        )

        assert [d["doc_id"] for d in normalize_json(path)] == ["a", "b"]

    def test_broken_json_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(IngestionError):
            normalize_json(path)

    def test_non_object_entries_dropped(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(json.dumps(["строка", {"doc_id": "a", "text": "A"}]), encoding="utf-8")

        assert normalize_json(path) == [{"doc_id": "a", "text": "A"}]


class TestBuildDocuments:
    def test_txt_gets_default_roles(self, tmp_path) -> None:
        (tmp_path / "hr.txt").write_text("Текст.", encoding="utf-8")

        documents = build_documents_from_path(tmp_path, allowed_roles=["hr"])

        assert documents[0]["metadata"]["allowed_roles"] == ["hr"]
        assert documents[0]["doc_id"] == "hr"

    def test_json_entry_inherits_default_roles(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(
            json.dumps([{"doc_id": "d1", "text": "Текст", "metadata": {"title": "T"}}]),
            encoding="utf-8",
        )

        document = build_documents_from_path(path, allowed_roles=["hr"])[0]

        assert document["metadata"]["allowed_roles"] == ["hr"]
        assert document["metadata"]["title"] == "T"

    def test_explicit_roles_not_overwritten(self, tmp_path) -> None:
        path = tmp_path / "docs.json"
        path.write_text(
            json.dumps([{"doc_id": "d1", "text": "T", "metadata": {"allowed_roles": ["fin"]}}]),
            encoding="utf-8",
        )

        document = build_documents_from_path(path, allowed_roles=["hr"])[0]

        assert document["metadata"]["allowed_roles"] == ["fin"]

    def test_document_without_roles_has_empty_roles(self, tmp_path) -> None:
        """Без default_roles поле остаётся пустым — ingest-gate отклонит документ."""
        (tmp_path / "orphan.txt").write_text("Текст.", encoding="utf-8")

        document = build_documents_from_path(tmp_path)[0]

        assert document["metadata"]["allowed_roles"] == []



class TestChunkerContract:
    def test_chunk_fields_match_schema(self) -> None:
        chunks = create_chunks(LONG_TEXT, doc_id="d1", chunk_size=200, chunk_overlap=20)

        assert chunks
        for index, chunk in enumerate(chunks):
            assert chunk["doc_id"] == "d1"
            assert chunk["index"] == index
            assert chunk["start_pos"] <= chunk["end_pos"]
            assert chunk["text"]

    def test_chunk_ids_are_deterministic(self) -> None:
        first = [c["chunk_id"] for c in create_chunks(LONG_TEXT, doc_id="d1", chunk_size=200)]
        second = [c["chunk_id"] for c in create_chunks(LONG_TEXT, doc_id="d1", chunk_size=200)]

        assert first == second

    def test_empty_text_yields_no_chunks(self) -> None:
        assert create_chunks("   ", doc_id="d1") == []
        assert create_chunks(None, doc_id="d1") == []

    def test_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_chunks("Текст для нарезки.", doc_id="d1", method="quantum")

    def test_metadata_is_propagated(self) -> None:
        chunks = create_chunks(
            "Один. Два. Три.", doc_id="d1", method="sentences",
            metadata={"allowed_roles": ["hr"]},
        )

        assert all(c["metadata"]["allowed_roles"] == ["hr"] for c in chunks)

    def test_headers_strategy_splits_by_headings(self) -> None:
        text = "# Раздел один\nТекст один.\n\n## Раздел два\nТекст два.\n"

        chunks = create_chunks(text, doc_id="d1", method="headers")

        assert len(chunks) == 2
        assert "Раздел один" in chunks[0]["text"]

    def test_headers_strategy_without_headings_single_chunk(self) -> None:
        chunks = create_chunks("Просто текст без заголовков.", doc_id="d1", method="headers")

        assert len(chunks) == 1

    def test_split_long_sentences_caps_length(self) -> None:
        """F-19: одно длинное предложение не может превысить лимит."""
        parts = split_long_sentences("я" * 250, max_size=100)

        assert all(len(p) <= 100 for p in parts)
        assert "".join(parts) == "я" * 250

    def test_split_long_sentences_passthrough(self) -> None:
        assert split_long_sentences("коротко", max_size=100) == ["коротко"]

    def test_sentence_chunks_respect_max_size(self) -> None:
        chunker = DocumentChunker(chunk_size=120, chunk_overlap=10)

        for chunk in chunker.chunk_by_sentences(LONG_TEXT, doc_id="d1", max_chunk_size=120):
            assert len(chunk["text"]) <= 120

    def test_text_chunker_preserves_roles_in_every_chunk(self, tmp_path) -> None:
        from secure_rag.config.settings import Settings

        settings = Settings(
            data_dir=tmp_path / "d", vector_db_path=tmp_path / "v", audit_secret="k" * 24
        )

        chunks = TextChunker(settings=settings).chunk_text(
            LONG_TEXT, metadata={"doc_id": "hr", "allowed_roles": ["hr"]}
        )

        assert chunks
        assert all(c["metadata"]["allowed_roles"] == ["hr"] for c in chunks)
        assert all(c["doc_id"] == "hr" for c in chunks)
        assert all("word_count" in c and "char_count" in c for c in chunks)

    def test_text_chunker_rejects_empty_text(self, tmp_path) -> None:
        from secure_rag.config.settings import Settings

        settings = Settings(
            data_dir=tmp_path / "d", vector_db_path=tmp_path / "v", audit_secret="k" * 24
        )

        assert TextChunker(settings=settings).chunk_text("   ") == []


class TestIngestDocuments:
    @staticmethod
    def _store(tmp_path, settings, acl, embedder):
        from conftest import FakeClient
        from secure_rag.storage.vector_db import VectorDatabase

        return VectorDatabase(
            settings=settings, embedder=embedder, acl=acl, client=FakeClient(metric="cosine")
        )

    def test_end_to_end_indexing(self, tmp_path, settings, acl, embedder) -> None:
        (tmp_path / "hr.txt").write_text(LONG_TEXT, encoding="utf-8")

        result = ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=["admin"],
            settings=settings,
            vector_db=self._store(tmp_path, settings, acl, embedder),
        )

        assert result["documents_total"] == 1
        assert result["indexed"] > 0
        assert result["documents_rejected"] == 0
        assert result["source"] == str(tmp_path)

    def test_document_without_roles_is_rejected(self, tmp_path, settings, acl, embedder) -> None:
        """F-02: без явных ролей документ не попадает в индекс."""
        (tmp_path / "orphan.txt").write_text(LONG_TEXT, encoding="utf-8")

        result = ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=None,
            settings=settings,
            vector_db=self._store(tmp_path, settings, acl, embedder),
        )

        assert result["indexed"] == 0
        assert result["documents_rejected"] == 1
        assert result["rejected_documents"][0]["issues"][0]["type"] == "Missing_Access_Control"

    def test_critical_secret_blocks_document(self, tmp_path, settings, acl, embedder) -> None:
        """F-09: CRITICAL-секрет блокирует индексацию, а не «мягко» редактируется."""
        (tmp_path / "leak.txt").write_text(
            "Пароль доступа: AKIAIOSFODNN7EXAMPLE и токен ghp_" + "a" * 36,
            encoding="utf-8",
        )
        store = self._store(tmp_path, settings, acl, embedder)

        result = ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=["admin"],
            settings=settings,
            vector_db=store,
        )

        assert result["indexed"] == 0
        types = {i["type"] for d in result["rejected_documents"] for i in d["issues"]}
        assert "Critical_Security_Violation" in types
        assert store.collection.count() == 0

    def test_redacted_pii_never_stored_verbatim(self, tmp_path, settings, acl, embedder) -> None:
        """Индексируется очищенный текст, а не исходный с PII."""
        (tmp_path / "pii.txt").write_text(
            "Контактный email: ivan.petrov@example.com " + LONG_TEXT, encoding="utf-8"
        )
        store = self._store(tmp_path, settings, acl, embedder)

        result = ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=["admin"],
            settings=settings,
            vector_db=store,
        )

        assert result["indexed"] > 0
        stored = " ".join(r["document"] for r in store.collection.rows.values())
        assert "ivan.petrov@example.com" not in stored

    def test_indexed_chunks_carry_roles(self, tmp_path, settings, acl, embedder) -> None:
        """Метаданные прав переезжают в каждый чанк (Zero-Trust в индексе)."""
        (tmp_path / "hr.txt").write_text(LONG_TEXT, encoding="utf-8")
        store = self._store(tmp_path, settings, acl, embedder)

        ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=["admin"],
            settings=settings,
            vector_db=store,
        )

        assert store.collection.rows
        assert all(
            "admin" in json.loads(r["metadata"]["allowed_roles"])
            for r in store.collection.rows.values()
        )

    def test_ingest_records_audit(self, tmp_path, settings, acl, embedder) -> None:
        from secure_rag.storage.audit import AuditTrail

        (tmp_path / "hr.txt").write_text(LONG_TEXT, encoding="utf-8")
        audit = AuditTrail(settings=settings)

        ingest_documents(
            str(tmp_path),
            user_id="admin",
            roles=["admin"],
            settings=settings,
            vector_db=self._store(tmp_path, settings, acl, embedder),
            audit=audit,
        )

        assert audit.read_records()

    def test_broken_json_fails_loudly(self, tmp_path, settings, acl, embedder) -> None:
        """Битый вход не игнорируется молча (fail-closed на всём пути)."""
        (tmp_path / "bad.json").write_text("{сломано", encoding="utf-8")

        with pytest.raises(IngestionError):
            ingest_documents(
                str(tmp_path),
                user_id="admin",
                roles=["admin"],
                settings=settings,
                vector_db=self._store(tmp_path, settings, acl, embedder),
            )

