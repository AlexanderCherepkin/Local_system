"""Тесты канонических контрактов данных (core/schemas.py)."""

from __future__ import annotations

import json

import pytest

from secure_rag.core.schemas import (
    AccessDecision,
    Chunk,
    ChunkMetadata,
    Document,
    RedactionFinding,
    RetrievedChunk,
    _coerce_roles,
    content_hash,
    make_chunk_id,
    normalize_chunk,
)


class TestHelpers:
    def test_content_hash_deterministic(self) -> None:
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")

    def test_content_hash_empty_string(self) -> None:
        assert content_hash("") == content_hash("")

    def test_make_chunk_id_format(self) -> None:
        assert make_chunk_id("doc", 3) == "doc::00003"

    def test_make_chunk_id_default_doc(self) -> None:
        assert make_chunk_id("", 0) == "doc::00000"


class TestCoerceRoles:
    def test_none_returns_empty(self) -> None:
        assert _coerce_roles(None) == []

    def test_json_string(self) -> None:
        assert _coerce_roles('["hr", "legal"]') == ["hr", "legal"]

    def test_broken_json_string_single(self) -> None:
        assert _coerce_roles("[broken") == ["[broken"]

    def test_comma_string(self) -> None:
        assert _coerce_roles("hr, legal") == ["hr", "legal"]

    def test_plain_string(self) -> None:
        assert _coerce_roles("hr") == ["hr"]

    def test_list(self) -> None:
        # Фильтруются только пустые значения; пробелы внутри элементов
        # сохраняются (контракт _coerce_roles).
        assert _coerce_roles(["hr", "", "legal"]) == ["hr", "legal"]

    def test_scalar(self) -> None:
        assert _coerce_roles(5) == ["5"]


class TestNormalizeChunk:
    def test_none_returns_empty(self) -> None:
        assert normalize_chunk(None) == {}

    def test_chunk_instance(self) -> None:
        chunk = Chunk(doc_id="d", text="текст", chunk_index=2)

        data = normalize_chunk(chunk)

        assert data["doc_id"] == "d"
        assert data["chunk_id"] == "d::00002"

    def test_legacy_aliases(self) -> None:
        data = normalize_chunk(
            {"document_id": "d", "start_position": 5, "end_position": 10, "text": "x"}
        )

        assert data["doc_id"] == "d"
        assert data["start_pos"] == 5
        assert data["end_pos"] == 10

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            normalize_chunk(42)

    def test_object_with_dict_attrs(self) -> None:
        class Obj:
            def __init__(self) -> None:
                self.doc_id = "d"
                self.text = "t"

        data = normalize_chunk(Obj())

        assert data["doc_id"] == "d"

    def test_metadata_from_dict(self) -> None:
        data = normalize_chunk(
            {"doc_id": "d", "text": "t", "metadata": {"allowed_roles": ["hr"]}}
        )

        assert data["metadata"]["allowed_roles"] == ["hr"]


class TestChunkMetadata:
    def test_roundtrip(self) -> None:
        meta = ChunkMetadata(doc_id="d", allowed_roles=["hr"], source_file="d.txt")

        restored = ChunkMetadata.from_dict(meta.to_dict())

        assert restored.doc_id == "d"
        assert restored.allowed_roles == ["hr"]

    def test_vector_metadata_serializes_roles_to_json(self) -> None:
        meta = ChunkMetadata(doc_id="d", allowed_roles=["hr", "legal"])

        vector_meta = meta.to_vector_metadata()

        assert json.loads(vector_meta["allowed_roles"]) == ["hr", "legal"]

    def test_extra_fields_preserved(self) -> None:
        meta = ChunkMetadata.from_dict({"doc_id": "d", "custom": "value"})

        assert meta.extra["custom"] == "value"


class TestChunk:
    def test_post_init_fills_metadata(self) -> None:
        chunk = Chunk(doc_id="d", text="текст")

        assert chunk.metadata.doc_id == "d"
        assert chunk.metadata.content_hash == content_hash("текст")
        assert chunk.chunk_id == "d::00000"

    def test_metadata_from_raw_dict(self) -> None:
        chunk = Chunk(doc_id="d", text="t", metadata={"allowed_roles": ["hr"]})

        assert isinstance(chunk.metadata, ChunkMetadata)
        assert chunk.metadata.allowed_roles == ["hr"]

    def test_counts(self) -> None:
        chunk = Chunk(text="три слова здесь")

        assert chunk.char_count == len("три слова здесь")
        assert chunk.word_count == 3

    def test_to_dict_canonical(self) -> None:
        data = Chunk(doc_id="d", text="t").to_dict()

        assert data["chunk_id"] == "d::00000"
        assert "start_position" not in data

    def test_to_vector_metadata(self) -> None:
        chunk = Chunk(doc_id="d", text="t", start_pos=1, end_pos=2)

        meta = chunk.to_vector_metadata()

        assert meta["start_pos"] == 1
        assert meta["end_pos"] == 2
        assert meta["chunk_id"] == "d::00000"

    def test_from_dict_roundtrip(self) -> None:
        original = Chunk(doc_id="d", text="текст", chunk_index=1)

        restored = Chunk.from_dict(original.to_dict())

        assert restored.doc_id == "d"
        assert restored.text == "текст"
        assert restored.chunk_index == 1


class TestRetrievedChunk:
    def test_from_chunk_carries_similarity(self) -> None:
        chunk = Chunk(doc_id="d", text="t")

        retrieved = RetrievedChunk.from_chunk(
            chunk.to_dict(), similarity=0.9, distance=0.1
        )

        assert retrieved.similarity == pytest.approx(0.9)
        assert retrieved.distance == pytest.approx(0.1)
        assert retrieved.text == "t"


class TestDocument:
    def test_roundtrip(self) -> None:
        doc = Document(doc_id="d", text="текст", title="Заголовок")

        restored = Document.from_dict(doc.to_dict())

        assert restored.doc_id == "d"
        assert restored.text == "текст"
        assert restored.char_count == len("текст")

    def test_legacy_document_id_alias(self) -> None:
        doc = Document.from_dict({"document_id": "legacy", "text": "t"})

        assert doc.doc_id == "legacy"


class TestSmallModels:
    def test_redaction_finding_roundtrip(self) -> None:
        finding = RedactionFinding(type="EMAIL", severity="MEDIUM", snippet="a***")

        restored = RedactionFinding.from_dict(finding.to_dict())

        assert restored.type == "EMAIL"

    def test_access_decision_defaults_denied(self) -> None:
        decision = AccessDecision.from_dict({})

        assert decision.allowed is False

    def test_access_decision_roundtrip(self) -> None:
        decision = AccessDecision(
            allowed=True, user_roles=["hr"], allowed_doc_ids=["d1"]
        )

        restored = AccessDecision.from_dict(decision.to_dict())

        assert restored.allowed is True
        assert restored.allowed_doc_ids == ["d1"]

        data = normalize_chunk(
            {"doc_id": "d", "text": "t", "metadata": {"allowed_roles": ["hr"]}}
        )

        assert data["metadata"]["allowed_roles"] == ["hr"]
