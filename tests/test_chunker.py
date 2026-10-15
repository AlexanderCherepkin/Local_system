"""Тесты DocumentChunker: стратегии нарезки и обёртка chunk_text."""

from __future__ import annotations

import pytest

from secure_rag.pipeline.chunker import DocumentChunker, chunk_text


@pytest.fixture()
def chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


class TestConstructor:
    def test_overlap_ge_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=50, chunk_overlap=50)


class TestChunkBySize:
    def test_empty_text(self, chunker) -> None:
        assert chunker.chunk_by_size("") == []

    def test_short_text_single_chunk(self, chunker) -> None:
        chunks = chunker.chunk_by_size("короткий текст", doc_id="d")

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "d::00000"
        assert chunks[0]["is_last_chunk"] is True

    def test_long_text_sliding_window(self, chunker) -> None:
        text = "а" * 250
        chunks = chunker.chunk_by_size(text, doc_id="d")

        assert len(chunks) == 3
        # Шаг окна = chunk_size - chunk_overlap = 80.
        assert chunks[1]["start_pos"] == 80
        assert chunks[-1]["is_last_chunk"] is True


class TestChunkByHeaders:
    def test_no_headers_single_chunk(self, chunker) -> None:
        chunks = chunker.chunk_by_headers("просто текст", doc_id="d")

        assert len(chunks) == 1
        assert chunks[0]["header_level"] == 0

    def test_split_by_headers(self, chunker) -> None:
        text = "# Первый\nТекст первого.\n\n## Второй\nТекст второго."
        chunks = chunker.chunk_by_headers(text, doc_id="d")

        assert len(chunks) == 2
        assert chunks[0]["header_level"] == 1
        assert chunks[1]["header_level"] == 2
        assert chunks[1]["header_text"] == "Второй"


class TestChunkDocument:
    def test_unknown_strategy_raises(self, chunker) -> None:
        with pytest.raises(ValueError):
            chunker.chunk_document({"text": "x"}, strategy="magic")

    def test_metadata_and_positions_propagated(self, chunker) -> None:
        doc = {
            "text": "Первое предложение. Второе предложение.",
            "metadata": {"doc_id": "d1", "allowed_roles": ["hr"]},
        }

        chunks = chunker.chunk_document(doc, strategy="sentences")

        assert chunks
        assert all(c["doc_id"] == "d1" for c in chunks)
        assert all(c["metadata"]["allowed_roles"] == ["hr"] for c in chunks)
        assert chunks[0]["is_first_chunk"] is True
        assert chunks[-1]["is_last_chunk"] is True

    def test_size_strategy(self, chunker) -> None:
        chunks = chunker.chunk_document(
            {"text": "б" * 300, "metadata": {"doc_id": "d"}}, strategy="size"
        )

        assert len(chunks) > 1

    def test_headers_strategy(self, chunker) -> None:
        chunks = chunker.chunk_document(
            {"text": "# A\nтекст", "metadata": {"doc_id": "d"}}, strategy="headers"
        )

        assert len(chunks) == 1


class TestProcessDocuments:
    def test_parent_fields_added(self, chunker) -> None:
        docs = [
            {
                "text": "Предложение одно. Предложение два.",
                "title": "Док",
                "metadata": {"doc_id": "d1"},
            }
        ]

        chunks = chunker.process_documents(docs, strategy="sentences")

        assert chunks
        assert all(c["parent_doc_id"] == "d1" for c in chunks)
        assert all(c["parent_doc_title"] == "Док" for c in chunks)


class TestChunkTextWrapper:
    def test_wrapper_uses_size_strategy_by_default(self) -> None:
        chunks = chunk_text("текст " * 200, size=100, overlap=20, doc_id="w")

        assert chunks
        assert chunks[0]["doc_id"] == "w"

    def test_wrapper_sentences(self) -> None:
        chunks = chunk_text(
            "Одно. Два. Три.", strategy="sentences", size=500, doc_id="w"
        )

        assert chunks
