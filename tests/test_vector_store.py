"""Тесты векторного хранилища (F-03, F-07, F-08, F-10).

Проверяется сквозной путь: чанк -> ACL-проверка -> redaction -> эмбеддинг ->
upsert с текстом -> поиск с фильтром по ролям и корректной метрикой.
"""

from __future__ import annotations

import pytest

from secure_rag.core.schemas import Chunk, ChunkMetadata
from secure_rag.security.acl import AccessDeniedError, Principal
from secure_rag.storage.vector_db import VectorDatabase, VectorStoreError

from conftest import FakeClient


def make_chunk(doc_id: str, text: str, roles: list, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{index:05d}",
        doc_id=doc_id,
        chunk_index=index,
        text=text,
        start_pos=0,
        end_pos=len(text),
        total_chunks=1,
        metadata=ChunkMetadata(doc_id=doc_id, source_file=f"{doc_id}.txt", allowed_roles=roles),
    )


class TestIndexing:
    def test_chunk_text_is_persisted(self, store, principal_hr) -> None:
        """Регресс F-07: текст документа должен попадать в коллекцию."""
        result = store.add_chunks(
            [make_chunk("doc1", "Отпуск составляет 28 календарных дней.", ["hr"])],
            principal=principal_hr,
        )

        assert result["indexed"] == 1
        stored = store.collection.rows["doc1::00000"]
        assert "28 календарных дней" in stored["document"]

    def test_chunk_without_acl_rejected(self, store, principal_hr) -> None:
        result = store.add_chunks([make_chunk("doc2", "Текст без прав доступа", [])], principal=principal_hr)

        assert result["indexed"] == 0
        assert result["rejected"] == 1
        assert result["rejected_details"][0]["reason"] == "missing_or_empty_allowed_roles"
        assert "doc2::00000" not in store.collection.rows

    def test_chunk_with_secret_rejected(self, store, principal_hr) -> None:
        result = store.add_chunks(
            [make_chunk("doc3", "Пароль доступа password=SuperSecret1 к системе", ["hr"])],
            principal=principal_hr,
        )

        assert result["indexed"] == 0
        assert result["rejected_details"][0]["reason"] == "critical_secrets_detected"
        assert "doc3::00000" not in store.collection.rows

    def test_pii_is_redacted_before_indexing(self, store, principal_hr) -> None:
        store.add_chunks([make_chunk("doc4", "Телефон HR: +7 916 123 45 67", ["hr"])], principal=principal_hr)

        stored = store.collection.rows["doc4::00000"]
        assert "916" not in stored["document"]
        assert "[REDACTED:PHONE_RU]" in stored["document"]

    def test_embedding_dimension_validated(self, store, principal_hr) -> None:
        """Регресс F-08: размерность эмбеддинга сверяется с конфигурацией."""
        store.embedder.dimension_value = 128  # расходится с settings.embedding_dimension

        result = store.add_chunks([make_chunk("doc5", "Текст", ["hr"])], principal=principal_hr)

        assert result["indexed"] == 0
        assert result["errors"]

    def test_reindex_is_idempotent(self, store, principal_hr) -> None:
        chunk = [make_chunk("doc6", "Одинаковый текст для повторной индексации", ["hr"])]

        store.add_chunks(chunk, principal=principal_hr)
        store.add_chunks(chunk, principal=principal_hr)

        assert store.count() == 1

    def test_empty_chunk_skipped(self, store, principal_hr) -> None:
        result = store.add_chunks([make_chunk("doc7", "   ", ["hr"])], principal=principal_hr)

        assert result["indexed"] == 0


class TestSearch:
    @pytest.fixture()
    def populated(self, store, principal_hr):
        store.add_chunks(
            [
                make_chunk("hr_doc", "Отпуск составляет 28 календарных дней.", ["hr"]),
                make_chunk("eng_doc", "Релиз новой версии запланирован на пятницу.", ["engineering"]),
                make_chunk("shared", "Общие правила внутреннего распорядка.", ["hr", "engineering"]),
            ],
            principal=principal_hr,
        )
        return store

    def test_user_sees_only_own_documents(self, populated, principal_hr, principal_eng) -> None:
        hr_result = populated.search("отпуск календарных дней", principal=principal_hr, top_k=5)
        eng_result = populated.search("релиз версии пятница", principal=principal_eng, top_k=5)

        hr_ids = {c["doc_id"] for c in hr_result["chunks"]}
        eng_ids = {c["doc_id"] for c in eng_result["chunks"]}

        assert "hr_doc" in hr_ids
        assert "eng_doc" not in hr_ids
        assert "eng_doc" in eng_ids
        assert "hr_doc" not in eng_ids

    def test_acl_filter_sent_to_database(self, populated, principal_hr) -> None:
        """Фильтр должен уходить в БД, а не применяться только постфактум (F-05)."""
        populated.search("вопрос", principal=principal_hr)

        assert populated.collection.last_query["where"] == {"allowed_roles": {"$in": ["hr"]}}

    def test_search_denied_without_roles(self, populated) -> None:
        with pytest.raises(AccessDeniedError):
            populated.search("вопрос", principal=Principal("noroles", ()))

    def test_search_denied_for_unknown_user(self, populated) -> None:
        with pytest.raises(AccessDeniedError):
            populated.search("вопрос", principal=Principal("ghost", ("admin",)))

    def test_retrieved_chunk_contains_text(self, populated, principal_hr) -> None:
        result = populated.search("отпуск", principal=principal_hr, top_k=3)

        assert result["chunks"]
        assert any("28 календарных дней" in c["text"] for c in result["chunks"])

    def test_empty_query_rejected(self, populated, principal_hr) -> None:
        with pytest.raises(ValueError):
            populated.search("   ", principal=principal_hr)

    def test_top_k_is_clamped(self, populated, principal_hr) -> None:
        result = populated.search("вопрос", principal=principal_hr, top_k=999)

        assert len(result["chunks"]) <= populated.settings.max_top_k


class TestMetricHandling:
    def test_metric_mismatch_detected(self, settings, acl, embedder) -> None:
        """Регресс F-10: индекс с другой метрикой использовать нельзя."""
        client = FakeClient(preexisting_metric="l2")

        with pytest.raises(VectorStoreError):
            VectorDatabase(settings=settings, embedder=embedder, acl=acl, client=client)

    def test_cosine_distance_converted_to_similarity(self, store) -> None:
        assert store._distance_to_similarity(0.0) == pytest.approx(1.0)
        assert store._distance_to_similarity(1.0) == pytest.approx(0.0)

    def test_l2_distance_converted_by_formula(self, settings, acl, embedder) -> None:
        l2_settings = settings.model_copy(update={"distance_metric": "l2"})
        store = VectorDatabase(settings=l2_settings, embedder=embedder, acl=acl, client=FakeClient(metric="l2"))

        assert store._distance_to_similarity(0.0) == pytest.approx(1.0)

    def test_unsupported_metric_rejected(self, settings, acl, embedder) -> None:
        bad = settings.model_copy(update={"distance_metric": "manhattan"})

        with pytest.raises(VectorStoreError):
            VectorDatabase(settings=bad, embedder=embedder, acl=acl, client=FakeClient())


class TestHealthcheck:
    def test_healthcheck_reports_metric(self, store) -> None:
        info = store.healthcheck()

        assert info["ready"] is True
        assert info["collection_metric"] == store.settings.distance_metric

