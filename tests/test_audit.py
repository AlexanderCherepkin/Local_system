"""Тесты аудита (F-15).

Проверяется fail-fast на небезопасном секрете, отсутствие исходного запроса в
журнале, детерминированность HMAC и читаемость отчёта.
"""

from __future__ import annotations

import json

import pytest

from secure_rag.storage.audit import (
    AuditConfigurationError,
    AuditRecord,
    AuditTrail,
    generate_audit_report,
)


@pytest.fixture()
def audit(settings, tmp_path):
    return AuditTrail(settings=settings, log_file=tmp_path / "audit" / "events.jsonl")


class TestSecretValidation:
    def test_default_secret_rejected(self, settings, tmp_path) -> None:
        with pytest.raises(AuditConfigurationError):
            AuditTrail(settings=settings, log_file=tmp_path / "a.jsonl", secret="fallback-secret-replace-in-production")

    def test_empty_secret_rejected(self, settings, tmp_path) -> None:
        with pytest.raises(AuditConfigurationError):
            AuditTrail(settings=settings, log_file=tmp_path / "a.jsonl", secret="")

    @pytest.mark.parametrize("secret", ["", "change_me", "fallback-secret-replace-in-production"])
    def test_insecure_secrets_rejected(self, secret: str) -> None:
        with pytest.raises(AuditConfigurationError):
            AuditTrail(secret=secret, log_file=None)

    def test_is_configured_helper(self, settings) -> None:
        assert AuditTrail.is_configured(settings) is True


class TestRequestHash:
    def test_hash_is_deterministic(self, audit) -> None:
        assert audit.compute_request_hash("u", "q") == audit.compute_request_hash("u", "q")

    def test_hash_differs_by_user_and_query(self, audit) -> None:
        assert audit.compute_request_hash("u1", "q") != audit.compute_request_hash("u2", "q")
        assert audit.compute_request_hash("u", "q1") != audit.compute_request_hash("u", "q2")

    def test_hash_does_not_contain_plaintext(self, audit) -> None:
        digest = audit.compute_request_hash("john.doe", "секретный вопрос")

        assert "john.doe" not in digest
        assert "секретный" not in digest


class TestEventWriting:
    def test_event_written_as_jsonl(self, audit) -> None:
        audit.log_event(user_id="hr_user", roles=["hr"], request_hash="abc", doc_ids_extracted=["d1"])

        records = audit.read_records()
        assert len(records) == 1
        assert records[0]["user_id"] == "hr_user"
        assert records[0]["doc_ids_extracted"] == ["d1"]
        assert records[0]["schema_version"] == "1.0"

    def test_prompt_is_not_stored(self, audit) -> None:
        audit.log_event(user_id="u", request_length=42, request_hash="h")

        raw = audit.log_file.read_text(encoding="utf-8")
        assert "request_length" in raw
        assert "prompt" not in raw.replace("prompt_eval", "")

    def test_acl_denial_recorded(self, audit) -> None:
        audit.log_event(user_id="u", acl_decision="deny", status="denied", error_type="AccessDeniedError")

        assert audit.read_records()[0]["acl_decision"] == "deny"

    def test_latency_timer(self) -> None:
        metrics: dict = {}
        with AuditTrail.stage_timer(metrics, "search"):
            pass

        assert "latency_search_ms" in metrics
        assert metrics["latency_search_ms"] >= 0.0

    def test_oversized_log_archived_not_deleted(self, settings, tmp_path) -> None:
        log_file = tmp_path / "audit" / "events.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("x" * 100, encoding="utf-8")

        AuditTrail(settings=settings, log_file=log_file, max_bytes=10)

        assert log_file.exists()
        assert (tmp_path / "audit" / "events_bak.jsonl").exists()


class TestAuditRecord:
    def test_roundtrip(self) -> None:
        record = AuditRecord(user_id="u", roles=["hr"], doc_ids_extracted=["d1"], status="success")

        restored = AuditRecord.from_dict(json.loads(record.to_json()))

        assert restored.user_id == "u"
        assert restored.roles == ["hr"]
        assert restored.doc_ids_extracted == ["d1"]

    def test_unknown_keys_ignored(self) -> None:
        record = AuditRecord.from_dict({"user_id": "u", "unexpected": "value"})

        assert record.user_id == "u"
        assert not hasattr(record, "unexpected")


class TestReport:
    def test_report_counts_events(self, audit, settings) -> None:
        audit.log_event(user_id="hr_user", acl_decision="allow", status="success", latency_total_ms=10)
        audit.log_event(user_id="eng_user", acl_decision="deny", status="denied", latency_total_ms=30)

        report = generate_audit_report(settings=settings, log_file=audit.log_file)

        assert report["total_events"] == 2
        assert report["acl_denied"] == 1
        assert sorted(report["unique_users"]) == ["eng_user", "hr_user"]
        assert report["latency_total_ms"]["count"] == 2

    def test_report_requires_secure_secret(self, tmp_path) -> None:
        from secure_rag.config.settings import Settings

        unsafe = Settings(audit_secret="fallback-secret-replace-in-production", audit_log_file=tmp_path / "a.jsonl")

        with pytest.raises(AuditConfigurationError):
            generate_audit_report(settings=unsafe)
