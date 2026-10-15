"""Тесты CLI Secure RAG (команды ingest/query/audit-report/selfcheck, парсер)."""

from __future__ import annotations

import json

import pytest

from secure_rag.cli import build_parser, main


class TestParser:
    def test_parser_requires_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_ingest_args(self) -> None:
        args = build_parser().parse_args(
            ["ingest", "--path", "docs", "--user", "admin", "--roles", "hr,viewer"]
        )

        assert args.command == "ingest"
        assert args.path == "docs"
        assert args.roles == "hr,viewer"

    def test_query_args(self) -> None:
        args = build_parser().parse_args(
            ["query", "--prompt", "вопрос", "--user", "u1", "--json"]
        )

        assert args.command == "query"
        assert args.json is True


class TestCommands:
    def test_ingest_command_prints_summary(
        self, tmp_path, settings, monkeypatch, capsys
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("Документ один." * 20, encoding="utf-8")

        recorded: dict = {}

        def fake_ingest(path, user_id="admin", roles=None, settings=None, **kwargs):
            recorded.update({"path": path, "user_id": user_id, "roles": roles})
            return {"indexed": 1, "rejected": 0, "errors": []}

        import secure_rag.ingestion.loader as loader_mod

        monkeypatch.setattr(loader_mod, "ingest_documents", fake_ingest)

        code = main(["ingest", "--path", str(docs), "--user", "admin"])

        assert code == 0
        assert recorded["path"] == str(docs)
        assert "indexed" in capsys.readouterr().out

    def test_ingest_command_fails_on_errors(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        def fake_ingest(*args, **kwargs):
            return {"indexed": 0, "errors": ["boom"]}

        import secure_rag.ingestion.loader as loader_mod

        monkeypatch.setattr(loader_mod, "ingest_documents", fake_ingest)

        code = main(["ingest", "--path", str(tmp_path)])

        assert code == 1

    def test_query_denied_without_roles(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_roles_for_user", lambda user, roles: [])

        code = main(["query", "--prompt", "вопрос", "--user", "norole"])

        assert code == 2
        assert "Отказано" in capsys.readouterr().err

    def test_query_access_denied(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_roles_for_user", lambda user, roles: ["hr"])

        class _Pipeline:
            def __init__(self, settings=None):
                pass

            def process_query(self, prompt, user):
                return {"error": "access_denied", "answer": ""}

        monkeypatch.setattr("secure_rag.pipeline.generator.RAGPipeline", _Pipeline)

        code = main(["query", "--prompt", "вопрос", "--user", "u1"])

        assert code == 2
        assert "Отказано в доступе" in capsys.readouterr().err

    def test_query_prompt_injection(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_roles_for_user", lambda user, roles: ["hr"])

        class _Pipeline:
            def __init__(self, settings=None):
                pass

            def process_query(self, prompt, user):
                return {"error": "prompt_injection_detected", "answer": ""}

        monkeypatch.setattr("secure_rag.pipeline.generator.RAGPipeline", _Pipeline)

        code = main(["query", "--prompt", "инъекция", "--user", "u1"])

        assert code == 2

    def test_query_success_json(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_roles_for_user", lambda user, roles: ["hr"])

        class _Pipeline:
            def __init__(self, settings=None):
                pass

            def process_query(self, prompt, user):
                return {
                    "answer": "Ответ [1].",
                    "sources": [
                        {"fragment": 1, "doc_id": "d", "source_file": "d.txt", "score": 0.9}
                    ],
                    "abstained": False,
                }

        monkeypatch.setattr("secure_rag.pipeline.generator.RAGPipeline", _Pipeline)

        code = main(["query", "--prompt", "вопрос", "--user", "u1", "--json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["answer"] == "Ответ [1]."

    def test_audit_report_command(self, settings, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        monkeypatch.setattr(
            cli_mod, "generate_audit_report", lambda settings=None: {"total_events": 0}
        )

        code = main(["audit-report"])

        assert code == 0
        assert "total_events" in capsys.readouterr().out

    def test_selfcheck_fails_without_users(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod

        class _EmptyAcl:
            def __init__(self, settings=None):
                pass

            def known_users(self):
                return []

        class _BrokenAudit:
            def __init__(self, settings=None):
                from secure_rag.storage.audit import AuditConfigurationError

                raise AuditConfigurationError("no secret")

        monkeypatch.setattr(cli_mod, "AccessControlList", _EmptyAcl)
        monkeypatch.setattr(cli_mod, "AuditTrail", _BrokenAudit)

        code = main(["selfcheck"])

        assert code == 1
        assert "failed" in capsys.readouterr().out

    def test_selfcheck_passes_when_ready(self, monkeypatch, capsys) -> None:
        import secure_rag.cli as cli_mod
        import secure_rag.storage.vector_db as vdb_mod

        class _Acl:
            def __init__(self, settings=None):
                pass

            def known_users(self):
                return ["u1"]

        class _Audit:
            def __init__(self, settings=None):
                pass

        class _Store:
            def __init__(self, settings=None):
                pass

            def healthcheck(self):
                return {"ready": True}

        monkeypatch.setattr(cli_mod, "AccessControlList", _Acl)
        monkeypatch.setattr(cli_mod, "AuditTrail", _Audit)
        monkeypatch.setattr(vdb_mod, "VectorDatabase", _Store)

        code = main(["selfcheck"])

        assert code == 0
        assert "passed" in capsys.readouterr().out

