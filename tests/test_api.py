"""Тесты HTTP API (F-06, F-13, F-16).

FastAPI в изолированном контуре может быть не установлен — тогда модуль
пропускается целиком через ``importorskip``, чтобы отсутствие зависимости
не выглядело как «успешно пройденный» тест.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi не установлен в окружении")

from fastapi.testclient import TestClient  # noqa: E402

from secure_rag.security.acl import Principal  # noqa: E402

OK_RESULT = {
    "answer": "Согласно документу [1].",
    "sources": [{"fragment": 1, "doc_id": "hr", "source_file": "hr.txt"}],
    "abstained": False,
    "metadata": {"request_id": "req-1"},
}


class _FakePipeline:
    """Замена RAGPipeline: фиксированный ответ без обращения к Ollama."""

    def __init__(self, result: dict) -> None:
        self._result = result
        self.calls: list = []

    def process_query(self, prompt, user, top_k=None, **kwargs):
        self.calls.append({"prompt": prompt, "user": user, "top_k": top_k})
        return dict(self._result)


@pytest.fixture()
def api_module(settings, monkeypatch):
    """Модуль API с подменённым singleton пайплайна."""
    from secure_rag import api as api_mod

    fake = _FakePipeline(OK_RESULT)
    monkeypatch.setattr(api_mod, "_pipeline", fake, raising=False)
    monkeypatch.setattr(api_mod, "get_pipeline_dep", lambda: fake)
    return api_mod, fake


@pytest.fixture()
def client(api_module) -> TestClient:
    api_mod, _ = api_module
    return TestClient(api_mod.app)


@pytest.fixture()
def token(settings) -> str:
    from secure_rag.security.auth import TokenService

    return TokenService(settings=settings).issue_token("hr_user", ["hr"])


class TestHealthEndpoints:
    def test_liveness_requires_no_dependencies(self, client) -> None:
        """F-16: liveness отвечает без проверок внешних сервисов."""
        response = client.get("/healthz")

        assert response.status_code == 200


class TestQueryAuthentication:
    def test_query_requires_token(self, client) -> None:
        response = client.post("/api/v1/query", json={"prompt": "вопрос"})

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_query_rejects_non_bearer(self, client) -> None:
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )

        assert response.status_code == 401

    def test_query_rejects_garbage_token(self, client) -> None:
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": "Bearer not.a.jwt"},
        )

        assert response.status_code == 401

    def test_valid_token_returns_answer(self, client, token) -> None:
        response = client.post(
            "/api/v1/query",
            json={"prompt": "Сколько дней отпуска?"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["sources"] == ["[1] hr (hr.txt)"]
        assert body["request_id"] == "req-1"

    def test_identity_comes_from_token_not_body(self, client, token, api_module) -> None:
        """F-06: пользователь не может заявить роль в теле запроса."""
        _, pipeline = api_module
        client.post(
            "/api/v1/query",
            json={"prompt": "вопрос", "top_k": 2},
            headers={"Authorization": f"Bearer {token}"},
        )

        user = pipeline.calls[-1]["user"]
        assert isinstance(user, Principal)
        assert user.user_id == "hr_user"
        assert set(user.roles) == {"hr"}

    def test_empty_prompt_rejected_by_schema(self, client, token) -> None:
        response = client.post(
            "/api/v1/query",
            json={"prompt": ""},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 422


class TestQueryErrorMapping:
    def test_access_denied_maps_to_403(self, client, token, api_module) -> None:
        api_module[1]._result = {"error": "access_denied", "answer": ""}
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403

    def test_prompt_injection_maps_to_400(self, client, token, api_module) -> None:
        api_module[1]._result = {"error": "prompt_injection_detected", "answer": ""}
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400

    def test_internal_error_hides_details(self, client, token, api_module) -> None:
        """F-13: внутренняя ошибка не раскрывает деталей клиенту."""
        api_module[1]._result = {
            "error": "internal_error",
            "answer": "",
            "metadata": {"request_id": "req-9"},
        }
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 503
        assert "internal_error" not in response.text

    def test_abstained_answer_returns_200(self, client, token, api_module) -> None:
        api_module[1]._result = {
            "answer": "Недостаточно информации.",
            "sources": [],
            "abstained": True,
            "metadata": {"request_id": "req-2"},
        }
        response = client.post(
            "/api/v1/query",
            json={"prompt": "вопрос"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["abstained"] is True


class TestRateLimit:
    def test_burst_is_limited(self, client, token, api_module) -> None:
        api_mod, _ = api_module
        api_mod._rate_buckets.clear()
        api_mod.RATE_LIMIT_REQUESTS = 2

        headers = {"Authorization": f"Bearer {token}"}
        codes = [
            client.post("/api/v1/query", json={"prompt": "вопрос"}, headers=headers).status_code
            for _ in range(4)
        ]

        assert codes == [200, 200, 429, 429]

    def test_limit_rejects_before_authorization(self, client, api_module) -> None:
        """Перебор токенов обрывается rate limiting до обращения к пайплайну."""
        api_mod, pipeline = api_module
        api_mod._rate_buckets.clear()
        api_mod.RATE_LIMIT_REQUESTS = 1

        client.post("/api/v1/query", json={"prompt": "в"}, headers={"Authorization": "Bearer bad"})
        response = client.post(
            "/api/v1/query", json={"prompt": "в"}, headers={"Authorization": "Bearer bad"}
        )

        assert response.status_code == 429
        assert pipeline.calls == []


class TestTokenIssuance:
    def test_token_endpoint_blocked_outside_debug(self, client) -> None:
        """Выпуск токенов остаётся за корпоративным IdP."""
        response = client.post("/api/v1/auth/token", json={"username": "hr_user"})

        assert response.status_code == 501

    def test_no_user_enumeration_oracle(self, client) -> None:
        """Ответ на неизвестного пользователя не отличается от общего режима."""
        unknown = client.post("/api/v1/auth/token", json={"username": "nobody"})

        assert unknown.status_code == 501


class TestReadiness:
    def test_readyz_reports_checks(self, client, settings, monkeypatch) -> None:
        """F-16: /readyz проверяет аудит, ACL, векторный индекс и Ollama."""
        import secure_rag.api as api_mod

        class _FakeAcl:
            def __init__(self, settings=None):
                pass

            def known_users(self):
                return ["u1"]

        class _FakeStore:
            def __init__(self, settings=None):
                pass

            def healthcheck(self):
                return {"ready": True}

        monkeypatch.setattr("secure_rag.security.acl.AccessControlList", _FakeAcl)
        monkeypatch.setattr("secure_rag.storage.vector_db.VectorDatabase", _FakeStore)

        class _Resp:
            status_code = 200

        monkeypatch.setattr(
            "requests.get", lambda url, timeout=0: _Resp()
        )

        response = client.get("/readyz")

        assert response.status_code == 200
        body = response.json()
        assert body["checks"]["acl"] == "ok"
        assert body["checks"]["ollama"] == "ok"
        assert body["checks"]["vector_store"]["ready"] is True


class TestDebugTokenIssuance:
    def test_token_issued_in_debug_mode(self, client, settings, monkeypatch) -> None:
        """В DEBUG-режиме токен выдаётся пользователю из ACL-хранилища."""
        import secure_rag.api as api_mod

        monkeypatch.setattr(settings, "log_level", "DEBUG")

        # jane.smith есть в пакетном acl_storage.json с ролью hr.
        response = client.post("/api/v1/auth/token", json={"username": "jane.smith"})

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]

    def test_unknown_user_in_debug_401(self, client, settings, monkeypatch) -> None:
        monkeypatch.setattr(settings, "log_level", "DEBUG")

        response = client.post("/api/v1/auth/token", json={"username": "ghost"})

        assert response.status_code == 401


