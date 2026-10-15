"""Тесты единой точки входа пайплайна (secure_rag.pipeline.runner)."""

from __future__ import annotations

from secure_rag.pipeline import runner
from secure_rag.security.acl import Principal


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list = []

    def process_query(self, prompt, user, **kwargs):
        self.calls.append({"prompt": prompt, "user": user, **kwargs})
        return {"answer": "ответ", "sources": []}


class TestGetPipeline:
    def setup_method(self) -> None:
        runner.set_pipeline(None)

    def teardown_method(self) -> None:
        runner.set_pipeline(None)

    def test_singleton_reused(self) -> None:
        fake = _FakePipeline()
        runner.set_pipeline(fake)

        assert runner.get_pipeline() is fake

    def test_set_pipeline_none_resets(self, settings, monkeypatch) -> None:
        import secure_rag.pipeline.runner as runner_mod

        created: list = []

        class _Stub:
            def __init__(self, settings=None):
                created.append(settings)

        monkeypatch.setattr(runner_mod, "RAGPipeline", _Stub)
        runner.set_pipeline(None)
        pipeline = runner.get_pipeline(settings=settings)

        assert pipeline is not None
        assert created == [settings]


class TestRunPipeline:
    def setup_method(self) -> None:
        runner.set_pipeline(None)

    def teardown_method(self) -> None:
        runner.set_pipeline(None)

    def test_wraps_identity_into_principal(self) -> None:
        fake = _FakePipeline()
        runner.set_pipeline(fake)

        runner.run_pipeline(prompt="вопрос", username="hr_user", roles=["hr"])

        call = fake.calls[0]
        assert isinstance(call["user"], Principal)
        assert call["user"].user_id == "hr_user"
        assert call["user"].roles == ("hr",)
        assert call["user"].trusted is True

    def test_roles_default_to_username(self) -> None:
        fake = _FakePipeline()
        runner.set_pipeline(fake)

        runner.run_pipeline(prompt="вопрос", username="service")

        assert fake.calls[0]["user"].roles == ("service",)

    def test_returns_pipeline_result(self) -> None:
        fake = _FakePipeline()
        runner.set_pipeline(fake)

        result = runner.run_pipeline(prompt="вопрос", username="u", roles=["r"])

        assert result == {"answer": "ответ", "sources": []}

    def test_kwargs_forwarded(self) -> None:
        fake = _FakePipeline()
        runner.set_pipeline(fake)

        runner.run_pipeline(prompt="вопрос", username="u", roles=["r"], top_k=3)

        assert fake.calls[0]["top_k"] == 3
