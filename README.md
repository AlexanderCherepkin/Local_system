# Secure RAG — Enterprise RAG-агент с Zero-Trust ACL (air-gapped)

Безопасный RAG-пайплайн для корпоративных документов: локальные модели
(sentence-transformers + Ollama), векторная база ChromaDB, строгий ACL
(deny-by-default), редакция PII/секретов на входе, JWT-аудит и проверка цитат.

## Быстрый старт

```bash
pip install -e .
cp .env.example .env          # задайте AUDIT_SECRET
secure-rag selfcheck
secure-rag ingest --path ./docs --roles hr
secure-rag query --prompt "Сколько дней отпуска?" --user hr_user
```

## Docker

```bash
docker compose up --build     # Ollama + API в изолированной сети
```

Сборка образа автоматически прогоняет тесты, ruff и bandit (multi-stage).

## API

- `POST /api/v1/query` — запрос к RAG (требует `Authorization: Bearer <JWT>`)
- `POST /api/v1/auth/token` — выпуск токена (только в DEBUG-режиме; в production — корпоративный IdP)
- `GET /healthz` — liveness
- `GET /readyz` — readiness (проверяет Ollama, индекс, ACL, аудит)

## Тесты

```bash
pytest --cov=secure_rag
```

## Правила проектирования

Смотрите [police.txt](police.txt): Zero-Trust ingestion, air-gapped deployment, audit trail.
