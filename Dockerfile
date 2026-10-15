# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY secure_rag/ secure_rag/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: Tests (unit + static analysis) — сборка падает, если тесты не проходят
FROM builder AS tester

RUN pip install --no-cache-dir pytest pytest-cov ruff bandit

COPY tests/ tests/

RUN ruff check secure_rag/ && \
    bandit -r secure_rag/ -ll && \
    pytest tests/ -q --cov=secure_rag --cov-fail-under=85

# Stage 3: Production image
FROM python:3.11-slim

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY secure_rag/ secure_rag/
COPY --from=tester /app/.coverage /app/.coverage

RUN mkdir -p /app/data /app/models /app/audit_logs /app/metrics && \
    chown -R appuser:appuser /app

USER appuser

ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "secure_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
