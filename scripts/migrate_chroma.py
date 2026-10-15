"""Миграция индекса ChromaDB под схему Secure RAG (Zero-Trust ACL).

Что делает скрипт:

1. Создаёт резервную копию каталога старого индекса (``--backup-dir``).
2. Читает все документы старой коллекции (``--source-collection``).
3. Для каждого документа:
   * дополняет метаданные ``allowed_roles`` (``--default-roles``);
   * прогоняет redaction-gate (CRITICAL-секреты -> документ пропускается
     и попадает в отчёт, но не в новый индекс);
4. Пересоздаёт целевую коллекцию с явной метрикой ``cosine`` и
   пересчитывает эмбеддинги ТЕМ ЖЕ эмбеддером, который используется при
   запросах (никаких псевдовекторов).
5. Пишет JSON-отчёт миграции (принято/отклонено и причины).

Использование:

    python -m scripts.migrate_chroma --data-dir ./data/chroma_db --dry-run

Откат: восстановите каталог из ``--backup-dir``
(скрипт ничего не удаляет из исходного индекса).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("migrate_chroma")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Миграция ChromaDB под Zero-Trust схему")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--source-collection", default=None)
    parser.add_argument("--target-collection", default=None)
    parser.add_argument("--default-roles", nargs="+", default=["viewer"])
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("migration_report.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def backup_index(data_dir: Path, backup_dir: Path) -> Path:
    """Скопировать индекс целиком. Исходные данные не изменяются."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Каталог индекса не найден: {data_dir}")
    logger.info("Резервное копирование %s -> %s", data_dir, backup_dir)
    shutil.copytree(data_dir, backup_dir, dirs_exist_ok=True)
    return backup_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Настройки приложения дают канонические имена коллекций и пути.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import chromadb

    from secure_rag.config.settings import get_settings
    from secure_rag.security.acl import AccessControlList, Principal
    from secure_rag.security.redaction import process_text
    from secure_rag.storage.audit import AuditTrail
    from secure_rag.storage.vector_db import VectorDatabase

    settings = get_settings()
    data_dir = args.data_dir or settings.vector_db_path
    source_name = args.source_collection or settings.collection_name
    target_name = args.target_collection or f"{settings.collection_name}_v2"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.backup_dir or data_dir.with_name(f"{data_dir.name}.bak-{ts}")

    # 1. Бэкап до любых изменений.
    backup_index(data_dir, backup_dir)

    # 2. Открыть источник.
    source_client = chromadb.PersistentClient(path=str(data_dir))
    source = source_client.get_collection(source_name)
    total = source.count()
    logger.info("Исходная коллекция %r: %d записей", source_name, total)

    # 3. Пересборка через единый VectorDatabase-путь (ACL + redaction +
    #    SentenceTransformerEmbedder + метрика cosine).
    acl = AccessControlList(settings=settings)
    store = VectorDatabase(
        settings=settings.model_copy(update={"collection_name": target_name}),
        acl=acl,
    )
    audit = AuditTrail(settings=settings)

    report: Dict[str, Any] = {
        "started_at": ts,
        "source_collection": source_name,
        "target_collection": target_name,
        "backup_dir": str(backup_dir),
        "total": total,
        "migrated": 0,
        "skipped": 0,
        "skipped_details": [],
        "dry_run": args.dry_run,
    }

    principal = Principal(user_id="migration", roles=tuple(args.default_roles))
    batch_size = 64
    for offset in range(0, total, batch_size):
        batch = source.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
        chunks: List[Dict[str, Any]] = []
        for chunk_id, text, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            meta = dict(meta or {})
            if not meta.get("allowed_roles"):
                meta["allowed_roles"] = list(args.default_roles)

            cleaned, _, severity = process_text(text or "", mode="redact")
            if severity == "CRITICAL":
                report["skipped"] += 1
                report["skipped_details"].append(
                    {"chunk_id": chunk_id, "reason": "critical_secrets_detected"}
                )
                logger.warning("Пропуск %s: обнаружены CRITICAL-секреты", chunk_id)
                continue

            chunks.append(
                {
                    "chunk_id": str(chunk_id),
                    "doc_id": str(meta.get("doc_id", "unknown")),
                    "text": cleaned,
                    "metadata": meta,
                }
            )

        if args.dry_run:
            report["migrated"] += len(chunks)
            continue

        result = store.add_chunks(chunks, principal=principal, audit=audit)
        report["migrated"] += int(result.get("indexed", 0))
        logger.info(
            "Партия %d..%d: indexed=%s rejected=%s",
            offset, offset + batch_size, result.get("indexed"), result.get("rejected"),
        )

    report["finished_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Отчёт: %s (migrated=%d, skipped=%d, dry_run=%s)",
        args.report, report["migrated"], report["skipped"], args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
