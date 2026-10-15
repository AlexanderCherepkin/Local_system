"""CLI Secure RAG (``secure-rag``).

Исторически ``pyproject.toml`` объявлял console script ``secure_rag.cli:main``,
но модуля не существовало (F-02). Здесь — единственная точка входа с командами
``ingest``, ``query``, ``audit-report`` и ``selfcheck``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from config.settings import Settings, get_settings, initialize_directories
from security.acl import AccessControlList, Principal
from storage.audit import AuditConfigurationError, AuditTrail, generate_audit_report


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _roles_for_user(user: str, roles: Optional[List[str]]) -> List[str]:
    """Роли берутся из ACL, если не заданы иным доверенным способом."""
    if roles:
        return list(roles)
    return AccessControlList().roles_of(user)


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from .ingestion.loader import ingest_documents

    roles = args.roles.split(",") if args.roles else None
    result = ingest_documents(path=args.path, user_id=args.user, roles=roles, settings=settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors") else 0


def cmd_query(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline.generator import RAGPipeline

    roles = _roles_for_user(args.user, args.roles.split(",") if args.roles else None)
    if not roles:
        print("Отказано: пользователю не назначены роли (Zero-Trust).", file=sys.stderr)
        return 2

    principal = Principal(user_id=args.user, roles=tuple(roles))
    pipeline = RAGPipeline(settings=settings)
    result = pipeline.process_query(prompt=args.prompt, user=principal)

    if result.get("error") == "access_denied":
        print("Отказано в доступе.", file=sys.stderr)
        return 2
    if result.get("error") == "prompt_injection_detected":
        print("Запрос отклонён: похоже на промпт-инъекцию.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("\nОтвет:\n" + str(result.get("answer", "")))
    sources = result.get("sources") or []
    if sources:
        print("\nИсточники:")
        for item in sources:
            print(f"  [{item['fragment']}] {item['doc_id']} — {item['source_file']} (score={item['score']})")
    if result.get("abstained"):
        print("\n(В доступных документах ответ не найден.)")
    return 0


def cmd_audit_report(args: argparse.Namespace, settings: Settings) -> int:
    print(json.dumps(generate_audit_report(settings=settings), ensure_ascii=False, indent=2))
    return 0

def cmd_selfcheck(args: argparse.Namespace, settings: Settings) -> int:
    """Проверка целостности: ACL, аудит, индекс, режим air-gap."""
    checks: dict = {}

    users = AccessControlList(settings=settings).known_users()
    checks["acl_users_loaded"] = len(users)

    try:
        AuditTrail(settings=settings)
        checks["audit_configured"] = True
    except AuditConfigurationError as exc:
        checks["audit_configured"] = False
        checks["audit_error"] = str(exc)

    try:
        from .storage.vector_db import VectorDatabase

        checks["vector_store"] = VectorDatabase(settings=settings).healthcheck()
    except Exception as exc:
        checks["vector_store"] = {"ready": False, "reason": f"{type(exc).__name__}: {exc}"}

    ok = bool(users) and checks.get("audit_configured") and bool(checks["vector_store"].get("ready"))
    print(
        json.dumps(
            {"selfcheck": "passed" if ok else "failed", "checks": checks},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secure-rag", description="Secure RAG CLI (air-gapped)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Индексировать документы")
    ingest.add_argument("--path", required=True, help="Файл или каталог с документами")
    ingest.add_argument("--user", default="admin", help="Кто выполняет индексацию")
    ingest.add_argument("--roles", default=None, help="Роли доступа к документам (через запятую)")

    query = subparsers.add_parser("query", help="Запросить RAG")
    query.add_argument("--prompt", required=True, help="Вопрос пользователя")
    query.add_argument("--user", required=True, help="Идентификатор пользователя")
    query.add_argument("--roles", default=None, help="Явно заданные роли (иначе из ACL)")
    query.add_argument("--json", action="store_true", help="Вывести полный результат в JSON")

    subparsers.add_parser("audit-report", help="Отчёт по аудит-журналу")
    subparsers.add_parser("selfcheck", help="Проверка целостности и режима air-gap")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    _configure_logging(settings)
    initialize_directories()

    handlers = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "audit-report": cmd_audit_report,
        "selfcheck": cmd_selfcheck,
    }
    try:
        return handlers[args.command](args, settings)
    except AuditConfigurationError as exc:
        print(f"Ошибка конфигурации аудита: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        logging.getLogger("secure_rag.cli").exception("Ошибка выполнения команды")
        print(f"Ошибка: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

