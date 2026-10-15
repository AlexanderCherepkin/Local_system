"""Аудит (facade).

``main.py`` исторически импортировал ``generate_audit_report`` из
``security.audit``, но такого модуля не было (F-02). Здесь — тонкая фасад над
единственной реализацией в ``secure_rag.storage.audit``.
"""

from storage.audit import (
    AuditConfigurationError,
    AuditRecord,
    AuditTrail,
    generate_audit_report,
)

__all__ = [
    "AuditTrail",
    "AuditRecord",
    "AuditConfigurationError",
    "generate_audit_report",
]
