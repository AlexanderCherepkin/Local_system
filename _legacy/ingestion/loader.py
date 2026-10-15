"""Модуль загрузки данных в векторную базу с Audit Trail."""

from typing import Dict, Any, List, Optional
from security.redaction import process_text

class AccessValidator:
    """Валидатор доступа и Zero-Trust детектор для документов."""
    
    def __init__(self):
        try:
            from security.acl import AccessControlList
            self.acl = AccessControlList()
        except ImportError:
            self.acl = None
        
    def validate_document_for_indexing(
        self, 
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Проверить документ перед добавлением в индекс через единый детектор.
        Возвращает результат с очищенным текстом (редакция), списком находок и флагом валидности.
        """
        result = {
            "doc_id": doc_id,
            "valid": True,
            "cleaned_text": text,
            "issues": [],
            "redactions_applied": []
        }
        
        # Защита от некорректного типа текста
        if not isinstance(text, str):
            text = str(text or "")
        
        # 1. Единый Zero-Trust детектор и редакция (режим 'redact')
        try:
            cleaned_text, redactions, max_severity = process_text(text, mode="redact")
            result["cleaned_text"] = cleaned_text
            result["redactions_applied"] = redactions
        except Exception as e:
            result["valid"] = False
            result["issues"].append({
                "type": "Redaction_Error",
                "message": f"Error during text redaction: {str(e)}",
                "severity": "CRITICAL",
                "action_required": True
            })
            return result
        
        # 2. Жёсткий Gate: при severity=CRITICAL документ не индексируется
        if max_severity == "CRITICAL":
            result["valid"] = False
            result["issues"].append({
                "type": "Critical_Security_Violation",
                "message": "Found critical secrets/tokens (Severity: CRITICAL). Document blocked.",
                "severity": "CRITICAL",
                "action_required": True
            })
            return result

        # Добавляем предупреждения для MEDIUM/LOW инцидентов, если они есть
        if redactions:
            result["issues"].append({
                "type": "Sensitive_Data_Redacted",
                "message": f"Applied {len(redactions)} redaction(s) during ingestion.",
                "severity": max_severity,
                "action_required": False
            })
        
        # 3. Безопасная проверка ACL метаданных (allowed_roles)
        has_valid_acl = True
        if self.acl and hasattr(self.acl, 'validate_chunk_metadata'):
            has_valid_acl = self.acl.validate_chunk_metadata(metadata)
        else:
            # Fallback проверка: убеждаемся, что allowed_roles присутствует в метаданных
            meta_dict = metadata if isinstance(metadata, dict) else getattr(metadata, "metadata", {})
            has_valid_acl = isinstance(meta_dict, dict) and "allowed_roles" in meta_dict and bool(meta_dict["allowed_roles"])

        if not has_valid_acl:
            result["valid"] = False
            result["issues"].append({
                "type": "Missing_Access_Control",
                "message": "Document metadata missing required 'allowed_roles' field",
                "severity": "CRITICAL"
            })
        
        return result