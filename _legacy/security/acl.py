"""Контроль доступа (ACL) для Enterprise RAG-системы."""

"""Контроль доступа (ACL) для Enterprise RAG-системы (Production Ready)."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from config.settings import Settings

logger = logging.getLogger("security.acl")


class ACLValidationError(Exception):
    """Специальное исключение для ошибок валидации метаданных и структуры ACL."""
    pass


class AccessControlList:
    """
    Класс контроля доступа (Access Control List).
    
    Реализует строгую Zero-Trust модель доступа:
    - Deny-by-default (пустой или отсутствующий allowed_roles блокирует доступ всем)
    - Транзитивная иерархия ролей с защитой от бесконечных циклов
    - Раздельные кэши (пользователи, разрешения ролей, метаданные документов)
    - Устранение N+1 IO за счет предварительной загрузки метаданных
    - Аудит отказов в доступе (кто, к чему, когда и почему)
    - Валидация существования ролей и соответствия департаментов
    """
    
    def __init__(self):
        self.settings = Settings.get_settings()
        self.security_config = self.settings.SECURITY_CONFIG
        
        # Получаем иерархию ролей из конфига или задаем безопасный дефолт
        self.role_hierarchy = getattr(self.security_config, "ROLE_HIERARCHY", {
            "admin": ["hr_manager", "hr_analyst", "hr_viewer"],
            "hr_manager": ["hr_analyst", "hr_viewer"],
            "hr_analyst": ["hr_viewer"],
            "hr_viewer": []
        })
        
        # Раздельные кэши для избежания путаницы смыслов
        self._user_roles_cache: Dict[str, Set[str]] = {}
        self._role_perms_cache: Dict[str, Set[str]] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        # Загрузка конфигураций при инициализации
        self._load_role_permissions()
        self._load_acl_storage()
        self._load_metadata_cache()

    def _load_role_permissions(self) -> None:
        """Загрузить разрешения ролей из хранилища."""
        try:
            storage_file = Path(__file__).parent / "acl_storage.json"
            if storage_file.exists():
                with open(storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for role, permissions in data.get("roles", {}).items():
                        self._role_perms_cache[role] = set(permissions)
                logger.info("Role permissions loaded successfully.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load role permissions: {e}")

    def _load_acl_storage(self) -> None:
        """Загрузить маппинг «пользователь → роли» из внешнего хранилища."""
        try:
            storage_file = Path(__file__).parent / "acl_storage.json"
            if storage_file.exists():
                with open(storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, roles in data.get("users", {}).items():
                        self._user_roles_cache[user_id] = set(roles)
                logger.info("ACL user storage loaded successfully.")
            else:
                logger.warning(f"ACL storage file not found at {storage_file}. Zero-Trust strict mode active.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load acl_storage.json users: {e}")

    def _load_metadata_cache(self) -> None:
        """Предзагрузка кэша метаданных документов для предотвращения N+1 IO."""
        try:
            metadata_store = Path(__file__).parent / "acl_metadata_cache.json"
            if metadata_store.exists():
                with open(metadata_store, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc in data.get("documents", []):
                        doc_id = doc.get("doc_id")
                        if doc_id:
                            self._metadata_cache[doc_id] = doc.get("metadata", {})
                logger.info("ACL metadata cache loaded successfully.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load metadata cache: {e}")

    def get_effective_roles(self, user_id: str) -> Set[str]:
        """
        Публичный API: получить эффективные роли пользователя с учетом 
        транзитивной иерархии и защитой от циклических зависимостей.
        """
        base_roles = self._user_roles_cache.get(user_id, set())
        effective_roles = set(base_roles)
        
        visited: Set[str] = set()
        queue = list(base_roles)
        
        while queue:
            current_role = queue.pop(0)
            if current_role in visited:
                continue
            visited.add(current_role)
            
            sub_roles = self.role_hierarchy.get(current_role, [])
            for sub_role in sub_roles:
                if sub_role not in visited:
                    effective_roles.add(sub_role)
                    queue.append(sub_role)
                    
        return effective_roles

    def _log_access_denied(self, user_id: str, doc_id: str, reason: str) -> None:
        """Аудит отказов в доступе (кто, к какому документу, когда, по какой причине)."""
        logger.warning(
            f"AUDIT_ACCESS_DENIED | User: '{user_id}' | Target Doc: '{doc_id}' | Reason: '{reason}'"
        )

    def validate_user_access(self, user_id: str, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Проверить доступ пользователя к списку документов по модели Zero-Trust.
        """
        if not getattr(self.security_config, "ACL_ENABLED", True):
            return {
                "allowed": True,
                "accessible_doc_ids": doc_ids,
                "user_roles": list(self.get_effective_roles(user_id)),
                "reason": "ACL disabled"
            }
        
        user_roles = self.get_effective_roles(user_id)
        if not user_roles:
            for doc_id in doc_ids:
                self._log_access_denied(user_id, doc_id, "User has no roles assigned")
            return {
                "allowed": False,
                "accessible_doc_ids": [],
                "denied_doc_ids": doc_ids,
                "user_roles": [],
                "reason": "User has no roles assigned"
            }
        
        accessible_docs = []
        denied_docs = []
        
        for doc_id in doc_ids:
            metadata = self._metadata_cache.get(doc_id)
            if not metadata:
                self._log_access_denied(user_id, doc_id, "Document metadata not found or restricted")
                denied_docs.append(doc_id)
                continue
            
            # Zero-Trust Deny-by-Default: пустой или отсутствующий allowed_roles запрещает доступ всем
            allowed_roles = metadata.get("allowed_roles", [])
            if not allowed_roles or not isinstance(allowed_roles, list):
                self._log_access_denied(user_id, doc_id, "Document has empty allowed_roles (Zero-Trust deny)")
                denied_docs.append(doc_id)
                continue
            
            # Валидация существования ролей в иерархии
            for role in allowed_roles:
                if role not in self.role_hierarchy and role not in self._role_perms_cache:
                    logger.error(f"Configuration error: Role '{role}' in doc '{doc_id}' metadata does not exist.")
            
            # Проверка department (если задан и для документа, и для пользователя)
            doc_department = metadata.get("department")
            user_department = metadata.get("user_department")
            if doc_department and user_department and doc_department != user_department:
                self._log_access_denied(user_id, doc_id, f"Department mismatch: user in '{user_department}', doc in '{doc_department}'")
                denied_docs.append(doc_id)
                continue
            
            # Проверка пересечения эффективных ролей пользователя и разрешенных ролей документа
            if user_roles & set(allowed_roles):
                accessible_docs.append({
                    "doc_id": doc_id,
                    "metadata": metadata
                })
            else:
                self._log_access_denied(user_id, doc_id, "User lacks required roles")
                denied_docs.append(doc_id)
        
        return {
            "allowed": len(accessible_docs) > 0,
            "accessible_doc_ids": [d["doc_id"] for d in accessible_docs],
            "denied_doc_ids": denied_docs,
            "user_roles": list(user_roles),
            "accessible_documents_count": len(accessible_docs),
            "denied_documents_count": len(denied_docs)
        }
    
    def validate_chunk_metadata(self, chunk: Dict[str, Any]) -> bool:
        """
        Валидация метаданных чанка. Выбрасывает ACLValidationError при нарушении контракта.
        Возвращает True в случае успешной проверки.
        """
        if not isinstance(chunk, dict):
            return False
        
        if "allowed_roles" not in chunk:
            raise ACLValidationError(
                "Chunk missing required field 'allowed_roles'. "
                "Zero-Trust requirement: all chunks must specify access permissions."
            )
        
        allowed_roles = chunk["allowed_roles"]
        if not isinstance(allowed_roles, list) or len(allowed_roles) == 0:
            raise ACLValidationError(
                "Chunk allowed_roles cannot be empty. Zero-Trust requirement: "
                "must specify at least one role with access permission."
            )
        
        for role in allowed_roles:
            if not isinstance(role, str) or not role.strip():
                raise ACLValidationError(
                    "Invalid allowed_roles format. Each role must be a non-empty string."
                )
        
        return True
    
    def add_role_to_user(self, user_id: str, role: str) -> bool:
        """Добавить роль пользователю в кэш."""
        if user_id not in self._user_roles_cache:
            self._user_roles_cache[user_id] = set()
        self._user_roles_cache[user_id].add(role)
        return True
    
    def remove_role_from_user(self, user_id: str, role: str) -> bool:
        """Удалить роль у пользователя из кэша."""
        if user_id in self._user_roles_cache and role in self._user_roles_cache[user_id]:
            self._user_roles_cache[user_id].discard(role)
            return True
        return False
    
    def clear_role_cache(self):
        """Очистить все кэши."""
        self._user_roles_cache.clear()
        self._role_perms_cache.clear()
        self._metadata_cache.clear()

