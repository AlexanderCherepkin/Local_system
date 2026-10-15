"""ACL контроль доступа по ролям."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class AccessDeniedError(Exception):
    """Ошибка отказа в доступе при нарушении ACL."""
    pass


class Principal:
    """Представление пользователя с ролями."""
    
    def __init__(self, username: str, roles: Optional[List[str]] = None):
        self.username = username
        self.roles = set(roles) if roles else set()
    
    @classmethod
    def from_dict(cls, data: dict) -> "Principal":
        return cls(username=data.get("username", ""), 
                   roles=data.get("roles", []))


class ACLManager:
    """Менеджер ACL конфигурации."""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "acl_storage.json"
        self.role_permissions: dict[str, Set[str]] = {}  # role → allowed_doc_ids
        self.user_roles: dict[str, List[str]] = {}      # username → roles
    
    def register_document(self, doc_id: str, allowed_roles: list[str], persist: bool = True):
        """Динамически зарегистрировать или обновить документ в ACL."""
        if isinstance(allowed_roles, str):
            allowed_roles = [allowed_roles]
        
        # Обновляем в памяти
        self.role_permissions[doc_id] = set(allowed_roles)
        logger.info(f"Документ {doc_id} зарегистрирован с ролями: {allowed_roles}")

        if persist:
            try:
                data = {}
                # Сохраняем текущие пользователи
                data["users"] = self.user_roles
                # Сохраняем документы (преобразуем множества в списки)
                data["documents"] = {
                    doc_id: {"allowed_roles": list(roles)} 
                    for doc_id, roles in self.role_permissions.items()
                }
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(f"ACL конфигурация сохранена в {self.config_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения ACL: {e}")
    
    def get_user_principal(self, username: str) -> Optional[Principal]:
        """Получить Principal для пользователя."""
        roles = self.user_roles.get(username, [])
        if not roles:
            # Гость без ролей
            return Principal(username=username, roles=["guest"])
        return Principal(username=username, roles=roles)


# Глобальный экземпляр
_acl_manager: Optional[ACLManager] = None

def get_acl_manager(config_path: str = None) -> ACLManager:
    """Получить или создать глобальный ACL менеджер."""
    global _acl_manager
    if _acl_manager is None:
        _acl_manager = ACLManager(config_path or "security/acl_storage.json")
        _acl_manager.load_config()
    return _acl_manager


def require_principal(username: str, required_roles: Optional[List[str]] = None) -> Principal:
    """Проверить наличие необходимых ролей.
    
    Args:
        username: Имя пользователя
        required_roles: Список требуемых ролей (если пустой — любой доступ разрешён)
        
    Returns:
        Principal объекта если доступ разрешён
        
    Raises:
        AccessDeniedError: Если пользователь не имеет нужных ролей
    """
    acl = get_acl_manager()
    principal = acl.get_user_principal(username)
    
    if not required_roles or not principal.roles:
        # Любой доступ без требований или у гостя
        return principal
    
    # Проверка через пересечение множеств (хотя бы одна роль должна совпасть)
    if not (set(required_roles) & principal.roles):
        raise AccessDeniedError(
            f"Отказ в доступе для {username}: пользователь не имеет ни одной из требуемых ролей {required_roles}"
        )
    
    return principal
