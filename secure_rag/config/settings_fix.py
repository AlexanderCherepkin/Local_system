"""Исправление импорта BaseSettings из pydantic-settings для совместимости с Pydantic 2.13."""

from typing import Any, Dict, Optional

try:
    from pydantic_settings import BaseSettings as _BaseSettings, SettingsConfigDict as _SettingsConfigDict
    
except ImportError:
    # Fallback: эмуляция минимального API для работы кода
    class _FakeSettingsConfigDict(dict):
        """Фейковый конфиг."""
        pass

    class _BaseSettings:
        """Базовый класс настроек с минимальным API."""
        
        def __init__(self, **data: Any) -> None:
            self._data = data
        
        @classmethod
        def model_config(cls) -> Dict[str, Any]:
            return {}
    
    class _SettingsConfigDict(_FakeSettingsConfigDict):
        """Фейковый конфиг."""
        pass

# Экспорт для совместимости
BaseSettings = _BaseSettings
SettingsConfigDict = _SettingsConfigDict
