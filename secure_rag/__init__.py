"""Secure RAG — air-gapped Enterprise RAG с Zero-Trust ACL и аудитом.

Единственный канонический пакет приложения. Все entry points
(``secure_rag.cli``, ``secure_rag.api``, ``pyproject.toml``, Dockerfile)
ссылаются ТОЛЬКО на этот пакет.
"""

from .config.settings import Settings, get_settings

__version__ = "0.1.0"

__all__ = ["__version__", "Settings", "get_settings"]
