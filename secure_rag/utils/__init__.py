"""Пакет утилит: обработка текста и вспомогательные инструменты."""

from .text_utils import TextUtils
from .chunking import create_chunks

__all__ = ["TextUtils", "create_chunks"]
