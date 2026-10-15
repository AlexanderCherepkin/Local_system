"""Модуль утилит для нарезки документов.

Централизирует доступ к функциям нарезки текста из pipeline.chunker,
чтобы обеспечить единый интерфейс и упростить переиспользование.

F-04: Заменён отсутствовавший ingest-путь (ingestion/chunker.py -> utils/chunking).
"""

from typing import Any, Dict, List, Optional

# Импорт основной функции нарезки из pipeline.chunker
# Это обеспечивает единую точку доступа и упрощает тестирование
from pipeline.chunker import create_chunks as _create_chunks


def create_chunks(
    text: str,
    doc_id: str = "doc_default",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    method: str = "sliding",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Нарезать текст на чанки и вернуть канонические dict'ы.

    Эта функция является обёрткой над pipeline.chunker.create_chunks()
    для удобства использования из других модулей (ingestion, tests и т.д.).

    Аргументы:
        text: Текст для нарезки на чанки.
        doc_id: Уникальный идентификатор документа по умолчанию "doc_default".
        chunk_size: Максимальная длина чанка в символах (по умолчанию 500).
        chunk_overlap: Пересечение между соседними чанками (по умолчанию 50).
        method: Метод нарезки - "sliding", "sentence" или "headers".
        metadata: Дополнительные метаданные, которые будут включены в каждый чанк.

    Возвращает:
        Список словарей с полями chunk_id, text, doc_id, start_pos, end_pos,
        index, strategy и (опционально) metadata.

    Raises:
        ValueError: Если указан неизвестный метод нарезки.
    """
    return _create_chunks(
        text=text,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method,
        metadata=metadata,
    )


__all__ = ["create_chunks"]
