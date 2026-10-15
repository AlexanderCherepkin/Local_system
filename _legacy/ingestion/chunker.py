"""Модуль разбиения текста на сегменты (chunking)."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import Settings
from pipeline.chunker import create_chunks


class TextChunker:
    """Класс-обертка для разбивки текста, использующий канонический модуль pipeline/chunker.py."""
    
    def __init__(self):
        self.settings = Settings.get_settings()
        self.chunk_size = self.settings.CHUNK_SIZE
        self.chunk_overlap = self.settings.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Разбить текст на сегменты с помощью канонического чанкера.
        
        Args:
            text: Текст для разбиения
            metadata: Метаданные документа (должны содержать doc_id, если требуется)
            
        Returns:
            Список словарей с сегментами и их метаданными
        """
        if not text or not isinstance(text, str):
            return []
            
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "default_doc")
        
        # Вызов канонического алгоритма из pipeline/chunker.py
        raw_chunks = create_chunks(
            text=text,
            doc_id=doc_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            method="sliding"  # Можно изменить на "sentence" при необходимости
        )
        
        # Преобразование в формат, ожидаемый текущим пайплайном ингеста
        chunks = []
        for rc in raw_chunks:
            chunk_dict = {
                "id": rc["chunk_id"],
                "text": rc["text"],
                "metadata": {
                    **metadata,
                    "start_pos": rc["start_pos"],
                    "end_pos": rc["end_pos"],
                    "index": rc["index"]
                },
                "word_count": len(rc["text"].split()),
                "char_count": len(rc["text"])
            }
            chunks.append(chunk_dict)
            
        return chunks
    
    def load_chunks_from_file(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """Загрузить сегменты из JSON-файла."""
        if not filepath:
            filepath = self.settings.CHUNKS_FILE
        
        path = Path(filepath)
        if not path.exists():
            return []
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data.get("chunks", [])
    
    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], filepath: Optional[str] = None):
        """Сохранить сегменты в JSON-файл."""
        if not filepath:
            filepath = self.settings.CHUNKS_FILE
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)


# Функция для быстрого использования (обратная совместимость)
def chunk_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Функция-обертка для разбивки текста."""
    chunker = TextChunker()
    return chunker.chunk_text(text, metadata)