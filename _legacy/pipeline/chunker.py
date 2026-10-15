"""Модуль разбиения документов на чанки для RAG."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import re


class DocumentChunker:
    """
    Класс для разбиения документов на семантические чанки.
    
    Поддерживает:
    - Скользящее окно с корректным overlap и проверкой лимитов
    - Разбиение по предложениям с сохранением реальных позиций start_pos/end_pos
    - Разбиение по заголовкам
    - Детерминированные уникальные ID (doc_id::index)
    """
    
    def __init__(
        self, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        language: str = "russian"
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) должен быть строго меньше chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        
        # Регулярное выражение для разделения предложений с сохранением границ
        self.sentence_pattern = re.compile(r'(?<=[.!?…])\s+')
    
    def chunk_by_sentences(
        self, 
        text: str,
        doc_id: str = "doc_default",
        max_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Разбить текст на чанки по предложениям с жёстким контролем лимита и реальными позициями.
        """
        max_size = max_chunk_size or self.chunk_size
        if not text:
            return []

        chunks = []
        # Находим предложения и их позиции в исходном тексте
        # Используем finditer по предложениям для точного отслеживания start_pos и end_pos
        sentence_regex = re.compile(r'[^.!?…]+[.!?…]+(?:\s+|$)|[^.!?…]+$')
        matches = list(sentence_regex.finditer(text))
        
        if not matches:
            return [{
                "chunk_id": f"{doc_id}::00000",
                "text": text,
                "sentence_count": 1,
                "start_pos": 0,
                "end_pos": len(text),
                "strategy": "sentences"
            }]

        current_sentences: List[str] = []
        current_len = 0
        start_pos = -1
        index = 0

        for match in matches:
            sent_text = match.group()
            sent_start = match.start()
            sent_end = match.end()

            if start_pos == -1:
                start_pos = sent_start

            # Если добавление предложения превышает лимит и уже есть накопленные предложения
            if current_len + len(sent_text) > max_size and current_sentences:
                chunk_text = "".join(current_sentences)
                end_pos = start_pos + len(chunk_text)
                
                chunk_id = f"{doc_id}::{index:05d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "sentence_count": len(current_sentences),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "strategy": "sentences"
                })
                index += 1

                # Начинаем новый чанк (с учётом возможного overlap на уровне предложений можно настроить отдельно, 
                # но для базового по предложениям держим чистое заполнение до лимита)
                current_sentences = [sent_text]
                current_len = len(sent_text)
                start_pos = sent_start
            else:
                current_sentences.append(sent_text)
                current_len += len(sent_text)

        # Сохраняем последний накопленный чанк
        if current_sentences:
            chunk_text = "".join(current_sentences)
            end_pos = start_pos + len(chunk_text)
            chunk_id = f"{doc_id}::{index:05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "sentence_count": len(current_sentences),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "strategy": "sentences"
            })

        return chunks
    
    def chunk_by_size(
        self, 
        text: str,
        doc_id: str = "doc_default"
    ) -> List[Dict[str, Any]]:
        """
        Разбить текст на фиксированные чанки по размеру с правильным скользящим окном.
        Формула: start += chunk_size - chunk_overlap
        """
        chunks = []
        if not text:
            return chunks

        start = 0
        index = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            chunk_id = f"{doc_id}::{index:05d}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "is_last_chunk": end == text_len,
                "strategy": "size"
            })
            
            if end == text_len:
                break
                
            # Корректный шаг скользящего окна
            start += self.chunk_size - self.chunk_overlap
            index += 1
            
        return chunks
    
    def chunk_by_headers(
        self, 
        text: str,
        doc_id: str = "doc_default"
    ) -> List[Dict[str, Any]]:
        """Разбить текст по заголовкам с вычислением позиций."""
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        chunks = []
        
        matches = list(header_pattern.finditer(text))
        if not matches:
            return [{
                "chunk_id": f"{doc_id}::00000",
                "text": text.strip(),
                "header_level": 0,
                "header_text": None,
                "start_pos": 0,
                "end_pos": len(text),
                "strategy": "headers"
            }]

        # Разобьем текст по позициям нахождени заголовков
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start_pos:end_pos].strip()
            
            if not chunk_text:
                continue

            chunks.append({
                "chunk_id": f"{doc_id}::{i:05d}",
                "text": chunk_text,
                "header_level": len(match.group(1)),
                "header_text": match.group(2),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "strategy": "headers"
            })
            
        return chunks
    
    def chunk_document(
        self, 
        document: Dict[str, Any],
        strategy: str = "sentences",
        max_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Разбить документ на чанки по заданной стратегии с полным сохранением метаданных."""
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        doc_id = metadata.get("doc_id", "doc_default")
        
        raw_chunks = []
        if strategy == "sentences":
            raw_chunks = self.chunk_by_sentences(text, doc_id=doc_id, max_chunk_size=max_chunk_size)
        elif strategy == "size":
            raw_chunks = self.chunk_by_size(text, doc_id=doc_id)
        elif strategy == "headers":
            raw_chunks = self.chunk_by_headers(text, doc_id=doc_id)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_with_meta = {
                **chunk,
                "metadata": metadata.copy(),
                "doc_id": doc_id,
                "chunk_index": i,
                "is_first_chunk": i == 0,
                "is_last_chunk": i == len(raw_chunks) - 1,
            }
            chunks.append(chunk_with_meta)
            
        return chunks

    def process_documents(
        self, 
        documents: List[Dict[str, Any]],
        strategy: str = "sentences"
    ) -> List[Dict[str, Any]]:
        """Обработать список документов и вернуть чанки."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, strategy=strategy)
            for chunk in chunks:
                chunk["parent_doc_id"] = doc.get("metadata", {}).get("doc_id")
                chunk["parent_doc_title"] = doc.get("title", "Untitled")
                all_chunks.append(chunk)
        return all_chunks


# Быстрая функция-обёртка для совместимости
def chunk_text(text: str, strategy: str = "sentences", size: int = 500, overlap: int = 50, doc_id: str = "doc_default") -> List[Dict[str, Any]]:
    """Быстрое разбиение текста на чанки."""
    chunker = DocumentChunker(chunk_size=size, chunk_overlap=overlap)
    doc = {"text": text, "metadata": {"doc_id": doc_id}}
    return chunker.chunk_document(doc, strategy=strategy)