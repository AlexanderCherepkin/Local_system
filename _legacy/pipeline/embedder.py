"""Модуль создания эмбеддингов (Air-Gapped)."""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from config.settings import Settings, EmbeddingConfig


class BaseEmbedder:
    """Интерфейс для всех эмбеддеров."""
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Продакшн-эмбеддер на базе SentenceTransformers.
    Никаких моков и silent-fallback'ов.
    """
    def __init__(self, model_name: Optional[str] = None):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "Библиотека 'sentence-transformers' не установлена. "
                "Установите её или используйте FakeEmbedder для тестов."
            )
        
        settings = Settings.get_settings()
        self.embedding_config = settings.EMBEDDING_CONFIG
        self.model_name = model_name or getattr(EmbeddingConfig, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
        
        cache_dir = Path(settings.MODEL_CACHE_DIR)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.model = SentenceTransformer(self.model_name)
        self._cache: Dict[str, List[float]] = {}

    def _get_cache_key(self, text: str) -> str:
        """Детерминированный ключ кэша на базе sha256 (не зависит от PYTHONHASHSEED)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст для эмбеддинга должен быть непустой строкой.")

        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        vector = self.model.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS
        )
        
        # Нормализация через явное приведение к np.asarray
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0 and not self.embedding_config.NORMALIZE_EMBEDDINGS:
            vector = vector / norm

        result = vector.tolist()
        self._cache[cache_key] = result
        return result

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Пакетная оптимизация через SentenceTransformers, если передано много текстов
        vectors = self.model.encode(
            texts, 
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS,
            batch_size=32
        )
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def dimension(self) -> int:
        dummy = self.model.encode("test", convert_to_numpy=True)
        return int(np.asarray(dummy).shape[-1])


class FakeEmbedder(BaseEmbedder):
    """
    Детерминированный эмбеддер для тестов и CI/CD.
    Проецирует любой текст в вектор размерности 384 с помощью sha256.
    """
    def __init__(self, dim: int = 384):
        self._dim = dim
        self._cache: Dict[str, List[float]] = {}

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            text = str(text)
            
        cache_key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Генерируем псевдослучайный, но строго детерминированный вектор на основе sha256 текста
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        vector = np.random.randn(self._dim).astype(np.float32)
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        result = vector.tolist()
        self._cache[cache_key] = result
        return result

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def dimension(self) -> int:
        return self._dim


class MultiLanguageEmbedder(SentenceTransformerEmbedder):
    """Мультиязычный эмбеддер с корректным выбором модели по языку."""
    
    def __init__(self, language: str = "multilingual"):
        supported_languages = {
            "english": "all-MiniLM-L6-v2",
            "russian": "cointegrated/rubert-t5",
            "multilingual": "BAAI/bge-m3",
            "chinese": "BAAI/bge-large-zh-v1.5"
        }
        model_name = supported_languages.get(language, "BAAI/bge-m3")
        logger.info(f"Using {model_name} for language: {language}")
        super().__init__(model_name=model_name)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Вычисление косинусного сходства с явным использованием np.asarray."""
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    
    if v1.shape != v2.shape:
        raise ValueError("Embeddings must have the same dimension")
        
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))


# --- Вспомогательные классы и функции для обратной совместимости ---

class Embedder(SentenceTransformerEmbedder):
    """Алиас для обратной совместимости со старым кодом приложения."""
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name=model_name)

    def embed_chunk(self, chunk: Dict[str, Any]) -> List[float]:
        """Создать эмбеддинг для чанка с учётом метаданных."""
        text = chunk.get("text", "")
        if not text:
            raise ValueError("Чанк не содержит текста для эмбеддинга.")
            
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("doc_id") or "unknown"
        context = f"[doc:{doc_id}] " if doc_id else "[chunk] "
        
        return self.embed_text(context + text)

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return cosine_similarity(embedding1, embedding2)

    def find_nearest(
        self, 
        query_embedding: List[float], 
        document_embeddings: Dict[str, Tuple[List[float], str]],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Найти наиболее похожие документы."""
        results = []
        
        for doc_id, (embedding, text) in document_embeddings.items():
            sim = self.similarity(query_embedding, embedding)
            
            if threshold is None or sim >= threshold:
                results.append({
                    "doc_id": doc_id,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "similarity": sim,
                    "embedding_dim": len(embedding)
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


def embed_text(text: str) -> List[float]:
    """Быстрое создание эмбеддинга."""
    embedder = Embedder()
    return embedder.embed_text(text)


def find_similar(
    query: str, 
    documents: Dict[str, Tuple[List[float], str]], 
    top_k: int = 5,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """Найти похожие документы по текстовому запросу без silent-fallback."""
    embedder = Embedder()
    query_embedding = embedder.embed_text(query)
    
    return embedder.find_nearest(
        query_embedding, 
        documents, 
        top_k=top_k,
        threshold=threshold
    )

from config import settings

# Использование параметров из конфига
host = settings.ollama_host