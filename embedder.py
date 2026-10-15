"""
Thread-safe SentenceTransformer-based text embedding service with caching.
Air-gapped mode: No network calls after initial model loading.
Batch vectorization for large datasets supported.
"""

import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbedderSettings:
    """Configuration for embedder."""
    
    def __init__(self):
        self.embedding_model = "BAAI/bge-m3"
        self.embedding_dimension = 1024
        self.model_cache_dir = Path.home() / ".cache" / "sentence-transformers"
        self.normalize_embeddings = False
        self.batch_size = 64
        self.max_memory_usage_mb = 512
    
    @classmethod
    def get_settings(cls) -> 'EmbedderSettings':
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance


class BaseEmbedder:
    """Abstract base class for text embedders."""
    
    def __init__(self):
        self.settings = EmbedderSettings.get_settings()
        self.embedding_config = self.settings
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Thread-safe embedding service with batch vectorization."""
    
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Batch processing configuration
        self.batch_size: int = 64
        self.max_memory_mb: int = 512
        
        self.model: Optional[SentenceTransformer] = None
        self.expected_dimension: int = 0
    
    def _load_model(self) -> SentenceTransformer:
        if self.model is not None:
            return self.model
        
        cache_dir = Path(self.settings.model_cache_dir)
        
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(cache_dir, 0o700)
        
        self.model = SentenceTransformer(
            self.settings.embedding_model, 
            cache_folder=str(cache_dir),
            trust_remote_code=self._is_trusted_model()
        )
        
        return self.model
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        
        stripped = text.strip()
        if not stripped:
            raise ValueError("Текст для эмбеддинга должен быть непустой строкой.")
        
        with self._cache_lock:
            cache_key = self._get_cache_key(text)
            
            if cache_key in self._cache:
                return self._cache[cache_key].copy()
        
        self._load_model()
        
        vector = self.model.encode(
            [stripped], 
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS,
            show_progress_bar=False
        )[0]
        
        with self._cache_lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = vector.tolist()
        
        return vector.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        cache_keys = [self._get_cache_key(text) for text in texts]
        uncached_indices = [i for i, key in enumerate(cache_keys) 
                           if key not in self._cache]
        uncached_texts = [texts[i] for i in uncached_indices]
        
        if uncached_texts:
            self._load_model()
            
            vectors = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS,
                show_progress_bar=False
            )
            
            for i, (idx, vector) in enumerate(zip(uncached_indices, vectors)):
                cache_key = cache_keys[idx]
                with self._cache_lock:
                    if cache_key not in self._cache:
                        self._cache[cache_key] = vector.tolist()
        
        return [self._cache.get(key, []) for key in cache_keys]
    
    def batch_vectorize(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        verbose: bool = False
    ) -> Generator[List[float], None, None]:
        """
        Stream embeddings one at a time from batch processing.
        
        Args:
            texts: Large list of text documents
            batch_size: Batch size for encoding (default from settings)
            verbose: Show progress
            
        Yields:
            One embedding vector per text document
        """
        if not texts:
            return
        
        effective_batch_size = batch_size or self.settings.batch_size
        
        # Pre-compute cache keys and identify uncached items
        cache_keys = [self._get_cache_key(text) for text in texts]
        
        total_texts = len(texts)
        processed_count = 0
        
        for start_idx in range(0, total_texts, effective_batch_size):
            end_idx = min(start_idx + effective_batch_size, total_texts)
            batch_texts = [texts[i] for i in range(start_idx, end_idx)]
            
            # Filter to uncached items only
            batch_cache_keys = cache_keys[start_idx:end_idx]
            uncached_mask = [key not in self._cache for key in batch_cache_keys]
            if not any(uncached_mask):
                continue
            
            uncached_texts = [batch_texts[i] for i, m in enumerate(uncached_mask) if m]
            
            # Load model and encode batch
            self._load_model()
            
            vectors = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS,
                show_progress_bar=False
            )
            
            # Cache results
            for i, vector in enumerate(vectors):
                original_idx = start_idx + sum(uncached_mask[:i])
                cache_key = cache_keys[original_idx]
                with self._cache_lock:
                    if cache_key not in self._cache:
                        self._cache[cache_key] = vector.tolist()
            
            processed_count += len(batch_texts)
            
            if verbose:
                print(f"Progress: {processed_count}/{total_texts} ({100*processed_count/total_texts:.1f}%)")
        
        # Yield cached items first (in order), then uncached ones
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key not in self._cache:
                vector = self.model.encode(
                    [text.strip()], 
                    convert_to_numpy=True,
                    normalize_embeddings=self.embedding_config.NORMALIZE_EMBEDDINGS,
                    show_progress_bar=False
                )[0]
                with self._cache_lock:
                    if cache_key not in self._cache:
                        self._cache[cache_key] = vector.tolist()
            
            yield self._cache[cache_key].tolist()
    
    def batch_vectorize_to_array(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        verbose: bool = False
    ) -> np.ndarray:
        """Vectorize entire dataset to numpy array.
        
        Args:
            texts: Large list of text documents
            batch_size: Batch size for encoding (default from settings)
            verbose: Show progress
            
        Returns:
            Numpy array of shape (n_texts, embedding_dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.settings.embedding_dimension)
        
        embeddings = []
        for vector in self.batch_vectorize(texts, batch_size=batch_size, verbose=verbose):
            embeddings.append(vector)
        
        return np.array(embeddings, dtype=np.float32)


# Thread-safe singleton instance
_embedder_instance: Optional[SentenceTransformerEmbedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> SentenceTransformerEmbedder:
    global _embedder_instance
    
    with _embedder_lock:
        if _embedder_instance is None:
            _embedder_instance = SentenceTransformerEmbedder()
        return _embedder_instance


class Embedder(BaseEmbedder):
    """Compatibility class - delegates to SentenceTransformerEmbedder."""
    
    def __init__(self):
        super().__init__()
        self._model = None
        
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            embedder = get_embedder()
            self._model = embedder.model
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        embedder = get_embedder()
        return embedder.embed_text(text)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = np.sqrt(sum(a ** 2 for a in vec1))
    norm2 = np.sqrt(sum(b ** 2 for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_nearest(
    embedder: BaseEmbedder,
    query_embedding: List[float],
    document_embeddings: Dict[str, tuple],
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[Dict]:
    similarities = []
    
    for doc_id, (doc_embedding, metadata) in document_embeddings.items():
        if len(doc_embedding) != len(query_embedding):
            continue
            
        sim = cosine_similarity(query_embedding, doc_embedding)
        
        if threshold is None or sim >= threshold:
            similarities.append({
                'doc_id': doc_id,
                'similarity': float(sim),
                **dict(metadata)
            })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]
