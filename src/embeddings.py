#!/usr/bin/env python3
"""Embedding Service - Local sentence transformers for RAG."""
import os
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise RuntimeError(f'sentence-transformers not installed: {e}')


class EmbeddingService:
    """Generate embeddings using local or cached models."""

    def __init__(self, model_name="BAAI/bge-m3", cache_dir=None):
        self.model_name = model_name
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def _ensure_model_loaded(self):
        """Load the sentence transformer model."""
        if not hasattr(self, "model") or self.model is None:
            print(f'Loading model: {self.model_name}...')
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str], batch_size=32) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        self._ensure_model_loaded()
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()


if __name__ == "__main__":
    service = EmbeddingService(cache_dir="./model_cache")
    texts = ["The quick brown fox jumps over the lazy dog", "Machine learning is a subset of artificial intelligence"]
    embeddings = service.generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings with dimension: {len(embeddings[0])}")
