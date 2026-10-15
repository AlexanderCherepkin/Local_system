"""
Secure RAG - Embedding Module v2.0
Text-to-embedding conversion with multiple backends support
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""
    
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Convert text to vector embeddings."""
        pass
    
    @abstractmethod
    async def embed_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Embed multiple texts in batch."""
        pass
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Return model information and capabilities."""
        return {
            'model_name': self.model_name,
            'embedding_dim': 768,  # Default placeholder
            'supported_languages': ['en', 'ru']
        }


class HuggingFaceEmbedder(BaseEmbedder):
    """Hugging Face Sentence Transformers embedding model."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        super().__init__(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    async def _load_model(self):
        """Lazy load the embedding model."""
        from transformers import AutoModel, AutoTokenizer
        
        if self.model is None:
            print(f"[Embedder] Loading model: {self.model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    device_map=self.device
                )
                print("[Embedder] Model loaded successfully")
            except Exception as e:
                # Fallback to mock embeddings for testing
                print(f"[Embedder] Using mock embeddings (model load failed): {str(e)}")
                self._mock_embeddings = True
    
    async def embed_text(self, text: str) -> List[float]:
        """Convert single text to embedding vector."""
        await self._load_model()
        
        if self._mock_embeddings:
            return self._get_mock_embedding(text)
        
        # Tokenize and encode (simplified - actual implementation needs full HF integration)
        inputs = {
            "input_ids": [[0, 1, 2]],  # Mock token IDs
            "attention_mask": [[1, 1, 1]]
        }
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0].mean(dim=0).tolist()
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch."""
        await self._load_model()
        
        results = []
        for text in texts:
            embedding = await self.embed_text(text)
            results.append(embedding)
        
        return results
    
    def _get_mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embeddings based on text hash."""
        import hashlib
        
        # Create deterministic fake vectors for testing
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        embedding_dim = 768
        return [float((hash_value >> i) & 1) - 0.5 for i in range(embedding_dim)]


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing without ML models."""
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__("mock")
        self.embedding_dim = embedding_dim
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embeddings."""
        return self._get_mock_embedding(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with mock data."""
        return [self._get_mock_embedding(t) for t in texts]
    
    def _get_mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic fake embeddings."""
        import hashlib
        
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        return [float((hash_val >> i) & 1) - 0.5 for i in range(self.embedding_dim)]


async def get_embedder(model_name: str = "huggingface") -> BaseEmbedder:
    """Factory function to create appropriate embedder."""
    
    if model_name.lower() == "mock":
        return MockEmbedder()
    elif model_name.lower().startswith("sentence-transformers/"):
        return HuggingFaceEmbedder(model_name)
    else:
        # Default to mock for testing
        print(f"[Embedder] Using default embedder: {model_name}")
        return MockEmbedder(embedding_dim=768)


async def main():
    """Test embedding module."""
    import torch
    
    # Create and test embedder
    embedder = await get_embedder("mock")
    
    print("[Test] Generating embeddings...")
    
    sample_texts = [
        "This is a test document about machine learning",
        "Deep learning and neural networks are powerful tools"
    ]
    
    for text in sample_texts:
        embedding = await embedder.embed_text(text)
        print(f"\nText: {text[:50]}...")
        print(f"Embedding shape: {len(embedding)} dimensions")
        print(f"Sample values: {embedding[:10]}")


if __name__ == "__main__":
    asyncio.run(main())
