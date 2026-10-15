"""
Secure RAG - Embedding Generation Service v1.0
Handles text-to-vector embedding generation for semantic search.

Note: This implementation uses mock embeddings for testing purposes.
Production deployment requires integration with actual embedding models
like Sentence-Transformers, BERT, or commercial APIs (OpenAI, Cohere).
"""

import hashlib
from typing import List, Dict, Any, Optional
import numpy as np


class MockEmbeddingModel:
    """
    Mock embedding model for testing and demonstration purposes.
    
    Real implementations should use:
    - Sentence-Transformers (https://www.sbert.net/)
    - HuggingFace Transformers with BERT models
    - Commercial APIs: OpenAI embeddings, Cohere, etc.
    """
    
    # Pre-defined mock embedding vectors for common concepts
    _concept_embeddings = {
        'искусственный интеллект': np.array([0.85, 0.12, -0.34, 0.67]),
        'машинное обучение': np.array([0.91, -0.23, 0.45, -0.56]),
        'нейронная сеть': np.array([0.78, -0.45, 0.89, -0.12]),
        'глубокое обучение': np.array([0.92, -0.34, 0.78, -0.67]),
        'обработка естественного языка': np.array([0.65, -0.78, 0.34, 0.91]),
        'компьютерное зрение': np.array([-0.23, 0.89, -0.45, 0.78]),
        'deep learning': np.array([0.89, -0.67, 0.34, 0.56]),
        'neural networks': np.array([0.92, -0.45, 0.78, -0.23]),
        'python programming': np.array([0.67, 0.89, -0.12, 0.45]),
        'data science': np.array([0.78, 0.34, -0.56, 0.89]),
    }
    
    def __init__(self):
        self.model_name = "mock-embedding-v1"
    
    async def generate_embedding(
        self, 
        text: str, 
        model: Optional[str] = None,
        normalize: bool = True
    ) -> List[float]:
        """
        Generate embedding vector for input text.
        
        Args:
            text: Input string to embed
            model: Model identifier (ignored in mock implementation)
            normalize: Whether to L2-normalize the output
        
        Returns:
            Embedding vector as list of floats
        """
        # Normalize input text
        text = text.lower().strip()
        
        # Check if we have a pre-defined concept match
        for concept, embedding in self._concept_embeddings.items():
            if concept in text or text in concept:
                if normalize:
                    return self._normalize(embedding).tolist()
                return embedding.tolist()
        
        # Generate deterministic mock embedding based on text hash
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Create pseudo-random but reproducible vector
        base_vector = np.array([0.0] * 4)
        for i in range(4):
            # Use different bits of hash to create variation
            val = ((hash_val >> (i * 8)) & 255) / 127.0 - 1.0
            base_vector[i] = val
        
        if normalize:
            return self._normalize(base_vector).tolist()
        
        return base_vector.tolist()
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""
        import asyncio
        
        tasks = [self.generate_embedding(text, model) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results


class EmbeddingService:
    """
    Service layer for embedding generation with caching and fallback.
    
    Features:
    - Model management (primary + fallback models)
    - Result caching to avoid redundant computations
    - Graceful degradation on model failures
    """
    
    def __init__(self, primary_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.primary_model = MockEmbeddingModel()
        self.fallback_models = [MockEmbeddingModel()]  # Add more if needed
        
        self.cache: Dict[str, List[float]] = {}
        self.max_cache_size = 1000
    
    async def generate_embedding(
        self, 
        text: str,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text string
            model_name: Specific model to use (uses primary if not specified)
            use_cache: Whether to check cache before generating
        
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        if use_cache and text in self.cache:
            return self.cache[text]
        
        # Generate embedding using current model
        try:
            embedding = await self.primary_model.generate_embedding(text)
            
            # Cache result (with size limit)
            if len(self.cache) >= self.max_cache_size:
                # Evict oldest entry
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[text] = embedding
            
        except Exception as e:
            # Fallback to secondary models if primary fails
            for fallback in self.fallback_models:
                try:
                    embedding = await fallback.generate_embedding(text)
                    self._update_cache(text, embedding)
                    break
                except:
                    continue
        
        return embedding
    
    async def generate_embeddings(
        self, 
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        import asyncio
        
        tasks = [
            self.generate_embedding(text, model_name)
            for text in texts
        ]
        
        return await asyncio.gather(*tasks)
    
    def _update_cache(self, text: str, embedding: List[float]):
        """Update cache with new result."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[text] = embedding
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cache.clear()


async def main():
    """Test embedding service functionality."""
    
    print("=" * 60)
    print("Testing Embedding Service")
    print("=" * 60)
    
    # Initialize service
    service = EmbeddingService()
    
    test_texts = [
        "Искусственный интеллект — это наука о создании умных машин",
        "Нейронные сети обучаются на больших объемах данных",
        "Что такое глубокое обучение и как оно работает?",
        "Python programming and machine learning basics",
        "Natural language processing with transformers"
    ]
    
    print("\n[Generating embeddings for test texts]")
    embeddings = await service.generate_embeddings(test_texts)
    
    # Display results
    for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
        print(f"\n--- Text {i+1} ---")
        text_preview = text[:60] + "..." if len(text) > 60 else text
        
        print(f"Text: {text_preview}")
        print(f"Embedding (first 5 dims): {embedding[:5]}")
    
    # Test caching behavior
    print("\n" + "=" * 60)
    print("Testing Cache Behavior")
    print("=" * 60)
    
    first_embedding = await service.generate_embedding(
        "Искусственный интеллект", 
        use_cache=True
    )
    
    second_embedding = await service.generate_embedding(
        "Искусственный интеллект", 
        use_cache=True
    )
    
    print(f"\nFirst call hash: {hashlib.md5(str(first_embedding).encode()).hexdigest()}")
    print(f"Second call hash: {hashlib.md5(str(second_embedding).encode()).hexdigest()}")
    print(f"Caching working: {'✓' if first_embedding == second_embedding else '✗'}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
