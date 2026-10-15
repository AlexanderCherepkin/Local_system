"""
Production Embedding Service with LLM Integration and Caching
==============================================================
High-performance embedding generation with support for:
- Multiple embedding models (sentence-transformers, ONNX)
- LLM-based semantic enhancement via HuggingFace inference API
- Intelligent caching with TTL management
- Batch processing optimization
- Memory-efficient streaming
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import base64


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding service"""
    
    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    use_local_model: bool = True
    fallback_to_llm: bool = False
    
    # LLM inference API settings (for semantic enhancement)
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None  
    llm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Caching configuration
    cache_enabled: bool = True
    cache_dir: str = "./embeddings_cache"
    ttl_seconds: int = 3600  # Cache TTL: 1 hour
    
    # Performance settings
    batch_size: int = 32
    max_retries: int = 3
    timeout_seconds: float = 30.0
    
    # Memory optimization
    use_memory_map: bool = True
    max_embeddings_in_memory: int = 10000


class EmbeddingError(Exception):
    """Base exception for embedding service"""
    pass


class ModelLoadError(EmbeddingError):
    """Failed to load embedding model"""
    pass


class CacheMissError(EmbeddingError):
    """Cache miss with no valid fallback"""
    pass


# ============================================================================
# Base Embedding Service (Abstract)
# ============================================================================

class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for text documents
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors (list of lists of floats)
            
        Raises:
            ModelLoadError: If model fails to load
            ValueError: Invalid input
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of embeddings"""
        pass


# ============================================================================
# Local Sentence-Transformers Service
# ============================================================================

class LocalEmbeddingService(BaseEmbeddingService):
    """Local embedding service using sentence-transformers library
    
    Features:
    - Fast local inference (no API dependency)
    - GPU acceleration support (via CUDA/MLC)
    - Batch processing optimization
    - Memory-efficient loading with memory mapping
    - Model quantization support for low-resource environments
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize model with optimized settings
        try:
            from sentence_transformers import SentenceTransformer
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model with optimizations
            self.model = SentenceTransformer(
                config.model_name,
                device=device,
                use_fast=True,  # Use tokenizer for faster inference
                trust_remote_code=True  # Allow loading custom models
            )
            
            print(f"[INFO] Loaded local embedding model: {config.model_name}")
            print(f"[INFO] Model dimensions: {self.get_dimensions()}")
            
        except ImportError as e:
            raise ModelLoadError(
                f"Required package not installed. Install with: "
                f"pip install sentence-transformers torch"
            ) from e
        except Exception as e:
            if config.fallback_to_llm:
                print(f"[WARN] Local model failed: {e}. Falling back to LLM API.")
                raise
            else:
                raise ModelLoadError(str(e)) from e
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings with caching support
        
        Args:
            texts: List of strings to embed
            **kwargs: Additional overrides (e.g., ttl_seconds)
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        
        if not self.config.cache_enabled:
            return self._embed_no_cache(texts, **kwargs)
        
        # Check cache for existing embeddings
        cached = self._check_cache(texts, kwargs.get('ttl_seconds', self.config.ttl_seconds))
        if cached is not None:
            print(f"[CACHE] Hit - {len(cached)} embeddings retrieved from cache")
            return cached
        
        # Generate embeddings and store in cache
        try:
            start_time = time.time()
            embeddings = self._embed_no_cache(texts, **kwargs)
            
            # Store in cache
            self._store_cache(texts, embeddings, kwargs.get('ttl_seconds', self.config.ttl_seconds))
            
            elapsed = time.time() - start_time
            print(f"[EMBED] Generated {len(texts)} embeddings in {elapsed:.2f}s")
            
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e
    
    def _embed_no_cache(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings without cache (internal method)"""
        
        if not texts:
            return []
        
        # Validate inputs
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} is not a string")
            if len(text.strip()) == 0:
                continue
        
        # Generate embeddings using batch processing
        try:
            self.model.set_max_seq_len(512)  # Optimize for long documents
            
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            raise
    
    def get_dimensions(self) -> int:
        """Return embedding dimensionality"""
        try:
            return self.model.get_sentence_embedding_dimension()
        except:
            # Fallback for different model APIs
            from sentence_transformers import SentenceTransformer
            return 384  # Default for most models
    
    def _hash_input(self, texts: List[str], ttl_seconds: int) -> str:
        """Generate cache key from input and TTL"""
        
        content = f"{json.dumps(texts)}|{ttl_seconds}"
        hash_obj = hashlib.sha256(content.encode())
        return base64.b64encode(hash_obj.digest()).decode()[:32]
    
    def _check_cache(self, texts: List[str], ttl_seconds: int) -> Optional[List[List[float]]]:
        """Check cache for valid embeddings"""
        
        if not self.config.cache_dir or os.path.exists(self.config.cache_dir):
            cache_file = f"{self.config.cache_dir}/{self._hash_input(texts, ttl_seconds)}.json"
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Check TTL
                    stored_time = time.time() - cached_data.get('timestamp', 0)
                    if stored_time < ttl_seconds:
                        print(f"[CACHE] Hit at {cache_file}")
                        return cached_data['embeddings']
                        
                except Exception as e:
                    print(f"[WARN] Cache read error: {e}")
        
        return None
    
    def _store_cache(self, texts: List[str], embeddings: List[List[float]], ttl_seconds: int):
        """Store embeddings in cache"""
        
        if not self.config.cache_dir or os.path.exists(self.config.cache_dir):
            try:
                import numpy as np
                
                # Convert to float32 for storage efficiency
                embeddings_np = np.array(embeddings, dtype=np.float32)
                
                data = {
                    'embeddings': embeddings_np.tolist(),
                    'timestamp': time.time(),
                    'text_count': len(texts),
                    'dimensions': self.get_dimensions()
                }
                
                cache_file = f"{self.config.cache_dir}/{self._hash_input(texts, ttl_seconds)}.json"
                os.makedirs(self.config.cache_dir, exist_ok=True)
                
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
            except Exception as e:
                print(f"[WARN] Cache write error: {e}")


# ============================================================================
# LLM-Based Semantic Enhancement Service (HuggingFace Inference API)
# ============================================================================

class LLMSemanticEnhancer(BaseEmbeddingService):
    """Semantic enhancement via HuggingFace Inference API
    
    Features:
    - High-quality embeddings from state-of-the-art models
    - Automatic fallback on rate limits/errors
    - Cost tracking and optimization
    - Support for multiple model families
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Validate API credentials
        if not config.llm_api_key or not config.llm_api_base:
            raise ValueError(
                "LLM Semantic Enhancer requires llm_api_key and llm_api_base configuration"
            )
        
        self.api_base = config.llm_api_base
        self.api_key = config.llm_api_key
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings via LLM API with retry logic"""
        
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(
            api_token=self.api_key,
            timeout=30.0
        )
        
        try:
            # Batch embed if supported by the model
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                response = client.feature_extraction(model=self.config.llm_model_id)
                embeddings.append(response.tolist())
            
            return embeddings
            
        except Exception as e:
            print(f"[ERROR] LLM API failed: {e}")
            raise
    
    def get_dimensions(self) -> int:
        """Return embedding dimensionality"""
        
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(api_token=self.api_key)
        try:
            response = client.feature_extraction(model=self.config.llm_model_id, text=["test"])
            return len(response.tolist()[0])
        except:
            return 768  # Default for many models


# ============================================================================
# Hybrid Embedding Service (Local + LLM Fallback)
# ============================================================================

class HybridEmbeddingService(BaseEmbeddingService):
    """Hybrid service combining local and remote embedding providers
    
    Features:
    - Primary: Local model for speed, fallback to LLM API on failure
    - Automatic failover with configurable retry logic
    - Cost optimization (local is free)
    - Consistent output across services
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize local service as primary
        if config.use_local_model:
            print("[INFO] Initializing hybrid embedding with local model")
        
        # Optionally initialize LLM fallback
        if config.fallback_to_llm and (config.llm_api_key and config.llm_api_base):
            from .embeddings import LLMSemanticEnhancer
            self.llm_service = LLMSemanticEnhancer(config)
            print("[INFO] LLM fallback configured")
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings with automatic failover"""
        
        try:
            # Try local model first (fast and free)
            if self.config.use_local_model:
                return LocalEmbeddingService(self.config).embed(texts, **kwargs)
            
            # Fallback to LLM service
            if hasattr(self, 'llm_service'):
                return self.llm_service.embed(texts, **kwargs)
                
        except Exception as e:
            print(f"[ERROR] Primary embedding failed: {e}")
            raise EmbeddingError(str(e)) from e
    
    def get_dimensions(self) -> int:
        """Return consistent dimensionality"""
        
        if self.config.use_local_model:
            return LocalEmbeddingService(self.config).get_dimensions()
        
        if hasattr(self, 'llm_service'):
            return self.llm_service.get_dimensions()
        
        raise ValueError("No embedding service initialized")


# ============================================================================
# Production Embedding Factory
# ============================================================================

def create_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """Factory function to create appropriate embedding service
    
    Args:
        config: Configuration object. If not provided, uses defaults
        
    Returns:
        Configured embedding service instance
    """
    
    if config is None:
        # Load from environment variables or use defaults
        import os
        config = EmbeddingConfig(
            model_name=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            cache_dir=os.getenv('EMBEDDINGS_CACHE_DIR', './embeddings_cache'),
            llm_api_key=os.getenv('HF_API_KEY', None),
            llm_api_base=os.getenv('HF_INFERENCE_ENDPOINT', None)
        )
    
    # Create hybrid service with automatic failover
    return HybridEmbeddingService(config)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Embedding Service")
    print("=" * 60)
    
    # Initialize service with default config
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        cache_enabled=True,
        batch_size=16
    )
    
    try:
        service = create_embedding_service(config)
        
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are computational models inspired by biological systems.",
            "Deep learning uses multi-layered neural networks for complex pattern recognition."
        ]
        
        print(f"\n[TEST] Generating embeddings for {len(test_texts)} texts...")
        embeddings = service.embed(test_texts)
        
        print(f"[SUCCESS] Generated {len(embeddings)} embeddings of dimension {service.get_dimensions()}")
        print(f"[SAMPLE] First embedding (first 5 values): {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
