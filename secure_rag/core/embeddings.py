"""
Secure RAG - Embedding Engine v2.0
Thread-safe batch vectorization with OpenAI-compatible API
"""

import asyncio
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import threading

from .schemas import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    BatchEmbeddingRequest,
    DocumentChunk
)


@dataclass
class EmbeddingEngine:
    """Thread-safe embedding engine with batch processing."""
    
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    max_batch_size: int = 50
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __post_init__(self):
        self._initialized = True
    
    async def initialize(self, api_key: str) -> None:
        """Initialize the engine with API credentials."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        self.api_key = api_key
        print(f"[Embedding Engine] Initialized with model: {self.model}")
    
    async def create_embedding(
        self, 
        text: str,
        dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """Create embedding for single text (synchronous simulation)."""
        if not self._initialized or not self.api_key:
            raise RuntimeError("API key required")
        
        # Simulate API call - replace with actual OpenAI SDK integration
        import hashlib
        vector = [hashlib.sha256(text.encode()).hexdigest()] * 1536
        
        response = EmbeddingResponse(
            object="embedding",
            model=self.model,
            data=[{
                "object": "embedding",
                "index": 0,
                "embedding": vector[:dimensions] if dimensions else vector
            }],
            usage={"prompt_tokens": len(text.split()), "total_tokens": len(text.split())}
        )
        
        return response
    
    async def batch_create_embeddings(
        self, 
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> BatchEmbeddingResponse:
        """Batch create embeddings with thread-safe processing."""
        if not self._initialized or not self.api_key:
            raise RuntimeError("API key required")
        
        # Thread-safe batching
        async with self._lock:
            responses = []
            
            for i, text in enumerate(texts):
                try:
                    response = await self.create_embedding(
                        text=text, 
                        dimensions=dimensions
                    )
                    response.index = i
                    
                    if isinstance(response, EmbeddingResponse):
                        responses.append({
                            "object": "embedding",
                            "index": i,
                            "embedding": response.data[0].embedding if response.data else []
                        })
                except Exception as e:
                    print(f"[Embedding Error] Text {i}: {str(e)}")
                    continue
            
            return BatchEmbeddingResponse(
                object="list",
                data=responses,
                model=self.model,
                usage={
                    "prompt_tokens": sum(len(t.split()) for t in texts if isinstance(t, str)),
                    "total_tokens": sum(len(t.split()) for t in texts if isinstance(t, str))
                }
            )


async def create_embedding_engine(api_key: Optional[str] = None) -> EmbeddingEngine:
    """Factory function to create embedding engine instance."""
    return EmbeddingEngine(
        api_key=api_key,
        max_batch_size=50
    )


@dataclass
class BatchEmbeddingResponse:
    """Batch embedding response from OpenAI API."""
    object: str = "list"
    model: str = ""
    data: List[Dict] = None
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = []
        if self.usage is None:
            self.usage = {}


def main():
    """Test embedding engine."""
    import asyncio
    
    async def test():
        engine = await create_embedding_engine()
        
        # Initialize (simulated)
        try:
            await engine.initialize("test-api-key")
            
            # Test single embedding
            response = await engine.create_embedding(
                text="This is a test document for secure RAG system."
            )
            print(f"[Test] Single embedding created: {len(response.data)} vectors")
            
            # Test batch embedding
            texts = [
                "Document 1 content",
                "Document 2 content", 
                "Document 3 content"
            ]
            batch_response = await engine.batch_create_embeddings(texts=texts)
            print(f"[Test] Batch embedding created: {len(batch_response.data)} vectors")
            
        except Exception as e:
            print(f"[Error] {str(e)}")
    
    asyncio.run(test())


if __name__ == "__main__":
    main()
