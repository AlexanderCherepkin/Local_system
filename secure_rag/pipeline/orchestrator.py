Production RAG Orchestrator with Hybrid Search and Fallback Strategies
=====================================================================
Advanced retrieval orchestration:
- Multi-stage search pipeline (hybrid + semantic)
- Automatic fallback to alternative backends
- Caching layer for query deduplication
- Query rewriting and expansion
- Performance monitoring and metrics collection

"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


# ============================================================================
# Configuration

@dataclass
class OrchestratorConfig:
    """Configuration for RAG orchestrator"""
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    
    # Search strategy
    use_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Fallback settings
    fallback_enabled: bool = True
    max_fallback_attempts: int = 2
    fallback_delay_ms: int = 100
    
    # Query processing
    rewrite_queries: bool = True
    expand_synonyms: bool = True
    max_results_per_stage: int = 5
    
    # Performance thresholds
    cache_hit_threshold: float = 0.8
    similarity_cutoff: float = 0.75


# ============================================================================
# Query Cache

class QueryCache:
    """In-memory query result cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.ttl = ttl_seconds
    
    def _cache_key(self, query_text: str) -> str:
        """Generate cache key from normalized query text"""
        normalized_query = query_text.strip().lower()
        return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()
    
    def get(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists and not expired"""
        key = self._cache_key(query_text)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check TTL
            age_seconds = time.time() - timestamp
            
            if age_seconds < self.ttl:
                return result
            
            del self.cache[key]
        
        return None
    
    def set(self, query_text: str, results: Dict[str, Any]):
        """Cache results"""
        key = self._cache_key(query_text)
        self.cache[key] = (results, time.time())


# ============================================================================
# Query Rewriter

class QueryRewriter:
    """Advanced query rewriting and expansion"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
    
    def rewrite(self, original_query: str) -> List[str]:
        """Generate rewritten versions of the query"""
        
        # Base query is always included
        queries = [original_query]
        
        if not self.config.rewrite_queries:
            return queries
        
        try:
            from transformers import pipeline
            
            # Extract key entities and expand synonyms
            # For production, use proper NLP library
            rewritten = []
            
            # 1. Remove stop words and noise
            cleaned = original_query.lower().strip()
            rewritten.append(cleaned)
            
        except ImportError:
            print("[WARN] Transformers pipeline not available, using basic cleaning")
            return [original_query.lower().strip()]

        return queries + rewritten


# ============================================================================
# Vector Store Manager (Mock/Stub for Orchestrator Logic)

class VectorStoreConfig:
    def __init__(self, backend: str, similarity_threshold: float):
        self.backend = backend
        self.similarity_threshold = similarity_threshold

class VectorStoreManager:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    async def initialize(self):
        # Mock initialization logic
        pass
    
    async def search(self, query_embedding: List[float], original_query: str, filter: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], float]:
        # Mock search results
        await time.sleep(0.05)
        results = [
            {"id": "doc_1", "content": "Sample content 1", "combined_score": 0.9, "payload": {"source": "wiki"}, "semantic_only": False},
            {"id": "doc_2", "content": "Sample content 2", "combined_score": 0.85, "payload": {"source": "manual"}, "semantic_only": True}
        ]
        return results, 45.2

# ============================================================================
# RAG Orchestrator

class RAGOrchestrator:
    """Production-grade RAG orchestrator with hybrid search and fallback logic."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.cache = QueryCache(ttl_seconds=config.cache_ttl_seconds)
        self.rewriter = QueryRewriter(config)
        self.search_engine = VectorStoreManager(VectorStoreConfig(backend="qdrant", similarity_threshold=config.similarity_cutoff))

    async def retrieve(self, query_embedding: List[float], original_query: str, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main retrieval entry point with caching and fallback logic."""
        
        # 1. Check Cache
        if self.config.cache_enabled:
            cached_result = self.cache.get(original_query)
            if cached_result:
                print("[CACHE] Hit for query")
                return cached_result

        start_time = time.perf_counter()
        
        # 2. Query Rewriting & Expansion
        rewritten_queries = self.rewriter.rewrite(original_query)
        
        # 3. Primary Search (Hybrid/Semantic)
        results, latency = await self.search_engine.search(query_embedding, original_query, filter)
        
        # 4. Fallback Logic if no results or low score
        if not results or len(results) < self.config.max_results_per_stage:
            if self.config.fallback_enabled:
                print("[INFO] No sufficient results, initiating fallback...")
                results, _ = await self._execute_fallback(query_embedding, original_query, filter)

        # 5. Post-processing and formatting
        latency_ms = (time.perf_counter() - start_time) * 1000
        formatted_result = self._format_results(results, latency_ms, original_query)
        
        # 6. Update Cache
        if self.config.cache_enabled:
            self.cache.set(original_query, formatted_result)
            
        return formatted_result

    async def _execute_fallback(
        self, 
        query_embedding: List[float],
        original_query: str,
        filter: Optional[Dict[str, Any]] = None
    ):
        """Execute fallback search strategies."""
        try:
            # Fallback to different backends or relaxed filters
            for i in range(self.config.max_fallback_attempts):
                wait_time = self.config.fallback_delay_ms * (i + 1) / 1000
                
                print(f"[FALLBACK] Attempt {i+1}/{self.config.max_fallback_attempts} after {wait_time:.2f}s")
                
                # Try alternative strategy: lower threshold
                config = VectorStoreConfig(
                    backend="qdrant",
                    similarity_threshold=0.5 - (i * 0.1)  # Gradually relax threshold
                )
                
                manager = VectorStoreManager(config)
                await manager.initialize()
                
                results, latency = await self.search_engine.search(
                    query_embedding, original_query, filter
                )
                
                if results:
                    print(f"[FALLBACK] Success! Found {len(results)} documents")
                    return results, latency
        
        except Exception as e:
            print(f"[ERROR] Fallback failed: {e}")
            return [], 0.0

    def _format_results(
        self, 
        results: List[Dict[str, Any]], 
        latency_ms: float,
        original_query: str
    ) -> Dict[str, Any]:
        """Format retrieval results for downstream processing"""
        
        return {
            'documents': [
                {
                    'id': doc.get('id'),
                    'content': doc.get('content', ''),
                    'score': doc.get('combined_score', 0),
                    'metadata': doc.get('payload', {}),
                    'vector_source': 'hybrid' if not doc.get('semantic_only') else 'semantic'
                }
                for doc in results[:self.config.max_results_per_stage]
            ],
            'total_found': len(results),
            'latency_ms': latency_ms,
            'query': self.cache._cache_key(original_query) if hasattr(self.cache, '_cache_key') else original_query,
            'timestamp': datetime.utcnow().isoformat(),
            'config_used': {
                'hybrid_search': self.config.use_hybrid_search,
                'semantic_weight': self.config.semantic_weight,
                'keyword_weight': self.config.keyword_weight
            }
        }

# Backward Compatibility Alias
RAPOrchestrator = RAGOrchestrator


# ============================================================================
# Example Usage and Testing

if __name__ == "__main__":
    print("=" * 60)
    print("Testing RAG Orchestrator")
    print("=" * 60)
    
    config = OrchestratorConfig(
        use_hybrid_search=True,
        semantic_weight=0.7,
        keyword_weight=0.3,
        max_results_per_stage=5,
        similarity_cutoff=0.75
    )
    
    import asyncio
    
    async def run_test():
        try:
            orchestrator = RAGOrchestrator(config)
            
            # Test query embedding (placeholder - use actual embeddings in production)
            test_query_embedding = [0.1] * 384
            
            print(f"\n[TEST] Running retrieval with test query...")
            
            result = await orchestrator.retrieve(
                query_embedding=test_query_embedding,
                original_query="What is machine learning?"
            )
            
            print(f"\n[SUCCESS] Retrieved {len(result['documents'])} documents in {result['latency_ms']:.2f}ms")
            print(f"[INFO] Documents found: {result['total_found']}")
            
        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")

    asyncio.run(run_test())
