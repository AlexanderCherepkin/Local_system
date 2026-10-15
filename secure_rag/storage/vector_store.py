"""
Secure RAG - Vector Storage Module v2.0
In-memory vector store with similarity search capabilities
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class VectorStore:
    """
    In-memory vector store for semantic similarity search.
    
    Features:
    - Embedding-based similarity search
    - Metadata filtering
    - Chunk management
    
    Note: This is a simplified in-memory implementation.
    Production systems should use FAISS, Pinecone, or Weaviate.
    """
    
    def __init__(self):
        self.chunks = {}  # chunk_id -> chunk_data
        self.index = {}   # embedding_hash -> chunk_ids
        self.chunk_counter = 0
    
    async def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 
                   'embedding', and 'metadata'
        
        Returns:
            Number of chunks added
        """
        count = 0
        
        for chunk_data in chunks:
            # Generate unique ID if not present
            content_hash = hashlib.md5(
                chunk_data['content'].encode()
            ).hexdigest()[:12]
            
            if 'id' not in chunk_data or chunk_data['id'] == 'doc_0':
                chunk_data['id'] = f"chunk_{self.chunk_counter}"
                self.chunk_counter += 1
            
            # Store full content and metadata
            self.chunks[chunk_data['id']] = {
                'content': chunk_data.get('content', ''),
                'embedding': np.array(chunk_data.get('embedding', [])),
                'metadata': chunk_data.get('metadata', {})
            }
            
            # Create embedding hash for fast lookup
            embed_hash = hashlib.md5(
                str(chunk_data['embedding']).encode()
            ).hexdigest()[:8]
            
            if embed_hash not in self.index:
                self.index[embed_hash] = []
            self.index[embed_hash].append(chunk_data['id'])
            
            count += 1
        
        return count
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a single chunk by ID."""
        if chunk_id in self.chunks:
            return self.chunks[chunk_id]
        return None
    
    async def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Semantic similarity search.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return
            metadata_filter: Optional filter on chunk metadata
        
        Returns:
            List of (chunk_id, similarity_score) tuples sorted by score
        """
        # Convert to numpy array and normalize
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        results = []
        
        for chunk_id, chunk_data in self.chunks.items():
            # Skip if metadata filter doesn't match
            if metadata_filter:
                meta = chunk_data.get('metadata', {})
                skip = False
                
                for key, value in metadata_filter.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                
                if skip:
                    continue
            
            # Calculate cosine similarity with mock embedding
            chunk_vec = chunk_data['embedding']
            norm_chunk = np.linalg.norm(chunk_vec)
            
            if norm > 0 and norm_chunk > 0:
                sim_score = np.dot(query_vec, chunk_vec / norm_chunk)
            else:
                # Fallback to deterministic score for mock embeddings
                sim_score = self._mock_similarity(
                    query_embedding, 
                    list(chunk_vec)
                )
            
            results.append((chunk_id, sim_score))
        
        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _mock_similarity(self, query: List[float], chunk: List[float]) -> float:
        """
        Generate deterministic mock similarity scores.
        
        This simulates actual cosine similarity when using 
        real embedding models for testing purposes.
        """
        import hashlib
        
        # Create hash-based pseudo-similarity
        combined = f"{query}:{chunk}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        
        return float((hash_val % 10000)) / 100.0
    
    def get_all_chunks(self, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Retrieve all chunks with optional filtering."""
        if not metadata_filter:
            return list(self.chunks.values())
        
        filtered = []
        for chunk_id, data in self.chunks.items():
            meta = data.get('metadata', {})
            
            # Check filter criteria
            match = True
            for key, value in metadata_filter.items():
                if meta.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(data)
        
        return filtered
    
    def clear(self):
        """Clear all stored chunks."""
        self.chunks.clear()
        self.index.clear()
        self.chunk_counter = 0


class HybridSearchStore(VectorStore):
    """
    Extended vector store with hybrid search capabilities.
    
    Combines:
    - Vector similarity (semantic)
    - Keyword matching (BM25-style)
    """
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha  # Weight for vector vs keyword search
    
    async def search(
        self, 
        query_embedding: List[float],
        query_text: str = "",
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Hybrid search combining vector and keyword scores."""
        
        # Get vector similarity results
        vector_results = await super().search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for hybrid scoring
            metadata_filter=metadata_filter
        )
        
        # Calculate keyword similarity (simple term matching)
        kw_results = self._keyword_similarity(query_text)
        
        # Combine scores using alpha weighting
        combined_results = []
        for chunk_id, vector_score in vector_results:
            kw_score = kw_results.get(chunk_id, 0.5)
            
            # Normalize keyword score to [0, 1] range
            normalized_kw = (kw_score - 0.5) * 2
            
            combined_score = self.alpha * vector_score + (1 - self.alpha) * normalized_kw
            combined_results.append((chunk_id, max(0, min(1, combined_score))))
        
        # Sort and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def _keyword_similarity(self, query_text: str) -> Dict[str, float]:
        """Simple keyword-based similarity scoring."""
        import re
        
        if not query_text:
            return {}
        
        # Tokenize query (simple word extraction)
        query_terms = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        results = {}
        for chunk_id, data in self.chunks.items():
            content = data.get('content', '').lower()
            
            if not content:
                continue
            
            # Calculate term overlap ratio (simplified BM25)
            content_terms = set(re.findall(r'\b\w+\b', content))
            
            if not query_terms or not content_terms:
                results[chunk_id] = 0.5
                continue
            
            # Jaccard-like similarity
            intersection = len(query_terms & content_terms)
            union = len(query_terms | content_terms)
            
            if union > 0:
                sim = intersection / union
            else:
                sim = 0.5
            
            results[chunk_id] = max(0, min(1, sim))
        
        return results


async def main():
    """Test vector store functionality."""
    
    print("=" * 60)
    print("Testing Vector Store")
    print("=" * 60)
    
    # Create store and add test chunks
    store = VectorStore()
    
    sample_chunks = [
        {
            'content': '''
            Искусственный интеллект (ИИ) — это раздел информатики, 
            занимающийся созданием систем, способных выполнять задачи, 
            требующие интеллекта. Машиное обучение и глубокое обучение 
            являются ключевыми областями ИИ.
            ''',
            'embedding': [0.1, 0.2, -0.3, 0.4],  # Mock embedding
            'metadata': {'source': 'ai_intro'}
        },
        {
            'content': '''
            Нейронные сети — это вычислительные модели, 
            вдохновленные работой мозга. Они состоят из множества узлов, 
            связанных весами, которые обучаются на данных.
            ''',
            "embedding": [0.5, -0.4, 0.6, -0.7],  # Mock embedding
            'metadata': {'source': 'neural_networks'}
        },
        {
            'content': '''
                Deep learning and neural networks are powerful tools 
                for natural language processing and computer vision tasks.
            ''',
            "embedding": [0.6, -0.5, 0.7, -0.8],
            'metadata': {'source': 'english_ml'}
        }
    ]
    
    # Add chunks to store
    count = await store.add_chunks(sample_chunks)
    print(f"\n[Store] Added {count} chunks")
    
    # Test query search (Russian query about neural networks)
    mock_query_embedding = [0.4, -0.3, 0.5, -0.6]
    
    print("\n[Query: Поиск по нейронным сетям]")
    results = await store.search(
        query_embedding=mock_query_embedding,
        top_k=2
    )
    
    for chunk_id, score in results:
        chunk_data = store.get_chunk(chunk_id)
        content_preview = (chunk_data['content'][:100] + '...') if len(chunk_data['content']) > 100 else chunk_data['content']
        
        print(f"\n--- Chunk {chunk_id} (score: {score:.4f}) ---")
        print(content_preview)
    
    # Test hybrid search with English query
    print("\n" + "=" * 60)
    print("Testing HybridSearchStore")
    print("=" * 60)
    
    hybrid_store = HybridSearchStore(alpha=0.7)
    await hybrid_store.add_chunks(sample_chunks)
    
    english_query = "What are neural networks?"
    results_hybrid = await hybrid_store.search(
        query_embedding=[0.5, -0.4, 0.6],
        query_text=english_query,
        top_k=2
    )
    
    for chunk_id, score in results_hybrid:
        chunk_data = hybrid_store.get_chunk(chunk_id)
        content_preview = (chunk_data['content'][:80] + '...') if len(chunk_data['content']) > 80 else chunk_data['content']
        
        print(f"\n[Hybrid Result {chunk_id}] Score: {score:.4f}")
        print(content_preview)


if __name__ == "__main__":
    asyncio.run(main())
