"""
Secure RAG - Pipeline Processor v2.0
Document chunking, embedding generation, and query processing
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    
    # Chunking parameters
    chunk_size: int = 1024          # Characters per chunk
    overlap: int = 64                # Overlap between chunks
    
    # Query parameters  
    top_k: int = 5                   # Number of results to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity score
    
    # Embedding settings
    embedding_dim: int = 768


@dataclass 
class ProcessedChunk:
    """A processed document chunk ready for storage."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    position: Tuple[int, int]      # (start_index, end_index) in original doc
    
    @classmethod
    def create(cls, 
               text: str, 
               start_idx: int = 0, 
               end_idx: int = None,
               metadata: Dict[str, Any] = None):
        """Create a chunk from text segment."""
        if end_idx is None:
            end_idx = len(text)
        
        content = text[start_idx:end_idx]
        
        return cls(
            id=f"chunk_{start_idx}_{end_idx}",
            content=content,
            metadata=metadata or {},
            position=(start_idx, end_idx)
        )


class DocumentProcessor:
    """Processes documents into chunks for RAG storage."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    async def process_document(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Dict]:
        """
        Split document into chunks and prepare for storage.
        
        Args:
            text: Full document text
            metadata: Document-level metadata
        
        Returns:
            List of chunk dictionaries ready for vector store
        """
        chunks = self._chunk_text(text)
        
        result_chunks = []
        for i, (start, end) in enumerate(chunks):
            chunk_meta = dict(metadata or {})
            chunk_meta['position'] = {'start': start, 'end': end}
            chunk_meta['chunk_index'] = i
            
            # Add deterministic mock embedding based on content hash
            embedding = self._generate_mock_embedding(text[start:end])
            
            result_chunk = {
                'id': f"doc_{len(result_chunks)}",
                'content': text[start:end],
                'embedding': embedding,
                'metadata': chunk_meta
            }
            
            result_chunks.append(result_chunk)
        
        return result_chunks
    
    def _chunk_text(self, text: str) -> List[Tuple[int, int]]:
        """Split text into overlapping chunks."""
        if len(text) <= self.config.chunk_size:
            return [(0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            
            # Handle overlap for non-final chunk
            if end < len(text):
                overlap_end = max(
                    end - self.config.overlap, 
                    start
                )
                chunks.append((start, overlap_end))
                start = overlap_end
            else:
                chunks.append((start, end))
                break
        
        return chunks
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embeddings."""
        import hashlib
        import numpy as np
        
        # Create hash-based embedding for testing
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        dim = self.config.embedding_dim
        return [float((hash_val >> i) & 1) - 0.5 for i in range(dim)]


class QueryProcessor:
    """Processes queries and retrieves relevant chunks."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    async def process_query(
        self, 
        query_text: str, 
        vector_store,
        embedding_model=None
    ) -> List[Dict]:
        """
        Process a query and retrieve relevant chunks.
        
        Args:
            query_text: User query string
            vector_store: Vector store instance
            embedding_model: Optional custom embedder
        
        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        if embedding_model is None:
            from embeddings import get_embedder
            embedder = await get_embedder("mock")
        else:
            embedder = embedding_model
        
        query_embedding = await embedder.embed_text(query_text)
        
        # Search vector store
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
            metadata_filter=None
        )
        
        # Filter by similarity threshold
        filtered_results = [
            (chunk_id, score) 
            for chunk_id, score in results 
            if score >= self.config.similarity_threshold
        ]
        
        # Fetch full chunks with context
        final_chunks = []
        for chunk_id, score in filtered_results:
            chunk_data = vector_store.get_chunk(chunk_id)
            
            if chunk_data and chunk_data.content:
                final_chunks.append({
                    'chunk': chunk_data.content,
                    'metadata': chunk_data.metadata,
                    'score': round(score, 4),
                    'context_window': self._get_context(
                        chunk_data.content, 
                        chunk_data.position
                    ) if hasattr(chunk_data, 'position') else None
                })
        
        return final_chunks
    
    def _get_context(self, text: str, position: Tuple[int, int]) -> Dict[str, str]:
        """Extract context window around the matched chunk."""
        start, end = position
        
        # Expand context window (simplified - in production would have actual doc)
        return {
            'start_position': start,
            'end_position': end,
            'length': end - start
        }


async def process_query_pipeline(
    query: str, 
    documents: List[Dict], 
    config: ProcessingConfig = None
) -> List[Dict]:
    """
    Full pipeline: document processing + query retrieval.
    
    Args:
        query: User question
        documents: List of raw document texts with metadata
        config: Optional processing configuration
    
    Returns:
        Retrieved chunks with scores and context
    """
    from storage.vector_store import VectorStore
    
    # Initialize components
    processor = DocumentProcessor(config)
    vector_store = VectorStore()
    
    # Step 1: Process documents into chunks
    print("[Pipeline] Processing documents...")
    all_chunks = []
    
    for doc in documents:
        text = doc.get('content', '') or doc.get('text', '')
        metadata = doc.get('metadata', {})
        
        if text:
            chunks = await processor.process_document(text, metadata)
            vector_store.add_chunks(chunks)
            all_chunks.extend(chunks)
    
    print(f"[Pipeline] Processed {len(all_chunks)} chunks")
    
    # Step 2: Query and retrieve
    query_processor = QueryProcessor(config)
    results = await query_processor.process_query(
        query, 
        vector_store
    )
    
    return results


def main():
    """Test RAG pipeline."""
    import asyncio
    
    async def run_test():
        # Sample documents (Russian content as per requirement)
        documents = [
            {
                'content': '''
                Искусственный интеллект — это раздел информатики, 
                занимающийся созданием систем, способных выполнять задачи, 
                требующие интеллекта. Машиное обучение и глубокое обучение 
                являются ключевыми областями ИИ.
                ''',
                'metadata': {'source': 'test_russian'}
            },
            {
                'content': '''
                Нейронные сети — это вычислительные модели, 
                вдохновленные работой мозга. Они состоят из множества узлов, 
                связанных весами, которые обучаются на данных.
                ''',
                'metadata': {'source': 'test_neural'}
            }
        ]
        
        query = "Что такое нейронные сети?"  # What are neural networks?
        
        print("=" * 60)
        print("Testing RAG Pipeline")
        print("=" * 60)
        
        results = await process_query_pipeline(query, documents)
        
        print(f"\n[Query: {query}]")
        print(f"Found {len(results)} relevant chunks:")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (score: {result['score']}) ---")
            print(result['chunk'])


    asyncio.run(run_test())


if __name__ == "__main__":
    main()
