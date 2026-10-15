"""Retriever Component with Hybrid Search Strategy."""
from typing import Dict, List, Any

class SecureRetriever:
    """Hybrid retriever combining dense vector search and keyword matching."""
    
    def __init__(self):
        self.vector_store = None  # Initialize in orchestrator
        
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute hybrid retrieval strategy."""
        
        # Step 1: Process and embed the query
        from pipeline.query_processor import QueryProcessor
        processor = QueryProcessor()
        
        processed_query = self._process_query(query)
        
        # Step 2: Generate embedding for query
        from pipeline.embedder import SecureEmbedder
        embedder = SecureEmbedder()
        query_embedding = embedder.generate_embedding(processed_query['normalized_query'])
        
        # Step 3: Retrieve using vector similarity (dense retrieval)
        if self.vector_store:
            vector_results = self._vector_search(query_embedding, top_k)
        else:
            vector_results = []
            
        # Step 4: Keyword-based retrieval (sparse retrieval)
        keyword_results = self._keyword_search(processed_query['key_terms'], top_k)
        
        # Step 5: Combine and rerank results
        combined_results = self._combine_and_rerank(
            vector_results, 
            keyword_results,
            processed_query
        )
        
        return {
            'original_query': query,
            'processed_query': processed_query['normalized_query'],
            'entities_found': processed_query.get('entities_found', []),
            'retrieved_chunks': combined_results[:top_k],
            'total_retrieved': len(combined_results)
        }
    
    def _process_query(self, query: str) -> Dict[str, Any]:
        """Normalize and prepare query for retrieval."""
        from pipeline.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        
        return {
            'original_query': query,
            'key_terms': processor._extract_key_terms(query.lower().split()),
            'normalized_query': ' '.join(processor._expand_terms(
                [t for t in query.lower().split() if len(t) > 2]
            ))
        }
    
    def _vector_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search."""
        # This would call the actual vector store implementation
        # Placeholder return value
        return []
    
    def _keyword_search(self, keywords: List[str], top_k: int = 5) -> List[Dict]:
        """Perform keyword-based retrieval (BM25 or similar)."""
        results = []
        
        for keyword in keywords[:3]:  # Search with top 3 keywords
            # Placeholder - would search actual document corpus
            mock_chunk = {
                'id': f"kw_{hash(keyword) % 10000:04d}",
                'text': f"Information about {keyword} and related security practices",
                'metadata': {'source': 'knowledge_base', 'keyword_match': keyword}
            }
            
            results.append(mock_chunk)
        
        return results[:top_k]
    
    def _combine_and_rerank(self, vector_results: List[Dict], 
                           keyword_results: List[Dict],
                           processed_query: Dict[str, Any]) -> List[Dict]:
        """Combine hybrid retrieval results and rerank by relevance."""
        
        # Create a combined score for each result (simplified)
        all_candidates = []
        
        # Add vector search results with weight
        for i, result in enumerate(vector_results[:3]):
            candidate = dict(result)
            candidate['_relevance_score'] = 0.6 * result.get('similarity_score', 0) + 0.4
            
            if '_keyword_match' not in candidate.get('metadata', {}):
                all_candidates.append(candidate)
        
        # Add keyword results with weight
        for i, result in enumerate(keyword_results[:3]):
            candidate = dict(result)
            candidate['_relevance_score'] = 0.4 + 0.6 * (1 - min(i / 3, 1))
            
            if '_keyword_match' not in candidate.get('metadata', {}):
                all_candidates.append(candidate)
        
        # Sort by relevance score and return unique results
        seen_ids = set()
        ranked_results = []
        
        for candidate in sorted(all_candidates, key=lambda x: x['_relevance_score'], reverse=True):
            chunk_id = candidate.get('id', 'unknown')
            
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                # Remove internal scoring field
                ranked_result = {k: v for k, v in candidate.items() if not k.startswith('_')}
                ranked_results.append(ranked_result)
        
        return ranked_results
