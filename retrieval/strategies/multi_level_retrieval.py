"""
Multi-Level Retrieval Strategy (F-10)
Implements 3-tier retrieval pipeline with hybrid search and reranking.

Level 1: Keyword + Vector Search (Hybrid)
Level 2: Cross-Encoder Reranking  
Level 3: Final Selection with Relevance Threshold
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results at each level."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "scores": self.scores,
            "metadata": self.metadata
        }


class MultiLevelRetrievalStrategy:
    """
    Implements multi-level retrieval with hybrid search and reranking.
    
    F-10 FIX: Combines keyword matching, semantic similarity, and cross-encoder
    reranking for optimal document selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "retrieval_levels": {
                "level_1": {  # Keyword + basic vector search
                    "enabled": True,
                    "keywords_threshold": 2,
                    "vector_top_k": 50,
                    "method": "hybrid"
                },
                "level_2": {  # Reranking stage
                    "enabled": True,
                    "reranker_model": "cross_encoder_bge",
                    "top_k": 10
                },
                "level_3": {  # Final selection
                    "enabled": True,
                    "final_top_k": 5
                }
            },
            "relevance_threshold": 0.7,
            "max_documents": 10
        }
        
        self._metrics = {
            "total_queries": 0,
            "documents_retrieved_level_1": 0,
            "documents_reranked_level_2": 0,
            "documents_selected_final": 0
        }
    
    def execute_retrieval(self, query: str) -> Dict[str, Any]:
        """Execute multi-level retrieval pipeline."""
        
        logger.info(f"Starting multi-level retrieval for query: {query[:50]}...")
        
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Level 1: Initial retrieval (keyword + vector)
        if self.config["retrieval_levels"]["level_1"]["enabled"]:
            level_1_results = self._execute_level_1(query)
            logger.info(f"Level 1 retrieved {len(level_1_results)} candidates")
            results["level_1"] = [d.to_dict() for d in level_1_results]
        
        # Level 2: Reranking
        if (self.config["retrieval_levels"]["level_2"]["enabled"] and 
            len(results.get("level_1", [])) > 0):
            
            reranked_results = self._execute_reranking(
                query, results["level_1"], level_3=False
            )
            logger.info(f"Level 2 (rerank) selected {len(reranked_results)} documents")
            results["level_2"] = [d.to_dict() for d in reranked_results]
        
        # Level 3: Final selection
        if self.config["retrieval_levels"]["level_3"]["enabled"]:
            
            final_results = self._execute_final_selection(
                query, results.get("level_2", []), level_2=True
            )
            logger.info(f"Level 3 (final) selected {len(final_results)} documents")
            results["level_3"] = [d.to_dict() for d in final_results]
        
        return results
    
    def _execute_level_1(self, query: str) -> List[RetrievalResult]:
        """Execute Level 1: Keyword + Vector search."""
        
        # This would integrate with vector database (e.g., Pinecone, Weaviate)
        # For demo, return mock results
        
        logger.info("Executing Level 1: Hybrid keyword-vector search")
        
        # Simulate hybrid search combining BM25 and dense vectors
        mock_documents = [
            {
                "id": f"doc_{i}",
                "score": 0.85 - i * 0.1,
                "content": f"Document content for query about {query[:30]}...",
                "metadata": {"source": "vector_db", "timestamp": None}
            }
            for i in range(self.config["retrieval_levels"]["level_1"]["vector_top_k"])
        ]
        
        # Apply keyword filter (mock)
        keywords = query.lower().split()[:3]
        filtered_docs = []
        
        for doc in mock_documents:
            doc_content = doc["content"].lower()
            matching_keywords = sum(1 for kw in keywords if kw in doc_content)
            
            if matching_keywords >= self.config["retrieval_levels"]["level_1"]["keywords_threshold"]:
                filtered_docs.append(doc)
        
        logger.info(f"Level 1 keyword filter: {len(mock_documents)} -> {len(filtered_docs)} docs")
        
        # Sort by score (descending) and apply top-k limit
        sorted_docs = sorted(filtered_docs, key=lambda x: x["score"], reverse=True)
        level_1_results = [RetrievalResult(
            documents=sorted_docs[:self.config["retrieval_levels"]["level_1"]["vector_top_k"]],
            scores=[d["score"] for d in sorted_docs[:self.config["retrieval_levels"]["level_1"]["vector_top_k"]]],
            metadata={"level": "level_1", "method": "hybrid_search"}
        )]
        
        self._metrics["documents_retrieved_level_1"] += len(level_1_results[0].documents)
        
        return level_1_results
    
    def _execute_reranking(self, 
                          query: str, 
                          documents: List[Dict[str, Any]], 
                          level_3: bool = False) -> List[RetrievalResult]:
        """Execute Level 2/3: Cross-encoder reranking."""
        
        logger.info("Executing Level 2/3: Cross-encoder reranking")
        
        # In production, use cross-encoder model (e.g., BGE-Reranker)
        # For demo, simulate reranking with modified scores
        
        if not documents:
            return []
        
        # Simulate cross-encoder scoring
        reranked_docs = []
        for i, doc in enumerate(documents):
            # Mock re-ranking score (in production: use actual cross-encoder model)
            mock_rerank_score = 0.9 - (i * 0.05)
            
            reranked_docs.append({
                "id": doc["id"],
                "score": mock_rerank_score,
                "content": doc.get("content", ""),
                "metadata": {**doc.get("metadata", {}), "reranking_stage": level_3}
            })
        
        # Sort by rerank score (descending)
        sorted_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
        
        top_k = self.config["retrieval_levels"]["level_2"]["top_k"] if not level_3 else \
                self.config["retrieval_levels"]["level_3"]["final_top_k"]
        
        reranked_results = [RetrievalResult(
            documents=sorted_docs[:top_k],
            scores=[d["score"] for d in sorted_docs[:top_k]],
            metadata={"level": "level_2" if not level_3 else "level_3", 
                     "method": "cross_encoder_reranking"}
        )]
        
        self._metrics["documents_reranked_level_2"] += len(reranked_results[0].documents)
        
        return reranked_results
    
    def _execute_final_selection(self, 
                                query: str, 
                                documents: List[Dict[str, Any]], 
                                level_2: bool = True) -> List[RetrievalResult]:
        """Execute Level 3: Final selection with relevance threshold."""
        
        logger.info("Executing Level 3: Final selection with threshold filtering")
        
        if not documents:
            return []
        
        # Filter by relevance threshold
        filtered_docs = [d for d in documents if d.get("score", 0) >= self.config["relevance_threshold"]]
        
        # Apply max documents limit
        final_docs = filtered_docs[:self.config["max_documents"]]
        
        logger.info(f"Final selection: {len(documents)} -> {len(filtered_docs)} (threshold) -> {len(final_docs)} (final)")
        
        result = RetrievalResult(
            documents=final_docs,
            scores=[d.get("score", 0) for d in final_docs],
            metadata={"level": "level_3", 
                     "method": "relevance_threshold_filtering",
                     "threshold_applied": self.config["relevance_threshold"]}
        )
        
        self._metrics["documents_selected_final"] += len(result.documents)
        
        return [result]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        
        total_queries = self._metrics.get("total_queries", 1)
        
        return {
            **self._metrics,
            "average_documents_per_query": (
                self._metrics["documents_retrieved_level_1"] / total_queries
            ) if total_queries > 0 else 0
        }


if __name__ == "__main__":
    import json
    
    # Test multi-level retrieval
    strategy = MultiLevelRetrievalStrategy({
        "relevance_threshold": 0.6,
        "max_documents": 5
    })
    
    test_query = "What is the latest news about SpaceX and Starship?"
    
    print(f"Query: {test_query}")
    print("=" * 60)
    
    results = strategy.execute_retrieval(test_query)
    
    for level, docs in results.items():
        print(f"\n{level.upper()} RESULTS ({len(docs)} documents):")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  [{i}] Score: {doc['scores'][0]:.4f}")
            print(f"      ID: {doc['documents'][0]['id']}")
            print(f"      Content preview: {doc['documents'][0]['content'][:80]}...")
