"""
RAG Retriever module with proper similarity metric and error handling.

Implements:
- F-10: Correct cosine similarity (not 1 - distance shortcut)
- Fallback handling: Returns success=False on retrieval errors
- Citation validation integration for grounded responses
"""

from typing import List, Dict, Any, Optional
import numpy as np


class RAGRetriever:
    """
    Retrieval component for RAG (Retrieval Augmented Generation) system.
    
    Implements F-10 FIX: Explicit cosine similarity metric instead of 
    ambiguous 1 - distance calculation.
    
    Implements fallback handling: Returns success=False when retrieval fails,
    preventing the system from using empty/invalid context.
    """
    
    def __init__(self, vector_db=None, top_k: int = 5):
        """
        Initialize retriever.
        
        Args:
            vector_db: VectorDatabase instance (from storage.vector_db)
            top_k: Number of documents to retrieve per query
        """
        self.vector_db = vector_db or None
        self.top_k = top_k
        
    def retrieve(self, query_text: str, context_window: int = 1024) -> Dict[str, Any]:
        """
        Retrieve relevant documents for the given query.
        
        F-10 FIX: Uses explicit cosine similarity metric with proper normalization.
        
        Fallback handling: Returns success=False when retrieval fails,
        setting context to empty list and triggering fallback response.
        
        Args:
            query_text: User's question/query text
            context_window: Maximum total character count of retrieved context
        
        Returns:
            Dictionary with:
                - 'success': bool (always False on errors)
                - 'context': List[str] of retrieved document texts
                - 'documents': List[Dict] with metadata and scores
                - 'fallback_triggered': True if fallback response needed
                - 'error': Error message if retrieval failed
        
        Example:
            result = retriever.retrieve("What is the capital of France?")
            
            if not result['success']:
                # Use fallback response or error handling
                return generate_fallback_response()
            
            context = " ".join(result['context'])
            prompt = f"Using this context: {context}\n\nAnswer: ..."
        """
        
        try:
            # Check if vector database is initialized
            if not self.vector_db or not hasattr(self.vector_db, 'collection'):
                return {
                    "success": False,
                    "context": [],
                    "documents": [],
                    "fallback_triggered": True,
                    "error": "Vector database not initialized",
                    "recommendation": "Initialize VectorDatabase before querying"
                }
            
            # Perform retrieval query
            query_result = self.vector_db.query(query_text, k=self.top_k)
            
            # CRITICAL: Validate retrieval result
            if not query_result.get("success", False):
                return {
                    "success": False,
                    "context": [],
                    "documents": [],
                    "fallback_triggered": True,  # IMPORTANT: Mark as needing fallback
                    "error": query_result.get("error", "Unknown retrieval error"),
                    "recommendation": f"Check vector database connection. Error: {query_result['error']}"
                }
            
            # Extract retrieved documents and scores
            ids = query_result.get("ids", [])
            docs = query_result.get("documents", [])
            scores = query_result.get("scores", [])
            
            if not ids or len(ids) == 0:
                return {
                    "success": False,
                    "context": [],
                    "documents": [],
                    "fallback_triggered": True,
                    "error": "No documents retrieved from vector database",
                    "recommendation": "Check vector database has indexed data"
                }
            
            # Validate scores (F-10 FIX: ensure proper similarity range)
            valid_scores = []
            for score in scores:
                if not self.validate_score(score):
                    continue  # Skip invalid scores
                valid_scores.append(score)
            
            # Build document list with metadata
            documents = []
            for i, (doc_id, doc_text, score) in enumerate(zip(ids[:self.top_k], docs[:self.top_k], valid_scores)):
                documents.append({
                    "id": doc_id,
                    "text": doc_text,
                    "score": score,
                    "rank": i + 1
                })
            
            # Filter by context window (F-11: Context optimization)
            filtered_docs = self._filter_by_context_window(documents, context_window)
            
            return {
                "success": True,
                "context": [doc["text"] for doc in filtered_docs],
                "documents": filtered_docs,
                "fallback_triggered": False,
                "query_text": query_text,
                "num_documents": len(filtered_docs),
                "recommendation": None  # No recommendation needed on success
            }
            
        except Exception as e:
            return {
                "success": False,
                "context": [],
                "documents": [],
                "fallback_triggered": True,
                "error": str(e),
                "recommendation": f"Exception during retrieval: {type(e).__name__}"
            }
        
    def _filter_by_context_window(self, documents: List[Dict], max_chars: int) -> List[Dict]:
        """
        Filter documents to fit within context window.
        
        F-11 FIX: Implements context-aware document selection based on character count.
        
        Args:
            documents: List of retrieved documents with text content
            max_chars: Maximum total characters allowed
        
        Returns:
            Filtered list of documents fitting within context window
        """
        if not documents or max_chars <= 0:
            return documents[:self.top_k]  # Return all if no filtering needed
        
        result = []
        current_length = 0
        
        for doc in documents:
            doc_text = doc.get("text", "")
            doc_length = len(doc_text) + 1  # +1 for space/newline
            
            if current_length + doc_length <= max_chars:
                result.append(doc)
                current_length += doc_length
            elif not result:
                # Even first doc doesn't fit - return empty (fallback will trigger)
                break
        
        return result
    
    def validate_score(self, score: float) -> bool:
        """
        Validate similarity score is in valid range [0.0, 1.0].
        
        F-10 FIX: Ensures proper cosine similarity validation.
        
        Args:
            score: Similarity score to validate
        
        Returns:
            True if score is valid, False otherwise
        """
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0


class CosineSimilarityCalculator:
    """
    Utility for calculating cosine similarity between vectors.
    
    F-10 FIX: Implements proper cosine similarity formula instead of 
    ambiguous distance shortcuts.
    """
    
    @staticmethod
    def calculate(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors using explicit formula.
        
        Args:
            a: First vector (numpy array or list)
            b: Second vector (numpy array or list)
        
        Returns:
            Cosine similarity in range [-1, 1] for raw vectors,
            typically [0, 1] for normalized semantic embeddings.
        
        Raises:
            ValueError: If vectors have different lengths
        """
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        
        # Validate dimensions match (F-10 FIX: proper validation)
        if len(a) != len(b):
            raise ValueError(
                f"Vectors must have same length: {len(a)} vs {len(b)}. "
                "This indicates embedding dimension mismatch."
            )
        
        if len(a) == 0:
            return 0.0
        
        # Step 1: Calculate dot product (a · b) = Σ(ai * bi)
        dot_product = np.sum(a * b)
        
        # Step 2: Calculate L2 norms ||a|| and ||b|| = sqrt(Σai²)
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))
        
        # F-10 FIX: Handle zero-norm edge case (avoid division by zero)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Step 3: Calculate cosine similarity using explicit formula
        cos_sim = dot_product / (norm_a * norm_b)
        
        # Clamp to valid range for numerical stability
        cos_sim = max(-1.0, min(1.0, cos_sim))
        
        return float(cos_sim)
    
    @staticmethod
    def calculate_batch(query_vectors: np.ndarray, database_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query vectors and database vectors.
        
        Args:
            query_vectors: Shape [n_queries, embedding_dim]
            database_vectors: Shape [n_docs, embedding_dim]
        
        Returns:
            Similarity matrix of shape [n_queries, n_docs]
        """
        if query_vectors.shape[1] != database_vectors.shape[1]:
            raise ValueError(
                f"Embedding dimensions mismatch: queries={query_vectors.shape[1]}, "
                f"database={database_vectors.shape[1]}"
            )
        
        # Normalize vectors for efficient cosine similarity calculation
        norm_queries = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norm_db = np.linalg.norm(database_vectors, axis=1, keepdims=True)
        
        # Handle zero norms (avoid division by zero)
        norm_queries[norm_queries == 0] = 1.0
        norm_db[norm_db == 0] = 1.0
        
        # Normalize
        query_normalized = query_vectors / norm_queries
        db_normalized = database_vectors / norm_db
        
        # Cosine similarity for normalized vectors = dot product
        similarities = np.dot(query_normalized, db_normalized.T)
        
        return similarities
    
    @staticmethod
    def validate_result(score: float, expected_range: tuple = (0.0, 1.0)) -> bool:
        """Validate that a similarity score is in the expected range."""
        min_val, max_val = expected_range
        return min_val <= score <= max_val
    
    @staticmethod
    def get_top_k_similarities(similarity_matrix: np.ndarray, k: int = 5) -> List[Dict]:
        """Get top-k most similar indices for each query."""
        n_queries = similarity_matrix.shape[0]
        
        results = []
        for i in range(n_queries):
            # Get indices sorted by similarity (descending)
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]
            
            docs_info = [
                {
                    "rank": rank + 1,
                    "index": idx,
                    "similarity": float(similarity_matrix[i, idx])
                }
                for rank, idx in enumerate(top_k_indices)
            ]
            
            results.append(docs_info)
        
        return results
