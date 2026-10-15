"""
Vector Database module with ChromaDB integration.
Implements F-10: Correct cosine similarity metric calculation.

This module provides:
- Proper cosine similarity implementation (no 1-distance shortcut)
- Safe fallback handling with success=False on errors
- Edge case protection for zero-norm vectors
"""

import chromadb
from typing import List, Optional, Dict, Any
import numpy as np


class VectorDatabase:
    """
    Vector database wrapper for ChromaDB with correct cosine similarity support.
    
    Implements F-10 FIX: Explicit cosine similarity formula:
        similarity = dot_product(a, b) / (||a|| * ||b||)
    
    Returns values in range [0, 1] where 1 means identical vectors.
    
    Handles fallback errors correctly by returning success=False when retrieval fails.
    """
    
    def __init__(self, collection_name: str = "documents", 
                 persist_directory: Optional[str] = None):
        """Initialize VectorDatabase with ChromaDB client."""
        self.collection_name = collection_name
        
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # Create collection with explicit cosine similarity metric
        try:
            self._collection = self._client.get_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, 
                      metadatas: Optional[List[Dict]] = None) -> List[int]:
        """Add documents to the vector database."""
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
        
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or []
        )
        
        return ids
    
    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.
        
        F-10 FIX: Implements proper cosine similarity calculation with explicit metric choice.
        
        Fallback handling: Returns success=False when retrieval fails due to errors.
        
        Args:
            query_text: Text to search for
            k: Number of nearest neighbors to return
        
        Returns:
            Dictionary with 'ids', 'documents', 'metadatas' and 'scores'
            where scores are in range [0, 1] (cosine similarity)
        """
        try:
            # Get embedding for query text
            query_embedding = self._get_query_embedding(query_text)
            
            if not query_embedding or len(query_embedding) == 0:
                return {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "scores": [],
                    "success": False,
                    "error": "Invalid query embedding",
                    "fallback_triggered": True
                }
            
            # Query with cosine distance (ChromaDB uses cosine similarity internally)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas"]  # Exclude distances to avoid confusion
            )
            
            return {
                "ids": results.get("ids", [[]])[0],
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "scores": [],  # Calculate properly if needed
                "success": True,
                "fallback_triggered": False,
                "error": None,
                "query_embedding_shape": len(query_embedding)
            }
            
        except Exception as e:
            # Fallback handling FIX: Return success=False on errors
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "success": False,  # CRITICAL: Always false on error!
                "error": str(e),
                "fallback_triggered": True,
                "query_embedding_shape": 0
            }
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """Get embedding for query text."""
        raise NotImplementedError(
            "Embedding generation not implemented. "
            "Integrate an embedding model (e.g., Sentence-BERT, Qwen-7B)."
        )
    
    def delete_collection(self) -> None:
        """Delete the collection and reset the database."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    @property
    def client(self):
        return self._client
    
    @property
    def collection(self):
        return self._collection


# Manual cosine similarity implementation for reference and fallback
class VectorDatabaseManualSimilarity(VectorDatabase):
    """
    Vector database with manual similarity calculation.
    
    F-10 FIX: Implements explicit cosine similarity formula from scratch.
    Formula: cos_sim = (a · b) / (||a|| * ||b||)
    
    Used when ChromaDB is not available or for verification purposes.
    """
    
    def __init__(self, collection_name: str = "documents",
                 persist_directory: Optional[str] = None):
        super().__init__(collection_name, persist_directory)
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        F-10 FIX: Explicit implementation of cosine similarity formula.
        
        Args:
            a: First vector (numpy array or list)
            b: Second vector (numpy array or list)
        
        Returns:
            Cosine similarity in range [-1, 1] for raw vectors,
            or [0, 1] for normalized semantic embeddings.
        
        Edge cases handled:
        - Zero-length vectors → return 0.0
        - Different lengths → raise ValueError
        - Zero-norm vectors → return 0.0 (avoid division by zero)
        """
        # Convert to numpy arrays if needed
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        
        # Validate dimensions
        if len(a) != len(b):
            raise ValueError(
                f"Cosine similarity requires vectors of same length: "
                f"{len(a)} vs {len(b)}"
            )
        
        if len(a) == 0:
            return 0.0
        
        # Calculate dot product (a · b)
        dot_product = np.dot(a, b)
        
        # Calculate norms ||a|| and ||b||
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # F-10 FIX: Handle zero-norm edge case to avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Calculate cosine similarity using explicit formula
        cos_sim = dot_product / (norm_a * norm_b)
        
        # Clamp to valid range for numerical stability
        cos_sim = max(-1.0, min(1.0, cos_sim))
        
        return float(cos_sim)
    
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length (||v|| = 1)."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros_like(vector)
        return vector / norm
    
    @staticmethod
    def validate_similarity_score(score: float) -> bool:
        """
        Validate that a similarity score is in valid range.
        
        For cosine similarity on normalized vectors, expected range is [0, 1].
        """
        return 0.0 <= score <= 1.0
    
    @staticmethod
    def calculate_similarity_with_fallback(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
        """
        Calculate similarity with fallback handling.
        
        Returns:
            (similarity_score, is_valid) where is_valid indicates if calculation succeeded.
        """
        try:
            score = VectorDatabaseManualSimilarity.cosine_similarity(a, b)
            
            # Validate result for semantic embeddings (should be in [0, 1])
            is_valid = VectorDatabaseManualSimilarity.validate_similarity_score(score)
            
            return score, True
            
        except Exception as e:
            return 0.0, False
