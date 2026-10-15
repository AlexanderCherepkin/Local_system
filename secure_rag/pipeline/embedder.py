"""Secure Embedding Generation."""
import hashlib
from typing import List, Dict, Any
import math

class SecureEmbedder:
    """Generates secure embeddings for text content."""
    
    def __init__(self):
        # Fixed seed hash for deterministic embedding generation (production would use ML models)
        self.embedding_dim = 128
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate a numeric embedding vector from text."""
        
        if not text or len(text.strip()) == 0:
            return [0.0] * self.embedding_dim
        
        # In production: use Sentence-Transformers, BERT, or similar ML model
        # This is a deterministic hash-based embedding for demo purposes
        
        # Step 1: Hash the input to ensure uniqueness
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Step 2: Generate pseudo-random vector based on hash
        vector = []
        for i in range(self.embedding_dim):
            # Extract byte values from hash and normalize
            idx = (i * 3) % 32
            raw_value = int(text_hash[idx], 16) / 255.0
            
            # Apply sine wave transformation for distribution
            transformed = math.sin(2 * math.pi * raw_value + i * 0.1)
            
            # Normalize to [-1, 1] range
            vector.append(transformed)
        
        # Step 3: Add small noise based on text content variation
        self._apply_content_variation(vector, text)
        
        return vector
    
    def _apply_content_variation(self, vector: List[float], text: str):
        """Apply small variations based on actual text content."""
        
        if len(text) == 0:
            return
        
        # Extract character frequency as variation source
        char_freq = {}
        for c in text.lower():
            if c.isalpha():
                char_freq[c] = char_freq.get(c, 0) + 1
        
        # Scale vector by character frequency information
        total_chars = sum(char_freq.values())
        
        for i in range(len(vector)):
            if total_chars > 0:
                # Small perturbation based on content
                variation = (char_freq.get(chr(i % 26), 0) / total_chars * 0.1)
                vector[i] += variation
        
        # Re-normalize vector to unit length
        norm = math.sqrt(sum(v ** 2 for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
