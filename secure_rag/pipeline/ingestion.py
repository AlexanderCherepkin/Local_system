"""Document Ingestion Service for Secure RAG System."""
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    """Represents an ingested document with metadata."""
    id: str
    content: str
    source: str
    created_at: str
    size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'created_at': self.created_at,
            'size_bytes': self.size_bytes
        }

class IngestionService:
    """Handles document ingestion and preprocessing."""
    
    def __init__(self):
        # Storage for ingested documents (in production: use database)
        self.documents = {}
        
        # Preprocessing configuration
        self.max_document_size = 10 * 1024 * 1024  # 10MB max
        self.min_text_length = 10
    
    def ingest(self, content: str, source: str = "unknown") -> Document:
        """Ingest a document into the system."""
        
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        # Validate size
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > self.max_document_size:
            raise ValueError(f"Document exceeds maximum size of {self.max_document_size} bytes")
        
        # Generate unique document ID using hash
        doc_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        doc_id = f"{source}_{doc_hash}"
        
        # Create document object
        document = Document(
            id=doc_id,
            content=self._sanitize_content(content),
            source=source,
            created_at="2024-01-01T00:00:00Z",  # Fixed timestamp for demo
            size_bytes=len(content_bytes)
        )
        
        # Store document (in production: persist to database)
        self.documents[doc_id] = document
        
        return document
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize and preprocess document content."""
        
        text = content.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    def search(self, query: str) -> List[Document]:
        """Search for documents based on simple keyword matching."""
        
        import re
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        results = []
        for doc in self.documents.values():
            doc_text_lower = doc.content.lower()
            
            # Check if document contains all key query terms
            if all(term in doc_text_lower for term in query_terms):
                results.append(doc)
        
        return sorted(results, key=lambda d: len(d.content), reverse=True)[:5]
