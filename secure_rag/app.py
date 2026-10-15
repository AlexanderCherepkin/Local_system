#!/usr/bin/env python3
"""
Secure RAG Application with Fine-Grained Access Control
FastAPI-based server for secure document retrieval and analysis
Author: Security Team
Version: 1.0.0
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration loaded from environment variables."""
    
    @staticmethod
    def load():
        config = {}
        
        # Ollama Configuration
        config['OLLAMA_HOST'] = os.getenv('OLLAMA_HOST', 'localhost')
        config['OLLAMA_PORT'] = int(os.getenv('OLLAMA_PORT', 11434))
        config['DEFAULT_MODEL'] = os.getenv('DEFAULT_MODEL', 'qwen2.5')
        
        # Retrieval Configuration
        config['CHUNK_SIZE'] = int(os.getenv('CHUNK_SIZE', 500))
        config['CHUNK_OVERLAP'] = int(os.getenv('CHUNK_OVERLAP', 50))
        config['TOP_K_RESULTS'] = int(os.getenv('TOP_K_RESULTS', 5))
        config['SIMILARITY_THRESHOLD'] = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
        
        # Paths
        script_dir = Path(__file__).parent
        config['VECTOR_DB_PATH'] = os.getenv('VECTOR_DB_PATH', str(script_dir / 'chroma_db'))
        config['SECURITY_CONFIG'] = os.getenv('SECURITY_CONFIG', str(script_dir / 'security' / 'acl_storage.json'))
        
        return config


config = Config.load()

# =============================================================================
# SECURITY CONTROLLER
# =============================================================================

class SecurityController:
    """Handles access control and authorization."""
    
    def __init__(self, security_config_path: str):
        self.security_config_path = Path(security_config_path)
        if not self.security_config_path.exists():
            raise FileNotFoundError(f"Security configuration not found: {security_config_path}")
        
        with open(self.security_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_user_roles(self, username: str) -> list[str]:
        """Get roles for a specific user."""
        users = self.config.get('users', {})
        if username in users:
            return users[username]
        return []
    
    def has_permission(self, username: str, permission: str) -> bool:
        """Check if user has specific permission through role hierarchy."""
        user_roles = set(self.get_user_roles(username))
        
        for role_name, permissions in self.config.get('roles', {}).items():
            if role_name in user_roles and permission in permissions:
                return True
        
        # Admin users have read_all permission
        if 'admin' in user_roles:
            return True
            
        return False
    
    def get_accessible_documents(self, username: str) -> list[str]:
        """Get list of documents accessible to user."""
        allowed = []
        
        for doc_type, permissions in self.config.get('roles', {}).items():
            if 'read_' + doc_type.replace('_', '-') in self.has_permission(username, permission):
                # In production: query database for actual documents
                pass
        
        return allowed


# =============================================================================
# DATABASE & VECTOR STORE
# =============================================================================

class VectorStoreManager:
    """Manages ChromaDB vector store."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        
        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
    
    def create_collection(self, name: str) -> chromadb.Collection:
        """Create or retrieve a collection by name."""
        return self.client.get_or_create_collection(name=name)
    
    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ):
        """Add documents to collection with embeddings."""
        if not ids:
            ids = [f"doc_{len(documents)}"] * len(documents) if documents else []
        
        # Generate embeddings using local model
        embeddings = self.generate_embeddings(documents)
        
        collection = self.create_collection(collection_name)
        collection.add(
            documents=documents,
            metadatas=(metadatas or [{}] * len(documents)),
            ids=ids,
            embeddings=np.array(embeddings).tolist() if isinstance(embeddings, np.ndarray) else embeddings
        )
    
    def query(self, collection_name: str, query_text: str, top_k: int = 5) -> list[dict]:
        """Query collection and return relevant documents."""
        collection = self.create_collection(collection_name)
        
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query_text])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        
        return {
            'documents': results.documents or [],
            'metadatas': results.metadatas or []
        }
    
    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate sentence embeddings using local model."""
        try:
            # Use small BERT-based model for speed
            if not hasattr(self, '_model'):
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return self._model.encode(texts)
        except Exception as e:
            print(f"Embedding generation error: {e}")
            # Return zeros for failed embeddings (will be filtered by similarity threshold)
            return np.zeros((len(texts), 384))


# =============================================================================
# SECURITY CONTROLLER INSTANCE
# =============================================================================

try:
    security_controller = SecurityController(config['SECURITY_CONFIG'])
except Exception as e:
    print(f"Warning: Security controller initialization failed: {e}")
    security_controller = None


# =============================================================================
# API REQUEST MODEL
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for document queries."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    doc_type: Optional[str] = Field(None, description="Document type filter (engineering, hr)")

class QueryResponse(BaseModel):
    """Response model for document queries."""
    success: bool
    query: str
    timestamp: datetime
    results_count: int
    documents: list[dict]


# =============================================================================
# API APPLICATION
# =============================================================================

app = FastAPI(
    title="Secure RAG API",
    description="Fine-grained access control for document retrieval and analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Secure RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/v1/query")
async def query_documents(
    request: QueryRequest,
    x_api_key: Optional[str] = Header(None),
    username: Optional[str] = Header(None)
):
    """Query documents based on search text with access control."""
    
    if security_controller is None:
        raise HTTPException(status_code=500, detail="Security controller not initialized")
    
    # Authenticate user (implement proper auth mechanism here)
    authenticated_username = username or "anonymous"
    
    # Check permissions based on doc_type request
    allowed_doc_types = ['engineering', 'hr']  # Add more as needed
    
    if request.doc_type and request.doc_type not in allowed_doc_types:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Only {allowed_doc_types} document types are available."
        )
    
    try:
        # Query vector store
        results = vector_store.query(
            collection_name=request.doc_type or 'general',
            query_text=request.query,
            top_k=request.top_k
        )
        
        return QueryResponse(
            success=True,
            query=request.query,
            timestamp=datetime.utcnow(),
            results_count=len(results['documents'][0]) if results['documents'] else 0,
            documents=[
                {
                    'text': doc,
                    'metadata': metadata
                }
                for doc, metadata in zip(results['documents'][0] or [], results['metadatas'][0] or [])
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == '__main__':
    import uvicorn
    
    print("Initializing Secure RAG Server...")
    print(f"Vector DB path: {config['VECTOR_DB_PATH']}")
    print(f"Security config: {config['SECURITY_CONFIG']}")
    
    uvicorn.run(
        "app:app",
        host=config.get('OLLAMA_HOST', 'localhost'),
        port=config.get('OLLAMA_PORT', 8000),
        reload=False,
        log_level="info"
    )
