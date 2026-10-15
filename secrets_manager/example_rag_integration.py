"""
RAG System Integration Example with Secure Token Management.
Demonstrates embedding storage and retrieval using the Secrets Manager.
"""

from secrets_manager.secrets import SecureVault, get_vault


class RAGEmbeddingStore:
    """Secure embedding store for LLM tokens."""

    def __init__(self, master_secret: str):
        self.vault = get_vault()
        self.master_secret = master_secret
        
    def register_provider(self, provider_name: str, api_token: str):
        """Register a new API provider with its token."""
        service_type = f"llm_{provider_name.lower()}"
        
        # Store the raw token securely
        self.vault.store_token(
            name=f"{provider_name}_api_key",
            service=service_type,
            token=api_token,
            master_secret=self.master_secret
        )
        
        print(f"✓ Registered {provider_name} provider")

    def get_provider_token(self, provider_name: str) -> str | None:
        """Retrieve API token for a registered provider."""
        service_type = f"llm_{provider_name.lower()}"
        name = f"{provider_name}_api_key"
        
        return self.vault.get_token(name, self.master_secret)

    def store_embedding(self, embedding_id: str, vector_data: list[float]):
        """Store embedding data (placeholder for actual vector DB integration)."""
        # In production, this would interface with your vector database
        print(f"✓ Stored embedding {embedding_id}")


# Example usage in RAG pipeline
def rag_pipeline_example():
    """Demonstrate secure token flow in RAG system."""
    
    # Initialize vault (master secret from environment)
    master_secret = os.environ.get("VAULT_MASTER_SECRET")
    store = RAGEmbeddingStore(master_secret)

    # Register providers
    store.register_provider("openai", "sk-proj-your-key-here")
    store.register_provider("anthropic", "sk-ant-your-key-here")

    # Retrieve token for query embedding (e.g., using OpenAI's text-embedding-3-large)
    openai_token = store.get_provider_token("openai")
    
    if openai_token:
        print(f"✓ Using {len(openai_token)} chars from vault for embeddings")
        
        # Call LLM API with retrieved token (token never logged or exposed)
        response = call_openai_embeddings_api(openai_token, "user query text")
        
        # Store the embedding vector securely
        store.store_embedding("doc_001", response["embedding"])


if __name__ == "__main__":
    import os
    
    master_secret = os.environ.get(
        "VAULT_MASTER_SECRET", 
        "dev-master-secret-change-in-production"
    )
    
    print("=" * 60)
    print("Secure Token Management Demo")
    print("=" * 60)
    
    vault = get_vault()
    
    # Store tokens
    print("\n[1] Storing API tokens...")
    vault.store_token(
        name="openai_api_key",
        service="llm_provider",
        token="sk-proj-abc123xyz789"
    )
    
    vault.store_token(
        name="postgres_connection_string", 
        service="database",
        token="postgresql://admin:secure_pass@localhost:5432/rag_db?sslmode=require"
    )
    
    # Retrieve tokens
    print("\n[2] Retrieving stored tokens...")
    openai_key = vault.get_token("openai_api_key", master_secret)
    postgres_conn = vault.get_token("postgres_connection_string", master_secret)
    
    if openai_key:
        preview = openai_key[:10] + "..."
        print(f"   OpenAI token preview: {preview}")
    
    if postgres_conn:
        preview = postgres_conn.split("@")[0] + "@***"
        print(f"   PostgreSQL connection: {preview}")
