import httpx
from markdownify import markdownify as md_to_markdown
from typing import List, Dict, Any
from .search.providers import get_search_provider

class WebKnowledgeEngine:
    """Handles fetching, cleaning, and processing of web content for permanent indexing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_provider = get_search_provider(config)
        self.client = httpx.Client(timeout=10.0, follow_redirects=True)

    async def fetch_and_index_web_content(self, query: str) -> List[Dict[str, Any]]:
        """Main entry point to search, fetch, and prepare web content for storage."""
        print(f"[WEB_ENGINE] Searching for: {query}")
        search_results = self.search_provider.search(query, max_results=self.config.get("max_web_results", 5))
        
        indexed_chunks = []
        for result in search_results:
            url = result.get("url") or result.get("link")
            if not url:
                continue
                
            try:
                # Fetch content
                response = self.client.get(url)
                response.raise_for_status()
                html_content = response.text
                
                # Clean and convert to markdown
                clean_markdown = md_to_markdown(html_content, strip=['script', 'style', 'nav', 'footer'])
                
                # Security Check (Placeholder for secure_rag/security calls)
                # In production: if not security.validate(clean_markdown): continue
                
                # Generate unique identifier and metadata
                from .embeddings import content_hash # Assuming this exists in embeddings.py
                content_hash_val = content_hash(clean_markdown)
                
                # Check for duplicates (Logic to be integrated with vector_store)
                # if not vector_store.exists(content_hash_val):
                #     chunks = self._chunk_content(clean_markdown)
                #     for chunk in chunks:
                #         chunk['metadata'].update({
                #             'source_type': 'web',
                #             'source_url': url,
                #             'fetched_at': datetime.utcnow().isoformat(),
                #             'content_hash': content_hash_val
                #         })
                #         vector_store.upsert(chunk)
                
                print(f"[WEB_ENGINE] Successfully processed: {url}")
            except Exception as e:
                print(f"[WEB_ENGINE] Failed to process {url}: {e}")
                continue
                
        return indexed_chunks

    def _chunk_content(self, text: str) -> List[Dict[str, Any]]:
        # Placeholder for chunking logic (should use secure_rag/pipeline/chunker.py)
        from .chunker import ChunkManager
        return ChunkManager.chunk(text)
