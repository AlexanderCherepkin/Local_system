from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx

class BaseSearchProvider(ABC):
    """Abstract base class for all search providers."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search and return a list of results."""
        pass

class DuckDuckGoProvider(BaseSearchProvider):
    """Free search provider using duckduckgo-search."""
    def __init__(self):
        # Note: Requires 'duckduckgo-search' library installed
        import duckduckgo_search as ddgs
        self.ddgs = ddgs

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            # Using the new DDGS context manager or direct call depending on version
            results = self.ddgs.duck()
            return results
        except Exception as e:
            print(f"Error using DuckDuckGo search: {e}")
            return []

class TavilyProvider(BaseSearchProvider):
    """Paid search provider using Tavily API."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("Tavily API key is required.")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"query": query, "max_results": max_results}
        
        try:
            with httpx.Client() as client:
                response = client.post(self.base_url, json=data, headers=headers)
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception as e:
            print(f"Error using Tavily search: {e}")
            return []

def get_search_provider(config: Dict[str, Any]) -> BaseSearchProvider:
    """Factory to return the correct search provider based on configuration."""
    provider_type = config.get("search_provider", "duckduckgo").lower()
    
    if provider_type == "tavily":
        api_key = config.get("tavily_api_key")
        return TavilyProvider(api_key=api_key)
    else:
        return DuckDuckGoProvider()
