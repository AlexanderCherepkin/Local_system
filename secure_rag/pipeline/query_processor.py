"""Query Processing Service for Secure RAG System."""
import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ProcessedQuery:
    """Represents a processed query with extracted entities and intent."""
    original_query: str
    key_terms: List[str]
    intent_category: str
    entities: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_query': self.original_query,
            'key_terms': self.key_terms,
            'intent_category': self.intent_category,
            'entities': self.entities
        }

class QueryProcessor:
    """Processes and analyzes user queries for semantic understanding."""
    
    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            'informational': [
                r'\bwhat\b', r'\bhow to\b', r'\bdefine\b', r'\blet me know\b'
            ],
            'instructional': [
                r'\bshow\b', r'\bdemonstrate\b', r'\bwalk through\b', r'\bguide\b'
            ],
            'analytical': [
                r'\banalyze\b', r'\bcompare\b', r'\breview\b', r'\bevaluate\b'
            ]
        }
        
        # Named entity patterns
        self.entity_patterns = {
            'technology': r'\b(Windows|Linux|macOS|Python|JavaScript|SQL)\b',
            'security_term': r'\b(ransomware|malware|phishing|DDoS|firewall|encryption)\b',
            'programming_language': r'\b(Python|Java|C\+\+|Ruby|Go|Rust)\b'
        }
        
    def process(self, query: str) -> ProcessedQuery:
        """Process a user query and extract meaningful components."""
        
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        # Extract key terms (significant words)
        key_terms = self._extract_key_terms(cleaned_query)
        
        # Classify intent
        intent_category = self._classify_intent(cleaned_query)
        
        # Extract entities
        entities = self._extract_entities(cleaned_query, query)
        
        return ProcessedQuery(
            original_query=query,
            key_terms=key_terms,
            intent_category=intent_category,
            entities=entities
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the input query."""
        
        # Remove extra whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for processing (but keep original for entities)
        return cleaned
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract significant key terms from the query."""
        
        import re
        
        # Remove stop words and punctuation
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'have', 'has', 'had',
                      'do', 'does', 'did', 'will', 'would', 'could',
                      'may', 'might', 'should', 'must', 'to', 'for',
                      'of', 'in', 'on', 'at', 'by', 'with', 'from'}
        
        # Extract words and filter
        words = set(re.findall(r'\b\w+\b', query.lower()))
        key_terms = [word for word in words if word not in stop_words]
        
        # Keep top 5 most significant terms
        return list(key_terms)[:5]
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query."""
        
        import re
        
        for category, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return category
        
        return 'general'  # Default classification
    
    def _extract_entities(self, cleaned_query: str, original_query: str) -> List[Dict[str, str]]:
        """Extract named entities from the query."""
        
        import re
        
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, original_query, re.IGNORECASE)
            if matches:
                # Get position info
                match_positions = list(re.finditer(pattern, original_query))
                
                for i, match in enumerate(matches):
                    entities.append({
                        'type': entity_type,
                        'value': match.group(0),
                        'position': (match_positions[i].start(), match_positions[i].end()) if i < len(match_positions) else None
                    })
        
        return entities
