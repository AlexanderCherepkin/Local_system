"""Response Formatting and Serialization."""
import json
from typing import Dict, Any, Optional

class OutputFormatter:
    """Formats pipeline results into various output formats."""
    
    def __init__(self):
        self.default_format = 'json'
    
    def format(self, result: Dict[str, Any], 
               output_format: str = None) -> str:
        """Format the RAG response according to specified format."""
        
        if output_format is None:
            output_format = self.default_format
        
        formatters = {
            'json': self._format_json,
            'text': self._format_text,
            'markdown': self._format_markdown,
            'compact_json': self._format_compact_json
        }
        
        formatter = formatters.get(output_format.lower(), self._format_json)
        return formatter(result)
    
    def _format_json(self, result: Dict[str, Any]) -> str:
        """Format as JSON with indentation."""
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def _format_text(self, result: Dict[str, Any]) -> str:
        """Format as plain text summary."""
        if not result.get('retrieved_docs'):
            return "No relevant information found."
        
        lines = ["=" * 60]
        lines.append("SECURE RAG RESPONSE")
        lines.append("=" * 60)
        lines.append(f"Query ID: {result.get('query_id', 'N/A')}")
        lines.append(f"Status: {'Success' if result.get('status') == 'success' else 'Failed'}")
        lines.append("")
        
        for doc in result.get('retrieved_docs', []):
            lines.append(f"[{doc.get('id', 'unknown')}]")
            content = doc.get('content', '')[:500]  # Truncate long content
            if len(content) < 500:
                content += "..."
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_markdown(self, result: Dict[str, Any]) -> str:
        """Format as Markdown."""
        if not result.get('retrieved_docs'):
            return "## No relevant information found."
        
        md = ["# RAG Response", ""]
        md.append(f"**Query ID:** {result.get('query_id', 'N/A')}")
        md.append(f"**Status:** {'✓ Success' if result.get('status') == 'success' else '✗ Failed'}")
        md.append(f"**Processing Time:** {result.get('processing_time_ms', 0)}ms")
        md.append("")
        md.append("## Retrieved Documents")
        md.append("")
        
        for doc in result.get('retrieved_docs', []):
            md.append(f"### [{doc.get('id', 'unknown')}]")
            content = doc.get('content', '')[:500]
            if len(content) < 500:
                content += "..."
            
            # Simple markdown formatting for key terms
            formatted_content = self._format_markdown_inline(content)
            md.append(formatted_content)
            md.append("")
        
        return "\n".join(md)
    
    def _format_compact_json(self, result: Dict[str, Any]) -> str:
        """Format as compact JSON (no whitespace)."""
        return json.dumps(result, ensure_ascii=False)
    
    def _format_markdown_inline(self, text: str) -> str:
        """Apply inline markdown formatting to text."""
        
        # Highlight key terms in bold
        import re
        words = text.split()
        formatted_words = []
        
        for word in words:
            cleaned_word = word.strip('.,!?;:')
            if len(cleaned_word) > 3 and not any(char.isupper() for char in cleaned_word[:1]):
                # Potential key term - bold it
                formatted_words.append(f"**{cleaned_word}**")
            else:
                formatted_words.append(word)
        
        return ' '.join(formatted_words)
