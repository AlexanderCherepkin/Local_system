#!/usr/bin/env python3
"""Pipeline Runner - Orchestration of RAG pipeline components."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("secure_rag.pipeline")


@dataclass
class PipelineResult:
    """Data class for pipeline execution results."""
    
    query: str
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class PipelineRunner:
    """Main pipeline orchestrator."""

    def __init__(self):
        logger.info("Pipeline Runner initialized")

    def run_pipeline(
        self,
        prompt: str,
        username: str = "anonymous",
        roles: List[str] = None
    ) -> PipelineResult:
        """Execute the complete RAG pipeline."""
        
        if roles is None:
            roles = [username]
        
        logger.info(f"Starting pipeline for user {username} with query length: {len(prompt)}")
        
        try:
            # Step 1: Query Analysis & Security Check
            security_result = self._check_security(prompt, username)
            
            if not security_result['allowed']:
                return PipelineResult(
                    query=prompt,
                    answer=None,
                    sources=[],
                    error=security_result['reason']
                )
            
        except Exception as e:
            logger.error(f"Security check failed: {str(e)}", exc_info=True)
            return PipelineResult(
                query=prompt,
                answer=None,
                sources=[],
                error=f"Pipeline execution error: {str(e)}"
            )

    def _check_security(self, prompt: str, username: str) -> Dict[str, Any]:
        """Perform security analysis on the query."""
        
        # Security checks (simplified implementation)
        blocked_terms = [
            "exploit", "vulnerability", "hack", "attack", 
            "malware", "trojan", "backdoor", "rootkit"
        ]
        
        for term in blocked_terms:
            if term.lower() in prompt.lower():
                logger.warning(f"Blocked term '{term}' detected in query")
                
                # Allow educational context with proper attribution
                educational_indicators = [
                    "educational", "learning", "study", "research", 
                    "security training", "vulnerability research"
                ]
                
                for indicator in educational_indicators:
                    if indicator.lower() in prompt.lower():
                        logger.info(f"Educational context detected, allowing query")
                        return {
                            'allowed': True,
                            'reason': None
                        }
                
                return {
                    'allowed': False,
                    'reason': f"Query blocked for security reasons: '{term}'"
                }
        
        logger.info(f"Security check passed for user {username}")
        return {'allowed': True, 'reason': None}

    def execute_pipeline(
        self,
        prompt: str,
        username: str = "anonymous",
        roles: List[str] = None
    ) -> PipelineResult:
        """Execute the full pipeline with all components."""
        
        if roles is None:
            roles = [username]

        logger.info(f"Executing pipeline for user {username}")
        
        # Step 2: Document Retrieval (placeholder)
        retrieved_docs = self._retrieve_documents(prompt, username)
        
        # Step 3: Context Processing (placeholder)
        context = self._process_context(retrieved_docs, prompt)
        
        # Step 4: Query Generation & Answering (placeholder)
        answer = self._generate_answer(context, prompt)
        
        # Step 5: Response Formatting (placeholder)
        formatted_response = self._format_response(answer, retrieved_docs)
        
        return PipelineResult(
            query=prompt,
            answer=formatted_response['answer'],
            sources=formatted_response['sources']
        )

    def _retrieve_documents(self, prompt: str, username: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query."""
        
        logger.info(f"Retrieving documents for query from user {username}")
        
        # Placeholder implementation - in production this would use vector search
        return [
            {
                'id': 1,
                'title': "Windows API Security Documentation",
                'content': "The Windows API provides various functions for system operations...",
                'score': 0.95,
                'username': username
            },
            {
                'id': 2,
                'title': "Security Best Practices",
                'content': "Follow security best practices when using Windows APIs...",
                'score': 0.87,
                'username': username
            }
        ]

    def _process_context(self, docs: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """Process retrieved documents into context for answering."""
        
        logger.info("Processing context from retrieved documents")
        
        # Placeholder implementation
        return {
            'documents': docs[:2],  # Top 2 documents
            'query': prompt
        }

    def _generate_answer(self, context: Dict[str, Any], prompt: str) -> Optional[str]:
        """Generate answer based on retrieved context."""
        
        logger.info("Generating answer from context")
        
        # Placeholder implementation - in production this would use LLM
        return f"This is a placeholder answer for the query about: {prompt[:50]}..."

    def _format_response(
        self, 
        answer: Optional[str], 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format the final response."""
        
        logger.info("Formatting response")
        
        # Format sources for display
        formatted_sources = []
        for source in sources[:3]:  # Top 3 sources
            formatted_sources.append({
                'title': source.get('title', 'Unknown'),
                'content_preview': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', ''),
                'score': source.get('score', 0),
                'id': source.get('id', -1)
            })
        
        return {
            'answer': answer,
            'sources': formatted_sources
        }


# Singleton instance for convenience
_pipeline_runner: Optional[PipelineRunner] = None

def get_pipeline_runner() -> PipelineRunner:
    """Get or create the singleton pipeline runner."""
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner()
    return _pipeline_runner


# Module-level function for API usage
def run_pipeline(
    prompt: str,
    username: str = "anonymous",
    roles: List[str] = None
) -> PipelineResult:
    """Convenience function to run the pipeline."""
    
    runner = get_pipeline_runner()
    return runner.run_pipeline(prompt, username, roles)
