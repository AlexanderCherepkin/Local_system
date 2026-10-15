"""
Fallback Handler for RAG responses.

Implements F-12: Handles cases where retrieval fails or citations are invalid.
Provides graceful degradation strategies when grounding is not possible.
"""

from typing import Dict, Any, List, Optional
import logging


logger = logging.getLogger(__name__)


class FallbackResponseStrategy:
    """
    Defines different fallback strategies for handling failed retrievals.
    
    F-12 FIX: Implements multiple fallback mechanisms when retrieval fails or
    citations cannot be validated against retrieved documents.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            "fallback_responses": {
                "retrieval_failure": None,  # Custom message override
                "citation_invalid": None,
                "empty_context": None,
                "out_of_scope": None
            },
            "graceful_degradation_enabled": True,
            "fallback_chain": [
                "direct_answer",
                "partial_retrieval", 
                "knowledge_based_response",
                "polite_decline"
            ]
        }
    
    def select_fallback_strategy(self, failure_type: str) -> Optional[Dict[str, Any]]:
        """Select appropriate fallback strategy based on failure type."""
        
        strategies = {
            "retrieval_failure": self._build_retrieval_failure_strategy(),
            "citation_invalid": self._build_citation_invalid_strategy(),
            "empty_context": self._build_empty_context_strategy(),
            "out_of_scope": self._build_out_of_scope_strategy()
        }
        
        return strategies.get(failure_type)
    
    def handle_retrieval_failure(self, query: str, original_query_text: str = None) -> Dict[str, Any]:
        """Handle case where retrieval completely fails."""
        
        strategy = self.select_fallback_strategy("retrieval_failure") or {}
        response_config = self.config["fallback_responses"]["retrieval_failure"]
        
        # Try custom message first, then fallback to standard strategies
        if response_config:
            return {
                "success": False,
                "response_type": "custom_fallback",
                "text": response_config.format(query=query) if isinstance(response_config, str) else response_config,
                "fallback_used": True,
                "strategy_applied": strategy.get("name"),
                "original_query": original_query_text or query
            }
        
        # Check configured strategies
        for step in self.config["fallback_chain"]:
            result = self._execute_fallback_step(step, query)
            if result and not result.get("error"):
                return result
        
        # Last resort: polite decline
        return {
            "success": False,
            "response_type": "polite_decline",
            "text": f"Sorry, I wasn't able to find information about '{query}'. Please try rephrasing your question.",
            "fallback_used": True,
            "strategy_applied": "polite_decline_last_resort",
            "original_query": original_query_text or query
        }
    
    def handle_invalid_citations(
        self, 
        response_text: str,
        validation_errors: List[Dict],
        retrieved_documents: List = None
    ) -> Dict[str, Any]:
        """Handle case where citations cannot be validated."""
        
        strategy = self.select_fallback_strategy("citation_invalid") or {}
        response_config = self.config["fallback_responses"]["citation_invalid"]
        
        # Try to salvage response by removing invalid citations
        if self.config.get("graceful_degradation_enabled", True):
            cleaned_response = self._remove_invalid_citations(response_text, validation_errors)
            
            return {
                "success": False,  # Not fully valid but degraded
                "response_type": "degraded_with_warning",
                "text": f"⚠️  WARNING: Some information may not be directly sourced from retrieved documents.\n\n{cleaned_response}",
                "fallback_used": True,
                "strategy_applied": "graceful_degradation_citation_removal",
                "validation_errors": validation_errors,
                "original_text_length": len(response_text),
                "cleaned_text_length": len(cleaned_response)
            }
        
        # Full failure if degradation disabled
        return {
            "success": False,
            "response_type": "citation_rejection",
            "text": f"Unable to verify citations for the generated response. Cannot provide answer with guaranteed grounding.",
            "fallback_used": True,
            "strategy_applied": "citation_rejection_strict_mode",
            "validation_errors": validation_errors
        }
    
    def handle_empty_context(self, query: str) -> Dict[str, Any]:
        """Handle case where no documents were retrieved."""
        
        strategy = self.select_fallback_strategy("empty_context") or {}
        response_config = self.config["fallback_responses"]["empty_context"]
        
        if response_config:
            return {
                "success": False,
                "response_type": "custom_fallback",
                "text": response_config.format(query=query) if isinstance(response_config, str) else response_config,
                "fallback_used": True,
                "strategy_applied": strategy.get("name") or "empty_context"
            }
        
        return {
            "success": False,
            "response_type": "polite_decline",
            "text": f"I don't have any relevant information to answer your question about '{query}'. Try asking a different question.",
            "fallback_used": True,
            "strategy_applied": "empty_context_polite_decline"
        }
    
    def _build_retrieval_failure_strategy(self) -> Dict[str, Any]:
        """Build strategy for complete retrieval failure."""
        return {
            "name": "retrieval_failure",
            "actions": [
                "log_error",
                "trigger_fallback_chain"
            ],
            "priority": 1
        }
    
    def _build_citation_invalid_strategy(self) -> Dict[str, Any]:
        """Build strategy for citation validation failures."""
        return {
            "name": "citation_invalid",
            "actions": [
                "log_warning", 
                "attempt_graceful_degradation",
                "remove_invalid_citations"
            ],
            "priority": 2
        }
    
    def _build_empty_context_strategy(self) -> Dict[str, Any]:
        """Build strategy for empty retrieval results."""
        return {
            "name": "empty_context",
            "actions": [
                "log_info",
                "trigger_fallback_chain"
            ],
            "priority": 3
        }
    
    def _build_out_of_scope_strategy(self) -> Dict[str, Any]:
        """Build strategy for out-of-scope queries."""
        return {
            "name": "out_of_scope",
            "actions": [
                "log_info",
                "suggest_alternative"
            ],
            "priority": 4
        }
    
    def _execute_fallback_step(self, step_name: str, query: str) -> Optional[Dict[str, Any]]:
        """Execute a single fallback step from the chain."""
        
        steps_config = {
            "direct_answer": self._generate_direct_answer,
            "partial_retrieval": lambda q: {"success": False, "text": f"I could only partially answer your question about '{q}'. Some information may be incomplete."},
            "knowledge_based_response": self._generate_knowledge_based_response,
            "polite_decline": self._generate_polite_decline
        }
        
        handler = steps_config.get(step_name)
        if handler:
            try:
                return handler(query)
            except Exception as e:
                logger.error(f"Fallback step '{step_name}' failed: {str(e)}")
        
        return None
    
    def _generate_direct_answer(self, query: str) -> Dict[str, Any]:
        """Generate direct answer without citations (when retrieval fails)."""
        response_config = self.config["fallback_responses"]["retrieval_failure"]
        
        if response_config:
            text = response_config.format(query=query) if isinstance(response_config, str) else response_config
            return {
                "success": False,
                "response_type": "direct_answer_no_citations",
                "text": text,
                "fallback_used": True
            }
        
        return {
            "success": False,
            "response_type": "generic_direct_answer",
            "text": f"I'm sorry, but I don't have enough information to answer your question about '{query}'. Please try a different approach.",
            "fallback_used": True
        }
    
    def _generate_knowledge_based_response(self, query: str) -> Dict[str, Any]:
        """Generate response based on internal knowledge (not from retrieval)."""
        
        # This would use the model's general knowledge
        return {
            "success": False,  # Not grounded in retrieved docs
            "response_type": "knowledge_based",
            "text": f"Based on my general knowledge: I don't have specific information about '{query}' from the retrieved documents. Please verify this information.",
            "fallback_used": True,
            "grounded_in_retrieved_docs": False
        }
    
    def _generate_polite_decline(self, query: str) -> Dict[str, Any]:
        """Generate polite response declining to answer."""
        
        return {
            "success": False,
            "response_type": "polite_decline",
            "text": f"I'm unable to provide a reliable answer about '{query}' without access to relevant documents. Please try rephrasing or providing more context.",
            "fallback_used": True
        }
    
    def _remove_invalid_citations(
        self, 
        text: str,
        invalid_citations: List[Dict]
    ) -> str:
        """Remove references to invalid citations from response text."""
        
        # Extract citation markers (e.g., [1], [2])
        import re
        
        for error in invalid_citations:
            citation = error.get("citation", "")
            
            # Remove citation markers and their content
            if re.search(rf'\[{re.escape(citation)}\]', text):
                text = re.sub(rf'\[{re.escape(citation)}\]', '', text)
        
        return text.strip()


class GracefulDegradationManager:
    """
    Manages graceful degradation of RAG responses when grounding is not possible.
    
    F-12 FIX: Implements multi-level fallback strategies for handling various failure modes.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            "degradation_levels": {
                "level_1": {  # Minor issues
                    "triggers": ["minor_citation_mismatch"],
                    "response_type": "proceed_with_warning"
                },
                "level_2": {  # Major issues
                    "triggers": ["no_documents", "all_citations_invalid"],
                    "response_type": "partial_answer"
                },
                "level_3": {  # Critical failures
                    "triggers": ["retrieval_complete_failure", "system_error"],
                    "response_type": "polite_decline"
                }
            },
            "warning_thresholds": {
                "citation_mismatch_ratio": 0.5,  # >50% invalid = major issue
                "max_invalid_citations": 3
            }
        }
    
    def determine_degradation_level(
        self, 
        validation_result: Dict,
        query: str
    ) -> tuple[str, str]:
        """Determine degradation level and response type."""
        
        issues = validation_result.get("issues_found", [])
        citation_issues = [i for i in issues if "citation" in i.get("reason", "").lower()]
        
        invalid_count = len(citation_issues)
        total_citations = len(issues) or 1
        
        mismatch_ratio = invalid_count / max(total_citations, 1)
        
        # Determine level based on triggers
        for level_name, config in self.config["degradation_levels"].items():
            if any(trigger in str(citation_issues) for trigger in config["triggers"]):
                return (level_name, config["response_type"])
        
        # Fallback to severity-based determination
        if mismatch_ratio > 0.5 or invalid_count >= self.config["warning_thresholds"]["max_invalid_citations"]:
            return ("2", "partial_answer")
        elif validation_result.get("valid", True):
            return ("1", "proceed_with_warning")
        else:
            return ("3", "polite_decline")
    
    def generate_degraded_response(
        self, 
        query: str,
        original_text: str,
        validation_errors: List[Dict],
        fallback_handler: FallbackResponseStrategy = None
    ) -> Dict[str, Any]:
        """Generate response with appropriate degradation level."""
        
        # Determine degradation level
        level_name, response_type = self.determine_degradation_level(
            {"issues_found": validation_errors}, query
        )
        
        result = {
            "success": False,  # Degraded success
            "degradation_applied": True,
            "level": level_name,
            "response_type": response_type,
            "original_text_length": len(original_text),
            "validation_errors_count": len(validation_errors)
        }
        
        if response_type == "proceed_with_warning":
            # Minor degradation - proceed with warning
            result["text"] = self._generate_proceed_with_warning(
                original_text, validation_errors
            )
            result["success"] = True  # Proceed but warn
        
        elif response_type == "partial_answer":
            # Major degradation - provide partial answer
            fallback_handler = fallback_handler or FallbackResponseStrategy()
            result["text"] = fallback_handler.handle_invalid_citations(
                original_text, validation_errors
            )
        
        else:  # polite_decline
            fallback_handler = fallback_handler or FallbackResponseStrategy()
            result["text"] = fallback_handler.handle_retrieval_failure(query)
        
        return result
    
    def _generate_proceed_with_warning(self, text: str, errors: List[Dict]) -> str:
        """Generate response with warning but proceed."""
        
        warnings = []
        for error in errors[:3]:  # Limit to top 3 errors
            reason = error.get("reason", "Unknown issue")
            warnings.append(f"- {reason}")
        
        warning_text = f"\n\n⚠️  SYSTEM WARNING:\nSome information may not be fully verified. Issues detected: {'; '.join(warnings)}"
        
        return text + warning_text


def main():
    """Example usage."""
    
    # Initialize handlers
    fallback_handler = FallbackResponseStrategy()
    degradation_manager = GracefulDegradationManager()
    
    # Example scenario 1: Retrieval failure
    query = "What is the latest news about SpaceX?"
    result = fallback_handler.handle_retrieval_failure(query)
    print(f"Retrieval failure response:\n{result['text']}\n")
    
    # Example scenario 2: Invalid citations
    invalid_validation = {
        "citations_found": ["[1]", "[99]"],
        "invalid_citations": [
            {"citation": "[99]", "reason": "Document not found"}
        ]
    }
    
    response_text = """SpaceX launched a new rocket yesterday. [1] 
The Falcon 9 reached orbit successfully. [2]"""
    
    result = fallback_handler.handle_invalid_citations(
        response_text, invalid_validation.get("invalid_citations", [])
    )
    print(f"Invalid citations response:\n{result['text']}\n")


if __name__ == "__main__":
    main()
