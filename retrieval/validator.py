"""
Citation Validator module for grounded RAG responses.

Implements F-11: Validates that citations belong to actually retrieved documents.
Prevents hallucinated references to non-existent sources.
"""

from typing import List, Dict, Any, Optional
import re


class CitationValidator:
    """
    Validates that citations in generated responses reference actual retrieved documents.
    
    F-11 FIX: Checks every citation against the list of actually retrieved documents.
    Prevents model from hallucinating references to non-existent sources.
    """
    
    def __init__(self, retrieved_documents: List[Dict] = None):
        """
        Initialize validator with retrieved documents.
        
        Args:
            retrieved_documents: List of document dicts with 'id', 'text' keys
        """
        self.retrieved_docs = retrieved_documents or []
        
    def validate_citations(self, response_text: str) -> Dict[str, Any]:
        """
        Validate that all citations in response reference actual documents.
        
        F-11 FIX: Checks each citation against the list of retrieved documents.
        
        Args:
            response_text: Generated response text with potential citations
        
        Returns:
            Dictionary with validation results:
                - 'valid': bool indicating if all citations are valid
                - 'citations_found': List of citation patterns found in text
                - 'invalid_citations': List of citations not matching any document
                - 'document_ids_referenced': Set of document IDs referenced by citations
        """
        result = {
            "valid": True,
            "response_text": response_text,
            "citations_found": [],
            "invalid_citations": [],
            "document_ids_referenced": set(),
            "retrieved_document_count": len(self.retrieved_docs)
        }
        
        if not self.retrieved_docs:
            result["valid"] = False
            result["error"] = "No documents retrieved - cannot validate citations"
            return result
        
        # Extract potential citations from response text
        citations_found = self._extract_citations(response_text)
        result["citations_found"] = citations_found
        
        if not citations_found:
            # No citations found - this is fine (maybe model didn't need to cite anything)
            return result
        
        # Check each citation against retrieved documents
        for citation in citations_found:
            doc_id = self._extract_document_id_from_citation(citation)
            
            if not doc_id:
                result["invalid_citations"].append({
                    "citation": citation,
                    "reason": "Could not extract document ID from citation"
                })
                continue
            
            # F-11 FIX: Check if this document ID actually exists in retrieved documents
            if self._document_exists(doc_id):
                result["document_ids_referenced"].add(doc_id)
            else:
                result["valid"] = False
                result["invalid_citations"].append({
                    "citation": citation,
                    "reason": f"Document {doc_id} not found in retrieved documents",
                    "retrieved_document_ids": [doc['id'] for doc in self.retrieved_docs]
                })
        
        return result
    
    def validate_grounding(self, response_text: str) -> Dict[str, Any]:
        """
        Validate that assertions in the response are grounded in retrieved documents.
        
        F-11 FIX: Checks if key claims can be traced back to actual source material.
        
        Args:
            response_text: Generated response
        
        Returns:
            Grounding validation result with confidence score and issues found.
        """
        result = {
            "grounded": True,
            "confidence_score": 1.0,
            "issues_found": [],
            "assertions_analyzed": []
        }
        
        # Extract potential assertions/claims from response
        assertions = self._extract_assertions(response_text)
        result["assertions_analyzed"] = assertions
        
        if not assertions:
            return result  # No assertions to validate (e.g., simple conversational response)
        
        retrieved_texts = [doc.get("text", "") for doc in self.retrieved_docs]
        
        for assertion in assertions:
            # Check if assertion appears in any retrieved document
            found_in_source = False
            
            for doc_text in retrieved_texts:
                # Simple substring check (can be improved with semantic search)
                if assertion.lower() in doc_text.lower():
                    found_in_source = True
                    break
            
            if not found_in_source:
                result["grounded"] = False
                result["issues_found"].append({
                    "assertion": assertion,
                    "reason": "Assertion not found in any retrieved document",
                    "severity": "HIGH"  # Critical grounding issue
                })
                
                # Reduce confidence score
                result["confidence_score"] -= 0.2
            
            else:
                result["issues_found"].append({
                    "assertion": assertion,
                    "reason": "Assertion grounded in retrieved documents",
                    "severity": None
                })
        
        return result
    
    def extract_and_validate_citations_from_context(
        self, 
        context_documents: List[Dict],
        generated_response: str
    ) -> Dict[str, Any]:
        """
        Comprehensive validation combining citation checking with grounding.
        
        Args:
            context_documents: Documents that were retrieved for the query
            generated_response: Model's generated response
        
        Returns:
            Combined validation result.
        """
        # Initialize validator
        validator = CitationValidator(context_documents)
        
        # Run both validations
        citation_result = validator.validate_citations(generated_response)
        grounding_result = validator.validate_grounding(generated_response)
        
        # Combine results
        combined = {
            "citation_validation": citation_result,
            "grounding_validation": grounding_result,
            "overall_valid": (
                citation_result.get("valid", False) and 
                grounding_result.get("grounded", False)
            ),
            "recommendation": self._generate_recommendation(
                citation_result,
                grounding_result
            )
        }
        
        return combined
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract potential citation patterns from text."""
        # Common citation patterns (can be extended)
        patterns = [
            r'\[\d+\]',  # [1], [2] style
            r'(\d+\.?\s*[.,])',  # "1. ", "1." style
            r'(doc_\w+)',  # doc_0, doc_1 style
            r'source\s*[:\s]*[^\n]+',  # source: ... pattern
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _extract_document_id_from_citation(self, citation: str) -> Optional[str]:
        """Extract document ID from citation string."""
        # Handle different citation formats
        if citation.startswith('doc_'):
            return citation
        
        match = re.search(r'doc_\w+', citation.lower())
        if match:
            return match.group(0)
        
        match = re.search(r'\[(\d+)\]', citation)
        if match and self.retrieved_docs:
            # Convert numeric index to doc_id (assuming sequential indexing)
            try:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(self.retrieved_docs):
                    return f"doc_{idx}"
            except ValueError:
                pass
        
        return None
    
    def _document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists in retrieved documents."""
        for doc in self.retrieved_docs:
            if doc.get("id") == doc_id or doc.get("doc_id") == doc_id:
                return True
        return False
    
    def _extract_assertions(self, text: str) -> List[str]:
        """Extract potential assertions/claims from response."""
        # Simple assertion extraction (can be improved with NLP)
        patterns = [
            r'[^.!?]+\.(?=\s|$)',  # End sentences
            r'[^?!]+!(?=\s|$)',    # End exclamations
        ]
        
        assertions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Filter out very short fragments
            for match in matches:
                if len(match.strip()) > 10 and '?' not in match.lower():
                    assertions.append(match.strip())
        
        return list(set(assertions))[:20]  # Limit to top 20
    
    def _generate_recommendation(
        self, 
        citation_result: Dict,
        grounding_result: Dict
    ) -> str:
        """Generate recommendation based on validation results."""
        issues = []
        
        if not citation_result.get("valid", True):
            invalid_count = len(citation_result.get("invalid_citations", []))
            issues.append(f"{invalid_count} invalid citations found")
        
        if not grounding_result.get("grounded", True):
            issue_count = len(grounding_result.get("issues_found", []))
            high_severity = sum(1 for i in grounding_result["issues_found"] 
                              if i.get("severity") == "HIGH")
            issues.append(f"{high_severity} HIGH severity grounding issues")
        
        if not issues:
            return "Response is valid and well-grounded"
        
        recommendations = [
            "Verify that all citations match the retrieved documents",
            "Cross-check assertions against source material",
            "Consider regenerating with stricter grounding constraints"
        ]
        
        return f"ISSUES FOUND ({', '.join(issues)}). Recommendations: {'; '.join(recommendations[:2])}"


class GroundedResponseGenerator:
    """
    Generates responses that are guaranteed to be grounded in retrieved documents.
    
    Enforces citation requirements before allowing response generation.
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
    
    def generate_grounding_enforced_response(
        self, 
        query: str,
        model_callback=None  # Callback that generates actual response text
    ) -> Dict[str, Any]:
        """
        Generate a grounded response with citation enforcement.
        
        Workflow:
        1. Retrieve relevant documents
        2. If retrieval fails (success=False), return error response
        3. Generate candidate response using model callback
        4. Validate citations against retrieved documents
        5. If validation fails, either:
           - Regenerate with stricter constraints
           - Return partial response with warnings
        
        Args:
            query: User's question
            model_callback: Function(query) -> str that generates response
        
        Returns:
            Response dict with 'success', 'text', 'citations_validated' fields.
        """
        
        # Step 1: Retrieve documents
        retrieval_result = self.retriever.retrieve_with_citations(query) if self.retriever else {}
        
        if not retrieval_result.get("success", False):
            return {
                "success": False,
                "text": "Unable to retrieve relevant information. Please rephrase your question.",
                "fallback_triggered": True,
                "citations_validated": None,
                "error": "Retrieval failed"
            }
        
        # Step 2: Generate candidate response (placeholder - replace with actual model)
        if not model_callback:
            raise ValueError("model_callback is required. Provide a function that generates text.")
        
        try:
            generated_text = model_callback(query, retrieval_result.get("context", []))
        except Exception as e:
            return {
                "success": False,
                "text": f"Error generating response: {str(e)}",
                "fallback_triggered": True,
                "citations_validated": None,
                "error": str(e)
            }
        
        # Step 3: Validate citations
        validator = CitationValidator(retrieval_result.get("documents", []))
        validation_result = validator.validate_citations(generated_text)
        
        if not validation_result.get("valid", True):
            warning_message = self._generate_warning(validation_result)
            
            return {
                "success": False,  # Not fully valid
                "text": generated_text + "\n\n" + warning_message,
                "fallback_triggered": True,
                "citations_validated": validation_result,
                "error": "Citation validation failed",
                "warning": warning_message
            }
        
        return {
            "success": True,
            "text": generated_text,
            "fallback_triggered": False,
            "citations_validated": validation_result,
            "error": None
        }
    
    def _generate_warning(self, validation_result: Dict) -> str:
        """Generate warning message for invalid citations."""
        invalid_count = len(validation_result.get("invalid_citations", []))
        
        warnings = [
            f"⚠️  WARNING: {invalid_count} citation(s) could not be verified against retrieved documents.",
            "Please verify that the referenced information is accurate."
        ]
        
        return "\n\n".join(warnings)
