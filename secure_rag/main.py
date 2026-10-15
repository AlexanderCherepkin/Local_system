#!/usr/bin/env python3
"""Secure RAG Pipeline - Main Entry Point."""
import sys
from typing import Dict, Any

def main():
    """Main entry point for the Secure RAG system."""
    
    # Import pipeline components
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.output_formatter import OutputFormatter
    
    print("=" * 60)
    print("SECURE RAG PIPELINE - MAIN ENTRY POINT")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    formatter = OutputFormatter()
    
    # Sample queries for demonstration
    sample_queries = [
        "What are the security vulnerabilities in Windows API?",
        "How to detect and prevent ransomware attacks?",
        "Best practices for secure authentication systems"
    ]
    
    try:
        results = []
        
        for query in sample_queries:
            print(f"\n--- Processing Query ---")
            print(f"Query: {query}")
            
            # Execute pipeline
            result = orchestrator.execute(query)
            
            # Format and display output
            formatted_output = formatter.format(result.to_dict(), 'markdown')
            results.append({
                'query': query,
                'result': result
            })
            
            print(formatted_output)
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print(f"Total Queries Processed: {len(results)}")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {str(e)}")
        print("[INFO] Make sure all pipeline modules are in the same directory.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
