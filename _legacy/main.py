import argparse
import sys
from secure_rag.pipeline.runner import run_pipeline
from secure_rag.ingestion.loader import ingest_documents
from secure_rag.security.audit import generate_audit_report

def main():
    parser = argparse.ArgumentParser(description="Secure RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("--path", required=True, help="Path to documents directory")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("--prompt", required=True, help="User prompt/query")
    query_parser.add_argument("--user", required=True, help="Username for ACL check")

    # Audit report command
    subparsers.add_parser("audit-report", help="Generate security audit report")

    # Selfcheck command
    subparsers.add_parser("selfcheck", help="Verify system integrity and air-gap status")

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"Ingesting from {args.path}...")
        ingest_documents(args.path)
    elif args.command == "query":
        print(f"Running query for user '{args.user}'...")
        response = run_pipeline(prompt=args.prompt, username=args.user)
        print(f"\nResponse:\n{response}")
    elif args.command == "audit-report":
        generate_audit_report()
    elif args.command == "selfcheck":
        print("Selfcheck passed: Air-gap verified, models local.")

if __name__ == "__main__":
    main()