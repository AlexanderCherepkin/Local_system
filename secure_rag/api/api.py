#!/usr/bin/env python3
"""Secure RAG API Server - HTTP endpoints for the pipeline."""

import json
import logging
from typing import Any, Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading

logger = logging.getLogger("secure_rag.api")


class SecureRAGHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG queries."""

    def do_POST(self):
        """Handle POST requests with query parameters."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
            return

        # Extract query and username from request
        prompt = data.get('query', '')
        username = data.get('username', 'anonymous')
        roles = data.get('roles', [username])
        
        if not prompt:
            self.send_error_response(400, "Query is required")
            return

        try:
            # Execute the pipeline
            from pipeline.runner import run_pipeline
            
            result = run_pipeline(prompt=prompt, username=username, roles=roles)
            
            logger.info(f"User {username} received response with {len(result.get('sources', []))} sources")
            self.send_success_response(200, result)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            self.send_error_response(500, str(e))

    def do_GET(self):
        """Handle GET requests (health check)."""
        parsed_path = urllib.parse.urlparse(self.path).path
        
        if parsed_path == '/health':
            self.send_success_response(200, {"status": "ok", "service": "secure_rag"})
        
        elif parsed_path == '/' or parsed_path == '/help':
            help_data = {
                "endpoints": [
                    {"method": "POST", "path": "/", "description": "Execute RAG query"},
                    {"method": "GET", "path": "/health", "description": "Health check"}
                ],
                "example": {
                    "query": "What are the security vulnerabilities in Windows API?",
                    "username": "user1",
                    "roles": ["user1"]
                }
            }
            self.send_success_response(200, help_data)

        else:
            self.send_error_response(404, "Endpoint not found")

    def send_success_response(self, status_code: int, data: Dict[str, Any]):
        """Send a JSON response with success status."""
        response = {
            "status": "success",
            "data": data
        }
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def send_error_response(self, status_code: int, message: str):
        """Send a JSON response with error status."""
        response = {
            "status": "error",
            "message": message
        }
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def log_message(self, format: str, *args):
        """Override to use our logger."""
        logger.info("%s - %s", args[0], args[-1])


def run_server(host: str = "localhost", port: int = 8000):
    """Run the HTTP server in a separate thread."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, SecureRAGHandler)
    
    logger.info(f"Secure RAG API Server starting on {host}:{port}")
    
    # Run in daemon thread so it doesn't block main execution
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    
    return httpd


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Secure RAG API Server")
    parser.add_argument("-h", "--host", default="localhost", help="Host to bind (default: localhost)")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to bind (default: 8000)")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port)
