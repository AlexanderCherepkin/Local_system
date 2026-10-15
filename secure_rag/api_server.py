#!/usr/bin/env python3
"""API Server - HTTP endpoint for Secure RAG Pipeline."""

import logging
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
from urllib.parse import urlparse, parse_qs

# Import pipeline runner
import sys
sys.path.insert(0, 'pipeline')
from runner import run_pipeline, PipelineResult

logger = logging.getLogger("secure_rag.api")


class SecureRAGAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the RAG API."""
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with proper headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
    
    def _parse_request(self) -> Dict[str, Any]:
        """Parse the incoming HTTP request."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in request body")
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        # Health check endpoint
        if parsed_path.path == '/health':
            self._send_json_response({
                'status': 'healthy',
                'service': 'secure-rag-api'
            })
            return
        
        # List available endpoints
        if parsed_path.path == '/api/v1/':
            self._send_json_response({
                'version': '1.0.0',
                'endpoints': [
                    {
                        'path': '/query',
                        'method': 'POST',
                        'description': 'Execute RAG pipeline for a query'
                    },
                    {
                        'path': '/health',
                        'method': 'GET',
                        'description': 'Health check endpoint'
                    }
                ]
            })
            return
        
        # Default 404
        self._send_json_response(
            {'error': 'Not found'}, 
            status_code=404
        )
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        
        try:
            # Parse request body
            data = self._parse_request()
            
            if parsed_path.path == '/api/v1/query':
                self._handle_query(data)
            else:
                self._send_json_response(
                    {'error': 'Not found'}, 
                    status_code=404
                )
        
        except ValueError as e:
            logger.error(f"Request parsing error: {str(e)}")
            self._send_json_response({'error': str(e)}, status_code=400)
        except Exception as e:
            logger.exception(f"Unexpected error handling request: {e}")
            self._send_json_response(
                {'error': 'Internal server error'}, 
                status_code=500
            )
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _handle_query(self, data: Dict[str, Any]):
        """Handle query execution."""
        
        # Extract parameters with defaults
        prompt = data.get('query', '')
        username = data.get('username', 'anonymous')
        
        # Validate prompt is not empty
        if not prompt or not isinstance(prompt, str):
            self._send_json_response(
                {'error': 'Query parameter is required and must be a string'}, 
                status_code=400
            )
            return
        
        logger.info(f"Processing query from user {username}: {prompt[:100]}...")
        
        try:
            # Execute pipeline
            result: PipelineResult = run_pipeline(prompt, username)
            
            response_data = {
                'success': True,
                'query': prompt,
                'answer': result.answer,
                'sources': result.sources,
                'error': None if not result.error else {'message': result.error}
            }
            
            logger.info(f"Query completed successfully for user {username}")
            self._send_json_response(response_data)
        
        except Exception as e:
            logger.exception(f"Error executing pipeline: {e}")
            self._send_json_response({
                'success': False,
                'error': {'message': str(e)}
            }, status_code=500)


def create_api_server(host: str = '127.0.0.1', port: int = 8000):
    """Create and start the API server."""
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, SecureRAGAPIHandler)
    
    logger.info(f"Secure RAG API Server starting on {host}:{port}")
    
    # Run server in a thread for graceful shutdown support
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    
    return httpd


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Secure RAG API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = create_api_server(args.host, args.port)
    logger.info("Server running. Press Ctrl+C to stop.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
