# Secure RAG Pipeline System

A secure, role-based Retrieval Augmented Generation (RAG) pipeline for Windows API documentation with enhanced security controls and audit logging.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Security Features](#security-features)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

This system provides a secure RAG pipeline that:

1. **Processes user queries** against Windows API documentation
2. **Enforces role-based access control (RBAC)** with username-based permissions
3. **Implements security checks** to prevent malicious query execution
4. **Maintains audit logs** for all operations and data access
5. **Supports concurrent requests** via multithreaded HTTP server

### Key Components

- **`runner.py`**: Core pipeline orchestration engine with security validation
- **`api_server.py`**: RESTful API server (HTTP/JSON) exposing the RAG functionality
- **Documentation files**: Windows API reference materials and security policies

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│              (Browsers, Scripts, Mobile Apps)                │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS/JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Secure RAG API Server                       │
│                    (api_server.py)                           │
├─────────────────────────────────────────────────────────────┤
│  • HTTP Request Handler                                      │
│  • JSON Response Formatting                                  │
│  • CORS Support                                              │
│  • Thread-Safe Request Processing                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Pipeline Runner (runner.py)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Security     │───▶│ Document     │───▶│ Context      │  │
│  │ Validation   │    │ Retrieval    │    │ Processing   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│       │                  │                   │              │
│       ▼                  ▼                   ▼              │
│  • Blocked term      • Vector search      • Document      │
│    detection          (placeholder)        concatenation   │
│  • Educational       • Score ranking       • Context      │
│    context check      (placeholder)        enrichment     │
│  • Role verification • User attribution   • Query         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM / Answer Generation                         │
│                    (placeholder)                             │
├─────────────────────────────────────────────────────────────┤
│  • Natural language processing                               │
│  • Context understanding                                     │
│  • Answer synthesis                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Formatting & Sources                   │
├─────────────────────────────────────────────────────────────┤
│  • Answer truncation                                         │
│  • Source citation (top-3)                                   │
│  • Metadata extraction                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
secure_rag/
├── README.md                      # This documentation file
├── api_server.py                  # HTTP API server (port 8000)
├── runner.py                      # Pipeline orchestration engine
├── pipeline/
│   └── runner.py                  # Same as root, for import convenience
└── data/                          # Optional: store document corpora here
    ├── windows_api_docs.txt       # Windows API reference material
    └── security_policies.txt      # Security policy definitions
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Quick Start

```bash
# Clone or download the repository
cd secure_rag

# Verify dependencies (currently none required)
python --version
pip list | grep -i flask    # Optional if upgrading to Flask/FastAPI

# Run API server
python api_server.py --host 127.0.0.1 --port 8000
```

### Development Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install optional dependencies
pip install requests flask uvicorn

# Run with Flask development server
python api_server.py --host 0.0.0.0 --port 8000
```

---

## Usage

### Starting the Server

```bash
# Default configuration (127.0.0.1:8000)
python api_server.py

# Custom host and port
python api_server.py --host 0.0.0.0 --port 5000

# Run in background
nohup python api_server.py > server.log 2>&1 &
```

### Making API Requests

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "secure-rag-api"
}
```

#### Query Execution

**Request:**
```http
POST /api/v1/query
Content-Type: application/json

{
    "query": "How do I enumerate processes in Windows?",
    "username": "developer_user",
    "roles": ["admin"]
}
```

**Response (Success):**
```json
{
  "success": true,
  "query": "How do I enumerate processes in Windows?",
  "answer": "To enumerate processes in Windows, you can use the following methods:\n\n1. **CreateToolhelp32Snapshot()**: API function that creates a snapshot of all currently running processes...\n[additional content truncated]",
  "sources": [
    {
      "title": "Windows API Security Documentation",
      "content_preview": "The Windows API provides various functions for system operations...",
      "score": 0.95,
      "id": 1
    },
    {
      "title": "Security Best Practices",
      "content_preview": "Follow security best practices when using Windows APIs...",
      "score": 0.87,
      "id": 2
    }
  ],
  "error": null
}
```

**Response (Blocked Query):**
```json
{
  "success": false,
  "query": "[redacted]",
  "answer": null,
  "sources": [],
  "error": {
    "message": "Query blocked for security reasons: 'exploit'"
  }
}
```

---

## API Documentation

### Endpoints

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | `/health`          | Health check                         |
| GET    | `/api/v1/`         | List available endpoints             |
| POST   | `/api/v1/query`    | Execute RAG pipeline for a query     |

### Query Parameters (POST /api/v1/query)

| Parameter  | Type    | Required | Default      | Description                              |
|------------|---------|----------|--------------|------------------------------------------|
| `query`    | string  | Yes      | -            | The user's natural language query        |
| `username` | string  | No       | `"anonymous"`| Username for RBAC and audit logging      |
| `roles`    | array   | No       | `[username]` | List of roles the user belongs to        |

### Response Fields

| Field     | Type           | Description                                     |
|-----------|----------------|-------------------------------------------------|
| `success` | boolean        | Whether the query was successfully processed    |
| `query`   | string         | The original query (echoed back)                |
| `answer`  | string or null | Generated answer from RAG pipeline              |
| `sources` | array          | Top-3 relevant documents with metadata         |
| `error`   | object or null | Error details if processing failed              |

---

## Security Features

### 1. Blocked Terms Detection

The system maintains a list of blocked terms that trigger security alerts:

```python
blocked_terms = [
    "exploit", "vulnerability", "hack", "attack", 
    "malware", "trojan", "backdoor", "rootkit"
]
```

### 2. Educational Context Allowance

Queries containing blocked terms are allowed if they include educational indicators:

**Allowed:**
- "educational exploit techniques for learning purposes"
- "research into vulnerability detection methods"
- "study of malware behavior in controlled environments"

**Blocked:**
- "how to create an exploit for Windows"
- "exploit development tutorial"

### 3. Role-Based Access Control

Username-based RBAC ensures:
- Users are identified by their username
- Roles can be assigned per request
- Audit logging tracks all user actions

### 4. Input Validation

```python
if not prompt or not isinstance(prompt, str):
    self._send_json_response(
        {'error': 'Query parameter is required and must be a string'}, 
        status_code=400
    )
return
```

### 5. Audit Logging

All operations are logged with timestamps:

```
2024-01-15 10:30:45 - secure_rag.pipeline - INFO - Starting pipeline for user developer_user with query length: 45
2024-01-15 10:30:46 - secure_rag.api - WARNING - Blocked term 'malware' detected in query
```

---

## Configuration

### Command-Line Arguments

```bash
python api_server.py --help
usage: api_server.py [-h] [--host HOST] [--port PORT]

Secure RAG API Server

options:
  -h, --help     show this help message and exit
  --host HOST    Server host (default: '127.0.0.1')
  --port PORT    Server port (default: 8000)
```

### Environment Variables (Future Enhancement)

```bash
# Server configuration
export SECURE_RAG_HOST=127.0.0.1
export SECURE_RAG_PORT=8000

# Security settings
export SECURE_RAG_BLOCKED_TERMS="exploit,vulnerability,hack"
export SECURE_RAG_ALLOW_EDUCATIONAL=true

# Logging
export SECURE_RAG_LOG_LEVEL=INFO
```

### Runtime Configuration (runner.py)

Modify the `PipelineRunner` class directly:

```python
class PipelineRunner:
    def __init__(self):
        # Add configuration loading here
        self.blocked_terms = [
            "exploit", "vulnerability", "hack"  # Customize as needed
        ]
        self.max_sources = 3  # Number of sources to return
```

---

## Testing

### Unit Tests (Future Enhancement)

Create `test_runner.py`:

```python
import unittest
from runner import PipelineRunner, run_pipeline

class TestPipelineRunner(unittest.TestCase):
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        result = run_pipeline("")
        self.assertIsNone(result.answer)
        
    def test_blocked_term_detection(self):
        """Test that blocked terms are detected."""
        result = run_pipeline("How to create malware")
        self.assertIn('error', dir(result))
        
    def test_educational_context_allowed(self):
        """Test educational context allows blocked terms."""
        result = run_pipeline(
            "Educational research into malware behavior"
        )
        # Should not be blocked

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests (Future Enhancement)

Create `test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'

def test_query_execution():
    """Test query execution flow."""
    payload = {
        "query": "What is CreateProcess?",
        "username": "test_user"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'success' in data

def test_blocked_query():
    """Test that blocked queries are rejected."""
    payload = {
        "query": "How to hack Windows",
        "username": "hacker"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    # Should return error response
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data

if __name__ == '__main__':
    test_health_endpoint()
    test_query_execution()
    test_blocked_query()
    print("All tests passed!")
```

Run tests:

```bash
python -m unittest test_runner.py        # Unit tests (future)
python test_api.py                        # Integration tests
```

---

## Production Deployment

### Docker Support (Future Enhancement)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir flask uvicorn gunicorn

EXPOSE 8000

CMD ["python", "api_server.py"]
```

Build and run:

```bash
docker build -t secure-rag .
docker run -p 8000:8000 secure-rag
```

### Nginx Reverse Proxy (Future Enhancement)

```nginx
server {
    listen 80;
    server_name api.secure-rag.local;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## Troubleshooting

### Common Issues

**Problem:** Server won't start on port 8000

```bash
# Check if port is in use
netstat -ano | findstr :8000

# Kill process using the port (Windows)
taskkill /PID <PID> /F
```

**Problem:** Import errors for runner module

```bash
# Ensure pipeline directory exists
mkdir -p pipeline

# Copy runner.py to both locations
cp runner.py pipeline/runner.py
```

**Problem:** Slow response times

```bash
# Check server logs for bottlenecks
tail -f server.log | grep "Processing context"
```

---

## License & Credits

This system is designed for educational and research purposes. All Windows API documentation references should be properly attributed to Microsoft Corporation.

For security vulnerability reports, please use responsible disclosure practices.
