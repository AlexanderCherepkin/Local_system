"""HTTP Server for Prometheus-compatible metrics endpoint."""

import sys
from flask import Flask, Response


# Configure the application factory function
def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    @app.route('/metrics')
    def get_metrics():
        """Prometheus-compatible metrics endpoint."""
        from .endpoint import get_metrics_endpoint
        
        content = get_metrics_endpoint()
        
        return Response(
            content,
            mimetype='text/plain; charset=utf-8',
        )
    
    @app.route('/health')
    def get_health():
        """Health status endpoint in JSON format."""
        from .endpoint import get_health_status_json
        
        data = get_health_status_json()
        
        return Response(
            json.dumps(data, indent=2),
            mimetype='application/json',
            status=503 if data.get('status') == 'unhealthy' else 200,
        )
    
    @app.route('/metrics/json')
    def get_metrics_json():
        """Metrics endpoint in JSON format."""
        from .endpoint import get_metrics_json
        
        data = get_metrics_json()
        
        return Response(
            json.dumps(data, indent=2),
            mimetype='application/json',
        )
    
    @app.route('/')
    def index():
        """Root endpoint with documentation."""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Secrets Manager - Metrics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        a { color: #0066cc; text-decoration: none; }
        .endpoint { padding: 10px; margin: 5px 0; background: #f5f5f5; border-left: 4px solid #0066cc; }
    </style>
</head>
<body>
    <h1>Secrets Manager - Metrics</h1>
    <p>Prometheus-compatible metrics endpoint for the Secrets Manager.</p>
    
    <div class="endpoint">
        <strong><a href="/metrics">/metrics</a></strong><br>
        Prometheus exposition format metrics.
    </div>
    
    <div class="endpoint">
        <strong><a href="/health">/health</a></strong><br>
        Health status in JSON format (200=healthy, 503=unhealthy).
    </div>
    
    <div class="endpoint">
        <strong><a href="/metrics/json">/metrics/json</a></strong><br>
        Metrics in JSON format.
    </div>
</body>
</html>'''
        
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


# Create the application instance for direct import
app = create_app()


if __name__ == '__main__':
    from .collector import MetricsCollector
    collector = MetricsCollector()

    # Start server
    app.run(host='0.0.0.0', port=9090)