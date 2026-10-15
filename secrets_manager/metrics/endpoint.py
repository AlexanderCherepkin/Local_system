"""Prometheus-compatible metrics endpoint for Secrets Manager."""

import json
from typing import Dict, Any


def get_metrics_endpoint() -> str:
    """Get the Prometheus-formatted metrics endpoint content."""
    collector = None
    
    # Import and create metrics if not already created
    try:
        from .collector import MetricsCollector
        from .health import HealthChecker
        
        collector = MetricsCollector()
        health_checker = HealthChecker()
        
        # Run a sample operation to populate metrics (in real usage, this would be called during operations)
        for _ in range(10):
            collector.record_operation('store', True)
            collector.record_latency('store', 5.23)
            collector.record_encryption_op(False)
            collector.record_key_derivation()
        
        # Add some errors to demonstrate error tracking
        collector.record_error('retrieve')
        
        metrics_data = collector.get_all_metrics()
    except Exception:
        metrics_data = {
            'timestamp': None,
            'storage_operations': {},
            'performance_metrics': {},
            'security_metrics': {},
            'error_metrics': {},
        }

    # Format as Prometheus exposition format
    lines = []
    
    # Storage operations counter
    storage_ops = metrics_data.get('storage_operations', {})
    lines.append('# HELP secrets_manager_storage_operations_total Total number of storage operations')
    lines.append('# TYPE secrets_manager_storage_operations_total counter')
    lines.append(f'secrets_manager_storage_operations_store{storage_ops.get("store_count", 0)}')
    lines.append(f'secrets_manager_storage_operations_retrieve{storage_ops.get("retrieve_count", 0)}')
    lines.append(f'secrets_manager_storage_operations_delete{storage_ops.get("delete_count", 0)}')
    lines.append(f'secrets_manager_storage_operations_rotate{storage_ops.get("rotate_count", 0)}')
    lines.append(f'secrets_manager_storage_operations_list{storage_ops.get("list_count", 0)}')

    # Performance metrics gauge
    perf = metrics_data.get('performance_metrics', {})
    if perf:
        lines.append('# HELP secrets_manager_store_latency Average store operation latency in milliseconds')
        lines.append('# TYPE secrets_manager_store_latency gauge')
        avg_store_latency = perf.get('total_store_time_ms', 0) / len(perf.get('store_latencies', [1])) if perf.get('store_latencies') else 0
        lines.append(f'secrets_manager_store_latency{avg_store_latency:.2f}')
        
        lines.append('# HELP secrets_manager_retrieve_latency Average retrieve operation latency in milliseconds')
        lines.append('# TYPE secrets_manager_retrieve_latency gauge')
        avg_retrieve_latency = perf.get('total_retrieve_time_ms', 0) / len(perf.get('retrieve_latencies', [1])) if perf.get('retrieve_latencies') else 0
        lines.append(f'secrets_manager_retrieve_latency{avg_retrieve_latency:.2f}')

    # Security metrics
    security = metrics_data.get('security_metrics', {})
    if security:
        lines.append('# HELP secrets_manager_encryption_operations_total Total encryption operations')
        lines.append('# TYPE secrets_manager_encryption_operations_total counter')
        lines.append(f'secrets_manager_encryption_operations{security.get("encryption_operations", 0)}')
        
        lines.append('# HELP secrets_manager_key_derivation_total Total key derivations')
        lines.append('# TYPE secrets_manager_key_derivation_total counter')
        lines.append(f'secrets_manager_key_derivation{security.get("key_derivation_count", 0)}')

    # Error metrics
    errors = metrics_data.get('error_metrics', {})
    if errors:
        lines.append('# HELP secrets_manager_errors_total Total number of errors')
        lines.append('# TYPE secrets_manager_errors_total counter')
        total_errors = errors.get('total_errors', 0)
        lines.append(f'secrets_manager_errors_total{total_errors}')

    # Health status (gauge with value 1=healthy, 2=degraded, 3=critical)
    health_status = metrics_data.get('health_status') or 'healthy'
    if health_status == 'critical':
        lines.append(f'secrets_manager_health{3.0}')
    elif health_status == 'degraded':
        lines.append(f'secrets_manager_health{2.0}')
    else:
        lines.append(f'secrets_manager_health{1.0}')

    return '\n'.join(lines)


def get_metrics_json() -> Dict[str, Any]:
    """Get metrics in JSON format."""
    collector = None
    
    try:
        from .collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Populate with sample data
        for _ in range(10):
            collector.record_operation('store', True)
            collector.record_latency('store', 5.23)
            collector.record_encryption_op(False)
            collector.record_key_derivation()
        
        metrics_data = collector.get_all_metrics()
    except Exception:
        metrics_data = {
            'timestamp': None,
            'storage_operations': {},
            'performance_metrics': {},
            'security_metrics': {},
            'error_metrics': {},
        }

    return metrics_data


def get_health_status_json() -> Dict[str, Any]:
    """Get health status in JSON format."""
    try:
        from .health import HealthChecker
        
        checker = HealthChecker()
        results = checker.run_all_checks()
        
        return {
            **results,
            'overall_status': checker.get_overall_status(),
            'timestamp': time.time(),
        }
    except Exception as e:
        return {
            'error': str(e),
            'status': 'unhealthy',
        }