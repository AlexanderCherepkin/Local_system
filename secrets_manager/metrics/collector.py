"""Prometheus-compatible metrics collection for Secrets Manager."""

import time
from typing import Dict, Any


class MetricsCollector:
    """Collects and exposes metrics for Prometheus monitoring."""

    def __init__(self):
        self._metrics = {
            'storage_operations': {
                'store_count': 0,
                'retrieve_count': 0,
                'delete_count': 0,
                'rotate_count': 0,
                'list_count': 0,
            },
            'performance_metrics': {
                'total_store_time_ms': 0.0,
                'total_retrieve_time_ms': 0.0,
                'total_delete_time_ms': 0.0,
                'store_latencies': [],
                'retrieve_latencies': [],
                'delete_latencies': [],
            },
            'security_metrics': {
                'encryption_operations': 0,
                'decryption_operations': 0,
                'key_derivation_count': 0,
            },
            'error_metrics': {
                'store_errors': 0,
                'retrieve_errors': 0,
                'delete_errors': 0,
                'total_errors': 0,
            },
        }

    def record_operation(self, operation: str, success: bool = True):
        """Record a storage operation."""
        if operation in ['store', 'rotate']:
            self._metrics['storage_operations']['store_count'] += 1
        elif operation == 'retrieve':
            self._metrics['storage_operations']['retrieve_count'] += 1
        elif operation == 'delete':
            self._metrics['storage_operations']['delete_count'] += 1
        elif operation == 'list':
            self._metrics['storage_operations']['list_count'] += 1

    def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency."""
        if operation in ['store', 'rotate']:
            self._metrics['performance_metrics']['total_store_time_ms'] += latency_ms
            self._metrics['performance_metrics']['store_latencies'].append(latency_ms)
        elif operation == 'retrieve':
            self._metrics['performance_metrics']['total_retrieve_time_ms'] += latency_ms
            self._metrics['performance_metrics']['retrieve_latencies'].append(latency_ms)
        elif operation == 'delete':
            self._metrics['performance_metrics']['total_delete_time_ms'] += latency_ms
            self._metrics['performance_metrics']['delete_latencies'].append(latency_ms)

    def record_encryption_op(self, is_decryption: bool = False):
        """Record encryption/decryption operation."""
        self._metrics['security_metrics']['encryption_operations'] += 1
        if is_decryption:
            self._metrics['security_metrics']['decryption_operations'] += 1

    def record_key_derivation(self, count: int = 1):
        """Record key derivation operations."""
        self._metrics['security_metrics']['key_derivation_count'] += count

    def record_error(self, operation: str):
        """Record an error."""
        if operation == 'store':
            self._metrics['error_metrics']['store_errors'] += 1
        elif operation == 'retrieve':
            self._metrics['error_metrics']['retrieve_errors'] += 1
        elif operation == 'delete':
            self._metrics['error_metrics']['delete_errors'] += 1
        self._metrics['error_metrics']['total_errors'] += 1

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            'timestamp': time.time(),
            'storage_operations': dict(self._metrics['storage_operations']),
            'performance_metrics': self._get_performance_summary(),
            'security_metrics': dict(self._metrics['security_metrics']),
            'error_metrics': dict(self._metrics['error_metrics']),
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics = {
            'storage_operations': {'store_count': 0, 'retrieve_count': 0, 'delete_count': 0, 'rotate_count': 0, 'list_count': 0},
            'performance_metrics': {'total_store_time_ms': 0.0, 'total_retrieve_time_ms': 0.0, 'total_delete_time_ms': 0.0, 'store_latencies': [], 'retrieve_latencies': [], 'delete_latencies': []},
            'security_metrics': {'encryption_operations': 0, 'decryption_operations': 0, 'key_derivation_count': 0},
            'error_metrics': {'store_errors': 0, 'retrieve_errors': 0, 'delete_errors': 0, 'total_errors': 0},
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status based on metrics."""
        errors = self._metrics['error_metrics']['total_errors']
        total_ops = (
            self._metrics['storage_operations']['store_count'] +
            self._metrics['storage_operations']['retrieve_count'] +
            self._metrics['storage_operations']['delete_count'] +
            self._metrics['storage_operations']['list_count']
        )

        error_rate = errors / total_ops if total_ops > 0 else 0.0
        health_status = 'healthy' if error_rate < 0.05 else ('degraded' if error_rate < 0.2 else 'critical')

        return {
            'status': health_status,
            'total_operations': total_ops,
            'error_count': errors,
            'error_rate': round(error_rate * 100, 2),
            'last_update': time.time(),
        }


_metrics_instance = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


def reset_metrics():
    """Reset the global metrics collector."""
    global _metrics_instance
    if _metrics_instance:
        _metrics_instance.reset_metrics()
        _metrics_instance = None