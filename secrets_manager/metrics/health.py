"""Health monitoring for Secrets Manager."""

import time
from typing import Dict, Any


class HealthChecker:
    """Monitors health of Secrets Manager components."""

    def __init__(self):
        self._checks = {
            'storage': {'status': 'unknown', 'last_check': None},
            'database': {'status': 'unknown', 'last_check': None, 'latency_ms': 0},
            'cache': {'status': 'unknown', 'last_check': None, 'hit_rate': 1.0},
            'encryption_service': {'status': 'unknown', 'last_check': None},
            'disk_usage': {'status': 'unknown', 'last_check': None, 'usage_percent': 0.0},
        }

    def check_storage(self) -> Dict[str, Any]:
        """Check storage health."""
        try:
            # Simulated check - replace with actual storage validation
            self._checks['storage']['status'] = 'healthy'
            self._checks['storage']['last_check'] = time.time()
            return {'service': 'storage', 'status': 'healthy'}
        except Exception as e:
            self._checks['storage']['status'] = 'unhealthy'
            self._checks['storage']['last_check'] = time.time()
            return {'service': 'storage', 'status': 'unhealthy', 'error': str(e)}

    def check_database(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            start = time.time()
            # Simulated check - replace with actual DB connection test
            latency = (time.time() - start) * 1000  # ms
            self._checks['database']['status'] = 'healthy' if latency < 500 else 'degraded'
            self._checks['database']['last_check'] = time.time()
            self._checks['database']['latency_ms'] = latency
            return {'service': 'database', 'status': 'healthy'}
        except Exception as e:
            self._checks['database']['status'] = 'unhealthy'
            self._checks['database']['last_check'] = time.time()
            return {'service': 'database', 'status': 'unhealthy', 'error': str(e)}

    def check_cache(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # Simulated check - replace with actual cache connection test
            self._checks['cache']['status'] = 'healthy'
            self._checks['cache']['last_check'] = time.time()
            return {'service': 'cache', 'status': 'healthy'}
        except Exception as e:
            self._checks['cache']['status'] = 'unhealthy'
            self._checks['cache']['last_check'] = time.time()
            return {'service': 'cache', 'status': 'unhealthy', 'error': str(e)}

    def check_encryption_service(self) -> Dict[str, Any]:
        """Check encryption service health."""
        try:
            # Simulated check - replace with actual KMS validation
            self._checks['encryption_service']['status'] = 'healthy'
            self._checks['encryption_service']['last_check'] = time.time()
            return {'service': 'encryption', 'status': 'healthy'}
        except Exception as e:
            self._checks['encryption_service']['status'] = 'unhealthy'
            self._checks['encryption_service']['last_check'] = time.time()
            return {'service': 'encryption', 'status': 'unhealthy', 'error': str(e)}

    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage health."""
        try:
            import os
            total, used, free = os.statvfs('/')
            usage_percent = (used / total) * 100
            self._checks['disk_usage']['usage_percent'] = usage_percent

            if usage_percent < 80:
                status = 'healthy'
            elif usage_percent < 90:
                status = 'warning'
            else:
                status = 'critical'

            self._checks['disk_usage']['status'] = status
            self._checks['disk_usage']['last_check'] = time.time()
            return {'service': 'disk', 'status': status, 'usage_percent': round(usage_percent, 2)}
        except Exception as e:
            self._checks['disk_usage']['status'] = 'unhealthy'
            self._checks['disk_usage']['last_check'] = time.time()
            return {'service': 'disk', 'status': 'unhealthy', 'error': str(e)}

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}

        results.update(self.check_storage())
        results.update(self.check_database())
        results.update(self.check_cache())
        results.update(self.check_encryption_service())
        results.update(self.check_disk_usage())

        return results

    def get_overall_status(self) -> str:
        """Get overall system health status."""
        statuses = [
            self._checks['storage']['status'],
            self._checks['database']['status'],
            self._checks['cache']['status'],
            self._checks['encryption_service']['status'],
            self._checks['disk_usage']['status'],
        ]

        if 'unhealthy' in statuses:
            return 'critical'
        elif 'degraded' in statuses or 'warning' in statuses:
            return 'degraded'
        else:
            return 'healthy'


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return results."""
    checker = HealthChecker()
    results = checker.run_all_checks()
    overall_status = checker.get_overall_status()

    return {
        **results,
        'overall_status': overall_status,
        'timestamp': time.time(),
    }


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    results = run_health_checks()
    return results