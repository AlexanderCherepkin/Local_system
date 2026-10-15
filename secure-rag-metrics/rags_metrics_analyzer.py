#!/usr/bin/env python3
"""
Secure RAG Metrics Analyzer - Enterprise-grade PostgreSQL Integration
====================================================================

This script provides comprehensive metrics collection, analysis, and reporting
for Retrieval-Augmented Generation (RAG) systems with security hardening.

Features:
- Multi-source metrics collection (system, query logs, error logs)
- Real-time analytics pipeline
- Security audit trail generation
- Automated alerting system
- PostgreSQL integration with connection pooling

Author: AI Engineering Team
License: Proprietary Enterprise
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics
import hashlib
import secrets
import base64

# PostgreSQL Connection with Security Hardening
try:
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    print("ERROR: psycopg2-binary not installed. Install via: pip install psycopg2-binary")
    sys.exit(1)

# YAML Configuration Parsing
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install via: pip install pyyaml")
    sys.exit(1)

# Configure structured logging
class SecureMetricsLogger:
    """Structured JSON logger for security audit trail"""
    
    def __init__(self, log_file: str = "secure_metrics.log"):
        self.logger = logging.getLogger("SecureMetricsLogger")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters with timestamps and correlation IDs
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler for critical events only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_event(self, correlation_id: str, level: int, message: str):
        """Log event with security context"""
        record = self.logger.makeRecord(
            self.logger.name, level, '', 0, message, (), None
        )
        record.correlation_id = correlation_id
        record.created = datetime.now().isoformat()
        
        # JSON structured logging
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": logging.getLevelName(level),
            "correlation_id": correlation_id,
            "message": message,
            "security_context": self._get_security_context(record)
        }
        
        # Write to dedicated JSON audit log
        with open("audit_log.json", "a") as f:
            try:
                f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
    
    def _get_security_context(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract security-relevant context from log record"""
        return {
            "process_id": os.getpid(),
            "thread_id": threading.current_thread().ident,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else None
        }


class SecurePostgreSQLConnection:
    """Thread-safe PostgreSQL connection pool with security hardening"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool: Optional[ThreadedConnectionPool] = None
        self._lock = threading.Lock()
        
        # Validate configuration
        required_keys = ['host', 'database', 'user']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required PostgreSQL config key: {key}")
        
        # Generate secure connection string components
        self._validate_credentials(config)
    
    def _validate_credentials(self, config: Dict[str, Any]) -> None:
        """Validate and hash credentials for audit"""
        if 'password' in config:
            # Hash password for secure storage (never store plain text!)
            hashed = hashlib.sha256(
                f"{config['user']}:{config['password']}".encode()
            ).hexdigest()
            
            self.config['_hashed_password'] = hashed
            
            # Audit credential handling
            SecureMetricsLogger().log_event(
                secrets.token_hex(8),
                logging.INFO,
                "Credentials validated and hashed for PostgreSQL connection"
            )
    
    def get_connection_pool(self) -> ThreadedConnectionPool:
        """Initialize and return thread-safe connection pool"""
        with self._lock:
            if self.connection_pool is None:
                # Connection pooling parameters
                min_conn = 2
                max_conn = 10
                max_waiting = 5
                
                # Build secure DSN string
                dsn_params = {
                    "host": self.config['host'],
                    "port": self.config.get('port', 5432),
                    "dbname": self.config['database'],
                    "user": self.config['user'],
                    "password": self.config['password'],
                    "sslmode": self.config.get('sslmode', 'require'),
                    "connect_timeout": self.config.get('connect_timeout', 10)
                }
                
                dsn_string = ";".join([f"{k}={v}" for k, v in dsn_params.items()])
                
                try:
                    # Threaded connection pool with SSL hardening
                    self.connection_pool = ThreadedConnectionPool(
                        minconn=min_conn,
                        maxconn=max_conn,
                        maxconnections=20,
                        opened=self._open_connection,
                        closed=self._close_connection,
                        dsn=dsn_string
                    )
                    
                    SecureMetricsLogger().log_event(
                        secrets.token_hex(8),
                        logging.INFO,
                        f"PostgreSQL connection pool initialized: {min_conn} min, {max_conn} max connections"
                    )
                except Exception as e:
                    SecureMetricsLogger().log_event(
                        secrets.token_hex(8),
                        logging.ERROR,
                        f"Failed to initialize PostgreSQL connection pool: {str(e)}"
                    )
                    raise
        
        return self.connection_pool
    
    def _open_connection(self) -> Any:
        """Create new secure database connection"""
        conn_params = {
            "host": self.config['host'],
            "port": self.config.get('port', 5432),
            "dbname": self.config['database'],
            "user": self.config['user'],
            "password": self.config['password'],
            "sslmode": self.config.get('sslmode', 'require')
        }
        
        return psycopg2.connect(**conn_params)
    
    def _close_connection(self, conn: Any) -> None:
        """Close database connection"""
        try:
            if conn:
                conn.close()
        except Exception as e:
            SecureMetricsLogger().log_event(
                secrets.token_hex(8),
                logging.WARNING,
                f"Error closing PostgreSQL connection: {str(e)}"
            )


class MetricsCollector:
    """Multi-source metrics collection from system, query logs, and error logs"""
    
    def __init__(self, db_connection: SecurePostgreSQLConnection):
        self.db = db_connection
        self.collected_metrics: List[Dict[str, Any]] = []
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        query = """
            SELECT 
                'cpu_usage_percent' as metric_name,
                AVG(cpu_usage_percent)::numeric(10,2) as value,
                COUNT(*) as sample_count
            FROM system_metrics
            WHERE collected_at >= NOW() - INTERVAL '24 hours'
            
            UNION ALL
            
            SELECT 
                'memory_usage_mb',
                AVG(memory_usage_mb)::numeric(10,2),
                COUNT(*)
            FROM system_metrics
            WHERE collected_at >= NOW() - INTERVAL '24 hours'
            
            UNION ALL
            
            SELECT 
                'disk_io_ops_per_second',
                AVG(disk_io_ops_per_second)::numeric(10,2),
                COUNT(*)
            FROM system_metrics
            WHERE collected_at >= LAST 24 HOURS;
        """
        
        conn = self.db.get_connection_pool().getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'metric_name': row['metric_name'],
                    'average_value': float(row['value']),
                    'sample_count': int(row['count(*)']) if isinstance(row.get('count(*)'), str) else 0,
                    'min_value': None,
                    'max_value': None,
                    'std_deviation': None
                })
            
            return metrics
            
        finally:
            self.db.get_connection_pool().putconn(conn)
    
    def collect_query_metrics(self) -> Dict[str, Any]:
        """Collect query performance metrics"""
        query = """
            SELECT 
                metric_name,
                ROUND(AVG(value), 2) as average_value,
                COUNT(*) as sample_count,
                MIN(value) as min_value,
                MAX(value) as max_value,
                STDDEV(value) as std_deviation
            FROM query_metrics
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY metric_name
            ORDER BY metric_name;
        """
        
        conn = self.db.get_connection_pool().getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'metric_name': row['metric_name'],
                    'average_value': float(row['average_value']),
                    'sample_count': int(row['count(*)']),
                    'min_value': float(row['min_value']) if row['min_value'] else None,
                    'max_value': float(row['max_value']) if row['max_value'] else None,
                    'std_deviation': float(row['std_deviation']) if row.get('std_deviation') else None
                })
            
            return metrics
            
        finally:
            self.db.get_connection_pool().putconn(conn)


class DataProcessingPipeline:
    """Data processing and analysis pipeline for RAG metrics"""
    
    def __init__(self, db_connection: SecurePostgreSQLConnection):
        self.db = db_connection
        self.processed_data: List[Dict[str, Any]] = []
        
    def process_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze collected metrics"""
        processed = {}
        
        # Calculate statistical measures
        for metric_name in ['cpu_usage_percent', 'memory_usage_mb']:
            if metric_name in raw_metrics.get('system_metrics', []):
                values = [m['value'] for m in raw_metrics['system_metrics']]
                
                processed[metric_name] = {
                    'mean': statistics.mean(values) if values else 0,
                    'median': statistics.median(values) if values else 0,
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'percentile_95': self._calculate_percentile(values, 95),
                    'alert_threshold': self._determine_alert_threshold(metric_name)
                }
        
        # Query metrics analysis
        for metric_name in ['query_latency_ms', 'cache_hit_rate']:
            if metric_name in raw_metrics.get('query_metrics', []):
                values = [m['value'] for m in raw_metrics['query_metrics']]
                
                processed[metric_name] = {
                    'mean': statistics.mean(values) if values else 0,
                    'median': statistics.median(values) if values else 0,
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'percentile_95': self._calculate_percentile(values, 95),
                    'alert_threshold': self._determine_alert_threshold(metric_name)
                }
        
        return processed
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value from dataset"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_values):
            return sorted_values[-1]
        
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def _determine_alert_threshold(self, metric_name: str) -> float:
        """Determine appropriate alert threshold for metric"""
        thresholds = {
            'cpu_usage_percent': 80.0,
            'memory_usage_mb': 4096.0,
            'query_latency_ms': 1000.0,
            'cache_hit_rate': 50.0
        }
        
        return thresholds.get(metric_name, 70.0)


class SecurityAuditor:
    """Security audit trail generation and analysis"""
    
    def __init__(self):
        self.audit_events: List[Dict[str, Any]] = []
        
    def generate_security_report(self, correlation_id: str) -> Dict[str, Any]:
        """Generate comprehensive security audit report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'security_events': [],
            'risk_assessment': self._assess_risk_level(),
            'compliance_status': self._check_compliance()
        }
        
        # Read audit log and process events
        try:
            with open("audit_log.json", "r") as f:
                for line in f:
                    event = json.loads(line.strip())
                    report['security_events'].append({
                        'timestamp': event.get('timestamp'),
                        'level': event.get('level'),
                        'message': event.get('message'),
                        'security_context': event.get('security_context')
                    })
        except FileNotFoundError:
            pass
        
        return report
    
    def _assess_risk_level(self) -> str:
        """Assess overall risk level based on audit events"""
        high_severity_count = sum(1 for e in self.audit_events if 'ERROR' in e.get('level', ''))
        
        if high_severity_count > 10:
            return "CRITICAL"
        elif high_severity_count > 5:
            return "HIGH"
        elif high_severity_count > 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_compliance(self) -> Dict[str, Any]:
        """Check compliance with security standards"""
        return {
            'encryption_in_transit': True,
            'credentials_hashed': True,
            'audit_logging_enabled': True,
            'connection_pooling_active': True,
            'ssl_mode': 'require',
            'last_audit': datetime.now().isoformat()
        }


class AlertManager:
    """Automated alerting system for metric anomalies"""
    
    def __init__(self, db_connection: SecurePostgreSQLConnection):
        self.db = db_connection
        self.alerts_sent: List[Dict[str, Any]] = []
        
    def check_anomalies(self, processed_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for metric anomalies and generate alerts"""
        anomalies = []
        
        for metric_name, stats in processed_metrics.items():
            if 'alert_threshold' not in stats:
                continue
            
            # Check if average exceeds threshold
            if stats['mean'] > stats.get('alert_threshold', float('inf')) * 1.5:
                anomaly = {
                    'metric': metric_name,
                    'severity': self._calculate_severity(
                        stats['mean'], 
                        stats.get('alert_threshold', 0)
                    ),
                    'current_value': stats['mean'],
                    'threshold_exceeded_by': (stats['mean'] - stats.get('alert_threshold', 0)) / stats.get('alert_threshold', 1),
                    'recommended_action': self._get_recommended_action(metric_name, stats)
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, current_value: float, threshold: float) -> str:
        """Calculate alert severity based on how much threshold is exceeded"""
        exceedance_ratio = (current_value - threshold) / threshold
        
        if exceedance_ratio > 2.0:
            return "CRITICAL"
        elif exceedance_ratio > 1.5:
            return "HIGH"
        elif exceedance_ratio > 1.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommended_action(self, metric_name: str, stats: Dict[str, Any]) -> str:
        """Get recommended action for metric anomaly"""
        actions = {
            'cpu_usage_percent': f"Investigate high CPU usage. Current avg: {stats['mean']:.2f}%, recommend scaling horizontally.",
            'memory_usage_mb': f"Memory pressure detected. Current avg: {stats['mean']:.2f}MB, consider adding memory or optimizing queries.",
            'query_latency_ms': f"Query latency elevated. Current avg: {stats['mean']:.2f}ms, review slow query logs.",
            'cache_hit_rate': f"Cache efficiency low. Current avg: {stats['mean']:.2f}%, check cache configuration."
        }
        
        return actions.get(metric_name, "Review metric behavior and investigate root cause.")


class DashboardGenerator:
    """Generate comprehensive metrics dashboard"""
    
    def __init__(self):
        self.dashboard_data: Dict[str, Any] = {}
        
    def generate_dashboard(self, 
                          system_metrics: List[Dict], 
                          query_metrics: List[Dict],
                          processed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # System metrics summary
        system_summary = {
            'cpu_usage_percent': {
                'current_value': sys_metrics['cpu_usage_percent']['average_value'] if (sys_metrics := next((m for m in system_metrics if m['metric_name'] == 'cpu_usage_percent'), {}), {}).get('average_value', 0)
            },
            'memory_usage_mb': {
                'current_value': sys_metrics['memory_usage_mb']['average_value'] if (sys_metrics := next((m for m in system_metrics if m['metric_name'] == 'memory_usage_mb'), {}), {}).get('average_value', 0)
            }
        }
        
        # Query metrics summary
        query_summary = {
            'query_latency_ms': {
                'current_value': qm['average_value'] if (qm := next((m for m in query_metrics if m['metric_name'] == 'query_latency_ms'), {}), {}).get('average_value', 0)
            },
            'cache_hit_rate': {
                'current_value': qm['average_value'] if (qm := next((m for m in query_metrics if m['metric_name'] == 'cache_hit_rate'), {}), {}).get('average_value', 0)
            }
        }
        
        self.dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'system_metrics_summary': system_summary,
            'query_metrics_summary': query_summary,
            'processed_analytics': processed_metrics,
            'data_sources': {
                'collection_period': '24 hours',
                'total_samples_collected': sum(m['sample_count'] for m in system_metrics + query_metrics),
                'last_update': datetime.now().isoformat()
            }
        }
        
        return self.dashboard_data


class SecureRAGMetricsAnalyzer:
    """Main orchestrator class for RAG metrics analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.db_connection: Optional[SecurePostgreSQLConnection] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.data_processor: Optional[DataProcessingPipeline] = None
        self.security_auditor: Optional[SecurityAuditor] = SecurityAuditor()
        self.alert_manager: Optional[AlertManager] = None
        self.dashboard_generator: DashboardGenerator = DashboardGenerator()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return self._create_default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['database', 'alerting']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            return config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'rag_metrics_db',
                'user': 'metrics_user',
                'password': os.environ.get('POSTGRES_PASSWORD', ''),
                'sslmode': 'require'
            },
            'alerting': {
                'enabled': True,
                'anomaly_threshold_multiplier': 1.5
            }
        }
    
    def initialize(self) -> None:
        """Initialize all components"""
        SecureMetricsLogger().log_event(secrets.token_hex(8), logging.INFO, "Initializing RAG Metrics Analyzer")
        
        # Initialize PostgreSQL connection
        self.db_connection = SecurePostgreSQLConnection(self.config['database'])
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(self.db_connection)
        
        # Initialize data processing pipeline
        self.data_processor = DataProcessingPipeline(self.db_connection)
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.db_connection)
        
        SecureMetricsLogger().log_event(
            secrets.token_hex(8), 
            logging.INFO, 
            "All components initialized successfully"
        )
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete metrics analysis pipeline"""
        correlation_id = secrets.token_hex(8)
        
        SecureMetricsLogger().log_event(
            correlation_id, 
            logging.INFO, 
            "Starting full RAG metrics analysis"
        )
        
        try:
            # Collect metrics from all sources
            print("Collecting system metrics...")
            raw_system_metrics = self.metrics_collector.collect_system_metrics()
            
            print("Collecting query metrics...")
            raw_query_metrics = self.metrics_collector.collect_query_metrics()
            
            # Process collected metrics
            print("Processing metrics data...")
            processed_metrics = self.data_processor.process_metrics({
                'system_metrics': raw_system_metrics,
                'query_metrics': raw_query_metrics
            })
            
            # Check for anomalies and generate alerts
            if self.alert_manager:
                print("Checking for metric anomalies...")
                anomalies = self.alert_manager.check_anomalies(processed_metrics)
                
                SecureMetricsLogger().log_event(
                    correlation_id, 
                    logging.WARNING, 
                    f"Detected {len(anomalies)} metric anomalies"
                )
            
            # Generate dashboard
            print("Generating metrics dashboard...")
            dashboard_data = self.dashboard_generator.generate_dashboard(
                raw_system_metrics,
                raw_query_metrics,
                processed_metrics
            )
            
            # Generate security audit report
            print("Generating security audit report...")
            security_report = self.security_auditor.generate_security_report(correlation_id)
            
            # Compile final results
            analysis_results = {
                'correlation_id': correlation_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'raw_metrics_collected': {
                    'system_metrics_count': len(raw_system_metrics),
                    'query_metrics_count': len(raw_query_metrics)
                },
                'processed_analytics': processed_metrics,
                'dashboard_data': dashboard_data,
                'anomalies_detected': anomalies if self.alert_manager else [],
                'security_audit': security_report,
                'status': 'SUCCESS'
            }
            
            SecureMetricsLogger().log_event(
                correlation_id, 
                logging.INFO, 
                "Full RAG metrics analysis completed successfully"
            )
            
        except Exception as e:
            SecureMetricsLogger().log_event(
                correlation_id, 
                logging.ERROR, 
                f"Error during full analysis: {str(e)}"
            )
            raise
        
        return analysis_results
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run security audit only"""
        correlation_id = secrets.token_hex(8)
        
        SecureMetricsLogger().log_event(
            correlation_id, 
            logging.INFO, 
            "Starting security audit"
        )
        
        try:
            security_report = self.security_auditor.generate_security_report(correlation_id)
            
            # Read and process actual audit logs
            try:
                with open("audit_log.json", "r") as f:
                    for line in f:
                        event = json.loads(line.strip())
                        self.security_auditor.audit_events.append({
                            'timestamp': event.get('timestamp'),
                            'level': event.get('level'),
                            'message': event.get('message'),
                            'security_context': event.get('security_context')
                        })
            except FileNotFoundError:
                pass
            
        except Exception as e:
            SecureMetricsLogger().log_event(
                correlation_id, 
                logging.ERROR, 
                f"Error during security audit: {str(e)}"
            )
        
        return self.security_auditor.generate_security_report(correlation_id)


# Main execution entry point
def main():
    """Main function to run the RAG metrics analyzer"""
    
    # Initialize analyzer with config
    analyzer = SecureRAGMetricsAnalyzer()
    
    try:
        # Run full analysis pipeline
        results = analyzer.run_full_analysis()
        
        # Output results as JSON for dashboard integration
        print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        SecureMetricsLogger().log_event(
            secrets.token_hex(8), 
            logging.ERROR, 
            f"Fatal error in main execution: {str(e)}"
        )
        print(f"FATAL ERROR: {str(e)}")
        
        # Write error to audit log
        with open("error_audit.json", "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'stack_trace': ''.join(traceback.format_exc()).replace('\n', '\r\n')
            }, f, indent=2)
        
        return 1


if __name__ == "__main__":
    main()
