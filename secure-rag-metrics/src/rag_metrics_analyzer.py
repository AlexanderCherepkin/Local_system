#!/usr/bin/env python3
"""
Secure RAG Metrics Analyzer System
===================================
Comprehensive metrics extraction, analysis, and visualization for RAG systems.
Features: PostgreSQL integration, detailed logging, security hardening, data validation.

Author: AI Security Analyst
Date: 2024-10-30
Security Level: ENTERPRISE-GRADE
"""

import os
import sys
import json
import time
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
import argparse
from pathlib import Path

# PostgreSQL imports with security considerations
try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)


# ============================================================================
# CONFIGURATION & SECURITY SETTINGS
# ============================================================================

@dataclass
class SecurityConfig:
    """Security configuration for the RAG metrics analyzer."""
    
    # Input validation settings
    MAX_QUERY_LENGTH: int = 5000
    ALLOWED_TABLES: List[str] = None
    
    # SQL injection prevention
    ENABLE_PARAM_STYLING: bool = True
    SANITIZE_INPUTS: bool = True
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "rag_metrics_analyzer.log"
    
    # Session management
    SESSION_TIMEOUT_SECONDS: int = 3600
    
    def __post_init__(self):
        if self.ALLOWED_TABLES is None:
            self.ALLOWED_TABLES = [
                'metrics', 
                'documents', 
                'chunks',
                'queries',
                'answers',
                'vector_embeddings',
                'security_logs'
            ]


# Global security configuration instance
SECURITY_CONFIG = SecurityConfig()

# ============================================================================
# LOGGER SETUP - COMPREHENSIVE LOGGING SYSTEM
# ============================================================================

def setup_logger(
    name: str = "RAGMetricsAnalyzer",
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up a comprehensive logger with multiple handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Use config default if not specified
    if log_file is None:
        log_file = SECURITY_CONFIG.LOG_FILE
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler (colorized output simulation)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler for persistent logging
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(getattr(logging, level.upper()))
    file_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()

# ============================================================================
# DATA CLASSES - METRIC STRUCTURES
# ============================================================================

@dataclass
class RAGMetric:
    """Represents a single RAG metric record."""
    id: int
    timestamp: datetime
    metric_type: str
    value: float
    context_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'error_details': self.error_details
        }


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

class InputValidator:
    """Validate and sanitize user inputs for security."""
    
    @staticmethod
    def validate_table_name(table_name: str) -> bool:
        """Validate table name against whitelist."""
        if not table_name or not isinstance(table_name, str):
            return False
        
        # Check against allowed tables
        normalized_name = table_name.lower().strip()
        if normalized_name in [t.lower().strip() for t in SECURITY_CONFIG.ALLOWED_TABLES]:
            return True
        
        logger.warning(f"Table '{table_name}' not in whitelist. Allowed: {SECURITY_CONFIG.ALLOWED_TABLES}")
        return False
    
    @staticmethod
    def sanitize_input(value: Any, max_length: int = None) -> str:
        """Sanitize input to prevent injection attacks."""
        if value is None:
            return ""
        
        if not isinstance(value, (str,)):
            value = str(value)
        
        # Remove potentially dangerous characters for string inputs
        dangerous_patterns = [
            r';\s*--',  # SQL comment
            r'/\*',     # Block comment start
            r'\*/',     # Block comment end
            r'\\x',     # Hex escape sequences
        ]
        
        safe_value = value
        
        for pattern in dangerous_patterns:
            import re
            safe_value = re.sub(pattern, '', safe_value)
        
        # Truncate if exceeds max length
        if max_length and len(safe_value) > max_length:
            logger.warning(f"Input truncated from {len(value)} to {max_length} characters")
            safe_value = safe_value[:max_length]
        
        return safe_value.strip()


class SQLInjectionPreventer:
    """Prevent SQL injection attacks using parameterized queries."""
    
    @staticmethod
    def validate_query(query: str, allowed_tables: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate query for potential SQL injection."""
        
        # Check table names in query
        import re
        
        # Extract table references from query
        table_pattern = r'\b(FROM|JOIN)\s+["\']?(\w+)["\']?'
        matches = re.findall(table_pattern, query.lower())
        
        for match in matches:
            table_name = match[1].lower().strip() if len(match) > 1 else ""
            
            # Check against whitelist
            normalized_allowed = [t.lower().strip() for t in allowed_tables]
            if table_name not in normalized_allowed:
                logger.error(f"SQL injection attempt detected: unauthorized table '{table_name}'")
                return False, f"Unauthorized table access: {table_name}"
        
        # Check for comment patterns
        dangerous_patterns = [
            r'--',      # SQL comment
            r'/\*.*?\*/',  # Block comment (non-greedy)
            r';\s*$',   # Statement terminator at end
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, query):
                logger.error(f"SQL injection attempt detected: {query[:100]}")
                return False, "Suspicious SQL patterns detected"
        
        return True, None


# ============================================================================
# POSTGRESQL DATABASE MANAGER - SECURE CONNECTION HANDLING
# ============================================================================

class PostgreSQLManager:
    """Secure PostgreSQL connection and query manager."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL manager with secure configuration.
        
        Args:
            config: Database configuration dictionary containing:
                - host: Database hostname
                - port: Database port
                - database: Database name
                - user: Username
                - password: Password (use environment variable in production)
        """
        self.config = config
        self.connection_pool = []
        
    def create_connection(self, connection_string: str) -> Tuple[bool, Optional[str]]:
        """
        Create a secure PostgreSQL connection.
        
        Args:
            connection_string: Connection string or tuple of parameters
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Validate connection details
            if isinstance(connection_string, str):
                # Parse connection string
                import re
                pattern = r'postgresql://([^:]+):([^@]+)@([^/]+)/(\d+)/([^?]+)'
                match = re.match(pattern, connection_string)
                
                if not match:
                    return False, "Invalid PostgreSQL connection string format"
                
                host, user, password, port, database = match.groups()
                
            elif isinstance(connection_string, dict):
                host = connection_string.get('host', 'localhost')
                user = connection_string.get('user', '')
                password = connection_string.get('password', '')
                port = int(connection_string.get('port', 5432))
                database = connection_string.get('database', '')
            else:
                return False, "Invalid connection string type"
            
            # Validate required fields
            if not all([host, user, database]):
                missing = [f for f in ['host', 'user', 'database'] 
                          if not self.config.get(f)]
                return False, f"Missing configuration: {missing}"
            
            # Create connection with security settings
            conn_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"
            )
            
            logger.info(f"Creating PostgreSQL connection to {self.config.get('host', 'localhost')}")
            
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                sslmode='require'  # Enforce SSL for security
            )
            
            self.connection_pool.append(conn)
            logger.info(f"Successfully connected to PostgreSQL database")
            
            return True, None
            
        except psycopg2.Error as e:
            error_msg = f"PostgreSQL connection error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected connection error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def execute_query_secure(self, query_template: str, params: Tuple, 
                            allowed_tables: List[str]) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a secure PostgreSQL query with parameterization.
        
        Args:
            query_template: SQL query with placeholders ($1, $2, etc.)
            params: Query parameters (tuples)
            allowed_tables: List of allowed table names
            
        Returns:
            Tuple of (success, result_or_error, error_message)
        """
        # Validate against SQL injection
        is_valid, validation_error = SQLInjectionPreventer.validate_query(
            query_template, allowed_tables
        )
        
        if not is_valid:
            logger.error(f"SQL validation failed: {validation_error}")
            return False, None, validation_error
        
        try:
            # Use connection from pool or create new one
            conn = self.connection_pool[0] if self.connection_pool else None
            
            if conn is None:
                logger.warning("No active database connection. Creating new connection...")
                success, error_msg = self.create_connection(self.config)
                if not success:
                    return False, None, error_msg
                conn = self.connection_pool[0]
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Use parameterized query for security
                logger.debug(f"Executing secure query (sanitized): {query_template}")
                cursor.execute(query_template, params)
                
                if query_template.lower().strip().endswith(('SELECT', 'EXPLAIN')):
                    result = cursor.fetchall()
                    conn.commit()
                    return True, result, None
                    
                elif query_template.lower().strip().startswith('INSERT') or \
                     query_template.lower().strip().startswith('UPDATE'):
                    affected_rows = cursor.rowcount
                    conn.commit()
                    return True, {'affected_rows': affected_rows}, None
                    
                else:
                    # For other queries (DDL, etc.)
                    conn.commit()
                    return True, {'status': 'executed'}, None
                    
        except psycopg2.Error as e:
            error_msg = f"PostgreSQL query execution error: {str(e)}"
            logger.error(error_msg)
            
            # Rollback on error
            try:
                if conn and conn.autocommit is False:
                    conn.rollback()
            except:
                pass
            
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected query execution error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def close_connections(self):
        """Close all database connections."""
        for conn in self.connection_pool:
            try:
                if not conn.closed:
                    conn.close()
                    logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
        
        self.connection_pool.clear()


# ============================================================================
# RAG METRICS DATA RETRIEVAL - SECURE QUERY EXECUTION
# ============================================================================

class RAGMetricsRetriever:
    """Secure retrieval of RAG metrics from PostgreSQL."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize the metrics retriever.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_manager = PostgreSQLManager(db_config)
        self.logger = logger
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        
        try:
            # Create or reuse connection
            if not self.db_manager.connection_pool:
                success, error_msg = self.db_manager.create_connection(self.db_config)
                if not success:
                    raise Exception(error_msg)
            
            conn = self.db_manager.connection_pool[0]
            yield conn
            
        finally:
            # Connection cleanup handled by the manager
            pass
    
    def get_metrics_summary(self, 
                           metric_types: List[str] = None,
                           start_date: str = None,
                           end_date: str = None) -> AnalysisResult:
        """
        Get a summary of RAG metrics.
        
        Args:
            metric_types: Filter by metric types (e.g., ['accuracy', 'latency'])
            start_date: Start date filter (YYYY-MM-DD format)
            end_date: End date filter (YYYY-MM-DD format)
            
        Returns:
            AnalysisResult with metrics summary or error details
        """
        timestamp = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{session_id}] Starting metrics retrieval")
        
        try:
            # Build query based on filters
            base_query = """
                SELECT 
                    id,
                    metric_type,
                    value,
                    timestamp::timestamp with time zone as normalized_timestamp,
                    context_id
                FROM metrics
                WHERE 1=1
            """
            
            params = []
            
            # Add metric type filter if specified
            if metric_types:
                metric_types_param = [f"'{type_}'" for type_ in metric_types]
                base_query += f"\nAND metric_type IN ({','.join(metric_types_param)})"
                params.append(tuple(metric_types))
            
            # Add date filters if specified
            if start_date or end_date:
                date_conditions = []
                
                if start_date:
                    date_conditions.append("(normalized_timestamp >= :start_date)")
                    params.append(start_date)
                
                if end_date:
                    date_conditions.append("(normalized_timestamp <= :end_date)")
                    params.append(end_date)
                
                if date_conditions:
                    base_query += "\nAND " + " AND ".join(date_conditions)
            
            # Execute secure query
            success, result, error = self.db_manager.execute_query_secure(
                base_query, 
                tuple(params),
                SECURITY_CONFIG.ALLOWED_TABLES
            )
            
            if not success:
                return AnalysisResult(
                    success=False,
                    message="Failed to retrieve metrics summary",
                    error_details=error
                )
            
            # Convert results to RAGMetric objects
            metrics_data = []
            for row in result:
                metric = RAGMetric(
                    id=int(row['id']),
                    timestamp=row['normalized_timestamp'],
                    metric_type=row['metric_type'] or '',
                    value=float(row['value']) if row.get('value') else 0.0,
                    context_id=int(row['context_id']) if row.get('context_id') else None
                )
                metrics_data.append(metric.to_dict())
            
            self.logger.info(f"[{session_id}] Successfully retrieved {len(metrics_data)} metrics")
            
            return AnalysisResult(
                success=True,
                message=f"Retrieved {len(metrics_data)} metrics successfully",
                data=metrics_data
            )
            
        except Exception as e:
            error_msg = f"Unexpected error in get_metrics_summary: {str(e)}"
            self.logger.error(error_msg)
            
            return AnalysisResult(
                success=False,
                message="Failed to retrieve metrics summary",
                error_details=error_msg
            )
    
    def get_document_statistics(self, 
                               document_id: int = None,
                               chunk_size: int = None) -> AnalysisResult:
        """
        Get statistics for documents and chunks.
        
        Args:
            document_id: Filter by specific document ID (optional)
            chunk_size: Minimum chunk size filter (optional)
            
        Returns:
            AnalysisResult with document/chunk statistics
        """
        timestamp = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{session_id}] Starting document statistics retrieval")
        
        try:
            base_query = """
                SELECT 
                    d.id as document_id,
                    COUNT(DISTINCT c.id) as chunk_count,
                    AVG(c.word_count) as avg_chunk_size,
                    MIN(c.created_at) as first_chunk_date,
                    MAX(c.created_at) as last_chunk_date
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id
            """
            
            params = []
            
            # Add document ID filter if specified
            if document_id:
                base_query += "\nWHERE d.id = :document_id"
                params.append(document_id)
            
            # Execute secure query
            success, result, error = self.db_manager.execute_query_secure(
                base_query,
                tuple(params),
                SECURITY_CONFIG.ALLOWED_TABLES
            )
            
            if not success:
                return AnalysisResult(
                    success=False,
                    message="Failed to retrieve document statistics",
                    error_details=error
                )
            
            # Convert results to dictionaries
            stats_data = []
            for row in result:
                stat = {
                    'document_id': int(row['document_id']),
                    'chunk_count': int(row['chunk_count']) if row.get('chunk_count') else 0,
                    'avg_chunk_size': float(row['avg_chunk_size']) if row.get('avg_chunk_size') else 0.0,
                    'first_chunk_date': str(row['first_chunk_date']),
                    'last_chunk_date': str(row['last_chunk_date'])
                }
                stats_data.append(stat)
            
            self.logger.info(f"[{session_id}] Successfully retrieved {len(stats_data)} document statistics")
            
            return AnalysisResult(
                success=True,
                message=f"Retrieved {len(stats_data)} document statistics successfully",
                data=stats_data
            )
            
        except Exception as e:
            error_msg = f"Unexpected error in get_document_statistics: {str(e)}"
            self.logger.error(error_msg)
            
            return AnalysisResult(
                success=False,
                message="Failed to retrieve document statistics",
                error_details=error_msg
            )


# ============================================================================
# METRICS ANALYZER - DATA PROCESSING AND VISUALIZATION PREPARATION
# ============================================================================

class RAGMetricsAnalyzer:
    """Analyze and process RAG metrics for visualization."""
    
    def __init__(self):
        """Initialize the metrics analyzer."""
        self.logger = logger
    
    def calculate_statistics(self, 
                            data: List[Dict[str, Any]], 
                            metric_type: str) -> Dict[str, float]:
        """
        Calculate statistical measures for a given metric type.
        
        Args:
            data: List of metric dictionaries
            metric_type: Type of metric to analyze
            
        Returns:
            Dictionary with calculated statistics
        """
        # Filter by metric type and extract values
        filtered_data = [item for item in data if item.get('metric_type') == metric_type]
        
        if not filtered_data:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std_deviation': 0.0
            }
        
        values = [item['value'] for item in filtered_data if isinstance(item.get('value'), (int, float))]
        
        if not values:
            return {
                'count': len(filtered_data),
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std_deviation': 0.0
            }
        
        import statistics
        
        count = len(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        
        try:
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        except statistics.StatisticsError:
            std_dev = 0.0
        
        return {
            'count': count,
            'mean': round(mean_val, 4),
            'median': round(median_val, 4),
            'min': round(min_val, 4),
            'max': round(max_val, 4),
            'std_deviation': round(std_dev, 4)
        }
    
    def generate_time_series_analysis(self, 
                                      data: List[Dict[str, Any]], 
                                      time_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Generate time series analysis for metrics.
        
        Args:
            data: List of metric dictionaries
            time_column: Column name containing timestamp
            
        Returns:
            Dictionary with time series analysis results
        """
        # Sort by time column
        sorted_data = sorted(data, key=lambda x: x.get(time_column))
        
        if not sorted_data:
            return {
                'trend': 'insufficient_data',
                'time_points': 0,
                'analysis': []
            }
        
        # Calculate metrics over time windows
        time_windows = self._calculate_time_windows(sorted_data)
        
        analysis_results = []
        for window in time_windows:
            stats = self.calculate_statistics(window['metrics'], window['metric_type'])
            
            analysis_results.append({
                'time_window': window['window_start'] if isinstance(window['window_start'], str) else str(window['window_start']),
                'metric_type': window['metric_type'],
                'statistics': stats,
                'trend_direction': self._determine_trend(stats)
            })
        
        return {
            'total_time_points': len(analysis_results),
            'time_windows': analysis_results
        }
    
    def _calculate_time_windows(self, data: List[Dict[str, Any]], 
                               window_size_seconds: int = 3600) -> List[Dict[str, Any]]:
        """Calculate time windows for metrics."""
        if not data or len(data) < 2:
            return [{'window_start': 'N/A', 'metrics': [], 'metric_type': ''}]
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.get('timestamp'))
        
        windows = []
        current_window_start = sorted_data[0].get('timestamp')
        window_metrics = []
        metric_types_seen = set()
        
        for item in sorted_data:
            ts = item.get('timestamp')
            
            # Check if we need to start a new window
            if isinstance(current_window_start, datetime) and isinstance(ts, datetime):
                time_diff = (ts - current_window_start).total_seconds()
                
                if time_diff >= window_size_seconds or len(window_metrics) > 0:
                    windows.append({
                        'window_start': current_window_start.isoformat(),
                        'metrics': window_metrics,
                        'metric_type': sorted(metric_types_seen)[0] if metric_types_seen else ''
                    })
                    
                    # Reset for new window
                    current_window_start = ts
                    window_metrics = []
                    metric_types_seen.clear()
            
            window_metrics.append(item)
            metric_types_seen.add(item.get('metric_type'))
        
        # Add last window
        if window_metrics:
            windows.append({
                'window_start': current_window_start.isoformat(),
                'metrics': window_metrics,
                'metric_type': sorted(metric_types_seen)[0] if metric_types_seen else ''
            })
        
        return windows
    
    def _determine_trend(self, stats: Dict[str, float]) -> str:
        """Determine trend direction based on statistics."""
        if stats['count'] < 2:
            return 'insufficient_data'
        
        # Simple linear regression to determine trend
        import math
        
        n = len(stats.get('values', [])) if hasattr(stats, 'get') else 0
        if n < 2:
            return 'no_trend'
        
        values = stats.get('values', [])
        x_values = list(range(len(values)))
        y_values = values
        
        # Calculate slope using least squares
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)
        
        denominator = n * sum_xx - sum_x ** 2
        
        if denominator == 0:
            return 'no_trend'
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        if abs(slope) < 1e-10:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'


# ============================================================================
# MAIN APPLICATION CLASS - ORCHESTRATION
# ============================================================================

class RAGMetricsApplication:
    """Main application class for the secure RAG metrics system."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize the main application.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.retriever = RAGMetricsRetriever(db_config)
        self.analyzer = RAGMetricsAnalyzer()
        
        # Generate session ID for tracking
        self.session_id = str(uuid.uuid4())[:8]
        
    def run_analysis(self, 
                     metric_types: List[str] = None,
                     start_date: str = None,
                     end_date: str = None) -> Dict[str, Any]:
        """
        Run complete RAG metrics analysis.
        
        Args:
            metric_types: Types of metrics to analyze
            start_date: Start date filter (YYYY-MM-DD format)
            end_date: End date filter (YYYY-MM-DD format)
            
        Returns:
            Dictionary containing analysis results
        """
        timestamp = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info(f"SESSION: {self.session_id}")
        self.logger.info("RAG METRICS ANALYSIS STARTED")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Retrieve metrics
            self.logger.info("\n[STEP 1] Retrieving metrics from database...")
            retrieval_result = self.retriever.get_metrics_summary(
                metric_types=metric_types,
                start_date=start_date,
                end_date=end_date
            )
            
            if not retrieval_result.success:
                error_msg = f"Metrics retrieval failed: {retrieval_result.error_details}"
                self.logger.error(error_msg)
                
                return {
                    'session_id': self.session_id,
                    'status': 'error',
                    'step': 1,
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            
            metrics_data = retrieval_result.data or []
            self.logger.info(f"Retrieved {len(metrics_data)} metric records")
            
            # Step 2: Analyze statistics by metric type
            self.logger.info("\n[STEP 2] Calculating statistics...")
            unique_types = set(item.get('metric_type') for item in metrics_data if item.get('metric_type'))
            
            stats_by_type = {}
            for mtype in unique_types:
                stats = self.analyzer.calculate_statistics(metrics_data, mtype)
                stats_by_type[mtype] = {
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'median': stats['median'],
                    'min': stats['min'],
                    'max': stats['max']
                }
            
            self.logger.info(f"Calculated statistics for {len(stats_by_type)} metric types")
            
            # Step 3: Generate time series analysis (if enough data)
            if len(metrics_data) >= 2:
                self.logger.info("\n[STEP 3] Generating time series analysis...")
                
                ts_analysis = {}
                for mtype in unique_types:
                    filtered_data = [item for item in metrics_data 
                                   if item.get('metric_type') == mtype]
                    
                    if len(filtered_data) >= 2:
                        ts_result = self.analyzer.generate_time_series_analysis(
                            filtered_data, 'timestamp'
                        )
                        ts_analysis[mtype] = ts_result
                
                self.logger.info(f"Time series analysis complete for {len(ts_analysis)} metric types")
            
            # Step 4: Document statistics (if available)
            self.logger.info("\n[STEP 4] Retrieving document statistics...")
            doc_stats_result = self.retriever.get_document_statistics()
            
            if doc_stats_result.success:
                self.logger.info(f"Retrieved {len(doc_stats_result.data or [])} document statistics")
            
            # Step 5: Compile final results
            self.logger.info("\n[STEP 5] Compiling analysis results...")
            
            final_results = {
                'session_id': self.session_id,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'query_parameters': {
                    'metric_types': metric_types or 'all',
                    'start_date': start_date,
                    'end_date': end_date
                },
                'metrics_summary': metrics_data[:100],  # Limit for output (first 100)
                'statistics_by_type': stats_by_type,
                'time_series_analysis': ts_analysis if len(metrics_data) >= 2 else None,
                'document_statistics': doc_stats_result.data or []
            }
            
            self.logger.info("=" * 80)
            self.logger.info("RAG METRICS ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Unexpected error in run_analysis: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'session_id': self.session_id,
                'status': 'error',
                'step': 5,
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }


# ============================================================================
# COMMAND LINE INTERFACE - USER-FRIENDLY INPUT HANDLING
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Secure RAG Metrics Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --host localhost --user admin --database rag_metrics
  %(prog)s --host localhost --user admin --database rag_metrics --metric_types accuracy,latency
  %(prog)s --config config.yaml
        """
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )
    
    parser.add_argument(
        '--database', 
        type=str, 
        required=True,
        help='PostgreSQL database name'
    )
    
    parser.add_argument(
        '--user', 
        type=str, 
        required=True,
        help='Database username'
    )
    
    parser.add_argument(
        '--password', 
        type=str, 
        default=None,
        help='Database password (optional - use environment variable RAG_DB_PASSWORD)'
    )
    
    parser.add_argument(
        '--metric-types', 
        type=str, 
        default=None,
        help='Comma-separated list of metric types to analyze'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        default=None,
        help='Start date filter (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        default=None,
        help='End date filter (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--config-file', 
        type=str, 
        default=None,
        help='Path to configuration YAML file'
    )
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a dictionary")
        
        return config
        
    except ImportError:
        print("ERROR: PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration from {config_path}: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the application."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get password from environment variable if not provided
    if not args.password:
        args.password = os.environ.get('RAG_DB_PASSWORD', None)
    
    if not args.password and args.host != 'localhost':
        print("WARNING: No password provided. Consider using environment variable RAG_DB_PASSWORD")
    
    # Load configuration from YAML file if specified
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password or ''  # Empty string for passwordless auth if needed
    }
    
    if args.config_file:
        try:
            loaded_config = load_config_from_yaml(args.config_file)
            
            # Merge with command line arguments (CLI takes precedence)
            db_config.update({
                'host': loaded_config.get('database', {}).get('host', args.host),
                'port': loaded_config.get('database', {}).get('port', args.port),
                'database': loaded_config.get('database', {}).get('name', args.database),
                'user': loaded_config.get('database', {}).get('username', args.user),
                'password': loaded_config.get('database', {}).get('password', '')
            })
        except Exception as e:
            print(f"WARNING: Could not load configuration file: {str(e)}")
    
    # Initialize and run application
    app = RAGMetricsApplication(db_config)
    
    # Parse metric types from comma-separated string
    metric_types = None
    if args.metric_types:
        metric_types = [t.strip() for t in args.metric_types.split(',')]
    
    # Run analysis
    results = app.run_analysis(
        metric_types=metric_types,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Output results as JSON
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    try:
        import json
        
        if 'status' in results and results['status'] == 'success':
            print(f"\nStatus: SUCCESS")
            print(f"Session ID: {results.get('session_id', 'N/A')}")
            print(f"Timestamp: {results.get('timestamp', 'N/A')}")
            
            # Output JSON for programmatic access
            output_data = {
                'status': results['status'],
                'metrics_count': len(results.get('metrics_summary', [])),
                'statistics_by_type': results.get('statistics_by_type', {}),
                'document_statistics_count': len(results.get('document_statistics', []))
            }
            
            print("\nJSON Output:")
            print(json.dumps(output_data, indent=2))
        
        else:
            print(f"\nStatus: ERROR")
            print(f"Message: {results.get('message', 'Unknown error')}")
            if 'error_details' in results and results['status'] != 'error':
                print(f"Details: {results['error_details']}")
    
    except Exception as e:
        print(f"\nERROR formatting output: {str(e)}")


if __name__ == "__main__":
    main()
