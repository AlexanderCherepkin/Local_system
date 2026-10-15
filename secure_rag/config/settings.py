"""Centralized configuration for Secure RAG.

This module provides centralized configuration management with:
- Environment variable support
- Type validation and constraints
- Multiple fallback strategies (Pydantic v2 > os.environ > defaults)
- Comprehensive error handling and logging

Environment Variables:
    OLLAMA_HOST: Ollama server host (default: localhost)
    OLLAMA_PORT: Ollama server port (default: 11434)
    DEFAULT_MODEL: Default LLM model name (default: qwen2.5)
    CHUNK_SIZE: Text chunk size in tokens (default: 500)
    CHUNK_OVERLAP: Overlap between chunks (default: 50)
    TOP_K_RESULTS: Number of retrieval results (default: 5)
    SIMILARITY_THRESHOLD: Similarity threshold 0-1 (default: 0.7)
    MAX_PROMPT_CHARS: Maximum prompt length (default: 4096)
    VECTOR_DB_PATH: Vector database path relative to cwd
    SECURITY_CONFIG: Security configuration file path
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from logging import getLogger

logger = getLogger(__name__)

try:
    from pydantic_settings import BaseSettings as _PydanticBaseSettings
    from pydantic import field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class SettingsConfigDict(dict):
    """Extended configuration dictionary with validation support."""
    pass


def _load_env_var(key: str) -> Optional[str]:
    """Load environment variable safely."""
    return os.environ.get(key)


def _parse_int(value: Any, default: int) -> int:
    """Parse integer with validation and fallback."""
    if isinstance(value, int):
        return value
    try:
        parsed = int(value)
        if parsed < 0 or parsed > 10**6:  # Reasonable bounds check
            raise ValueError(f"Value {parsed} out of valid range")
        return parsed
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid integer value for '{value}': {e}")
        return default


def _parse_float(value: Any, default: float) -> float:
    """Parse float with validation and fallback."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = float(value)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"Value {parsed} out of valid range [0, 1]")
        return parsed
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid float value for '{value}': {e}")
        return default


def _validate_path(path: str, env_key: Optional[str] = None) -> str:
    """Validate and resolve path to absolute form."""
    if not path or not Path(path).exists():
        raise ValueError(f"Invalid path '{path}'")
    return os.path.abspath(path)


class Settings:
    """Centralized configuration for Secure RAG system.

    Attributes:
        ollama_host: Ollama server host address
        ollama_port: Ollama server port number
        default_model: Default LLM model name identifier
        chunk_size: Text document chunk size in tokens/symbols
        chunk_overlap: Overlap between consecutive chunks
        top_k_results: Number of retrieval results to return
        similarity_threshold: Similarity score threshold (0.0-1.0)
        max_prompt_chars: Maximum allowed prompt character count
        vector_db_path: Absolute path to ChromaDB storage directory
        security_config: Absolute path to ACL configuration file
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize settings from environment variables with validation.
        
        Args:
            config_file: Optional path to configuration file (future extensibility)
        
        Raises:
            ValueError: If critical required parameters are missing
        """
        if PYDANTIC_AVAILABLE and config_file:
            logger.info("Initializing Pydantic-based settings")
            self._load_from_pydantic(config_file)
        else:
            logger.warning("Using fallback environment variable loading")
            self._load_from_env()
    
    def _load_from_pydantic(self, config_file: str) -> None:
        """Load settings using Pydantic (if available)."""
        try:
            from pydantic_settings import SettingsConfigDict as PydanticSettingsConfig
            from pydantic import Field
            
            class PydanticSettings(_PydanticBaseSettings):
                model_config = PydanticSettingsConfig(
                    env_prefix="",
                    extra="ignore"
                )
                ollama_host: str = Field(default="localhost", description="Ollama server host")
                ollama_port: int = Field(default=11434, ge=0, le=65535)
                default_model: str = Field(default="qwen2.5", min_length=1)
                chunk_size: int = Field(default=500, gt=0, lt=10**6)
                chunk_overlap: int = Field(default=50, ge=0)
                top_k_results: int = Field(default=5, gt=0)
                similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
                max_prompt_chars: int = Field(default=4096, gt=0)
                vector_db_path: str = "./chroma_db"
                security_config: str = "./security/acl_storage.json"
            
            self._settings_obj = PydanticSettings()
            for attr in dir(self._settings_obj):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(self._settings_obj, attr))
        except Exception as e:
            logger.error(f"Pydantic settings initialization failed: {e}")
            self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load settings from environment variables with fallback defaults."""
        # Ollama server configuration
        host = _load_env_var("OLLAMA_HOST") or "localhost"
        port_str = _load_env_var("OLLAMA_PORT")
        self.ollama_port = _parse_int(port_str, 11434)
        self.ollama_host = host
        
        # Model configuration
        model = _load_env_var("DEFAULT_MODEL") or "qwen2.5"
        if not model:
            raise ValueError("DEFAULT_MODEL environment variable cannot be empty")
        self.default_model = model
        
        # Chunking parameters
        self.chunk_size = _parse_int(_load_env_var("CHUNK_SIZE"), 500)
        self.chunk_overlap = _parse_int(_load_env_var("CHUNK_OVERLAP"), 50)
        
        # Retrieval settings
        self.top_k_results = _parse_int(_load_env_var("TOP_K_RESULTS"), 5)
        threshold_str = _load_env_var("SIMILARITY_THRESHOLD")
        self.similarity_threshold = _parse_float(threshold_str, 0.7)
        
        # Prompt constraints
        max_chars_str = _load_env_var("MAX_PROMPT_CHARS")
        self.max_prompt_chars = _parse_int(max_chars_str, 4096)
        
        # Path configuration with validation
        vector_path = _load_env_var("VECTOR_DB_PATH") or "./chroma_db"
        security_path = _load_env_var("SECURITY_CONFIG") or "./security/acl_storage.json"
        
        try:
            self.vector_db_path = _validate_path(vector_path)
            logger.debug(f"Resolved vector DB path: {self.vector_db_path}")
            self.security_config = _validate_path(security_path)
            logger.debug(f"Resolved security config path: {self.security_config}")
        except ValueError as e:
            raise RuntimeError(f"Invalid configuration paths: {e}") from e
    
    @property
    def ollama_url(self) -> str:
        """Generate full Ollama server URL."""
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    @property
    def is_valid(self) -> bool:
        """Check if all configuration values are valid and complete."""
        required_attrs = [
            'ollama_host', 'ollama_port', 'default_model',
            'chunk_size', 'top_k_results', 'similarity_threshold'
        ]
        return all(hasattr(self, attr) for attr in required_attrs)


# Module-level singleton instance (created on first access or import)
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance.
    
    This ensures a single configuration instance per process.
    Callers should use this function rather than creating their own instances.
    
    Returns:
        Settings: The singleton configuration instance
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def set_settings(config: Settings) -> None:
    """Set a custom settings instance.
    
    Args:
        config: New settings configuration to use globally
    """
    global _settings_instance
    _settings_instance = config


# Initialize singleton at module load time (optional - can be delayed)
try:
    settings = get_settings()
except Exception as e:
    logger.error(f"Failed to initialize default settings: {e}")
    settings = None  # type: ignore[assignment, misc]
