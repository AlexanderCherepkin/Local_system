"""Application settings configuration module."""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class BaseSettings:
    """Base class for all settings with Pydantic-like validation support."""

    def __init__(self):
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self):
        """Initialize the settings instance."""
        if not self._initialized:
            self._initialized = True


class Settings(BaseSettings):
    """Main application settings configuration."""

    # Application
    APP_NAME: str = "RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "models")))
    
    @property
    def logs_dir(self) -> Path:
        return self.BASE_DIR / ".logs"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def log_handlers(self) -> List[str]:
        return ["file", "console"] if self.DEBUG else ["file"]

    # Database/Storage (placeholder for future expansion)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///app.db")
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")

    # Ollama/RAG Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    @property
    def is_ollama_available(self) -> bool:
        return self.OLLAMA_HOST.startswith("http://") or self.OLLAMA_HOST.startswith("https://")

    # Security
    AUDIT_SECRET: str = os.getenv("AUDIT_SECRET", "change_me_to_a_secure_random_secret_string")
    
    @property
    def is_audit_configured(self) -> bool:
        return len(self.AUDIT_SECRET) > 20 and self.AUDIT_SECRET != "change_me_to_a_secure_random_secret_string"

    # Hugging Face
    HF_HUB_OFFLINE: str = os.getenv("HF_HUB_OFFLINE", "0")
    
    @property
    def is_offline_mode(self) -> bool:
        return self.HF_HUB_OFFLINE.lower() in ("1", "true", "yes")

    # CORS (placeholder for future API development)
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    CORS_ENABLE_CREDENTIALS: bool = True


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.initialize()
    return _settings_instance
