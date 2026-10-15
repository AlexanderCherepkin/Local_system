"""
Secrets Manager Package
Provides secure token storage with AES-256-GCM encryption.
"""

from .secrets import SecureVault, get_vault
from .cli import secret_cli

__version__ = "1.0.0"
__all__ = ["SecureVault", "get_vault", "secret_cli"]
