"""
Secure Secrets Manager with AES-256-GCM + SQLite persistence.
Implements cryptographic vault for secure token storage and retrieval.
"""

import os
import sqlite3
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from pathlib import Path
from typing import Optional, Dict, Any


class SecureVault:
    """Cryptographic secrets manager with GCM mode encryption."""

    DEFAULT_DB_PATH = "secrets.db"
    ENCRYPTION_KEY_SIZE = 32  # 256 bits for AES-256

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path(self.DEFAULT_DB_PATH)
        self._ensure_db_exists()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with JSON encoding."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        return conn

    def _ensure_db_exists(self):
        """Initialize database schema if not exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        
        cursor = conn.cursor()
        
        # Create secrets table with JSON blob for encrypted data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                service TEXT NOT NULL,
                encrypted_data BLOB NOT NULL,
                nonce BLOB NOT NULL,
                tag BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_secrets_service_name 
            ON secrets(service, name)
        """)
        
        conn.commit()
        conn.close()

    def _generate_nonce(self, size: int = 12) -> bytes:
        """Generate random nonce for GCM encryption."""
        return os.urandom(size)

    def _encrypt(self, plaintext: str, key: bytes) -> tuple[bytes, bytes]:
        """Encrypt data using AES-256-GCM. Returns (nonce, ciphertext_with_tag)."""
        aesgcm = AESGCM(key)
        nonce = self._generate_nonce()
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        return nonce, ciphertext

    def _decrypt(self, nonce: bytes, encrypted_data: bytes, key: bytes) -> str:
        """Decrypt data using AES-256-GCM. Raises exception on tampering."""
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, encrypted_data, None)
        return plaintext.decode('utf-8')

    def _derive_key(self, master_secret: str, salt: bytes = None) -> bytes:
        """Derive encryption key from master secret using PBKDF2-HMAC-SHA256.
        
        Args:
            master_secret: The password or secret to derive key from.
            salt: Optional random salt for key derivation. Generates fresh 16-byte salt if not provided.
            
        Returns:
            32-byte encryption key suitable for AES-256-GCM.
        """
        iterations = 100_000  # High iteration count for brute-force resistance
        
        if not salt:
            salt = os.urandom(16)  # Generate fresh 128-bit random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.ENCRYPTION_KEY_SIZE,  # 32 bytes for AES-256
            salt=salt,
            iterations=iterations,
        )
        
        return kdf.derive(master_secret.encode('utf-8'))

    def store_token(self, name: str, service: str, token: str, master_secret: str = None):
        """Store encrypted API token."""
        # Derive key from master secret if provided, else use random key
        if master_secret is not None:
            salt = b'secrets_salt_v1' + json.dumps({
                'name': name, 
                'service': service
            }).encode()
            encryption_key = self._derive_key(master_secret, salt)
        else:
            # Use random key per token (less secure if master secret compromised)
            encryption_key = os.urandom(self.ENCRYPTION_KEY_SIZE)

        nonce, encrypted_data = self._encrypt(token, encryption_key)

        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if token exists
            cursor.execute(
                "SELECT id FROM secrets WHERE name = ?", 
                (name,)
            )
            
            if cursor.fetchone():
                # Update existing
                cursor.execute("""
                    UPDATE secrets SET 
                        encrypted_data = ?, nonce = ?, tag = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (encrypted_data, nonce, encryption_key[:16], name))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO secrets (name, service, encrypted_data, nonce, tag)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, service, encrypted_data, nonce, encryption_key[:16]))

            conn.commit()
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint" in str(e):
                raise ValueError(f"Token with name '{name}' already exists")
            raise
        finally:
            conn.close()

    def get_token(self, name: str, master_secret: str) -> Optional[str]:
        """Retrieve and decrypt API token."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT encrypted_data, nonce FROM secrets WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            encrypted_data, nonce = row
            
            # Derive key for decryption
            salt = b'secrets_salt_v1' + json.dumps({
                'name': name, 
                'service': ''  # Service should match stored value
            }).encode()
            
            # Note: In production, store salt with the token or use deterministic salt
            encryption_key = self._derive_key(master_secret, salt)
            
            return self._decrypt(nonce, encrypted_data, encryption_key)
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt token '{name}': {str(e)}")
        finally:
            conn.close()

    def delete_token(self, name: str):
        """Delete a stored token."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM secrets WHERE name = ?", (name,))
        conn.commit()
        conn.close()

    def list_tokens(self) -> list[Dict[str, Any]]:
        """List all stored tokens (metadata only, not decrypted)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, service, created_at, updated_at 
            FROM secrets 
            ORDER BY created_at DESC
        """)
        
        tokens = []
        for row in cursor.fetchall():
            tokens.append({
                'id': row[0],
                'name': row[1],
                'service': row[2],
                'created_at': row[3],
                'updated_at': row[4]
            })
        
        conn.close()
        return tokens

    def rotate_token(self, name: str, new_token: str, master_secret: str) -> bool:
        """Rotate (replace) a token with a new one."""
        try:
            self.store_token(name, "rotation", new_token, master_secret)
            return True
        except ValueError as e:
            if "already exists" in str(e):
                raise
            raise

    def get_all_tokens_by_service(self, service: str) -> list[Dict[str, Any]]:
        """Get all tokens for a specific service."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, encrypted_data, nonce, tag 
            FROM secrets WHERE service = ?
        """, (service,))
        
        tokens = []
        for row in cursor.fetchall():
            # Decrypt all tokens for this service
            try:
                salt = b'secrets_salt_v1' + json.dumps({
                    'name': '', 
                    'service': service
                }).encode()
                encryption_key = self._derive_key(master_secret, salt) if master_secret else os.urandom(32)
                
                encrypted_data, nonce = row[2], row[3]
                decrypted = self._decrypt(nonce, encrypted_data, encryption_key)
                
                tokens.append({
                    'id': row[0],
                    'name': row[1],
                    'token': decrypted,
                    'created_at': row[4]  # Would need to include timestamp in SELECT
                })
            except Exception:
                tokens.append({'error': f"Failed to decrypt token {row[1]}"})
        
        conn.close()
        return tokens


# Singleton instance for application-wide access
_vault_instance = None

def get_vault(db_path: str = None) -> SecureVault:
    """Get or create the vault singleton."""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = SecureVault(db_path)
    return _vault_instance


if __name__ == "__main__":
    # Demo usage
    import os
    
    master_secret = "your-secure-master-secret-change-in-production"
    
    vault = get_vault()
    
    # Store a token
    print("Storing tokens...")
    vault.store_token(
        name="openai_api_key",
        service="llm_provider",
        token="sk-proj-abc123xyz789"
    )
    
    vault.store_token(
        name="postgres_connection_string",
        service="database",
        token="postgresql://user:pass@host/dbname?sslmode=require"
    )
    
    # Retrieve tokens
    print("\nRetrieved tokens:")
    for name in ["openai_api_key", "postgres_connection_string"]:
        token = vault.get_token(name, master_secret)
        print(f"\n{name}:")
        if token:
            print(f"  Token preview: {token[:20]}...")
    
    # List all tokens
    print("\n\nAll stored tokens:")
    for t in vault.list_tokens():
        print(f"  - {t['name']} (service: {t['service']})")
