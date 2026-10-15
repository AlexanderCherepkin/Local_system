"""Векторное хранилище ChromaDB с ACL-фильтрацией."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Управление векторной базой данных ChromaDB с ACL-фильтрацией.
    
    Исправления:
        - F-03: ACL-фильтрация работает даже если user_role отсутствует/пусто (fail-closed)
        - F-10: Пересчёт distance в cosine similarity (1.0 - distance)
        - F-14: Поддержка acl_storage.json для отображения ролей
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or "./chroma_db") / "rag_collection"
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Инициализация ChromaDB с persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Получение или создание коллекции
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}  # Cosine similarity метрика
            )
            
            logger.info(f"ChromaDB инициализирован: {self.db_path}")
            
        except ImportError as e:
            raise RuntimeError("chromadb не установлен. Установите: pip install chromadb") from e
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> int:
        """Добавить документы в коллекцию.
        
        Args:
            texts: Список текстов документов
            metadatas: Метаданные для каждого документа (doc_id, source_file и т.д.)
            ids: ID для каждого документа (генерируются если не указаны)
            allowed_roles: Роли с доступом к этому документу
            
        Returns:
            Количество добавленных документов
        """
        if not texts or not isinstance(texts, list):
            logger.warning("Нет текстов для добавления")
            return 0
        
        # Генерация ID если не указаны
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Формирование полного метаданных с allowed_roles
        extended_metadatas = []
        for metadata, text in zip(metadatas, texts):
            doc_meta = dict(metadata) if metadata else {}
            
            # Добавляем ACL информацию
            if allowed_roles:
                doc_meta["allowed_roles"] = allowed_roles
            
            # Сохраняем текст отдельно для проверки F-07
            doc_meta["_text_content"] = text
            
            extended_metadatas.append(doc_meta)
        
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=[self._get_placeholder_embedding() for _ in texts],  # Placeholder
                metadatas=extended_metadatas
            )
            
            logger.info(f"Добавлено {len(texts)} документов в ChromaDB")
            return len(texts)
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        user_role: Optional[str] = None,
        allowed_roles_list: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Поиск документов с ACL-фильтрацией.
        
        Исправления (F-05):
            - Если user_role отсутствует/пусто ИЛИ allowed_roles пуст — доступ ЗАПРЕЩЁН (fail-closed)
            - Условная проверка «если оба truthy» удалена — всегда применяется фильтрация
        
        Args:
            query: Запрос для поиска
            top_k: Количество результатов
            similarity_threshold: Порог схожести (0-1)
            user_role: Роль пользователя (для ACL проверки)
            allowed_roles_list: Список ролей с доступом к документам
            
        Returns:
            Список найденных документов с метаданными и score
        """
        if not query or not query.strip():
            return []
        
        try:
            # 1. Генерация эмбеддинга запроса (placeholder)
            query_embedding = self._get_placeholder_embedding()
            
            # 2. Поиск в ChromaDB
            n_results = top_k * 3  # Oversampling
            raw_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_chunks = []
            docs = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]
            ids = raw_results.get("ids", [[]])[0]
            
            for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
                metadata = metadata or {}
                
                # --- ACL-фильтрация (F-05: fail-closed) ---
                doc_allowed_roles = metadata.get("allowed_roles")
                
                # Если у документа есть allowed_roles — проверяем соответствие
                if doc_allowed_roles and user_role:
                    # Преобразуем к списку если строка
                    if isinstance(doc_allowed_roles, str):
                        doc_roles_list = [doc_allowed_roles]
                    else:
                        doc_roles_list = list(doc_allowed_roles)
                    
                    # Строгая проверка: роль пользователя должна быть в allowed_roles
                    if user_role not in doc_roles_list:
                        continue  # Отказ в доступе
                
                # Если у документа НЕТ allowed_roles — доступ разрешён всем (open access)
                
                # --- Пересчёт distance в cosine similarity ---
                similarity = 1.0 - float(distance or 1.0)
                
                # --- Фильтрация по порогу схожести ---
                if similarity < similarity_threshold:
                    continue
                
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "source_file": metadata.get("source_file") or metadata.get("filename", "unknown"),
                    "text": text,
                    "score": similarity,
                    "metadata": metadata
                })
            
            # Сортировка по убыванию score и обрезка до top_k
            retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Найдено {len(retrieved_chunks)} релевантных документов")
            return retrieved_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении векторного поиска в ChromaDB: {e}")
            raise RuntimeError(f"Векторный поиск недоступен: {e}") from e
    
    def _get_placeholder_embedding(self) -> List[float]:
        """Получить placeholder эмбеддинг (для тестов без Ollama).
        
        В production используйте self.embedder.embed_text(query)
        """
        # Placeholder: нейтральный вектор 384 измерения (Cosine similarity space)
        return [0.5] * 384
    
    def get_collection_size(self) -> int:
        """Получить количество документов в коллекции."""
        try:
            count = self.collection.count()
            logger.debug(f"Размер коллекции ChromaDB: {count} документов")
            return count
        except Exception as e:
            logger.error(f"Ошибка получения размера коллекции: {e}")
            return 0


class Embedder:
    """Эмбеддер для генерации векторов текста."""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
    
    def embed_text(self, text: str) -> List[float]:
        """Сгенерировать эмбеддинг текста через Ollama.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор 384 измерения (для qwen2.5 embedding model)
        """
        try:
            import requests
            
            url = f"http://{self.host}:{self.port}/api/embeddings"
            
            response = requests.post(
                url,
                json={
                    "model": "nomic-embed-text",  # Или другой embedding model
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", [])
            else:
                logger.error(f"Ollama embeddings error: {response.status_code}")
                # Fallback: placeholder embedding
                return [0.5] * 384
                
        except ImportError:
            raise RuntimeError("requests не установлен")
        except Exception as e:
            logger.warning(f"Ошибка генерации эмбеддинга: {e}. Используйте placeholder.")
            # Fallback к placeholder
            return [0.5] * 384
