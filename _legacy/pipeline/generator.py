"""Оллатма генерация ответов для Enterprise RAG системы."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import requests

logger = logging.getLogger(__name__)

# Корректный импорт через pydantic-settings
from config.settings import Settings
from pipeline.embedder import Embedder
from storage.vector_db import VectorDatabase

# Инициализация глобального инстанса настроек
settings = Settings.get_settings()


class OllamaGenerator:
    """Класс генерации ответов через локальный Ollama сервер с поддержкой RAG-контекста и цитирования."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k_rag: Optional[int] = None,
        top_k_sampling: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        num_ctx: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        # Получаем параметры из pydantic-settings
        self.base_url = (base_url or settings.ollama_host).rstrip("/")
        self.model_name = model_name or settings.default_model
        
        # Разделяем RAG top_k и sampling top_k модели
        self.top_k_rag = top_k_rag if top_k_rag is not None else settings.top_k_results
        self.top_k_sampling = top_k_sampling if top_k_sampling is not None else 40
        
        self.temperature = temperature if temperature is not None else settings.temperature
        self.top_p = top_p if top_p is not None else 0.9
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else 1.1
        self.num_ctx = num_ctx if num_ctx is not None else settings.context_window
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.similarity_threshold = settings.similarity_threshold

        # Проверка доступности модели при инициализации (Fail-Fast)
        self._verify_model_availability()

    def _verify_model_availability(self) -> None:
        """Проверяет через /api/tags, установлена ли модель в Ollama."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(tags_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name") for m in models]
            
            if not any(self.model_name in name for name in model_names):
                raise RuntimeError(
                    f"Модель '{self.model_name}' не найдена на сервере Ollama ({self.base_url}). "
                    f"Доступные модели: {model_names}. Убедитесь, что модель установлена (ollama pull {self.model_name})."
                )
        except requests.RequestException as e:
            logger.error("Не удалось подключиться к Ollama по адресу %s: %s", self.base_url, e)
            raise RuntimeError(f"Ошибка соединения с Ollama ({self.base_url}): {e}") from e

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Получить список доступных моделей Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    models.append({
                        "name": model["name"],
                        "size": model.get("size"),
                        "modified_at": model.get("modified_at")
                    })
                logger.info(f"Найдено доступных моделей: {[m['name'] for m in models]}")
                return models
            return []
        except Exception as e:
            logger.error(f"Не удалось получить список моделей Ollama: {e}")
            return []

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Форматирует найденные чанки в нумерованные фрагменты с doc_id."""
        if not context:
            return "Контекст не найден."

        formatted_parts = []
        for idx, chunk in enumerate(context[:self.top_k_rag], start=1):
            doc_id = chunk.get("doc_id", "unknown")
            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})
            meta_str = f"\nМетаданные: {json.dumps(metadata, ensure_ascii=False)}" if metadata else ""
            formatted_parts.append(f"[{idx}] (doc_id: {doc_id}){meta_str}\n{text}")

        return "\n\n".join(formatted_parts)

    def _build_system_prompt(self, user_role: Optional[str] = None) -> str:
        base_prompt = (
            "Ты — умная система поддержки решений на основе Enterprise RAG. Твоя задача — отвечать на вопросы пользователей, "
            "опираясь исключительно на предоставленный контекст из документов организации.\n\n"
            "Правила ответа:\n"
            "1. Используй ТОЛЬКО информацию из предоставленного контекста.\n"
            "2. В тексте ответа обязательно используй ссылки на источники в формате квадратных скобок с номерами фрагментов, например [1], [2].\n"
            "3. Если информация отсутствует в контексте, скажи «К сожалению, я не нашел эту информацию в доступных документах».\n"
            "4. Не придумывай факты и не используй внешние знания.\n"
            "5. Отвечай четко и структурировано на русском языке."
        )
        
        if user_role:
            base_prompt += f"\n\nТвой уровень доступа: {user_role}\nТы можешь использовать информацию только из документов, доступных для твоей роли."
        
        return base_prompt

    def generate_response(
        self, 
        prompt: str,
        context: List[Dict[str, Any]],
        user_role: Optional[str] = None,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Сгенерировать ответ на запрос с контекстом через POST /api/chat с учетом роли и цитирования."""
        chat_url = f"{self.base_url}/api/chat"
        system_prompt = self._build_system_prompt(user_role)
        
        context_str = self._format_context(context)
        
        user_content = (
            f"Контекст:\n{context_str}\n\n"
            f"Запрос пользователя: {prompt}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_k": self.top_k_sampling,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            }
        }

        result: Dict[str, Any] = {
            "success": False,
            "response": "",
            "metadata": {}
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(chat_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("message", {}).get("content", "")
                result["success"] = True
                result["response"] = answer
                result["usage"] = data.get("usage")
                result["metadata"] = {
                    "model_used": self.model_name,
                    "context_chunks_used": len(context[:self.top_k_rag]),
                    "user_role": user_role,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "attempt": attempt,
                }
                return result

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Нет соединения с сервером Ollama по адресу {self.base_url}: {e}")
                if attempt == self.max_retries:
                    result["metadata"] = {"error": f"No connection to Ollama: {str(e)}"}
                    result["suggestion"] = "Запустите службу Ollama и убедитесь, что сервер доступен"
                    return result
            except requests.exceptions.Timeout:
                logger.error("Тайм-аут запроса к Ollama")
                if attempt == self.max_retries:
                    result["metadata"] = {"error": "Request timed out. Server may be overloaded."}
                    return result
            except Exception as e:
                logger.warning(f"Попытка генерации {attempt}/{self.max_retries} не удалась: {e}")
                if attempt == self.max_retries:
                    logger.exception("Все попытки генерации через Ollama исчерпаны.")
                    result["metadata"] = {"error": str(e)}
                    return result
            
            time.sleep(2 ** attempt)

        return result

    def generate_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Сгенерировать ответ при отсутствии контекста."""
        fallback_message = f"""Пользователь задал вопрос: "{prompt}"

Однако в доступных документах не найдена релевантная информация.

Рекомендации для администратора системы:
1. Проверить, загружены ли необходимые документы в базу знаний.
2. Убедиться, что пользователь имеет права доступа к нужным документам.
3. Рассмотреть возможность расширения базы документов."""
        
        return {
            "success": True,
            "response": fallback_message,
            "metadata": {
                "context_used": 0,
                "fallback_reason": "no_relevant_context"
            }
        }


class RAGPipeline:
    """Полный пайплайн RAG системы с реальным векторным поиском."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.settings = settings
        
        # Инициализация ядра RAG
        try:
            self.vector_db = VectorDatabase()
            self.embedder = Embedder()
        except Exception as e:
            logger.warning(f"Не удалось инициализировать векторную базу или эмбеддер: {e}")
            self.vector_db = None
            self.embedder = None
            
        self.generator = OllamaGenerator()
    
    def process_query(
        self, 
        user_question: str,
        user_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """Обработать запрос пользователя через полный RAG пайплайн."""
        result = {
            "query": user_question,
            "role": user_role,
            "success": False,
            "steps": {},
            "final_answer": None,
            "metadata": {}
        }
        
        try:
            # Шаг 1: Поиск релевантных документов через ChromaDB
            logger.info("Шаг 1: Поиск релевантных документов через векторный индекс...")
            retrieved_chunks = self._retrieve_relevant_documents(user_question, user_role)
            
            result["steps"]["retrieval"] = {
                "success": len(retrieved_chunks) > 0,
                "chunks_count": len(retrieved_chunks),
                "chunk_ids": [c.get("chunk_id") for c in retrieved_chunks]
            }
            
            if not retrieved_chunks:
                logger.warning(f"Не найдено релевантных документов для запроса: {user_question}")
                fallback_result = self.generator.generate_fallback_response(user_question)
                
                result["success"] = True
                result["steps"]["fallback"] = {"used": True}
                result["final_answer"] = fallback_result.get("response")
                result["metadata"] = fallback_result.get("metadata", {})
                return result
            
            # Шаг 2: Генерация ответа с контекстом
            logger.info(f"Шаг 2: Генерация ответа ({len(retrieved_chunks)} чанков)...")
            generation_result = self.generator.generate_response(
                prompt=user_question,
                context=retrieved_chunks,
                user_role=user_role
            )
            
            result["steps"]["generation"] = {
                "success": generation_result.get("success", False),
                "metadata": generation_result.get("metadata", {})
            }
            
            result["metadata"] = generation_result.get("metadata", {})
            
            if generation_result.get("success"):
                result["success"] = True
                result["final_answer"] = generation_result.get("response")
                result["retrieved_chunks_count"] = len(retrieved_chunks)
                logger.info("Ответ сгенерирован успешно.")
            else:
                result["steps"]["error"] = generation_result.get("metadata", {}).get("error", "Unknown error")
                
        except Exception as e:
            logger.exception(f"Ошибка в RAG пайплайне: {e}")
            result["success"] = False
            result["metadata"] = {"error": str(e)}
        
        return result
    
    def _retrieve_relevant_documents(
        self, 
        query: str,
        user_role: Optional[str] = None,
        oversample_factor: int = 3
    ) -> List[Dict[str, Any]]:
        """Полноценный векторный поиск с ACL-фильтрацией."""
        if not query or not query.strip():
            return []
            
        if not self.vector_db or not self.embedder:
            logger.error("Векторная база данных или эмбеддер не инициализированы.")
            return []

        try:
            top_k = self.settings.top_k_results
            threshold = self.settings.similarity_threshold
            n_results = top_k * oversample_factor
            
            # 1. Генерация эмбеддинга запроса
            query_embedding = self.embedder.embed_text(query)
            
            # 2. Запрос в коллекцию ChromaDB
            query_results = self.vector_db.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_chunks = []
            docs = query_results.get("documents", [[]])[0]
            metadatas = query_results.get("metadatas", [[]])[0]
            distances = query_results.get("distances", [[]])[0]
            ids = query_results.get("ids", [[]])[0]

            for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
                metadata = metadata or {}
                
                # --- 3. ACL-фильтрация в Python ---
                doc_roles = metadata.get("allowed_roles")
                if doc_roles and user_role:
                    if isinstance(doc_roles, list) and user_role not in doc_roles:
                        continue
                    elif isinstance(doc_roles, str) and doc_roles != user_role:
                        continue
                
                # --- 4. Пересчет расстояния в косинусное сходство ---
                similarity = 1.0 - float(distance)
                
                # --- 5. Фильтрация по порогу схожести ---
                if similarity < threshold:
                    continue
                
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", "unknown"),
                    "source_file": metadata.get("source_file", metadata.get("filename", "unknown")),
                    "text": text,
                    "score": similarity,
                    "metadata": metadata
                })

            # Сортировка по убыванию score и обрезка до top_k
            retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
            return retrieved_chunks[:top_k]

        except Exception as e:
            logger.error(f"Ошибка при выполнении векторного поиска в ChromaDB: {e}")
            raise RuntimeError(f"Векторный поиск недоступен: {e}") from e


def main():
    """Тестирование RAG пайплайна."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Инициализация RAG системы...")
    pipeline = RAGPipeline()
    
    models = pipeline.generator.get_available_models()
    if not models:
        print("\n⚠️ Сервер Ollama недоступен. Запустите:")
        print("  1. ollama pull qwen2.5")
        print("  2. ollama serve")
        return
    
    print(f"\nДоступные модели Ollama: {[m['name'] for m in models]}")
    
    test_query = "Какие документы доступны в системе?"
    print(f"\nОбработка запроса: {test_query}")
    result = pipeline.process_query(test_query)
    
    if result.get("success"):
        print("\n✅ Ответ:")
        print(result["final_answer"])
    else:
        print("\n❌ Ошибка:")
        print(result.get("metadata", {}).get("error"))


if __name__ == "__main__":
    main()