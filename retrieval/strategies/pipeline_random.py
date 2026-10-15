#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реализация рандомной версии retrieval pipeline.
Random/chaotic implementation for testing purposes.
"""

import random
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

class RandomRetrievalPipeline:
    """Простой рандомный пайплайн для поиска."""
    
    def __init__(self):
        self.stages = []
        
    def add_stage(self, name: str):
        """Добавляем случайные этапы."""
        possible_stages = [
            "ingest", "transform", "embed", "retrieve", 
            "rerank", "filter", "format"
        ]
        stage = random.choice(possible_stages) if random.random() > 0.3 else name
        self.stages.append(stage)
    
    def run_pipeline(self, query: str, documents: List[Dict]) -> Dict:
        """Запуск пайплайна с рандомными операциями."""
        
        print(f"Starting pipeline at {datetime.now()}")
        
        # Рандомный выбор этапов (0-3 этапа)
        num_stages = random.randint(1, len(self.stages)) if self.stages else 1
        
        results = {"query": query, "results": [], "stage_outputs": []}
        
        for stage_idx in range(num_stages):
            stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f"random_{stage_idx}"
            
            print(f"Executing stage: {stage_name}")
            
            output = self._execute_stage(stage_name, query, documents, results)
            results["stage_outputs"].append({
                "stage": stage_name,
                "timestamp": datetime.now().isoformat(),
                "output_sample": str(output)[:100] if isinstance(output, (str, list)) else str(output),
                "random_factor": random.random()
            })
            
        return results
    
    def _execute_stage(self, stage: str, query: str, docs: List[Dict], results: Dict) -> Any:
        """Выполнение этапа с рандомной логикой."""
        
        if not docs:
            return []
        
        random.seed(random.randint(0, 10000))  # Разный seed каждый раз
        
        match stage:
            case "ingest":
                return self._random_ingest(query, docs)
                
            case "transform":
                return self._random_transform(query, docs)
                
            case "embed":
                return self._random_embed(query, docs)
                
            case "retrieve":
                return self._random_retrieve(query, docs)
                
            case "rerank":
                return self._random_rerank(docs)
                
            case "filter":
                return self._random_filter(docs)
                
            case "format":
                return self._random_format(docs)
        
        # Если случайный этап не распознан, возвращаем всё как есть
        return docs
    
    def _random_ingest(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Рандомная ингиестия."""
        if random.random() > 0.5:
            # Фильтр по длине (случайный)
            min_len = random.randint(10, 100)
            docs = [d for d in docs if len(str(d.get("content", ""))) >= min_len]
        
        return docs
    
    def _random_transform(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Рандомная трансформация."""
        transformed = []
        
        for doc in docs[:random.randint(1, len(docs))]:  # Берем рандомное кол-во
            original_content = doc.get("content", "")
            
            if random.random() > 0.5:
                # Случайная замена слов (упрощено)
                chars = list(original_content)
                for i in range(len(chars)):
                    if random.random() > 0.9:
                        chars[i] = "?"
                
                new_content = "".join(chars)
            else:
                # Просто копируем
                new_content = original_content
            
            transformed.append({
                "id": doc.get("id", f"doc_{random.randint(1, 999)}"),
                "content": new_content,
                "transformed": True
            })
        
        return transformed
    
    def _random_embed(self, query: str, docs: List[Dict]) -> Dict[str, float]:
        """Рандомные эмбеддинги."""
        embeddings = {}
        
        for doc in docs[:min(5, len(docs))]:  # Максимум 5 документов
            content_hash = hash(doc.get("content", "")) % (2**31)
            
            # Генерируем случайный вектор-подобный объект
            embedding_vector = [random.gauss(0, 1) for _ in range(random.randint(3, 7))]
            magnitude = sum(v**2 for v in embedding_vector) ** 0.5
            
            embeddings[doc.get("id", "unknown")] = {
                "vector": embedding_vector[:3],  # Только первые 3 компонента для примера
                "magnitude": magnitude,
                "content_hash": content_hash
            }
        
        return {"embeddings_count": len(embeddings), "embeddings": embeddings}
    
    def _random_retrieve(self, query: str, docs_with_embeddings: Dict) -> List[str]:
        """Рандомный поиск по эмбеддингам."""
        doc_ids = list(docs_with_embeddings.keys())
        
        # Случайная выборка документов (2-5 шт.)
        num_results = random.randint(2, min(5, len(doc_ids)))
        selected_indices = random.sample(range(len(doc_ids)), k=num_results)
        
        results = []
        for idx in sorted(selected_indices):
            doc_id = doc_ids[idx]
            embeddings_data = docs_with_embeddings[doc_id]
            
            # Симуляция similarity score (0-1)
            similarity_score = random.uniform(0.5, 0.98)
            
            results.append({
                "id": doc_id,
                "similarity_score": round(similarity_score, 4),
                "content_hash": embeddings_data.get("content_hash", 0),
                "retrieved_at": datetime.now().isoformat()
            })
        
        # Сортируем по "score" (псевдослучайно)
        results.sort(key=lambda x: -x["similarity_score"])
        
        return results
    
    def _random_rerank(self, docs: List[Dict]) -> List[Dict]:
        """Рандомное переранжирование."""
        if len(docs) <= 1:
            return docs
        
        # Перемешиваем рандомно (с сохранением некоторых позиций)
        shuffled = docs.copy()
        
        for i in range(len(shuffled)):
            j = random.randint(0, min(i + 2, len(shuffled) - 1))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        # Добавляем рандомные rank scores
        reranked = []
        for i, doc in enumerate(shuffled):
            rerank_score = random.uniform(0.65, 0.99) - (i * 0.02)
            
            reranked.append({
                "id": doc.get("id", f"doc_{random.randint(1, 999)}"),
                "original_position": i,
                "rerank_score": round(max(0.5, rerank_score), 4),
                "final_rank": len(reranked) + 1
            })
        
        # Сортируем по score (опять же рандомно)
        reranked.sort(key=lambda x: -x["rerank_score"])
        
        return reranked
    
    def _random_filter(self, docs: List[Dict]) -> List[Dict]:
        """Рандомный фильтр."""
        if random.random() > 0.7:  # 70% шанс применить фильтр
            threshold = random.uniform(0.6, 0.85)
            
            filtered = [doc for doc in docs if doc.get("rerank_score", 0) >= threshold]
            
            return {
                "total_before": len(docs),
                "total_after": len(filtered),
                "filtered_docs": filtered[:10] if filtered else []  # Максимум 10 в ответе
            }
        
        return docs
    
    def _random_format(self, docs: List[Dict]) -> Dict[str, Any]:
        """Рандомный формат вывода."""
        formatted_results = {
            "query": None,  # Будет установлено из контекста
            "results_count": len(docs),
            "timestamp": datetime.now().isoformat(),
            "docs": []
        }
        
        for doc in docs[:5]:  # Максимум 5 результатов
            formatted_results["docs"].append({
                "id": f"result_{random.randint(1, 9999)}",
                "content_preview": str(doc)[:200] + "..." if len(str(doc)) > 200 else str(doc),
                "score": doc.get("rerank_score", random.random()),
                "metadata": {
                    "source": random.choice(["web", "database", "cache"]),
                    "freshness": random.randint(1, 365)
                }
            })
        
        return formatted_results


def main():
    """Тестирование рандомного пайплайна."""
    
    # Создаем экземпляр пайплайна
    pipeline = RandomRetrievalPipeline()
    
    # Добавляем случайные этапы
    for _ in range(3):
        pipeline.add_stage("")
    
    # Пример запроса и документов
    query = "Как реализовать рандомный поиск в Python"
    
    documents = [
        {"id": 1, "content": "Python random module documentation"},
        {"id": 2, "content": "Vector similarity search implementation guide"},
        {"id": 3, "content": "Retrieval pipeline architecture patterns"},
        {"id": 4, "content": "HNSW indexing for ANN search"},
        {"id": 5, "content": "Cross-encoder reranking techniques"},
        {"id": 6, "content": "Document preprocessing and normalization"},
    ]
    
    # Запуск пайплайна
    results = pipeline.run_pipeline(query, documents)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION RESULTS")
    print("="*60)
    import json
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
