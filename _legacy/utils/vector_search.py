"""Векторный поиск для Enterprise RAG системы."""

import numpy as np
from typing import List, Dict, Any, Tuple

class VectorSearch:
    def __init__(self):
        self.vector_index: List[np.ndarray] = []
        self.metadata_list: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Any], embeddings: List[np.ndarray]) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self.vector_index.append(emb)
            meta = chunk.metadata if hasattr(chunk, 'metadata') else {"text": getattr(chunk, 'text', str(chunk))}
            self.metadata_list.append(meta)

    def filter_index(self, filter_conditions: Dict[str, Any]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Безопасная фильтрация без изменения индекса на лету во время итерации."""
        filtered_vectors = []
        filtered_metadata = []
        
        for vec, meta in zip(self.vector_index, self.metadata_list):
            match = True
            for key, val in filter_conditions.items():
                if meta.get(key) != val:
                    match = False
                    break
            if match:
                filtered_vectors.append(vec)
                filtered_metadata.append(meta)
                
        return filtered_vectors, filtered_metadata

    def delete_document(self, document_id: str) -> None:
        """Точечное удаление фрагментов конкретного документа."""
        new_vectors = []
        new_metadata = []
        
        for vec, meta in zip(self.vector_index, self.metadata_list):
            if meta.get('document_id') != document_id:
                new_vectors.append(vec)
                new_metadata.append(meta)
                
        self.vector_index = new_vectors
        self.metadata_list = new_metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск по косинусному сходству с нормировкой векторов."""
        if not self.vector_index:
            return []
            
        scores = []
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        for i, vec in enumerate(self.vector_index):
            v_norm = vec / (np.linalg.norm(vec) + 1e-8)
            score = float(np.dot(q_norm, v_norm))
            scores.append((score, i))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "score": score,
                "metadata": self.metadata_list[idx]
            })
            
        return results