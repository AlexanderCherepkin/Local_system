"""Document processing and parsing for Enterprise RAG system."""

from settings import DEFAULT_ALLOWED_ROLES

class DocumentProcessor:
    def split_into_chunks(self, document) -> list:
        # Универсальное получение метаданных и контента (поддержка объектов и словарей)
        if isinstance(document, dict):
            metadata = document.get("metadata", {})
            content = document.get("content", "")
            file_type = metadata.get("file_type", "unknown")
        else:
            metadata = getattr(document, "metadata", {})
            content = getattr(document, "content", "")
            file_type = getattr(document, "file_type", metadata.get("file_type", "unknown"))

        # Получаем allowed_roles или применяем DEFAULT_ALLOWED_ROLES как маркер проверки
        doc_roles = metadata.get("allowed_roles")
        if not doc_roles:
            doc_roles = DEFAULT_ALLOWED_ROLES
            metadata["requires_role_confirmation"] = True

        chunks = []
        raw_chunks = self._chunk_text(content)
        
        for chunk_text in raw_chunks:
            chunk_metadata = {
                **metadata,
                "file_type": file_type,
                "allowed_roles": doc_roles
            }
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
        return chunks

    def _chunk_text(self, text: str) -> list:
        # Пример реализации разбиения текста
        chunk_size = getattr(self, "chunk_size", 500)
        overlap = getattr(self, "chunk_overlap", 50)
        return [text[i:i + chunk_size] for i in range(0, max(1, len(text)), chunk_size - overlap)]