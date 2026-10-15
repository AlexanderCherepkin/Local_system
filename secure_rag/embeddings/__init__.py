"""Secure RAG Embedding Module."""
from .embedder import BaseEmbedder, HuggingFaceEmbedder

__all__ = ['BaseEmbedder', 'HuggingFaceEmbedder']
