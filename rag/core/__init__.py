"""
RAG (Retrieval-Augmented Generation) module for efficient text chunking, 
embedding, and retrieval using ChromaDB and LangChain.
"""

from .retriever import RAGRetriever, create_rag_retriever
from .chunker import ChunkingStrategy, TextChunker
from .embedder import TextEmbedder
from .vector_store import ChromaVectorStore
from .config import RAGConfig 

__version__ = "0.1.0" 