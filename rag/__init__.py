from .core import (
    RAGRetriever,
    create_rag_retriever,
    ChunkingStrategy,
    TextChunker,
    TextEmbedder,
    ChromaVectorStore,
    RAGConfig,
    __version__
)

# Expose version at the top level
__all__ = [
    "RAGRetriever",
    "create_rag_retriever",
    "ChunkingStrategy", 
    "TextChunker",
    "TextEmbedder",
    "ChromaVectorStore",
    "RAGConfig"
] 