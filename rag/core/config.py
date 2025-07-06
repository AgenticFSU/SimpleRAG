"""
Configuration settings for the RAG system.
"""

from enum import Enum
from dataclasses import dataclass


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"
    CHARACTER = "character"


@dataclass
class RAGConfig:
    """Configuration class for RAG system."""
    
    # Chunking configurations
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Recursive chunking separators
    RECURSIVE_SEPARATORS: list = None
    
    # Embedding model configuration
    # Original models: Sentence Transformers Hugging Face organization https://huggingface.co/models?library=sentence-transformers&author=sentence-transformers
    # Community models: All Sentence Transformer models on Hugging Face https://huggingface.co/models?library=sentence-transformers
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # ChromaDB configuration
    CHROMA_PERSIST_DIRECTORY: str = "../data/chroma_db"
    COLLECTION_NAME: str = "rag_documents"
    
    # Retrieval configuration
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Performance settings
    BATCH_SIZE: int = 100
    MAX_WORKERS: int = 4
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.RECURSIVE_SEPARATORS is None:
            self.RECURSIVE_SEPARATORS = [
                "\n\n",  # Double newlines (paragraphs)
                "\n",    # Single newlines
                " ",     # Spaces
                ".",     # Periods
                ",",     # Commas
                "\u200B", # Zero-width space
                "\uff0c", # Fullwidth comma
                "\u3001", # Ideographic comma
                "\uff0e", # Fullwidth full stop
                "\u3002", # Ideographic full stop
                "",      # Empty string as fallback
            ]


# Default configuration instance
DEFAULT_CONFIG = RAGConfig()