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
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and reliable sentence-transformers model
    EMBEDDING_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # ChromaDB configuration
    CHROMA_PERSIST_DIRECTORY: str = "../data/chroma_db"
    COLLECTION_NAME: str = "rag_documents"
    
    # Retrieval configuration
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
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


# Model configurations for different embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient model for general purpose"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2", 
        "dimension": 768,
        "description": "Higher quality embeddings, slower"
    },
    "distilbert-base-nli-mean-tokens": {
        "name": "sentence-transformers/distilbert-base-nli-mean-tokens",
        "dimension": 768,
        "description": "Good balance of speed and quality"
    }
} 