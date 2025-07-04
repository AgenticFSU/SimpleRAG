"""
Text chunking strategies for the RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
import logging

from .config import ChunkingStrategy, RAGConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Chunk the input text into smaller pieces."""
        pass


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter implementation."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        super().__init__(config)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.config.RECURSIVE_SEPARATORS,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using recursive character splitting."""
        if not text.strip():
            logger.warning("Input text is empty or whitespace. Returning no chunks.")
            return []
        try:
            chunks = self.splitter.split_text(text)
            # Filter out empty or whitespace-only chunks
            result = [chunk.strip() for chunk in chunks if chunk.strip()]
            logger.info(f"Chunked text into {len(result)} chunks using RecursiveChunker.")
            return result
        except Exception as e:
            logger.error(f"Failed to chunk text in RecursiveChunker: {e}")
            return []


class CharacterChunker(BaseChunker):
    """Simple character-based text splitter implementation."""
    
    def __init__(self, config: Optional[RAGConfig] = None, separator: str = "\n\n"):
        super().__init__(config)
        self.separator = separator
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using simple character splitting."""
        if not text.strip():
            logger.warning("Input text is empty or whitespace. Returning no chunks.")
            return []
        try:
            chunks = self.splitter.split_text(text)
            result = [chunk.strip() for chunk in chunks if chunk.strip()]
            logger.info(f"Chunked text into {len(result)} chunks using CharacterChunker.")
            return result
        except Exception as e:
            logger.error(f"Failed to chunk text in CharacterChunker: {e}")
            return []

class TextChunker:
    """Main text chunker class that supports multiple strategies."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._chunkers = {}
    
    def _get_chunker(self, strategy: ChunkingStrategy) -> BaseChunker:
        """Get or create a chunker for the specified strategy."""
        if strategy not in self._chunkers:
            if strategy == ChunkingStrategy.RECURSIVE:
                self._chunkers[strategy] = RecursiveChunker(self.config)
            elif strategy == ChunkingStrategy.CHARACTER:
                self._chunkers[strategy] = CharacterChunker(self.config)
            else:
                raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
        return self._chunkers[strategy]
    
    def chunk_text(
        self, 
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> List[str]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy to use
            
        Returns:
            List of text chunks
        """
        if not isinstance(strategy, ChunkingStrategy):
            # Try to convert string to enum
            try:
                strategy = ChunkingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Invalid chunking strategy: {strategy}")
        
        chunker = self._get_chunker(strategy)
        try:
            result = chunker.chunk_text(text)
            logger.info(f"Chunked text using strategy '{strategy.name}': {len(result)} chunks.")
            return result
        except Exception as e:
            logger.error(f"Failed to chunk text using strategy '{strategy}': {e}")
            return []
    
    def chunk_documents(
        self, 
        documents: List[str], 
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> List[str]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy to use
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.chunk_text(doc, strategy)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk a document: {e}")
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "total_characters": 0
            }
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        } 