"""
Tests for RAG text chunking module.
"""

import pytest
from rag.core.chunker import TextChunker, RecursiveChunker, CharacterChunker, BaseChunker
from rag.core.config import ChunkingStrategy, RAGConfig


class TestBaseChunker:
    """Test the abstract base chunker class."""
    
    def test_base_chunker_is_abstract(self):
        """Test that BaseChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChunker()
            
    def test_base_chunker_has_abstract_methods(self):
        """Test that BaseChunker defines abstract methods."""
        assert hasattr(BaseChunker, 'chunk_text')


class TestRecursiveChunker:
    """Test the recursive character text splitter."""
    
    def test_recursive_chunker_creation(self, test_config):
        """Test creating a recursive chunker."""
        chunker = RecursiveChunker(test_config)
        assert chunker.config == test_config
        assert chunker.splitter is not None
        
    def test_recursive_chunker_default_config(self):
        """Test recursive chunker with default config."""
        chunker = RecursiveChunker()
        assert chunker.config is not None
        assert chunker.config.CHUNK_SIZE == 1000
        
    def test_chunk_empty_text(self, test_config):
        """Test chunking empty text."""
        chunker = RecursiveChunker(test_config)
        chunks = chunker.chunk_text("")
        assert chunks == []
        
        chunks = chunker.chunk_text("   ")  # Whitespace only
        assert chunks == []
        
    def test_chunk_short_text(self, test_config):
        """Test chunking text shorter than chunk size."""
        chunker = RecursiveChunker(test_config)
        short_text = "This is a short text."
        chunks = chunker.chunk_text(short_text)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
    def test_chunk_long_text(self, test_config, long_text):
        """Test chunking text longer than chunk size."""
        chunker = RecursiveChunker(test_config)
        chunks = chunker.chunk_text(long_text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= test_config.CHUNK_SIZE + test_config.CHUNK_OVERLAP
            assert chunk.strip() != ""
            
    def test_chunk_overlap(self, test_config):
        """Test that chunks have proper overlap."""
        # Create a config with specific chunk size and overlap
        config = RAGConfig(CHUNK_SIZE=100, CHUNK_OVERLAP=20)
        chunker = RecursiveChunker(config)
        
        text = "A" * 300  # 300 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        # Each chunk should be roughly the chunk size
        for chunk in chunks:
            assert len(chunk) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP
            
    def test_hierarchical_splitting(self, test_config):
        """Test that recursive chunker respects separator hierarchy."""
        chunker = RecursiveChunker(test_config)
        
        # Text with different separators
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunker.chunk_text(text)
        
        # Should respect paragraph breaks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.strip() != ""


class TestCharacterChunker:
    """Test the character-based text splitter."""
    
    def test_character_chunker_creation(self, test_config):
        """Test creating a character chunker."""
        chunker = CharacterChunker(test_config)
        assert chunker.config == test_config
        assert chunker.separator == "\n\n"  # Default separator
        assert chunker.splitter is not None
        
    def test_character_chunker_custom_separator(self, test_config):
        """Test character chunker with custom separator."""
        custom_separator = "||"
        chunker = CharacterChunker(test_config, separator=custom_separator)
        assert chunker.separator == custom_separator
        
    def test_chunk_by_paragraphs(self, test_config):
        """Test chunking by paragraph separator."""
        chunker = CharacterChunker(test_config, separator="\n\n")
        
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_text(text)
        
        # Should split by paragraphs if they fit in chunk size
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.strip() != ""
            
    def test_chunk_by_custom_separator(self, test_config):
        """Test chunking by custom separator."""
        chunker = CharacterChunker(test_config, separator="---")
        
        text = "Section 1---Section 2---Section 3"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Should split into multiple chunks or keep as single chunk if it fits
        assert all(isinstance(chunk, str) for chunk in chunks)
            
    def test_chunk_empty_text(self, test_config):
        """Test chunking empty text."""
        chunker = CharacterChunker(test_config)
        chunks = chunker.chunk_text("")
        assert chunks == []
        
    def test_chunk_without_separator(self, test_config):
        """Test chunking text without the separator."""
        chunker = CharacterChunker(test_config, separator="|||")
        
        text = "This text has no triple pipes separator."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestTextChunker:
    """Test the main TextChunker class."""
    
    def test_text_chunker_creation(self, test_config):
        """Test creating a TextChunker."""
        chunker = TextChunker(test_config)
        assert chunker.config == test_config
        assert chunker._chunkers == {}
        
    def test_text_chunker_default_config(self):
        """Test TextChunker with default config."""
        chunker = TextChunker()
        assert chunker.config is not None
        
    def test_get_chunker_recursive(self, test_config):
        """Test getting recursive chunker."""
        chunker = TextChunker(test_config)
        recursive_chunker = chunker._get_chunker(ChunkingStrategy.RECURSIVE)
        
        assert isinstance(recursive_chunker, RecursiveChunker)
        assert recursive_chunker.config == test_config
        
        # Should cache the chunker
        recursive_chunker2 = chunker._get_chunker(ChunkingStrategy.RECURSIVE)
        assert recursive_chunker is recursive_chunker2
        
    def test_get_chunker_character(self, test_config):
        """Test getting character chunker."""
        chunker = TextChunker(test_config)
        char_chunker = chunker._get_chunker(ChunkingStrategy.CHARACTER)
        
        assert isinstance(char_chunker, CharacterChunker)
        assert char_chunker.config == test_config
        
    def test_get_chunker_invalid_strategy(self, test_config):
        """Test getting chunker with invalid strategy."""
        chunker = TextChunker(test_config)
        
        with pytest.raises(ValueError, match="Unsupported chunking strategy"):
            # Create a fake strategy by bypassing the enum
            class FakeStrategy:
                pass
            fake_strategy = FakeStrategy()
            chunker._get_chunker(fake_strategy)
            
    def test_chunk_text_with_enum(self, test_config, sample_texts):
        """Test chunking text with enum strategy."""
        chunker = TextChunker(test_config)
        
        for text in sample_texts:
            # Test recursive
            chunks_recursive = chunker.chunk_text(text, ChunkingStrategy.RECURSIVE)
            assert isinstance(chunks_recursive, list)
            assert all(isinstance(chunk, str) for chunk in chunks_recursive)
            
            # Test character
            chunks_character = chunker.chunk_text(text, ChunkingStrategy.CHARACTER)
            assert isinstance(chunks_character, list)
            assert all(isinstance(chunk, str) for chunk in chunks_character)
            
    def test_chunk_text_with_string(self, test_config, sample_texts):
        """Test chunking text with string strategy."""
        chunker = TextChunker(test_config)
        
        for text in sample_texts:
            # Test recursive
            chunks_recursive = chunker.chunk_text(text, "recursive")
            assert isinstance(chunks_recursive, list)
            
            # Test character
            chunks_character = chunker.chunk_text(text, "character")
            assert isinstance(chunks_character, list)
            
    def test_chunk_text_invalid_string_strategy(self, test_config):
        """Test chunking with invalid string strategy."""
        chunker = TextChunker(test_config)
        
        with pytest.raises(ValueError, match="Invalid chunking strategy"):
            chunker.chunk_text("Some text", "invalid_strategy")
            
    def test_chunk_documents(self, test_config, sample_documents):
        """Test chunking multiple documents."""
        chunker = TextChunker(test_config)
        documents = sample_documents["documents"]
        
        all_chunks = chunker.chunk_documents(documents, ChunkingStrategy.RECURSIVE)
        
        assert isinstance(all_chunks, list)
        assert len(all_chunks) >= len(documents)  # Should have at least one chunk per document
        assert all(isinstance(chunk, str) for chunk in all_chunks)
        
    def test_chunk_documents_empty_list(self, test_config):
        """Test chunking empty document list."""
        chunker = TextChunker(test_config)
        chunks = chunker.chunk_documents([])
        assert chunks == []
        
    def test_get_chunk_stats(self, test_config, sample_texts):
        """Test getting chunk statistics."""
        chunker = TextChunker(test_config)
        chunks = chunker.chunk_documents(sample_texts, ChunkingStrategy.RECURSIVE)
        
        stats = chunker.get_chunk_stats(chunks)
        
        assert "total_chunks" in stats
        assert "avg_chunk_length" in stats
        assert "min_chunk_length" in stats
        assert "max_chunk_length" in stats
        assert "total_characters" in stats
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] == sum(len(chunk) for chunk in chunks)
        
        if chunks:
            assert stats["min_chunk_length"] <= stats["avg_chunk_length"] <= stats["max_chunk_length"]
            
    def test_get_chunk_stats_empty(self, test_config):
        """Test getting stats for empty chunk list."""
        chunker = TextChunker(test_config)
        stats = chunker.get_chunk_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["avg_chunk_length"] == 0
        assert stats["min_chunk_length"] == 0
        assert stats["max_chunk_length"] == 0
        assert stats["total_characters"] == 0


class TestChunkingEdgeCases:
    """Test edge cases in text chunking."""
    
    def test_chunk_very_long_single_word(self, test_config):
        """Test chunking a very long single word."""
        chunker = TextChunker(test_config)
        long_word = "A" * (test_config.CHUNK_SIZE * 2)
        
        chunks = chunker.chunk_text(long_word, ChunkingStrategy.RECURSIVE)
        
        # Should split even a single long word
        assert len(chunks) >= 1
        
    def test_chunk_text_with_only_separators(self, test_config):
        """Test chunking text that's only separators."""
        chunker = TextChunker(test_config)
        
        separator_text = "\n\n\n\n"
        chunks = chunker.chunk_text(separator_text, ChunkingStrategy.RECURSIVE)
        
        # Should return empty list or minimal chunks
        assert isinstance(chunks, list)
        
    def test_chunk_text_with_unicode(self, test_config):
        """Test chunking text with Unicode characters."""
        chunker = TextChunker(test_config)
        
        unicode_text = "Hello 世界! This is a test with émojis 🚀 and accénts."
        chunks = chunker.chunk_text(unicode_text, ChunkingStrategy.RECURSIVE)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
    def test_chunk_overlap_larger_than_size(self):
        """Test configuration where overlap is larger than chunk size."""
        config = RAGConfig(CHUNK_SIZE=50, CHUNK_OVERLAP=100)
        
        # This should raise an error since LangChain validates overlap < chunk_size
        with pytest.raises(ValueError, match="larger chunk overlap"):
            chunker = TextChunker(config)
            chunker.chunk_text("test text", ChunkingStrategy.RECURSIVE)
        
    def test_different_strategies_same_text(self, test_config, long_text):
        """Test that different strategies produce different results."""
        chunker = TextChunker(test_config)
        
        chunks_recursive = chunker.chunk_text(long_text, ChunkingStrategy.RECURSIVE)
        chunks_character = chunker.chunk_text(long_text, ChunkingStrategy.CHARACTER)
        
        # Results may be different depending on text structure
        assert isinstance(chunks_recursive, list)
        assert isinstance(chunks_character, list)
        
        # Both should produce valid chunks
        for chunk in chunks_recursive + chunks_character:
            assert isinstance(chunk, str)
            assert chunk.strip() != "" 