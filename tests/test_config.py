"""
Tests for RAG configuration module.
"""

import pytest
from rag.core.config import RAGConfig, ChunkingStrategy, DEFAULT_CONFIG


class TestRAGConfig:
    """Test the RAGConfig class."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = RAGConfig()
        
        assert config.CHUNK_SIZE == 1000
        assert config.CHUNK_OVERLAP == 200
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert config.DEFAULT_TOP_K == 5
        assert config.SIMILARITY_THRESHOLD == 0.5
        assert config.BATCH_SIZE == 100
        assert config.MAX_WORKERS == 4
        
    def test_custom_config_creation(self):
        """Test creating a custom configuration."""
        config = RAGConfig(
            CHUNK_SIZE=346,
            CHUNK_OVERLAP=13,
            EMBEDDING_MODEL="herty-base-v2",
            DEFAULT_TOP_K=4,
            SIMILARITY_THRESHOLD=0.573
        )
        
        assert config.CHUNK_SIZE == 346
        assert config.CHUNK_OVERLAP == 13
        assert config.EMBEDDING_MODEL == "herty-base-v2"
        assert config.DEFAULT_TOP_K == 4
        assert config.SIMILARITY_THRESHOLD == 0.573
        
    def test_recursive_separators_initialization(self):
        """Test that recursive separators are properly initialized."""
        config = RAGConfig()
        
        assert config.RECURSIVE_SEPARATORS is not None
        assert len(config.RECURSIVE_SEPARATORS) > 0
        assert "\n\n" in config.RECURSIVE_SEPARATORS
        assert "\n" in config.RECURSIVE_SEPARATORS
        assert " " in config.RECURSIVE_SEPARATORS
        assert "" in config.RECURSIVE_SEPARATORS  # Fallback
        
    def test_custom_recursive_separators(self):
        """Test setting custom recursive separators."""
        custom_separators = ["\n\n", "\n", ".", ","]
        config = RAGConfig(RECURSIVE_SEPARATORS=custom_separators)
        
        assert config.RECURSIVE_SEPARATORS == custom_separators
        
    def test_default_config_instance(self):
        """Test the DEFAULT_CONFIG instance."""
        assert isinstance(DEFAULT_CONFIG, RAGConfig)
        assert DEFAULT_CONFIG.CHUNK_SIZE == 1000
        assert DEFAULT_CONFIG.RECURSIVE_SEPARATORS is not None


class TestChunkingStrategy:
    """Test the ChunkingStrategy enum."""
    
    def test_chunking_strategy_values(self):
        """Test that chunking strategies have correct values."""
        assert ChunkingStrategy.RECURSIVE.value == "recursive"
        assert ChunkingStrategy.CHARACTER.value == "character"
        
    def test_chunking_strategy_from_string(self):
        """Test creating chunking strategy from string."""
        assert ChunkingStrategy("recursive") == ChunkingStrategy.RECURSIVE
        assert ChunkingStrategy("character") == ChunkingStrategy.CHARACTER
        
    def test_invalid_chunking_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            ChunkingStrategy("invalid_strategy")
            
    def test_chunking_strategy_membership(self):
        """Test membership in ChunkingStrategy enum."""
        strategies = list(ChunkingStrategy)
        assert ChunkingStrategy.RECURSIVE in strategies
        assert ChunkingStrategy.CHARACTER in strategies
        assert len(strategies) == 2  # Only 2 strategies currently


class TestConfigValidation:
    """Test configuration validation and edge cases."""
    
    def test_chunk_size_validation(self):
        """Test chunk size edge cases."""
        # Very small chunk size
        config = RAGConfig(CHUNK_SIZE=1)
        assert config.CHUNK_SIZE == 1
        
        # Large chunk size
        config = RAGConfig(CHUNK_SIZE=10000)
        assert config.CHUNK_SIZE == 10000
        
    def test_chunk_overlap_validation(self):
        """Test chunk overlap edge cases."""
        # Zero overlap
        config = RAGConfig(CHUNK_OVERLAP=0)
        assert config.CHUNK_OVERLAP == 0
        
        # Overlap larger than chunk size (should be allowed for flexibility)
        config = RAGConfig(CHUNK_SIZE=100, CHUNK_OVERLAP=150)
        assert config.CHUNK_OVERLAP == 150
        
    def test_similarity_threshold_range(self):
        """Test similarity threshold values."""
        # Minimum threshold
        config = RAGConfig(SIMILARITY_THRESHOLD=0.0)
        assert config.SIMILARITY_THRESHOLD == 0.0
        
        # Maximum threshold
        config = RAGConfig(SIMILARITY_THRESHOLD=1.0)
        assert config.SIMILARITY_THRESHOLD == 1.0
        
        # Negative threshold (should be allowed for flexibility)
        config = RAGConfig(SIMILARITY_THRESHOLD=-0.1)
        assert config.SIMILARITY_THRESHOLD == -0.1
        
    def test_top_k_validation(self):
        """Test top-k values."""
        # Minimum top-k
        config = RAGConfig(DEFAULT_TOP_K=1)
        assert config.DEFAULT_TOP_K == 1
        
        # Large top-k
        config = RAGConfig(DEFAULT_TOP_K=1000)
        assert config.DEFAULT_TOP_K == 1000
        
    def test_batch_size_and_workers(self):
        """Test batch size and worker configurations."""
        config = RAGConfig(BATCH_SIZE=1, MAX_WORKERS=1)
        assert config.BATCH_SIZE == 1
        assert config.MAX_WORKERS == 1
        
        config = RAGConfig(BATCH_SIZE=1000, MAX_WORKERS=16)
        assert config.BATCH_SIZE == 1000
        assert config.MAX_WORKERS == 16 