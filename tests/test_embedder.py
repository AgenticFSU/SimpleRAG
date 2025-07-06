"""
Tests for RAG text embedding module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag.core.embedder import TextEmbedder
from rag.core.config import RAGConfig


class TestTextEmbedder:
    """Test the TextEmbedder class."""
    
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedder_creation_default(self, mock_sentence_transformer):
        """Test creating embedder with default config."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder()
        assert embedder.config is not None
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is mock_model  # Model is loaded immediately
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedder_creation_custom_config(self, mock_sentence_transformer, test_config):
        """Test creating embedder with custom config."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        assert embedder.config == test_config
        assert embedder.model_name == test_config.EMBEDDING_MODEL
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedder_creation_custom_model(self, mock_sentence_transformer, test_config):
        """Test creating embedder with custom model override."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        custom_model = "all-mpnet-base-v2"
        embedder = TextEmbedder(test_config, model_name=custom_model)
        assert embedder.model_name == custom_model

    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedding_dimension_property(self, mock_sentence_transformer, test_config):
        """Test embedding dimension property."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        # Should return the dimension from the model
        dimension = embedder.embedding_dimension
        assert dimension == 768
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_text_single(self, mock_sentence_transformer, test_config):
        """Test embedding a single text."""
        mock_model = MagicMock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        text = "Test text"
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert np.array_equal(embedding, mock_embedding)
        mock_model.encode.assert_called_once_with(text, convert_to_numpy=True)
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_empty_text(self, mock_sentence_transformer, test_config):
        """Test embedding empty text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embedding = embedder.embed_text("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.allclose(embedding, np.zeros(384))
        
        # Should not call the model for empty text
        mock_model.encode.assert_not_called()
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_whitespace_only_text(self, mock_sentence_transformer, test_config):
        """Test embedding whitespace-only text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embedding = embedder.embed_text("   \n\t  ")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.allclose(embedding, np.zeros(384))
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_multiple(self, mock_sentence_transformer, test_config, sample_texts):
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embeddings = embedder.embed_texts(sample_texts[:3])
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (2,)
            
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_empty_list(self, mock_sentence_transformer, test_config):
        """Test embedding empty text list."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embeddings = embedder.embed_texts([])
        assert embeddings == []
        mock_model.encode.assert_not_called()
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_with_empty_strings(self, mock_sentence_transformer, test_config):
        """Test embedding texts with some empty strings."""
        mock_model = MagicMock()
        # Use the actual embedding dimension from test_config
        embedding_dim = 384
        mock_embedding = np.random.rand(1, embedding_dim)  # Create valid embedding
        mock_model.encode.return_value = mock_embedding
        mock_model.get_sentence_embedding_dimension.return_value = embedding_dim
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        texts = ["", "Valid text", "   "]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 3
        
        # First and third should be zero vectors with correct dimension
        assert np.allclose(embeddings[0], np.zeros(embedding_dim))
        assert np.allclose(embeddings[2], np.zeros(embedding_dim))
        
        # Second should be the actual embedding
        assert np.array_equal(embeddings[1], mock_embedding[0])
        
        # Model should only be called with non-empty text
        mock_model.encode.assert_called_once()
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_single_text_1d_result(self, mock_sentence_transformer, test_config):
        """Test handling of 1D result from single text batch."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2])  # 1D array
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embeddings = embedder.embed_texts(["Single text"])
        
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape == (2,)
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_batch_processing(self, mock_sentence_transformer, test_config):
        """Test batch processing of texts."""
        mock_model = MagicMock()
        # Mock multiple batch calls
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # First batch
            np.array([[0.5, 0.6]])                # Second batch
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sentence_transformer.return_value = mock_model
        
        # Use small batch size to force multiple batches
        config = RAGConfig(BATCH_SIZE=2)
        embedder = TextEmbedder(config)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert mock_model.encode.call_count == 2  # Two batches
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_get_model_info(self, mock_sentence_transformer, test_config):
        """Test getting model information."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        info = embedder.get_model_info()
        
        assert "model_name" in info
        assert "embedding_dimension" in info
        
        assert info["model_name"] == test_config.EMBEDDING_MODEL
        assert info["embedding_dimension"] == 384

    @patch('rag.core.embedder.SentenceTransformer')
    def test_model_loading_failure(self, mock_sentence_transformer, test_config):
        """Test handling of model loading failure."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            TextEmbedder(test_config)  # Exception happens in __init__ now
            
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedding_failure(self, mock_sentence_transformer, test_config):
        """Test handling of embedding failure."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        with pytest.raises(Exception, match="Encoding failed"):
            embedder.embed_text("Test text")
            
        with pytest.raises(Exception, match="Encoding failed"):
            embedder.embed_texts(["Test text"])


class TestEmbeddingIntegration:
    """Integration tests for text embedding (requires actual model loading)."""
    
    @pytest.mark.integration
    def test_real_embedding_single_text(self, test_config):
        """Test real embedding of single text."""
        embedder = TextEmbedder(test_config)
        
        text = "This is a test sentence for embedding."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dimension,)
        assert not np.allclose(embedding, np.zeros_like(embedding))
        
    @pytest.mark.integration  
    def test_real_embedding_multiple_texts(self, test_config, sample_texts):
        """Test real embedding of multiple texts."""
        embedder = TextEmbedder(test_config)
        
        embeddings = embedder.embed_texts(sample_texts)
        
        assert len(embeddings) == len(sample_texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedder.embedding_dimension,)
