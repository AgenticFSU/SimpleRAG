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
    
    def test_embedder_creation_default(self):
        """Test creating embedder with default config."""
        embedder = TextEmbedder()
        assert embedder.config is not None
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder._model is None  # Lazy loading
        
    def test_embedder_creation_custom_config(self, test_config):
        """Test creating embedder with custom config."""
        embedder = TextEmbedder(test_config)
        assert embedder.config == test_config
        assert embedder.model_name == test_config.EMBEDDING_MODEL
        
    def test_embedder_creation_custom_model(self, test_config):
        """Test creating embedder with custom model override."""
        custom_model = "all-mpnet-base-v2"
        embedder = TextEmbedder(test_config, model_name=custom_model)
        assert embedder.model_name == custom_model
        
    def test_embedder_model_info_validation(self):
        """Test model info structure validation."""
        embedder = TextEmbedder()
        model_info = embedder._model_info
        
        assert "name" in model_info
        assert "dimension" in model_info
        assert model_info["name"].startswith("sentence-transformers/")
        
    def test_embedder_unknown_model_warning(self, test_config):
        """Test warning for unknown model."""
        unknown_model = "unknown-model"
        embedder = TextEmbedder(test_config, model_name=unknown_model)
        
        # Should still work but with warning logged
        assert embedder.model_name == unknown_model
        assert embedder._model_info["name"] == f"sentence-transformers/{unknown_model}"
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_model_lazy_loading(self, mock_sentence_transformer, test_config):
        """Test that model is loaded lazily."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        assert embedder._model is None
        
        # Access model property to trigger loading
        model = embedder.model
        assert model is mock_model
        assert embedder._model is mock_model
        
        # Second access should return cached model
        model2 = embedder.model
        assert model2 is mock_model
        mock_sentence_transformer.assert_called_once()
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embedding_dimension_property(self, mock_sentence_transformer, test_config):
        """Test embedding dimension property."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        # Should trigger model loading and return dimension
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
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        embeddings = embedder.embed_texts([])
        assert embeddings == []
        mock_model.encode.assert_not_called()
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_embed_texts_with_empty_strings(self, mock_sentence_transformer, test_config):
        """Test embedding texts with some empty strings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        texts = ["", "Valid text", "   "]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 3
        
        # First and third should be zero vectors
        assert np.allclose(embeddings[0], np.zeros(2))
        assert np.allclose(embeddings[2], np.zeros(2))
        
        # Second should be the actual embedding
        assert np.array_equal(embeddings[1], np.array([0.1, 0.2]))
        
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
        
    def test_similarity_calculation(self, test_config):
        """Test cosine similarity calculation."""
        embedder = TextEmbedder(test_config)
        
        # Test vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        # Orthogonal vectors should have similarity 0
        sim1 = embedder.similarity(vec1, vec2)
        assert abs(sim1 - 0.0) < 1e-10
        
        # Identical vectors should have similarity 1
        sim2 = embedder.similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 1e-10
        
    def test_similarity_zero_vectors(self, test_config):
        """Test similarity with zero vectors."""
        embedder = TextEmbedder(test_config)
        
        zero_vec = np.array([0, 0, 0])
        normal_vec = np.array([1, 0, 0])
        
        # Zero vector should have similarity 0 with any vector
        sim = embedder.similarity(zero_vec, normal_vec)
        assert sim == 0.0
        
        sim = embedder.similarity(normal_vec, zero_vec)
        assert sim == 0.0
        
        sim = embedder.similarity(zero_vec, zero_vec)
        assert sim == 0.0
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_get_model_info(self, mock_sentence_transformer, test_config):
        """Test getting model information."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = TextEmbedder(test_config)
        
        info = embedder.get_model_info()
        
        assert "model_name" in info
        assert "model_path" in info
        assert "embedding_dimension" in info
        assert "description" in info
        assert "loaded" in info
        
        assert info["model_name"] == test_config.EMBEDDING_MODEL
        assert not info["loaded"]  # Model not loaded yet
        
        # Load model and check again
        _ = embedder.model
        info2 = embedder.get_model_info()
        assert info2["loaded"]
        
    @patch('rag.core.embedder.SentenceTransformer')
    def test_model_loading_failure(self, mock_sentence_transformer, test_config):
        """Test handling of model loading failure."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        embedder = TextEmbedder(test_config)
        
        with pytest.raises(Exception, match="Model loading failed"):
            _ = embedder.model
            
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
            
    @pytest.mark.integration
    def test_real_similarity_calculation(self, test_config):
        """Test real similarity calculation with actual embeddings."""
        embedder = TextEmbedder(test_config)
        
        # Similar texts
        text1 = "The cat is sleeping."
        text2 = "A cat is taking a nap."
        
        # Different text
        text3 = "Machine learning algorithms are powerful."
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        
        # Similar texts should have higher similarity
        sim_similar = embedder.similarity(emb1, emb2)
        sim_different = embedder.similarity(emb1, emb3)
        
        assert sim_similar > sim_different
        assert 0 <= sim_similar <= 1
        assert -1 <= sim_different <= 1 