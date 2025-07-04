"""
Tests for RAG vector store module.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from rag.core.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    """Test the ChromaVectorStore class."""
    
    @patch('rag.core.vector_store.chromadb.PersistentClient')
    def test_vector_store_creation(self, mock_client, test_config):
        """Test creating vector store."""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(test_config)
        
        assert store.config == test_config
        # Access the collection property to trigger lazy loading
        assert store.collection == mock_collection
        
    @patch('rag.core.vector_store.chromadb.PersistentClient')
    @patch('rag.core.vector_store.TextEmbedder')
    def test_add_chunks_to_db_single(self, mock_embedder, mock_client, test_config):
        """Test adding a single text with automatic batching."""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        # Mock embedder
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embed_texts.return_value = [np.array([0.1, 0.2, 0.3])]
        
        store = ChromaVectorStore(test_config)
        
        texts = ["Test document text"]
        metadatas = [{"source": "test"}]
        
        result = store.add_chunks_to_db(texts, metadatas)
        
        assert isinstance(result, list)
        assert len(result) == 1
        
    @patch('rag.core.vector_store.chromadb.PersistentClient')
    @patch('rag.core.vector_store.TextEmbedder')
    def test_add_chunks_to_db_batching(self, mock_embedder, mock_client, test_config):
        """Test adding multiple texts with automatic batching."""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        # Mock embedder - return different embeddings for each batch
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        # Create a list of 5 texts to test batching with batch_size=2
        texts = [f"Test document {i}" for i in range(5)]
        
        # Mock embed_texts to return appropriate embeddings for each batch call
        mock_embedder_instance.embed_texts.side_effect = [
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])],  # First batch (2 items)
            [np.array([0.7, 0.8, 0.9]), np.array([1.0, 1.1, 1.2])],  # Second batch (2 items)  
            [np.array([1.3, 1.4, 1.5])]                               # Third batch (1 item)
        ]
        
        store = ChromaVectorStore(test_config)
        
        # Test with small batch size to force batching
        result = store.add_chunks_to_db(texts, batch_size=2)
        
        assert isinstance(result, list)
        assert len(result) == 5  # Should return 5 IDs
        
        # Verify embed_texts was called 3 times (for 3 batches)
        assert mock_embedder_instance.embed_texts.call_count == 3
        
        # Verify collection.add was called 3 times (once per batch)
        assert mock_collection.add.call_count == 3
        
    @patch('rag.core.vector_store.chromadb.PersistentClient')
    @patch('rag.core.vector_store.TextEmbedder')
    def test_similarity_search(self, mock_embedder, mock_client, test_config):
        """Test similarity search in the vector store."""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        mock_results = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1 text', 'Document 2 text']],
            'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
            'distances': [[0.1, 0.3]]
        }
        mock_collection.query.return_value = mock_results
        
        # Mock embedder
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        
        store = ChromaVectorStore(test_config)
        
        results = store.similarity_search("test query", k=2)
        
        assert len(results) == 2
        assert results[0][0] == "Document 1 text"  # document
        assert results[0][1] == 0.9  # similarity (1 - 0.1)
        
    @patch('rag.core.vector_store.chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_client, test_config):
        """Test getting collection statistics."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(test_config)
        
        stats = store.get_collection_stats()
        
        assert stats["document_count"] == 42
        assert stats["collection_name"] == test_config.COLLECTION_NAME 