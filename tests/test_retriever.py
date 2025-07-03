"""
Tests for RAG retriever module.
"""

from unittest.mock import patch, MagicMock
from rag import RAGRetriever, create_rag_retriever, ChunkingStrategy
from rag.core.retriever import RAGResult


class TestRAGRetriever:
    """Test the main RAGRetriever class."""
    
    def test_rag_retriever_creation(self, test_config):
        """Test creating RAG retriever."""
        retriever = RAGRetriever(test_config)
        
        assert retriever.config == test_config
        assert retriever.chunker is not None
        assert retriever.embedder is not None
        assert retriever.vector_store is not None
        
    def test_rag_retriever_default_config(self):
        """Test RAG retriever with default config."""
        retriever = RAGRetriever()
        assert retriever.config is not None
        
    @patch('rag.core.retriever.ChromaVectorStore')
    def test_ingest_text_single(self, mock_vector_store, test_config):
        """Test ingesting a single text."""
        mock_store_instance = MagicMock()
        mock_store_instance.add_documents.return_value = {"success": True, "documents_added": 2}
        mock_vector_store.return_value = mock_store_instance
        
        retriever = RAGRetriever(test_config)
        
        text = "This is a test document for ingestion."
        result = retriever.ingest_text(text, ChunkingStrategy.RECURSIVE)
        
        assert result["success"]
        assert result["chunks_added"] == 2
        
    def test_ingest_empty_text(self, test_config):
        """Test ingesting empty text."""
        retriever = RAGRetriever(test_config)
        
        result = retriever.ingest_text("", ChunkingStrategy.RECURSIVE)
        
        assert result["success"]
        assert result["chunks_added"] == 0
        
    @patch('rag.core.retriever.ChromaVectorStore')
    def test_ingest_documents_multiple(self, mock_vector_store, test_config, sample_documents):
        """Test ingesting multiple documents."""
        mock_store_instance = MagicMock()
        mock_store_instance.add_documents.return_value = {"success": True, "documents_added": 5}
        mock_vector_store.return_value = mock_store_instance
        
        retriever = RAGRetriever(test_config)
        
        documents = sample_documents["documents"]
        metadatas = sample_documents["metadatas"]
        
        result = retriever.ingest_documents(
            documents, 
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            metadatas=metadatas
        )
        
        assert result["success"]
        assert result["total_chunks_added"] == 5
        
    @patch('rag.core.retriever.ChromaVectorStore')
    def test_retrieve_documents(self, mock_vector_store, test_config):
        """Test document retrieval."""
        mock_store_instance = MagicMock()
        mock_store_instance.query.return_value = [
            {"document": "Test doc 1", "similarity": 0.9, "metadata": {}},
            {"document": "Test doc 2", "similarity": 0.8, "metadata": {}}
        ]
        mock_vector_store.return_value = mock_store_instance
        
        retriever = RAGRetriever(test_config)
        
        result = retriever.retrieve("test query", k=2)
        
        assert isinstance(result, RAGResult)
        assert result.total_chunks_found == 2
        assert len(result.chunks) == 2
        assert result.retrieval_time > 0
        
    def test_get_context(self, test_config):
        """Test context generation."""
        retriever = RAGRetriever(test_config)
        
        # Mock the retrieve method  
        mock_result = RAGResult(
            query="test query",
            chunks=["First chunk", "Second chunk"],
            similarities=[0.9, 0.8],
            metadatas=[{}, {}],
            total_chunks_found=2,
            retrieval_time=0.05
        )
        
        with patch.object(retriever, 'retrieve', return_value=mock_result):
            context = retriever.get_context("test query", k=2)
            
            assert "First chunk" in context
            assert "Second chunk" in context
            
    def test_get_system_stats(self, test_config):
        """Test getting system statistics."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store stats
        with patch.object(retriever.vector_store, 'get_collection_stats', 
                         return_value={"document_count": 10}):
            stats = retriever.get_system_stats()
            
            assert "chunker" in stats
            assert "embedder" in stats
            assert "vector_store" in stats
            assert stats["vector_store"]["document_count"] == 10


class TestCreateRAGRetriever:
    """Test the create_rag_retriever convenience function."""
    
    def test_create_rag_retriever_default(self):
        """Test creating RAG retriever with defaults."""
        retriever = create_rag_retriever()
        
        assert isinstance(retriever, RAGRetriever)
        assert retriever.config is not None
        
    def test_create_rag_retriever_custom_params(self, temp_dir):
        """Test creating RAG retriever with custom parameters."""
        retriever = create_rag_retriever(
            collection_name="custom_test",
            embedding_model="all-mpnet-base-v2",
            chunk_size=500,
            top_k=10
        )
        
        assert retriever.config.COLLECTION_NAME == "custom_test"
        assert retriever.config.EMBEDDING_MODEL == "all-mpnet-base-v2"
        assert retriever.config.CHUNK_SIZE == 500
        assert retriever.config.DEFAULT_TOP_K == 10


class TestRAGResult:
    """Test the RAGResult class."""
    
    def test_rag_result_creation(self):
        """Test creating a RAG result."""
        chunks = ["Test chunk"]
        similarities = [0.9]
        metadatas = [{"test": "value"}]
        
        result = RAGResult(
            query="test query",
            chunks=chunks,
            similarities=similarities,
            metadatas=metadatas,
            total_chunks_found=1,
            retrieval_time=0.05
        )
        
        assert result.chunks == chunks
        assert result.similarities == similarities
        assert result.metadatas == metadatas
        assert result.total_chunks_found == 1
        assert result.retrieval_time == 0.05
        
    def test_rag_result_empty(self):
        """Test empty RAG result."""
        result = RAGResult(
            query="test query",
            chunks=[],
            similarities=[],
            metadatas=[],
            total_chunks_found=0,
            retrieval_time=0.01
        )
        
        assert len(result.chunks) == 0
        assert result.total_chunks_found == 0 