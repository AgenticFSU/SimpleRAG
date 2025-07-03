"""
Tests for RAG CLI module.
"""

from unittest.mock import patch, MagicMock
from rag import cli


class TestCLICommands:
    """Test CLI command functions."""
    
    @patch('rag.cli.create_rag_retriever')
    def test_ingest_command_text(self, mock_create_rag):
        """Test CLI ingest command with text."""
        mock_retriever = MagicMock()
        mock_retriever.ingest_text.return_value = {"success": True, "chunks_added": 3}
        mock_create_rag.return_value = mock_retriever
        
        # Mock command line arguments
        args = MagicMock()
        args.text = "Test text for ingestion"
        args.files = None
        args.strategy = "recursive"
        args.collection = "test_collection"
        args.model = "all-MiniLM-L6-v2"
        args.chunk_size = 1000
        args.chunk_overlap = 200
        args.top_k = 5
        args.clear = False
        
        cli.ingest_command(args)
        
        mock_retriever.ingest_text.assert_called_once()
        
    @patch('rag.cli.create_rag_retriever')
    @patch('rag.cli.load_text_files')
    def test_ingest_command_files(self, mock_load_files, mock_create_rag):
        """Test CLI ingest command with files."""
        mock_retriever = MagicMock()
        mock_retriever.ingest_documents.return_value = {"success": True, "total_chunks_added": 5}
        mock_create_rag.return_value = mock_retriever
        
        mock_load_files.return_value = ["File content"]
        
        args = MagicMock()
        args.text = None
        args.files = ["test.txt"]
        args.strategy = "character"
        args.collection = "test_collection"
        args.model = "all-MiniLM-L6-v2"
        args.chunk_size = 1000
        args.chunk_overlap = 200
        args.top_k = 5
        args.clear = False
        
        cli.ingest_command(args)
        
        mock_load_files.assert_called_once_with(["test.txt"])
        mock_retriever.ingest_documents.assert_called_once()
        
    @patch('rag.cli.create_rag_retriever')
    def test_query_command(self, mock_create_rag):
        """Test CLI query command."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(
            chunks=["Test result"],
            similarities=[0.9],
            total_chunks_found=1,
            retrieval_time=0.05
        )
        mock_retriever.get_system_stats.return_value = {
            'vector_store': {'document_count': 10}
        }
        mock_create_rag.return_value = mock_retriever
        
        args = MagicMock()
        args.query = "test query"
        args.k = 5
        args.collection = "test_collection"
        args.model = "all-MiniLM-L6-v2"
        args.top_k = 5
        args.context = False
        
        cli.query_command(args)
        
        mock_retriever.retrieve.assert_called_once_with("test query", k=5)
        
    @patch('rag.cli.create_rag_retriever')
    def test_stats_command(self, mock_create_rag):
        """Test CLI stats command."""
        mock_retriever = MagicMock()
        mock_retriever.get_system_stats.return_value = {
            "vector_store": {"document_count": 10, "collection_name": "test", "persistence_path": "/tmp"},
            "embedder": {"model_name": "test-model", "embedding_dimension": 384, "loaded": True},
            "configuration": {"chunk_size": 1000, "chunk_overlap": 200, "default_top_k": 5, "similarity_threshold": 0.7},
            "available_chunking_strategies": ["recursive", "character"]
        }
        mock_create_rag.return_value = mock_retriever
        
        args = MagicMock()
        args.collection = "test_collection"
        
        cli.stats_command(args)
        
        mock_retriever.get_system_stats.assert_called_once()
        
    @patch('rag.cli.create_rag_retriever')
    def test_clear_command(self, mock_create_rag):
        """Test CLI clear command."""
        mock_retriever = MagicMock()
        mock_retriever.clear_data.return_value = True
        mock_create_rag.return_value = mock_retriever
        
        args = MagicMock()
        args.collection = "test_collection"
        args.reset = False
        
        cli.clear_command(args)
        
        mock_retriever.clear_data.assert_called_once()


class TestCLIUtilities:
    """Test CLI utility functions."""
    
    def test_format_similarity_score(self):
        """Test similarity score formatting."""
        # This would test utility functions if they exist
        pass 