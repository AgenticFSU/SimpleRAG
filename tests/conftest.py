"""
Pytest configuration and shared fixtures for RAG system tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Import from parent package
from rag import RAGConfig, create_rag_retriever, ChunkingStrategy


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for each test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_config(temp_dir):
    """Create a test configuration with temporary directories."""
    return RAGConfig(
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=50,
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHROMA_PERSIST_DIRECTORY=os.path.join(temp_dir, "test_chroma"),
        COLLECTION_NAME="test_collection",
        DEFAULT_TOP_K=3,
        SIMILARITY_THRESHOLD=0.5,
        BATCH_SIZE=10,
        MAX_WORKERS=2
    )


@pytest.fixture(scope="function") 
def sample_texts():
    """Sample texts for testing."""
    return [
        "Artificial Intelligence is a branch of computer science.",
        "Machine learning algorithms can learn from data automatically.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information."
    ]


@pytest.fixture(scope="function")
def sample_documents():
    """Sample documents with metadata for testing."""
    return {
        "documents": [
            "Python is a versatile programming language used for AI, web development, and data science.",
            "JavaScript is essential for modern web development and can run on both frontend and backend.",
            "Machine learning algorithms analyze large datasets to find patterns and make predictions.",
            "Deep learning models require significant computational resources and large amounts of data.",
            "Data science combines statistics, programming, and domain expertise to extract insights."
        ],
        "metadatas": [
            {"category": "programming", "language": "python", "difficulty": "beginner"},
            {"category": "programming", "language": "javascript", "difficulty": "intermediate"},
            {"category": "ai", "topic": "machine_learning", "difficulty": "advanced"},
            {"category": "ai", "topic": "deep_learning", "difficulty": "advanced"},
            {"category": "data", "topic": "data_science", "difficulty": "intermediate"}
        ]
    }


@pytest.fixture(scope="function")
def rag_instance(test_config):
    """Create a RAG instance for testing."""
    from .. import RAGRetriever
    return RAGRetriever(config=test_config)


@pytest.fixture(scope="function")
def populated_rag(rag_instance, sample_documents):
    """RAG instance with sample data pre-loaded."""
    rag_instance.ingest_documents(
        documents=sample_documents["documents"],
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        metadatas=sample_documents["metadatas"]
    )
    return rag_instance


@pytest.fixture(scope="function")
def long_text():
    """Long text for chunking tests."""
    return """
    The field of artificial intelligence has evolved dramatically over the past several decades. 
    Starting from simple rule-based systems, AI has progressed to sophisticated machine learning 
    algorithms that can process vast amounts of data and make complex decisions.
    
    Machine learning, a subset of artificial intelligence, encompasses various techniques including 
    supervised learning, unsupervised learning, and reinforcement learning. Each approach has its 
    own strengths and is suited to different types of problems.
    
    Deep learning, which uses neural networks with multiple hidden layers, has revolutionized 
    many fields including computer vision, natural language processing, and speech recognition. 
    These models can automatically learn hierarchical representations from raw data.
    
    Natural language processing enables computers to understand, interpret, and generate human 
    language. Applications include chatbots, machine translation, sentiment analysis, and 
    document summarization.
    
    Computer vision allows machines to interpret and analyze visual information from images 
    and videos. This technology is used in autonomous vehicles, medical imaging, facial 
    recognition systems, and quality control in manufacturing.
    
    The future of AI holds great promise but also presents significant challenges. Issues such 
    as algorithmic bias, data privacy, job displacement, and the need for explainable AI 
    systems require careful consideration as the technology continues to advance.
    """


@pytest.fixture(autouse=True)
def cleanup_collections():
    """Cleanup any test collections after each test."""
    yield
    # This runs after each test
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Try to clean up any test collections
        temp_clients = []
        test_paths = [
            "./test_chroma",
            "../test_chroma", 
            "/tmp/test_chroma"
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                try:
                    client = chromadb.PersistentClient(
                        path=path,
                        settings=Settings(allow_reset=True, anonymized_telemetry=False)
                    )
                    collections = client.list_collections()
                    for collection in collections:
                        if "test" in collection.name.lower():
                            client.delete_collection(collection.name)
                except Exception:
                    pass  # Ignore cleanup errors
                    
                # Remove the directory
                shutil.rmtree(path, ignore_errors=True)
                
    except Exception:
        pass  # Ignore cleanup errors 