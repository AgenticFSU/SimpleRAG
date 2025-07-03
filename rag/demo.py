"""
Demonstration script for the RAG system.

This script shows how to use the RAG system with different chunking strategies,
ingest documents, and perform retrieval queries.
"""

import logging
from rag.core import ChunkingStrategy, create_rag_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demo_basic_usage():
    """Demonstrate basic RAG functionality."""
    print("=" * 60)
    print("RAG System Demo - Basic Usage")
    print("=" * 60)
    
    # Create a RAG retriever
    rag = create_rag_retriever(
        collection_name="demo_collection",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3
    )
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        machines that can perform tasks that typically require human intelligence. These 
        tasks include learning, reasoning, problem-solving, perception, and language 
        understanding. AI has applications in various fields including healthcare, 
        finance, transportation, and entertainment.
        """,
        """
        Machine Learning is a subset of AI that enables computers to learn and improve 
        from experience without being explicitly programmed. It uses algorithms to 
        analyze data, identify patterns, and make predictions or decisions. Common 
        types include supervised learning, unsupervised learning, and reinforcement 
        learning.
        """,
        """
        Natural Language Processing (NLP) is a field of AI that focuses on the 
        interaction between computers and human language. It involves teaching 
        machines to understand, interpret, and generate human language in a 
        valuable way. Applications include chatbots, language translation, 
        sentiment analysis, and text summarization.
        """,
        """
        Deep Learning is a subset of machine learning that uses neural networks 
        with multiple layers to model and understand complex patterns in data. 
        It has achieved remarkable success in areas like image recognition, 
        speech recognition, and natural language processing. Deep learning 
        models require large amounts of data and computational power.
        """
    ]
    
    # Clear existing data
    rag.reset_system()
    
    print("Ingesting documents...")
    result = rag.ingest_documents(
        documents=documents,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        metadatas=[
            {"topic": "AI Overview", "source": "demo"},
            {"topic": "Machine Learning", "source": "demo"},
            {"topic": "NLP", "source": "demo"},
            {"topic": "Deep Learning", "source": "demo"}
        ]
    )
    
    print(f"Ingestion result: {result}")
    print()
    
    # Perform some queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Tell me about neural networks",
        "What are applications of NLP?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        rag_result = rag.retrieve(query, k=2)
        
        print(f"Found {rag_result.total_chunks_found} relevant chunks:")
        for i, (chunk, similarity) in enumerate(zip(rag_result.chunks, rag_result.similarities)):
            print(f"  Chunk {i+1} (similarity: {similarity:.3f}):")
            print(f"    {chunk[:200]}...")
        print()
    
    # Get system stats
    stats = rag.get_system_stats()
    print("System Statistics:")
    print(f"  Documents in vector store: {stats['vector_store']['document_count']}")
    print(f"  Embedding model: {stats['embedder']['model_name']}")
    print(f"  Embedding dimension: {stats['embedder']['embedding_dimension']}")
    print()


def demo_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("=" * 60)
    print("RAG System Demo - Chunking Strategies")
    print("=" * 60)
    
    # Sample long text
    sample_text = """
    The field of artificial intelligence has undergone tremendous growth and transformation 
    over the past decade. Machine learning, which is a subset of AI, has become increasingly 
    sophisticated with the development of deep learning techniques.
    
    Deep learning models, particularly neural networks with multiple hidden layers, have 
    achieved remarkable performance in various domains. Computer vision applications now 
    can recognize objects in images with superhuman accuracy. Natural language processing 
    systems can understand and generate human-like text.
    
    The impact of AI extends beyond technology companies. Healthcare institutions use AI 
    for medical diagnosis and drug discovery. Financial firms employ machine learning 
    algorithms for fraud detection and algorithmic trading. Autonomous vehicles rely on 
    AI systems for navigation and decision-making.
    
    However, the rapid advancement of AI also raises important ethical and societal 
    questions. Issues such as algorithmic bias, job displacement, and privacy concerns 
    need to be carefully addressed. The development of AI must be guided by principles 
    that ensure its benefits are distributed fairly across society.
    """
    
    strategies = [
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.CHARACTER
    ]
    
    for strategy in strategies:
        print(f"Testing {strategy.value} chunking:")
        
        # Create a new RAG instance for each strategy
        rag = create_rag_retriever(
            collection_name=f"demo_{strategy.value}",
            chunk_size=300,
            chunk_overlap=50
        )
        
        # Clear and ingest with current strategy
        rag.reset_system()
        result = rag.ingest_text(
            text=sample_text,
            chunking_strategy=strategy,
            metadata={"strategy": strategy.value}
        )
        
        print(f"  Chunks created: {result['chunks_added']}")
        print(f"  Average chunk length: {result['chunk_stats']['avg_chunk_length']:.1f}")
        print(f"  Min/Max chunk length: {result['chunk_stats']['min_chunk_length']}/{result['chunk_stats']['max_chunk_length']}")
        
        # Test retrieval
        rag_result = rag.retrieve("How is AI used in healthcare?", k=2)
        print(f"  Retrieved {rag_result.total_chunks_found} chunks for healthcare query")
        print()


def demo_advanced_features():
    """Demonstrate advanced RAG features."""
    print("=" * 60)
    print("RAG System Demo - Advanced Features")
    print("=" * 60)
    
    rag = create_rag_retriever(collection_name="advanced_demo")
    rag.reset_system()
    
    # Ingest documents with rich metadata
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is the programming language of the web, used for both frontend and backend development.",
        "Machine learning algorithms can be implemented in various programming languages including Python and R.",
        "Data science involves collecting, analyzing, and interpreting complex data sets.",
        "Web development includes frontend technologies like HTML, CSS, and JavaScript."
    ]
    
    metadatas = [
        {"category": "programming", "language": "python", "difficulty": "beginner"},
        {"category": "programming", "language": "javascript", "difficulty": "intermediate"},
        {"category": "ai", "language": "python", "difficulty": "advanced"},
        {"category": "data", "language": "multiple", "difficulty": "intermediate"},
        {"category": "web", "language": "javascript", "difficulty": "beginner"}
    ]
    
    rag.ingest_documents(documents, metadatas=metadatas)
    
    print("Testing filtered queries:")
    
    # Query with category filter
    print("1. Programming-related documents only:")
    result = rag.retrieve(
        "programming languages",
        k=5,
        filter_dict={"category": "programming"}
    )
    for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities)):
        print(f"   {i+1}. {chunk} (similarity: {similarity:.3f})")
    print()
    
    # Query with difficulty filter
    print("2. Beginner-level documents only:")
    result = rag.retrieve(
        "programming",
        k=5,
        filter_dict={"difficulty": "beginner"}
    )
    for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities)):
        print(f"   {i+1}. {chunk} (similarity: {similarity:.3f})")
    print()
    
    # Get context string
    print("3. Getting context for a query:")
    context = rag.get_context("web development", k=3, separator="\n---\n")
    print(f"Context:\n{context}")
    print()


def main():
    """Run all demonstrations."""
    try:
        demo_basic_usage()
        demo_chunking_strategies()
        demo_advanced_features()
        
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 