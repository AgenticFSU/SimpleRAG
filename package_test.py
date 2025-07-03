#!/usr/bin/env python3
"""
Quick test script to verify the RAG package works as a standalone unit.
"""

def test_package_completeness():
    """Test that all main components can be imported and basic functionality works."""
    print("Testing RAG package completeness...")
    
    # Test imports
    try:
        from rag import RAGConfig, ChunkingStrategy, create_rag_retriever
        print("✓ Main components imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test configuration
    try:
        config = RAGConfig(CHUNK_SIZE=100, CHUNK_OVERLAP=20)
        print("✓ Configuration created successfully")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False
    
    # Test chunking strategies
    try:
        strategies = list(ChunkingStrategy)
        print(f"✓ Chunking strategies available: {[s.value for s in strategies]}")
    except Exception as e:
        print(f"✗ Chunking strategies failed: {e}")
        return False
    
    # Test RAG retriever creation
    try:
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        config = RAGConfig(
            CHROMA_PERSIST_DIRECTORY=os.path.join(temp_dir, "test_chroma"),
            COLLECTION_NAME="test_collection"
        )
        _ = create_rag_retriever(config)
        print("✓ RAG retriever created successfully")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"✗ RAG retriever creation failed: {e}")
        return False
    
    print("\n🎉 Package completeness test PASSED!")
    return True

if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports when running as script
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Import and run test
    success = test_package_completeness()
    sys.exit(0 if success else 1) 