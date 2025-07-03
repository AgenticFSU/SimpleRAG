"""
Main RAG retriever class that orchestrates chunking, embedding, and retrieval.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .config import ChunkingStrategy, RAGConfig, DEFAULT_CONFIG
from .chunker import TextChunker
from .embedder import TextEmbedder
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG retrieval."""
    query: str
    chunks: List[str]
    similarities: List[float]
    metadatas: List[Dict[str, Any]]
    total_chunks_found: int
    retrieval_time: float


class RAGRetriever:
    """
    Main RAG retriever class that provides a complete RAG pipeline.
    
    This class orchestrates text chunking, embedding, storage, and retrieval
    using ChromaDB and LangChain.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            config: RAG configuration object
            collection_name: Name of the ChromaDB collection
            embedding_model: Override embedding model name
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.chunker = TextChunker(self.config)
        self.embedder = TextEmbedder(self.config, embedding_model)
        self.vector_store = ChromaVectorStore(
            self.config, 
            collection_name, 
            self.embedder
        )
        
        logger.info("RAG Retriever initialized successfully")
    
    def ingest_text(
        self,
        text: str,
        chunking_strategy: Union[ChunkingStrategy, str] = ChunkingStrategy.RECURSIVE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single text document into the RAG system.
        
        Args:
            text: Text to ingest
            chunking_strategy: Strategy for chunking the text
            metadata: Optional metadata for the document
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Convert string to enum if needed
            if isinstance(chunking_strategy, str):
                chunking_strategy = ChunkingStrategy(chunking_strategy)
            
            # Chunk the text
            chunks = self.chunker.chunk_text(text, chunking_strategy)
            
            if not chunks:
                logger.warning("No chunks generated from the input text")
                return {
                    "success": False,
                    "message": "No chunks generated from input text",
                    "chunks_added": 0
                }
            
            # Prepare metadata for each chunk
            chunk_metadatas = []
            base_metadata = metadata or {}
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "chunking_strategy": chunking_strategy.value,
                    "total_chunks": len(chunks)
                })
                chunk_metadatas.append(chunk_metadata)
            
            # Add chunks to vector store
            chunk_ids = self.vector_store.add_texts(chunks, chunk_metadatas)
            
            # Get chunk statistics
            chunk_stats = self.chunker.get_chunk_stats(chunks)
            
            logger.info(f"Successfully ingested text with {len(chunks)} chunks")
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "chunk_ids": chunk_ids,
                "chunking_strategy": chunking_strategy.value,
                "chunk_stats": chunk_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            return {
                "success": False,
                "message": f"Failed to ingest text: {str(e)}",
                "chunks_added": 0
            }
    
    def ingest_documents(
        self,
        documents: List[str],
        chunking_strategy: Union[ChunkingStrategy, str] = ChunkingStrategy.RECURSIVE,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents into the RAG system.
        
        Args:
            documents: List of documents to ingest
            chunking_strategy: Strategy for chunking the documents
            metadatas: Optional list of metadata for each document
            
        Returns:
            Dictionary with ingestion results
        """
        if not documents:
            return {
                "success": False,
                "message": "No documents provided",
                "total_chunks_added": 0
            }
        
        try:
            # Convert string to enum if needed
            if isinstance(chunking_strategy, str):
                try:
                    chunking_strategy = ChunkingStrategy(chunking_strategy)
                except ValueError:
                    raise ValueError(f"Invalid chunking strategy: {chunking_strategy}")
            
            all_chunks = []
            all_metadatas = []
            doc_chunk_counts = []
            
            # Process each document
            for doc_idx, document in enumerate(documents):
                # Chunk the document
                chunks = self.chunker.chunk_text(document, chunking_strategy)
                doc_chunk_counts.append(len(chunks))
                
                # Prepare metadata
                base_metadata = {}
                if metadatas and doc_idx < len(metadatas):
                    base_metadata = metadatas[doc_idx].copy()
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "document_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "chunk_length": len(chunk),
                        "chunking_strategy": chunking_strategy.value,
                        "total_chunks_in_doc": len(chunks)
                    })
                    
                    all_chunks.append(chunk)
                    all_metadatas.append(chunk_metadata)
            
            if not all_chunks:
                logger.warning("No chunks generated from any document")
                return {
                    "success": False,
                    "message": "No chunks generated from any document",
                    "total_chunks_added": 0
                }
            
            # Add all chunks to vector store in batches
            chunk_ids = self.vector_store.add_documents_batch(all_chunks, all_metadatas)
            
            # Get overall statistics
            chunk_stats = self.chunker.get_chunk_stats(all_chunks)
            
            logger.info(f"Successfully ingested {len(documents)} documents with {len(all_chunks)} total chunks")
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "total_chunks_added": len(all_chunks),
                "chunks_per_document": doc_chunk_counts,
                "chunk_ids": chunk_ids,
                "chunking_strategy": chunking_strategy.value,
                "chunk_stats": chunk_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            return {
                "success": False,
                "message": f"Failed to ingest documents: {str(e)}",
                "total_chunks_added": 0
            }
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_metadata: bool = True
    ) -> RAGResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            filter_dict: Optional metadata filter
            return_metadata: Whether to include metadata in results
            
        Returns:
            RAGResult object with retrieved chunks and metadata
        """
        import time
        
        start_time = time.time()
        
        try:
            k = k or self.config.DEFAULT_TOP_K
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict
            )
            
            # Extract components
            chunks = [result[0] for result in results]
            similarities = [result[1] for result in results]
            metadatas = [result[2] for result in results] if return_metadata else []
            
            retrieval_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(chunks)} chunks for query in {retrieval_time:.3f}s")
            
            return RAGResult(
                query=query,
                chunks=chunks,
                similarities=similarities,
                metadatas=metadatas,
                total_chunks_found=len(chunks),
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            retrieval_time = time.time() - start_time
            
            return RAGResult(
                query=query,
                chunks=[],
                similarities=[],
                metadatas=[],
                total_chunks_found=0,
                retrieval_time=retrieval_time
            )
    
    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        separator: str = "\n\n",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get concatenated context for a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            separator: Separator between chunks
            filter_dict: Optional metadata filter
            
        Returns:
            Concatenated context string
        """
        result = self.retrieve(query, k, filter_dict, return_metadata=False)
        return separator.join(result.chunks)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get embedder info
            embedder_info = self.embedder.get_model_info()
            
            # Get configuration info
            config_info = {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "default_top_k": self.config.DEFAULT_TOP_K,
                "similarity_threshold": self.config.SIMILARITY_THRESHOLD,
                "batch_size": self.config.BATCH_SIZE
            }
            
            return {
                "vector_store": vector_stats,
                "embedder": embedder_info,
                "configuration": config_info,
                "available_chunking_strategies": [strategy.value for strategy in ChunkingStrategy]
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def clear_data(self) -> bool:
        """
        Clear all data from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        return self.vector_store.clear_collection()
    
    def reset_system(self) -> bool:
        """
        Reset the entire RAG system (clear all data and recreate collection).
        
        Returns:
            True if successful, False otherwise
        """
        return self.vector_store.reset_collection()


# Convenience function for quick RAG setup
def create_rag_retriever(
    collection_name: str = "default_rag",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 5
) -> RAGRetriever:
    """
    Create a RAG retriever with custom configuration.
    
    Args:
        collection_name: Name for the ChromaDB collection
        embedding_model: Embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        top_k: Default number of chunks to retrieve
        
    Returns:
        Configured RAGRetriever instance
    """
    config = RAGConfig(
        CHUNK_SIZE=chunk_size,
        CHUNK_OVERLAP=chunk_overlap,
        EMBEDDING_MODEL=embedding_model,
        DEFAULT_TOP_K=top_k,
        COLLECTION_NAME=collection_name
    )
    
    return RAGRetriever(config=config) 