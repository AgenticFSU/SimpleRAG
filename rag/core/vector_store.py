"""
Vector store implementation using ChromaDB for the RAG system.
"""

import os
import logging
from typing import List, Optional, Dict, Tuple, Any
import uuid
import chromadb
from chromadb.config import Settings

from .config import RAGConfig, DEFAULT_CONFIG
from .embedder import TextEmbedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store for efficient similarity search."""
    
    def __init__(
        self, 
        config: Optional[RAGConfig] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[TextEmbedder] = None
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            config: RAG configuration object
            collection_name: Name of the collection to use
            embedder: Text embedder instance
        """
        self.config = config or DEFAULT_CONFIG
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        self.embedder = embedder or TextEmbedder(self.config)
        
        # Initialize ChromaDB client
        self._client = None
        self._collection = None
        
        # Ensure persistence directory exists
        os.makedirs(self.config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    @property
    def client(self) -> chromadb.Client:
        """Lazy loading of ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=self.config.CHROMA_PERSIST_DIRECTORY,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                logger.info(f"Connected to ChromaDB at: {self.config.CHROMA_PERSIST_DIRECTORY}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise
        
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Got/Created collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to get or create collection: {e}")
                raise
        
        return self._collection
    
    def add_chunks_to_db(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Add text chunks to the vector store with automatic batching.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            batch_size: Batch size for processing
            
        Returns:
            List of all document IDs
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.BATCH_SIZE
        all_ids = []
        
        # Process in batches - loop automatically handles single batch if texts <= batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Handle batch metadata
            batch_metadatas = None
            if metadatas:
                batch_metadatas = metadatas[i:i + batch_size]
            else:
                batch_metadatas = [{"text_length": len(text)} for text in batch_texts]
            
            # Handle batch IDs
            batch_ids = None
            if ids:
                batch_ids = ids[i:i + batch_size]
            else:
                batch_ids = [str(uuid.uuid4()) for _ in batch_texts]
            
            # Ensure we have the right number of items for this batch
            assert len(batch_texts) == len(batch_ids), "Number of texts and IDs must match"
            assert len(batch_texts) == len(batch_metadatas), "Number of texts and metadatas must match"
            
            try:
                # Generate embeddings for this batch
                embeddings = self.embedder.embed_texts(batch_texts, show_progress_bar=True)
                
                # Convert numpy arrays to lists for ChromaDB
                embedding_lists = [emb.tolist() for emb in embeddings]
                
                # Add batch to collection
                self.collection.add(
                    embeddings=embedding_lists,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                all_ids.extend(batch_ids)
                
                # Log progress only if we have multiple batches
                total_batches = (len(texts) + batch_size - 1) // batch_size
                if total_batches > 1:
                    logger.info(f"Processed batch {i//batch_size + 1}/{total_batches}")
                else:
                    logger.info(f"Added {len(batch_texts)} chunks to collection")
                
            except Exception as e:
                logger.error(f"Failed to add batch to vector store: {e}")
                raise
        
        return all_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None, 
        embedding: Optional[bool] = False
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples containing (document, similarity_score, metadata)
        """
        k = k or self.config.DEFAULT_TOP_K
        
        try:
            # Generate query embedding
            if not embedding:
                query = self.embedder.embed_text(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query.tolist()],
                n_results=k,
                where=filter_dict
            )
            
            # Format results
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            # Convert distances to similarity scores (ChromaDB returns distances)
            # For cosine distance, similarity = 1 - distance
            similarities = [1 - dist for dist in distances]
            
            # Combine results
            search_results = []
            for doc, sim, meta in zip(documents, similarities, metadatas):
                if sim >= self.config.SIMILARITY_THRESHOLD:
                    search_results.append((doc, sim, meta or {}))
            
            logger.info(f"Found {len(search_results)} similar documents for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedder.embedding_dimension,
                "persistence_path": self.config.CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs and delete them
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
            logger.info("Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing collection
            self.client.delete_collection(name=self.collection_name)
            self._collection = None  # Reset cached collection
            
            # Recreate collection (will be created on next access)
            _ = self.collection
            
            logger.info("Reset collection successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False 