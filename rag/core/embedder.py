"""
Text embedding functionality for the RAG system.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RAGConfig, DEFAULT_CONFIG, EMBEDDING_MODELS

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text embedder using SentenceTransformers for fast and reliable embeddings."""
    
    def __init__(self, config: Optional[RAGConfig] = None, model_name: Optional[str] = None):
        """
        Initialize the text embedder.
        
        Args:
            config: RAG configuration object
            model_name: Override the embedding model name from config
        """
        self.config = config or DEFAULT_CONFIG
        self.model_name = model_name or self.config.EMBEDDING_MODEL
        
        # Validate model name
        if self.model_name not in EMBEDDING_MODELS:
            available_models = list(EMBEDDING_MODELS.keys())
            logger.warning(
                f"Model '{self.model_name}' not in predefined models. "
                f"Available models: {available_models}. "
                f"Will attempt to load anyway."
            )
        
        self._model = None
        self._model_info = EMBEDDING_MODELS.get(
            self.model_name, 
            {"name": f"sentence-transformers/{self.model_name}", "dimension": None}
        )
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the embedding model."""
        if self._model is None:
            try:
                model_path = self._model_info["name"]
                self._model = SentenceTransformer(model_path)
                logger.info(f"Loaded embedding model: {model_path}")
                
                # Update dimension if not set
                if self._model_info["dimension"] is None:
                    self._model_info["dimension"] = self._model.get_sentence_embedding_dimension()
                    
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise
        
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._model_info["dimension"] is None:
            # Force model loading to get dimension
            _ = self.model
        return self._model_info["dimension"]
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dimension)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> List[np.ndarray]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            show_progress_bar: Whether to show progress bar
            
        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.BATCH_SIZE
        
        try:
            # Filter out empty texts and keep track of original indices
            non_empty_texts = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if text.strip():
                    non_empty_texts.append(text)
                    text_indices.append(i)
            
            if not non_empty_texts:
                # All texts are empty, return zero vectors
                return [np.zeros(self.embedding_dimension) for _ in texts]
            
            # Process texts in batches
            embeddings = []
            for i in range(0, len(non_empty_texts), batch_size):
                batch = non_empty_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress_bar and i == 0  # Only show for first batch
                )
                
                # Handle single text case (returns 1D array)
                if len(batch) == 1 and batch_embeddings.ndim == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)
                
                embeddings.extend(batch_embeddings)
            
            # Reconstruct full embedding list with zero vectors for empty texts
            full_embeddings = []
            embedding_idx = 0
            
            for i in range(len(texts)):
                if i in text_indices:
                    full_embeddings.append(embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    full_embeddings.append(np.zeros(self.embedding_dimension))
            
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
        
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_path": self._model_info["name"],
            "embedding_dimension": self.embedding_dimension,
            "description": self._model_info.get("description", "No description available"),
            "loaded": self._model is not None
        } 