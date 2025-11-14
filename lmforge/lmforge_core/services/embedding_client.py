"""
Embedding Client for Ollama Integration
Handles embedding generation through Ollama API
"""
import logging
from typing import List, Dict, Any
from django.conf import settings
from .clients.ollama_client import ollama_client

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings via Ollama"""
    
    def __init__(self):
        self.ollama = ollama_client
    
    def get_available_chunkers(self) -> Dict[str, Any]:
        """Get chunker info - For API compatibility"""
        return {
            "semantic": {
                "name": "Recursive Semantic Chunking",
                "description": "Word-based recursive chunking optimized for speed",
                "target_size": "200-250 words",
                "optimal_for": "Academic PDFs, technical documents"
            }
        }
    
    async def generate_embeddings_batch(self, texts: List[str], use_gpu: bool = True) -> List[List[float]]:
        """
        Generate embeddings for batch of texts
        
        Args:
            texts: List of text strings to embed
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            List of embedding vectors
            
        Raises:
            ConnectionError: If Ollama service is not available
            RuntimeError: If embedding generation fails
        """
        try:
            # Check Ollama service connection first
            if not ollama_client._check_service():
                logger.error("Ollama service is not available")
                raise ConnectionError("Ollama service is offline. Please ensure Ollama is running and accessible.")
            
            logger.info(f"Generating {len(texts)} embeddings using {'GPU' if use_gpu else 'CPU'} Ollama")
            
            embeddings = ollama_client.generate_embeddings(texts, use_gpu=use_gpu)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                raise RuntimeError("Failed to generate embeddings from Ollama service")
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except ConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], use_gpu: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks - simplified version for analysis mode
        
        Args:
            chunks: List of chunk dictionaries
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            List of chunks with embedding metadata (embeddings disabled in analysis mode)
        """
        if not chunks:
            return []
        
        logger.info(f"Processing {len(chunks)} chunks for embedding analysis")
        
        # Return chunks without embeddings but with analysis metadata
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()
            enhanced_chunk["embedding"] = None
            enhanced_chunk["embedding_model"] = "DISABLED_FOR_ANALYSIS"
            enhanced_chunk["embedding_dim"] = 0
            enhanced_chunk["service_used"] = "chunking_analysis_mode"
            enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Analysis complete for {len(enhanced_chunks)} chunks")
        return enhanced_chunks


# Global instance
embedding_client = EmbeddingClient()
