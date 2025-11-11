"""
Chunking utilities for different text splitting strategies
Simplified implementation for testing
"""
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChunker:
    """Base class for all chunkers"""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """Get information about this chunker"""
        return {
            "name": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "description": self.__doc__ or "No description available"
        }


class RecursiveCharacterChunker(BaseChunker):
    """Simple recursive character-based chunker"""
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence ending
                sentence_break = text.rfind('.', start, end)
                if sentence_break > start:
                    end = sentence_break + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'index': chunk_index,
                    'length': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'sentences': len([s for s in chunk_text.split('.') if s.strip()])
                })
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start < 0:
                start = end
        
        return chunks


class DocumentSpecificChunker(BaseChunker):
    """Document-aware chunker for academic content"""
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        # Use paragraph-based chunking with target size
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed target size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                chunks.append({
                    'content': current_chunk.strip(),
                    'index': chunk_index,
                    'length': len(current_chunk.strip()),
                    'word_count': len(current_chunk.strip().split()),
                    'sentences': len([s for s in current_chunk.strip().split('.') if s.strip()])
                })
                chunk_index += 1
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'index': chunk_index,
                'length': len(current_chunk.strip()),
                'word_count': len(current_chunk.strip().split()),
                'sentences': len([s for s in current_chunk.strip().split('.') if s.strip()])
            })
        
        return chunks


class SemanticBasedChunker(BaseChunker):
    """Sentence-aware semantic chunker"""
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append({
                    'content': current_chunk.strip(),
                    'index': chunk_index,
                    'length': len(current_chunk.strip()),
                    'word_count': len(current_chunk.strip().split()),
                    'sentences': len([s for s in current_chunk.strip().split('.') if s.strip()])
                })
                chunk_index += 1
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'index': chunk_index,
                'length': len(current_chunk.strip()),
                'word_count': len(current_chunk.strip().split()),
                'sentences': len([s for s in current_chunk.strip().split('.') if s.strip()])
            })
        
        return chunks


class SemanticEmbeddingChunker(BaseChunker):
    """Advanced semantic chunker with embeddings (simplified for testing)"""
    
    def __init__(self, embedding_function=None, chunk_size: int = 200, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_function = embedding_function
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        # Fallback to semantic chunker if no embedding function
        semantic_chunker = SemanticBasedChunker(self.chunk_size, self.chunk_overlap)
        return semantic_chunker.chunk(text)