"""
PDF Processor for RAG Pipeline
Handles PDF text extraction and chunking operations
"""
import time
import uuid
import logging
from typing import List, Dict, Any
from .chunking_service import chunking_service

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDFs for RAG system"""
    
    def __init__(self):
        self.chunking_service = chunking_service
    
    def extract_text_from_pdfs(self, pdf_files) -> str:
        """
        Extract text from multiple PDF files
        
        Args:
            pdf_files: List of PDF file objects
            
        Returns:
            Combined text from all PDFs
        """
        text = ""
        total_files = len(pdf_files)
        logger.info(f"Extracting text from {total_files} PDF files")
        
        for i, pdf in enumerate(pdf_files):
            try:
                extracted_text, _, _ = chunking_service.extract_text_from_pdf(pdf)
                
                if extracted_text.strip():
                    text += extracted_text + "\n\n"
                    logger.info(f"PDF {i+1}: text extracted successfully")
                
            except Exception as e:
                logger.error(f"Error reading PDF {i+1}: {e}")
                continue
        
        logger.info(f"Text extraction complete: {len(text)} characters")
        return text if text.strip() else "No text could be extracted from PDFs"
    
    def recursive_semantic_chunking(self, text: str, filename: str = "") -> List[Dict[str, Any]]:
        """
        Fast word-based recursive semantic chunking
        
        Args:
            text: Text to chunk
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries
        """
        return chunking_service.word_based_recursive_chunking(text, filename)
    
    async def process_pdfs_complete(self, pdf_files) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_files: List of PDF file objects
            
        Returns:
            Processing result with chunks and statistics
        """
        try:
            session_id = str(uuid.uuid4())
            
            # Extract text
            logger.info("Extracting text from PDFs...")
            raw_text = self.extract_text_from_pdfs(pdf_files)
            text_length = len(raw_text.strip())
            
            # Create chunks
            logger.info("Creating chunks...")
            all_chunks = self.recursive_semantic_chunking(raw_text)
            
            if not all_chunks:
                # Fallback for empty chunks
                all_chunks = [{
                    'text': raw_text[:1000] if len(raw_text) > 1000 else raw_text,
                    'content': raw_text[:1000] if len(raw_text) > 1000 else raw_text,
                    'chunk_index': 0,
                    'length': min(len(raw_text), 1000),
                    'word_count': len(raw_text.split()) if raw_text else 0,
                    'type': 'fallback',
                    'chunk_method': 'fallback'
                }]
            
            # Create previews
            chunk_previews = [chunk.get("text", "")[:200] for chunk in all_chunks[:10]]
            
            result = {
                "success": True,
                "session_id": session_id,
                "total_chunks": len(all_chunks),
                "chunks_stored": 0,  # Placeholder
                "documents_stored": 1,
                "embedding_service": "ollama",
                "original_text_length": text_length,
                "chunking_method": "recursive_semantic",
                "chunks": chunk_previews,
                "processing_stats": {
                    "pdf_count": len(pdf_files),
                    "total_chunks": len(all_chunks),
                    "avg_chunk_length": sum(chunk["length"] for chunk in all_chunks) / len(all_chunks) if all_chunks else 0,
                    "chunking_method_used": "recursive_semantic",
                    "text_extraction_success": text_length > 0
                }
            }
            
            logger.info(f"PDF processing complete: {len(all_chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "total_chunks": 0
            }
    
    async def process_pdfs_chunk_only(self, pdf_files: List) -> Dict[str, Any]:
        """
        Chunk PDFs only (no embeddings)
        
        Args:
            pdf_files: List of PDF file objects
            
        Returns:
            Chunking result with statistics
        """
        try:
            session_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(f"Chunking {len(pdf_files)} files")
            
            total_chunks = 0
            documents_data = []
            
            for pdf_file in pdf_files:
                file_start_time = time.time()
                
                # Use chunking service
                result = chunking_service.chunk_pdf_file(pdf_file, getattr(pdf_file, 'name', 'unknown.pdf'))
                
                if result.get('success'):
                    file_chunk_count = result['chunk_count']
                    total_chunks += file_chunk_count
                    
                    file_processing_time = time.time() - file_start_time
                    
                    document_data = {
                        'filename': result['filename'],
                        'file_size': getattr(pdf_file, 'size', 0),
                        'chunks': result['chunks'],
                        'chunk_count': file_chunk_count,
                        'processing_time': file_processing_time
                    }
                    documents_data.append(document_data)
                    
                    logger.info(f"{result['filename']}: {file_chunk_count} chunks in {file_processing_time:.2f}s")
            
            processing_time = time.time() - start_time
            
            # Preview chunks
            preview_chunks = []
            for doc_data in documents_data:
                for chunk in doc_data['chunks'][:5]:
                    preview_chunks.append(chunk['content'][:200])
            
            return {
                "success": True,
                "session_id": session_id,
                "total_chunks": total_chunks,
                "documents_stored": len(documents_data),
                "processing_stats": {
                    "total_processing_time": round(processing_time, 2),
                    "avg_chunks_per_file": round(total_chunks / len(documents_data), 2) if documents_data else 0,
                    "files_processed": len(documents_data)
                },
                "chunks": preview_chunks[:15]
            }
            
        except Exception as e:
            logger.error(f"Error in chunk-only processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
pdf_processor = PDFProcessor()
