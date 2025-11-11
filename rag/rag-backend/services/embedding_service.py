"""
Simplified Embedding Service for RAG Pipeline
Contains only essential functions used by frontend API:
- PDF text extraction
- Recursive semantic chunking  
- Embedding generation
- Database storage
- API export functions
"""
import logging
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
import json
import time
import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from clients.ollama_client import ollama_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Simplified Embedding Service - Core functions for frontend API"""
    
    def __init__(self):
        self.ollama = ollama_client
        
        # Database connection parameters
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'pdf_rag_db'),
            'user': os.getenv('DB_USER', 'pdf_rag_user'),
            'password': os.getenv('DB_PASSWORD', 'pdf_rag_password')
        }
    
    def get_db_connection(self):
        """Get database connection - Used by main.py health check"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def get_available_chunkers(self) -> Dict[str, Any]:
        """Get chunker info - Required by main.py API endpoints"""
        return {
            "semantic": {
                "name": "Recursive Semantic Chunking",
                "description": "Word-based recursive chunking optimized for speed",
                "target_size": "200-250 words",
                "optimal_for": "Academic PDFs, technical documents"
            }
        }
    
    def extract_text_from_pdfs(self, pdf_files) -> str:
        """Extract text from multiple PDF files"""
        text = ""
        total_files = len(pdf_files)
        logger.info(f"Extracting text from {total_files} PDF files")
        
        for i, pdf in enumerate(pdf_files):
            try:
                # Handle both UploadFile objects and raw file objects
                if hasattr(pdf, 'file'):
                    pdf_reader = PdfReader(pdf.file)
                else:
                    pdf_reader = PdfReader(pdf)
                
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF {i+1} has {num_pages} pages")
                
                page_count = 0
                extracted_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if len(page_text.strip()) > 50:
                            extracted_text += page_text.strip() + "\n\n"
                            page_count += 1
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                
                if extracted_text.strip():
                    text += extracted_text
                    logger.info(f"PDF {i+1}: {page_count} pages processed")
                
            except Exception as e:
                logger.error(f"Error reading PDF {i+1}: {e}")
                continue
        
        logger.info(f"Text extraction complete: {len(text)} characters")
        return text if text.strip() else "No text could be extracted from PDFs"
    
    def recursive_semantic_chunking(self, text: str, filename: str = "") -> List[Dict[str, Any]]:
        """
        Fast word-based recursive semantic chunking
        Target: 200-250 words per chunk (~1000-1250 chars)
        """
        if not text or not text.strip():
            logger.warning("Empty text for chunking")
            return []
        
        start_time = time.time()
        logger.info(f"Fast chunking for {len(text)} characters from {filename}")
        
        # Configuration
        target_words = 200
        max_words = 300
        min_words = 50
        overlap_words = 30
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        current_words = []
        
        for para in paragraphs:
            para_words = para.split()
            
            # If adding paragraph keeps us under target, add it
            if len(current_words) + len(para_words) <= target_words:
                current_words.extend(para_words)
                current_words.append('\n\n')
            # If current chunk is good size, save it and start new
            elif len(current_words) >= min_words:
                chunk_text = ' '.join(current_words).replace(' \n\n ', '\n\n').strip()
                if chunk_text:
                    chunks.append(self._create_chunk_dict(chunk_text, len(chunks), filename))
                # Start new chunk with overlap
                if len(current_words) > overlap_words:
                    current_words = current_words[-overlap_words:] + para_words
                else:
                    current_words = para_words
            else:
                current_words.extend(para_words)
        
        # Add final chunk
        if current_words:
            final_text = ' '.join(current_words).replace(' \n\n ', '\n\n').strip()
            if final_text:
                chunks.append(self._create_chunk_dict(final_text, len(chunks), filename))
        
        processing_time = time.time() - start_time
        
        if chunks:
            avg_words = sum(c['word_count'] for c in chunks) / len(chunks)
            logger.info(f"✅ Fast chunking: {len(chunks)} chunks, avg {avg_words:.1f} words, {processing_time:.2f}s")
        
        return chunks
    
    def _create_chunk_dict(self, text: str, index: int, filename: str = "") -> Dict[str, Any]:
        """Create standardized chunk dictionary"""
        word_count = len(text.split())
        return {
            'text': text,
            'content': text,  # For compatibility
            'chunk_index': index,
            'index': index,
            'length': len(text),
            'word_count': word_count,
            'type': 'recursive_semantic',
            'chunk_method': 'word_based_recursive_semantic',
            'filename': filename,
            'metadata': {
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'paragraph_count': text.count('\n\n') + 1,
                'is_optimal_size': 50 <= word_count <= 300
            }
        }
    
    async def generate_embeddings_batch(self, texts: List[str], use_gpu: bool = True) -> List[List[float]]:
        """Generate embeddings for batch of texts - Used by main.py"""
        try:
            logger.info(f"Generating {len(texts)} embeddings using {'GPU' if use_gpu else 'CPU'} Ollama")
            
            embeddings = ollama_client.generate_embeddings(texts, use_gpu=use_gpu)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return [[0.0] * 768 for _ in texts]
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[0.0] * 768 for _ in texts]
    
    async def store_chunks_with_embeddings(
        self, 
        filename: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store chunks with embeddings in pgVector database - Used by main.py"""
        conn = None
        cursor = None
        
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "postgres"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "pdf_rag_db"),
                user=os.getenv("DB_USER", "pdf_rag_user"),
                password=os.getenv("DB_PASSWORD", "pdf_rag_password")
            )
            
            cursor = conn.cursor()
            
            # Create document entry
            document_id = filename.replace('.pdf', '').replace(' ', '_').lower()
            session_id = metadata.get('session_id', str(uuid.uuid4())) if metadata else str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO documents (
                    document_id, filename, file_size, session_id, 
                    processed, total_chunks, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET metadata = EXCLUDED.metadata,
                    total_chunks = EXCLUDED.total_chunks,
                    processed = EXCLUDED.processed,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                document_id, 
                filename, 
                metadata.get('file_size', 0) if metadata else 0,
                session_id,
                True,
                len(chunks),
                json.dumps(metadata or {})
            ))
            
            document_pk = cursor.fetchone()[0]
            logger.info(f"Document '{filename}' stored with PK: {document_pk}")
            
            # Store chunks with embeddings - batch approach with proper error handling
            stored_count = 0
            failed_chunks = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    # Validate chunk data
                    if not chunk.get('text') or not chunk['text'].strip():
                        logger.warning(f"Skipping empty chunk {i}")
                        continue
                    
                    if not embedding or len(embedding) == 0:
                        logger.warning(f"Skipping chunk {i} - no embedding")
                        continue
                    
                    chunk_metadata = {
                        'chunk_method': chunk.get('chunk_method', 'word_based_recursive_semantic'),
                        'is_optimal_size': chunk.get('metadata', {}).get('is_optimal_size', True),
                        'sentence_count': chunk.get('metadata', {}).get('sentence_count', 0),
                        'paragraph_count': chunk.get('metadata', {}).get('paragraph_count', 0)
                    }
                    
                    # Convert embedding to pgvector format
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Insert chunk - if this fails, the whole transaction will be aborted
                    cursor.execute("""
                        INSERT INTO chunks (
                            document_id, chunk_index, text_content, 
                            embedding, chunk_size, word_count, metadata
                        )
                        VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                    """, (
                        document_pk,
                        chunk.get('chunk_index', i),
                        chunk['text'][:50000],  # Limit text length to avoid errors
                        embedding_str,
                        chunk.get('length', len(chunk['text'])),
                        chunk.get('word_count', len(chunk['text'].split())),
                        json.dumps(chunk_metadata)
                    ))
                    
                    stored_count += 1
                    
                    # Commit every 100 chunks to avoid long transactions
                    if stored_count % 100 == 0:
                        conn.commit()
                        logger.info(f"Committed batch: {stored_count} chunks stored")
                    
                except Exception as e:
                    logger.error(f"Error storing chunk {i}: {e}")
                    failed_chunks.append(i)
                    
                    # Roll back the current transaction and start a new one
                    conn.rollback()
                    
                    # Log the problematic chunk details for debugging
                    logger.error(f"Problematic chunk {i} details:")
                    logger.error(f"  Text length: {len(chunk.get('text', ''))}")
                    logger.error(f"  Embedding length: {len(embedding) if embedding else 0}")
                    logger.error(f"  Chunk keys: {list(chunk.keys())}")
                    
                    # Continue with next chunk (transaction is clean now)
                    continue
            
            # Final commit for remaining chunks
            conn.commit()
            
            if failed_chunks:
                logger.warning(f"Failed to store {len(failed_chunks)} chunks: {failed_chunks[:10]}...")
            
            logger.info(f"✅ Stored {stored_count} chunks for {filename} (failed: {len(failed_chunks)})")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    async def process_pdfs_complete(self, pdf_files) -> Dict[str, Any]:
        """Complete PDF processing pipeline - Used by main.py /process-pdfs endpoint"""
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
                all_chunks = [{
                    'text': raw_text[:1000] if len(raw_text) > 1000 else raw_text,
                    'content': raw_text[:1000] if len(raw_text) > 1000 else raw_text,
                    'chunk_index': 0,
                    'length': min(len(raw_text), 1000),
                    'word_count': len(raw_text.split()) if raw_text else 0,
                    'type': 'fallback',
                    'chunk_method': 'fallback'
                }]
            
            # Generate embeddings (disabled for analysis)
            logger.info("Generating embeddings...")
            enhanced_chunks = await self.generate_embeddings(all_chunks)
            
            # Store in database (placeholder)
            storage_result = {"success": True, "total_chunks_stored": 0, "documents_stored": 1}
            
            # Create previews
            chunk_previews = [chunk.get("text", "")[:200] for chunk in enhanced_chunks[:10]]
            
            result = {
                "success": True,
                "session_id": session_id,
                "total_chunks": len(enhanced_chunks),
                "chunks_stored": storage_result.get("total_chunks_stored", 0),
                "documents_stored": storage_result.get("documents_stored", 0),
                "embedding_service": "ollama",
                "original_text_length": text_length,
                "chunking_method": "recursive_semantic",
                "chunks": chunk_previews,
                "processing_stats": {
                    "pdf_count": len(pdf_files),
                    "total_chunks": len(enhanced_chunks),
                    "avg_chunk_length": sum(chunk["length"] for chunk in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0,
                    "chunking_method_used": "recursive_semantic",
                    "text_extraction_success": text_length > 0
                }
            }
            
            logger.info(f"PDF processing complete: {len(enhanced_chunks)} chunks")
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
        """Chunk PDFs only - Used by main.py /api/chunk-pdfs endpoint"""
        try:
            session_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(f"Chunking {len(pdf_files)} files")
            
            total_chunks = 0
            documents_data = []
            
            for pdf_file in pdf_files:
                file_start_time = time.time()
                
                # Extract text
                text_content = self.extract_text_from_pdfs([pdf_file])
                if not text_content.strip():
                    continue
                
                # Create chunks
                chunks = self.recursive_semantic_chunking(text_content, pdf_file.filename)
                file_chunk_count = len(chunks)
                total_chunks += file_chunk_count
                
                file_processing_time = time.time() - file_start_time
                
                document_data = {
                    'filename': pdf_file.filename,
                    'file_size': pdf_file.size if hasattr(pdf_file, 'size') else 0,
                    'chunks': chunks,
                    'chunk_count': file_chunk_count,
                    'processing_time': file_processing_time
                }
                documents_data.append(document_data)
                
                logger.info(f"{pdf_file.filename}: {file_chunk_count} chunks in {file_processing_time:.2f}s")
            
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
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], use_gpu: bool = True) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks - simplified version"""
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
        
        logger.info(f"✅ Analysis complete for {len(enhanced_chunks)} chunks")
        return enhanced_chunks
    
    async def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5, 
                                   threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar chunks to return
            threshold: Minimum similarity threshold
        Returns:
            List of similar chunks with metadata
        """
        try:
            conn = self.get_db_connection()
            if not conn:
                logger.error("No database connection available")
                return []
            
            with conn.cursor() as cur:
                # Convert embedding to string format for pgvector
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Search for similar chunks using cosine similarity
                search_query = """
                SELECT 
                    c.id,
                    c.document_id,
                    c.text_content,
                    c.chunk_index,
                    c.metadata,
                    d.filename as document_name,
                    d.metadata as document_metadata,
                    1 - (c.embedding <=> %s::vector) as similarity_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                    AND 1 - (c.embedding <=> %s::vector) >= %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """
                
                cur.execute(search_query, (embedding_str, embedding_str, threshold, embedding_str, top_k))
                results = cur.fetchall()
                
                similar_chunks = []
                for row in results:
                    chunk = {
                        "chunk_id": row[0],
                        "document_id": row[1],
                        "content": row[2],
                        "chunk_index": row[3],
                        "metadata": row[4] if row[4] else {},
                        "document_name": row[5],
                        "document_metadata": row[6] if row[6] else {},
                        "similarity_score": float(row[7])
                    }
                    similar_chunks.append(chunk)
                
                logger.info(f"Found {len(similar_chunks)} similar chunks (threshold: {threshold})")
                return similar_chunks
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
        finally:
            if conn:
                conn.close()


# Global embedding service instance
embedding_service = EmbeddingService()