"""
Initialize RAG Vector Storage from JSON Files
This script processes JSON files from the root-level jsons/ directory,
generates embeddings using Ollama, and stores them in pgvector.
"""
import os
import sys
import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.ollama_client import ollama_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGVectorInitializer:
    """Initialize RAG vector storage from JSON files"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'pdf_rag_db'),
            'user': os.getenv('DB_USER', 'pdf_rag_user'),
            'password': os.getenv('DB_PASSWORD', 'pdf_rag_password')
        }
        # JSON directory - check if running in Docker container or local development
        if os.path.exists('/app/jsons'):
            # Running in Docker container
            self.json_directory = '/app/jsons'
        else:
            # Running locally - go up two directories from rag-backend to find jsons
            self.json_directory = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'jsons'
            )
        
        logger.info(f"JSON directory set to: {self.json_directory}")
        self.embedding_model = "all-minilm:33m"
        self.chunk_size = 800  # Target chunk size in characters
        self.session_id = str(uuid.uuid4())
        
    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def check_vector_storage_exists(self) -> bool:
        """Check if vector storage is already initialized"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                # Check if pgvector extension exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """)
                vector_exists = cur.fetchone()[0]
                
                # Check if tables exist
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'documents'
                    )
                """)
                documents_exists = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'chunks'
                    )
                """)
                chunks_exists = cur.fetchone()[0]
                
                # Check if there are already chunks with embeddings
                if chunks_exists:
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                    chunk_count = cur.fetchone()[0]
                else:
                    chunk_count = 0
                
            conn.close()
            
            if vector_exists and documents_exists and chunks_exists:
                logger.info(f"Vector storage exists with {chunk_count} chunks")
                return chunk_count > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking vector storage: {e}")
            return False
    
    def find_json_files(self) -> List[str]:
        """Find all JSON files in the jsons directory"""
        try:
            if not os.path.exists(self.json_directory):
                logger.error(f"JSON directory not found: {self.json_directory}")
                return []
            
            json_files = []
            for filename in os.listdir(self.json_directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.json_directory, filename)
                    json_files.append(filepath)
                    logger.info(f"Found JSON file: {filename}")
            
            return json_files
            
        except Exception as e:
            logger.error(f"Error finding JSON files: {e}")
            return []
    
    def extract_chunks_from_json(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract text chunks from JSON data"""
        chunks = []
        
        try:
            # Get metadata
            metadata = json_data.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            # Process chapters
            chapters = json_data.get('chapters', [])
            
            for chapter_idx, chapter in enumerate(chapters):
                chapter_title = chapter.get('title', f'Chapter {chapter_idx + 1}')
                chapter_content = chapter.get('content', '')
                
                # Create chunks from chapter content
                if chapter_content:
                    chapter_chunks = self._split_text_into_chunks(
                        chapter_content, 
                        chapter_title,
                        filename
                    )
                    chunks.extend(chapter_chunks)
                
                # Process subsections
                subsections = chapter.get('subsections', [])
                for subsection in subsections:
                    subsection_title = subsection.get('title', 'Subsection')
                    subsection_content = subsection.get('content', '')
                    
                    if subsection_content:
                        subsection_chunks = self._split_text_into_chunks(
                            subsection_content,
                            f"{chapter_title} - {subsection_title}",
                            filename
                        )
                        chunks.extend(subsection_chunks)
                    
                    # Process learning objectives
                    learning_objectives = subsection.get('learning_objectives', [])
                    if learning_objectives:
                        objectives_text = "Learning Objectives:\n" + "\n".join(
                            f"- {obj}" for obj in learning_objectives
                        )
                        chunks.append({
                            'content': objectives_text,
                            'section_title': f"{chapter_title} - {subsection_title}",
                            'source_file': filename,
                            'type': 'learning_objectives',
                            'word_count': len(objectives_text.split()),
                            'length': len(objectives_text)
                        })
                    
                    # Process key takeaways
                    key_takeaways = subsection.get('key_takeaways', [])
                    if key_takeaways:
                        takeaways_text = "Key Takeaways:\n" + "\n".join(
                            f"- {takeaway}" for takeaway in key_takeaways
                        )
                        chunks.append({
                            'content': takeaways_text,
                            'section_title': f"{chapter_title} - {subsection_title}",
                            'source_file': filename,
                            'type': 'key_takeaways',
                            'word_count': len(takeaways_text.split()),
                            'length': len(takeaways_text)
                        })
            
            logger.info(f"Extracted {len(chunks)} chunks from {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks from JSON: {e}")
            return []
    
    def _split_text_into_chunks(self, text: str, section_title: str, 
                                source_file: str) -> List[Dict[str, Any]]:
        """Split text into chunks of target size"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                chunks.append({
                    'content': current_chunk.strip(),
                    'section_title': section_title,
                    'source_file': source_file,
                    'type': 'content',
                    'word_count': len(current_chunk.strip().split()),
                    'length': len(current_chunk.strip())
                })
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
                'section_title': section_title,
                'source_file': source_file,
                'type': 'content',
                'word_count': len(current_chunk.strip().split()),
                'length': len(current_chunk.strip())
            })
        
        return chunks
    
    def wait_for_ollama(self, max_retries: int = 30, retry_interval: int = 5) -> bool:
        """Wait for Ollama service to be ready with embedding model"""
        logger.info("Waiting for Ollama service to be ready...")
        
        for attempt in range(max_retries):
            try:
                # Check service status using simplified client
                status = ollama_client.get_service_status()
                
                if status.get('service_available', False):
                    logger.info(f"✅ Ollama service ready (attempt {attempt + 1})")
                    return True
                
                logger.info(f"Ollama not ready yet (attempt {attempt + 1}/{max_retries}), waiting {retry_interval}s...")
                time.sleep(retry_interval)
                
            except Exception as e:
                logger.warning(f"Error checking Ollama (attempt {attempt + 1}): {e}")
                time.sleep(retry_interval)
        
        logger.error(f"❌ Ollama service not ready after {max_retries} attempts")
        return False
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], 
                          use_gpu: bool = True) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks using Ollama"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Wait for Ollama to be ready
        if not self.wait_for_ollama():
            logger.error("Ollama service not available, cannot generate embeddings")
            return []
        
        enhanced_chunks = []
        failed_count = 0
        
        for idx, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = ollama_client.generate_embedding(
                    text=chunk['content']
                )
                
                if embedding:
                    chunk['embedding'] = embedding
                    chunk['embedding_dim'] = len(embedding)
                    enhanced_chunks.append(chunk)
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Generated embeddings for {idx + 1}/{len(chunks)} chunks")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate embedding for chunk {idx + 1}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating embedding for chunk {idx + 1}: {e}")
        
        logger.info(f"Successfully generated {len(enhanced_chunks)} embeddings, {failed_count} failed")
        return enhanced_chunks
    
    def store_in_database(self, json_file: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store chunks with embeddings in PostgreSQL database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                # Create document record
                filename = os.path.basename(json_file)
                file_size = os.path.getsize(json_file)
                
                # Check if document already exists
                cur.execute("""
                    SELECT id FROM documents 
                    WHERE filename = %s AND session_id = %s
                """, (filename, self.session_id))
                
                existing = cur.fetchone()
                if existing:
                    logger.info(f"Document {filename} already exists, skipping...")
                    conn.close()
                    return True
                
                # Insert document
                doc_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO documents (
                        document_id, filename, file_size, session_id,
                        processed, total_chunks, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                """, (
                    doc_id,
                    filename,
                    file_size,
                    self.session_id,
                    True,
                    len(chunks),
                    json.dumps({
                        'source': 'json_initialization',
                        'embedding_model': self.embedding_model,
                        'chunk_count': len(chunks)
                    })
                ))
                
                document_pk = cur.fetchone()[0]
                
                # Insert chunks with embeddings
                chunks_stored = 0
                for idx, chunk in enumerate(chunks):
                    if 'embedding' not in chunk:
                        continue
                    
                    embedding_list = chunk['embedding']
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
                    
                    cur.execute("""
                        INSERT INTO chunks (
                            document_id, chunk_index, text_content,
                            embedding, word_count, chunk_size, metadata
                        ) VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                    """, (
                        document_pk,
                        idx,
                        chunk['content'],
                        embedding_str,
                        chunk.get('word_count', 0),
                        chunk.get('length', 0),
                        json.dumps({
                            'section_title': chunk.get('section_title', ''),
                            'source_file': chunk.get('source_file', ''),
                            'type': chunk.get('type', 'content'),
                            'embedding_dim': chunk.get('embedding_dim', 384)
                        })
                    ))
                    chunks_stored += 1
                
                # Update document with actual stored chunk count
                cur.execute("""
                    UPDATE documents 
                    SET total_chunks = %s 
                    WHERE id = %s
                """, (chunks_stored, document_pk))
                
                conn.commit()
                logger.info(f"Stored {chunks_stored} chunks for document: {filename}")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing in database: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def initialize(self, force: bool = False, use_gpu: bool = True) -> Dict[str, Any]:
        """
        Initialize RAG vector storage from JSON files
        
        Args:
            force: Force reinitialization even if storage exists
            use_gpu: Use GPU for embedding generation
        
        Returns:
            Dictionary with initialization results
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting RAG Vector Storage Initialization")
        logger.info("=" * 60)
        
        # Check if already initialized
        if not force and self.check_vector_storage_exists():
            logger.info("Vector storage already initialized. Use force=True to reinitialize.")
            return {
                'success': True,
                'message': 'Vector storage already initialized',
                'already_exists': True
            }
        
        # Find JSON files
        json_files = self.find_json_files()
        if not json_files:
            logger.error("No JSON files found to process")
            return {
                'success': False,
                'message': 'No JSON files found',
                'files_processed': 0
            }
        
        # Process each JSON file
        total_chunks = 0
        files_processed = 0
        
        for json_file in json_files:
            logger.info(f"\nProcessing: {os.path.basename(json_file)}")
            
            try:
                # Load JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Extract chunks
                chunks = self.extract_chunks_from_json(json_data)
                if not chunks:
                    logger.warning(f"No chunks extracted from {json_file}")
                    continue
                
                # Generate embeddings
                enhanced_chunks = self.generate_embeddings(chunks, use_gpu=use_gpu)
                if not enhanced_chunks:
                    logger.warning(f"No embeddings generated for {json_file}")
                    continue
                
                # Store in database
                success = self.store_in_database(json_file, enhanced_chunks)
                if success:
                    total_chunks += len(enhanced_chunks)
                    files_processed += 1
                    logger.info(f"✅ Successfully processed {os.path.basename(json_file)}")
                else:
                    logger.error(f"❌ Failed to store data for {json_file}")
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("RAG Vector Storage Initialization Complete")
        logger.info(f"Files processed: {files_processed}/{len(json_files)}")
        logger.info(f"Total chunks stored: {total_chunks}")
        logger.info(f"Time elapsed: {elapsed_time:.2f}s")
        logger.info("=" * 60)
        
        return {
            'success': True,
            'files_processed': files_processed,
            'total_files': len(json_files),
            'total_chunks': total_chunks,
            'elapsed_time': elapsed_time,
            'session_id': self.session_id
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Initialize RAG vector storage from JSON files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinitialization even if storage exists'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU for embeddings'
    )
    
    args = parser.parse_args()
    
    # Create initializer
    initializer = RAGVectorInitializer()
    
    # Run initialization
    result = initializer.initialize(
        force=args.force,
        use_gpu=not args.cpu
    )
    
    # Print results
    if result['success']:
        if result.get('already_exists'):
            print("\n✅ Vector storage already initialized")
            print("   Use --force to reinitialize")
        else:
            print(f"\n✅ Initialization successful!")
            print(f"   Files processed: {result['files_processed']}/{result['total_files']}")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Time elapsed: {result['elapsed_time']:.2f}s")
            print(f"   Session ID: {result['session_id']}")
    else:
        print(f"\n❌ Initialization failed: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
