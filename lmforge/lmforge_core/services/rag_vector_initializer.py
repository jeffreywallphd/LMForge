"""
RAG Vector Storage Initializer Service
Handles initialization of RAG vector storage from JSON files
"""
import os
import json
import time
import uuid
import logging
from typing import List, Dict, Any
from django.conf import settings
from django.db import connection

from .clients.ollama_client import ollama_client

logger = logging.getLogger(__name__)


class RAGVectorInitializer:
    """Initialize RAG vector storage from JSON files for LMForge"""
    
    def __init__(self, stdout=None, style=None):
        self.stdout = stdout
        self.style = style
        
        # JSON directory - LMForge media/JSON directory
        self.json_directory = os.path.join(settings.MEDIA_ROOT, 'JSON')
        
        self.log(f"JSON directory set to: {self.json_directory}")
        self.embedding_model = "all-minilm:33m"
        self.chunk_size = 800  # Target chunk size in characters
        self.session_id = str(uuid.uuid4())
        
    def log(self, message, level="INFO"):
        """Log message to both logger and stdout if available"""
        if level == "ERROR":
            logger.error(message)
            if self.stdout and self.style:
                self.stdout.write(self.style.ERROR(message))
        elif level == "WARNING":
            logger.warning(message)
            if self.stdout and self.style:
                self.stdout.write(self.style.WARNING(message))
        else:
            logger.info(message)
            if self.stdout:
                self.stdout.write(message)
    
    def check_vector_storage_exists(self) -> bool:
        """Check if vector storage is already initialized"""
        try:
            with connection.cursor() as cursor:
                # Check if pgvector extension exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """)
                result = cursor.fetchone()
                vector_exists = result[0] if result else False
                
                # Check if tables exist and have data
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'documents'
                    )
                """)
                result = cursor.fetchone()
                documents_exists = result[0] if result else False
                
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'chunks'
                    )
                """)
                result = cursor.fetchone()
                chunks_exists = result[0] if result else False
                
                if documents_exists and chunks_exists:
                    cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                    result = cursor.fetchone()
                    chunk_count = result[0] if result else 0
                    if chunk_count > 0:
                        self.log(f"Vector storage already initialized with {chunk_count} chunks")
                        return True
                
            if vector_exists and documents_exists and chunks_exists:
                self.log("Vector storage tables exist but no embeddings found")
                return False
            
            return False
            
        except Exception as e:
            self.log(f"Error checking vector storage: {e}", "ERROR")
            return False
    
    def find_json_files(self) -> List[str]:
        """Find all JSON files in the JSON directory"""
        try:
            if not os.path.exists(self.json_directory):
                self.log(f"JSON directory not found: {self.json_directory}", "ERROR")
                return []
            
            json_files = []
            for filename in os.listdir(self.json_directory):
                if filename.lower().endswith('.json'):
                    json_files.append(os.path.join(self.json_directory, filename))
                    self.log(f"Found JSON file: {filename}")
            
            return json_files
            
        except Exception as e:
            self.log(f"Error finding JSON files: {e}", "ERROR")
            return []
    
    def extract_chunks_from_json(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract text chunks from JSON data"""
        chunks = []
        
        try:
            # Handle different JSON structures
            if 'chapters' in json_data:
                # Structured document with chapters
                chunks = self._extract_from_chapters(json_data)
            elif 'content' in json_data:
                # Simple content structure
                chunks = self._extract_from_content(json_data)
            elif isinstance(json_data, list):
                # Array of content items
                chunks = self._extract_from_array(json_data)
            else:
                # Try to extract from any text fields
                chunks = self._extract_from_generic(json_data)
            
            return chunks
            
        except Exception as e:
            self.log(f"Error extracting chunks from JSON: {e}", "ERROR")
            return []
    
    def _extract_from_chapters(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract chunks from chapter-based structure"""
        chunks = []
        metadata = json_data.get('metadata', {})
        filename = metadata.get('filename', 'Unknown')
        
        chapters = json_data.get('chapters', [])
        
        for chapter_idx, chapter in enumerate(chapters):
            # Handle different chapter title fields
            chapter_title = (chapter.get('chapter_title') or 
                           chapter.get('title') or 
                           f'Chapter {chapter_idx + 1}')
            
            # Process main chapter content
            chapter_content = chapter.get('content', '')
            if chapter_content and chapter_content.strip():
                chapter_chunks = self._split_text_into_chunks(
                    chapter_content, 
                    chapter_title,
                    filename
                )
                chunks.extend(chapter_chunks)
            
            # Process sections within chapter (if they exist)
            sections = chapter.get('sections', [])
            
            for section_idx, section in enumerate(sections):
                section_title = section.get('section_title', f'Section {section_idx + 1}')
                section_content = section.get('content', '')
                
                if not section_content.strip():
                    continue
                
                # Split content into smaller chunks
                section_chunks = self._split_text_into_chunks(
                    section_content, 
                    f"{chapter_title} - {section_title}",
                    filename
                )
                chunks.extend(section_chunks)
            
            # Process subsections (if they exist)
            subsections = chapter.get('subsections', [])
            for subsec_idx, subsec in enumerate(subsections):
                subsec_title = subsec.get('title', f'Subsection {subsec_idx + 1}')
                subsec_content = subsec.get('content', '')
                
                if subsec_content and subsec_content.strip():
                    subsec_chunks = self._split_text_into_chunks(
                        subsec_content,
                        f"{chapter_title} - {subsec_title}",
                        filename
                    )
                    chunks.extend(subsec_chunks)
            
            # Process chapter summary if exists
            if chapter.get('summary'):
                summary_chunks = self._split_text_into_chunks(
                    chapter['summary'],
                    f"{chapter_title} - Summary",
                    filename
                )
                chunks.extend(summary_chunks)
        
        self.log(f"Extracted {len(chunks)} chunks from {filename}")
        return chunks
    
    def _extract_from_content(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract chunks from simple content structure"""
        chunks = []
        content = json_data.get('content', '')
        title = json_data.get('title', 'Document')
        
        if content.strip():
            chunks = self._split_text_into_chunks(content, title, title)
        
        return chunks
    
    def _extract_from_array(self, json_data: List) -> List[Dict[str, Any]]:
        """Extract chunks from array structure"""
        chunks = []
        
        for idx, item in enumerate(json_data):
            if isinstance(item, dict):
                # Handle Q&A format
                if 'question' in item and 'answer' in item:
                    # Direct Q&A format
                    qa_content = f"Q: {item['question']}\n\nA: {item['answer']}"
                    title = f"Q&A {idx + 1}"
                    item_chunks = self._split_text_into_chunks(qa_content, title, "Q&A Dataset")
                    chunks.extend(item_chunks)
                
                # Handle parts with questions (Mock_up_2 format)
                elif 'part' in item and 'questions' in item:
                    part_num = item.get('part', idx + 1)
                    questions = item.get('questions', [])
                    
                    for q_idx, qa in enumerate(questions):
                        if 'question' in qa and 'answer' in qa:
                            qa_content = f"Q: {qa['question']}\n\nA: {qa['answer']}"
                            title = f"Part {part_num} - Q&A {q_idx + 1}"
                            item_chunks = self._split_text_into_chunks(qa_content, title, f"Part {part_num}")
                            chunks.extend(item_chunks)
                
                # Handle mixed format with text and Q&A (Introduction to Text Segmentation)
                elif 'part' in item and isinstance(item['part'], list):
                    part_items = item['part']
                    for part_idx, part_item in enumerate(part_items):
                        if isinstance(part_item, dict):
                            # Text content
                            if 'text' in part_item:
                                content = part_item['text'].strip()
                                if content:
                                    title = f"Section {idx + 1}.{part_idx + 1}"
                                    item_chunks = self._split_text_into_chunks(content, title, f"Document {idx + 1}")
                                    chunks.extend(item_chunks)
                            
                            # Q&A content
                            elif 'question' in part_item and 'answer' in part_item:
                                qa_content = f"Q: {part_item['question']}\n\nA: {part_item['answer']}"
                                title = f"Q&A {idx + 1}.{part_idx + 1}"
                                item_chunks = self._split_text_into_chunks(qa_content, title, f"Q&A {idx + 1}")
                                chunks.extend(item_chunks)
                
                # Handle generic content
                else:
                    content = item.get('content', item.get('text', ''))
                    title = item.get('title', f'Item {idx + 1}')
                    
                    if content.strip():
                        item_chunks = self._split_text_into_chunks(content, title, title)
                        chunks.extend(item_chunks)
        
        return chunks
    
    def _extract_from_generic(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract chunks from generic JSON structure"""
        chunks = []
        
        # Try to find text content in common fields
        text_fields = ['text', 'content', 'description', 'body', 'data']
        
        for field in text_fields:
            if field in json_data and isinstance(json_data[field], str):
                content = json_data[field].strip()
                if content:
                    title = json_data.get('title', json_data.get('name', field.title()))
                    item_chunks = self._split_text_into_chunks(content, title, title)
                    chunks.extend(item_chunks)
        
        return chunks
    
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
                current_chunk += f"\n\n{paragraph}" if current_chunk else paragraph
        
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
        """Wait for Ollama service to be ready"""
        self.log("Waiting for Ollama service to be ready...")
        
        for attempt in range(max_retries):
            try:
                # Try to generate a test embedding
                test_embedding = ollama_client.generate_embedding("test")
                if test_embedding:
                    self.log(f"✅ Ollama service ready on attempt {attempt + 1}")
                    return True
                else:
                    self.log(f"Attempt {attempt + 1}/{max_retries}: Ollama not ready, waiting...")
                    time.sleep(retry_interval)
                    
            except Exception as e:
                self.log(f"Attempt {attempt + 1}: {e}", "WARNING")
                time.sleep(retry_interval)
        
        self.log(f"❌ Ollama service not ready after {max_retries} attempts", "ERROR")
        return False
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], 
                          use_gpu: bool = True) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks using Ollama with batch processing"""
        self.log(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Wait for Ollama to be ready
        if not self.wait_for_ollama():
            self.log("Ollama service not available, cannot generate embeddings", "ERROR")
            return []
        
        enhanced_chunks = []
        failed_count = 0
        batch_size = 100  # Process 50 chunks at a time for better throughput
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Extract texts for batch processing
            batch_texts = [chunk.get('content', '') for chunk in batch_chunks]
            
            try:
                # Generate embeddings in batch
                batch_embeddings = ollama_client.generate_embeddings(batch_texts, use_gpu=use_gpu)
                
                if batch_embeddings and len(batch_embeddings) == len(batch_chunks):
                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        if embedding:
                            enhanced_chunks.append({
                                **chunk,
                                'embedding': embedding,
                                'embedding_model': self.embedding_model,
                                'embedding_dim': len(embedding)
                            })
                        else:
                            failed_count += 1
                else:
                    # Fallback to individual processing for this batch
                    for idx, chunk in enumerate(batch_chunks):
                        try:
                            content = chunk.get('content', '')
                            embedding = ollama_client.generate_embedding(content)
                            
                            if embedding:
                                enhanced_chunks.append({
                                    **chunk,
                                    'embedding': embedding,
                                    'embedding_model': self.embedding_model,
                                    'embedding_dim': len(embedding)
                                })
                            else:
                                failed_count += 1
                        except Exception as e:
                            failed_count += 1
                            self.log(f"Error processing chunk {batch_start + idx}: {e}", "ERROR")
                
                self.log(f"Processed {batch_end}/{len(chunks)} chunks")
                    
            except Exception as e:
                failed_count += len(batch_chunks)
                self.log(f"Error processing batch {batch_start}-{batch_end}: {e}", "ERROR")
        
        self.log(f"Successfully generated {len(enhanced_chunks)} embeddings, {failed_count} failed")
        return enhanced_chunks
    
    def store_in_database(self, json_file: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store chunks with embeddings in PostgreSQL database"""
        try:
            with connection.cursor() as cursor:
                # Create document entry
                document_id = os.path.basename(json_file).replace('.json', '').replace(' ', '_').lower()
                filename = os.path.basename(json_file)
                
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
                """, [
                    document_id, 
                    filename, 
                    os.path.getsize(json_file) if os.path.exists(json_file) else 0,
                    self.session_id,
                    True,
                    len(chunks),
                    json.dumps({'source_type': 'json_import', 'chunks_count': len(chunks)})
                ])
                
                result = cursor.fetchone()
                if result:
                    document_pk = result[0]
                    self.log(f"Document '{filename}' stored with PK: {document_pk}")
                else:
                    self.log(f"Failed to create document entry for {filename}", "ERROR")
                    return False
                
                # Store chunks with embeddings using bulk insert
                stored_count = 0
                failed_chunks = []
                
                # Prepare bulk insert data
                bulk_data = []
                for i, chunk in enumerate(chunks):
                    try:
                        # Validate chunk data
                        if not chunk.get('content') or not chunk['content'].strip():
                            self.log(f"Skipping empty chunk {i}", "WARNING")
                            failed_chunks.append(i)
                            continue
                        
                        embedding = chunk.get('embedding')
                        if not embedding or len(embedding) == 0:
                            self.log(f"Skipping chunk {i} - no embedding", "WARNING")
                            failed_chunks.append(i)
                            continue
                        
                        chunk_metadata = {
                            'section_title': chunk.get('section_title', ''),
                            'source_file': chunk.get('source_file', ''),
                            'type': chunk.get('type', 'content'),
                            'embedding_model': chunk.get('embedding_model', self.embedding_model)
                        }
                        
                        # Convert embedding to pgvector format
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        bulk_data.append((
                            document_pk,
                            i,
                            chunk['content'][:50000],  # Limit text length
                            embedding_str,
                            chunk.get('length', len(chunk['content'])),
                            chunk.get('word_count', len(chunk['content'].split())),
                            json.dumps(chunk_metadata)
                        ))
                        
                    except Exception as e:
                        self.log(f"Error preparing chunk {i}: {e}", "ERROR")
                        failed_chunks.append(i)
                        continue
                
                # Bulk insert in batches of 100
                batch_size = 100
                for batch_start in range(0, len(bulk_data), batch_size):
                    batch = bulk_data[batch_start:batch_start + batch_size]
                    try:
                        cursor.executemany("""
                            INSERT INTO chunks (
                                document_id, chunk_index, text_content, 
                                embedding, chunk_size, word_count, metadata
                            )
                            VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                        """, batch)
                        stored_count += len(batch)
                    except Exception as e:
                        self.log(f"Error bulk inserting batch {batch_start}: {e}", "ERROR")
                        # Fallback to individual inserts for this batch
                        for row in batch:
                            try:
                                cursor.execute("""
                                    INSERT INTO chunks (
                                        document_id, chunk_index, text_content, 
                                        embedding, chunk_size, word_count, metadata
                                    )
                                    VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                                """, row)
                                stored_count += 1
                            except Exception as e2:
                                self.log(f"Error inserting individual chunk: {e2}", "ERROR")
                                failed_chunks.append(row[1])
                
                if failed_chunks:
                    self.log(f"Failed to store {len(failed_chunks)} chunks: {failed_chunks[:10]}...", "WARNING")
                
                self.log(f"Stored {stored_count} chunks for {filename} (failed: {len(failed_chunks)})")
            
            return True
            
        except Exception as e:
            self.log(f"Error storing in database: {e}", "ERROR")
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
        
        self.log("=" * 60)
        self.log("Starting RAG Vector Storage Initialization for LMForge")
        self.log("=" * 60)
        
        # Check if already initialized
        if not force and self.check_vector_storage_exists():
            self.log("Vector storage already initialized. Use --force to reinitialize.")
            return {
                'success': True,
                'message': 'Vector storage already initialized',
                'already_exists': True
            }
        
        # Find JSON files
        json_files = self.find_json_files()
        if not json_files:
            self.log("No JSON files found to process", "ERROR")
            return {
                'success': False,
                'message': 'No JSON files found',
                'files_processed': 0
            }
        
        # Process each JSON file
        total_chunks = 0
        files_processed = 0

        for json_file in json_files:
            self.log(f"\nProcessing: {os.path.basename(json_file)}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Extract chunks
                chunks = self.extract_chunks_from_json(json_data)
                if not chunks:
                    self.log(f"No chunks extracted from {json_file}", "WARNING")
                    continue

                # Generate embeddings
                enhanced_chunks = self.generate_embeddings(chunks, use_gpu)
                if not enhanced_chunks:
                    self.log(f"No embeddings generated for {json_file}", "WARNING")
                    continue

                # Store in database
                success = self.store_in_database(json_file, enhanced_chunks)

                if success:
                    files_processed += 1
                    total_chunks += len(enhanced_chunks)
                    self.log(f"✅ Successfully processed {os.path.basename(json_file)}")
                else:
                    self.log(f"❌ Failed to store {os.path.basename(json_file)}", "ERROR")

            except Exception as e:
                self.log(f"❌ Error processing {json_file}: {e}", "ERROR")
        
        elapsed_time = time.time() - start_time
        
        self.log("=" * 60)
        self.log("RAG Vector Storage Initialization Complete")
        self.log(f"Files processed: {files_processed}/{len(json_files)}")
        self.log(f"Total chunks stored: {total_chunks}")
        self.log(f"Time elapsed: {elapsed_time:.2f}s")
        self.log("=" * 60)
        
        return {
            'success': True,
            'files_processed': files_processed,
            'total_files': len(json_files),
            'total_chunks': total_chunks,
            'elapsed_time': elapsed_time
        }


# Global instance
rag_vector_initializer = RAGVectorInitializer()
