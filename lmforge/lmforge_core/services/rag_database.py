"""
RAG Database Operations
Handles storage and retrieval of chunks with embeddings
Supports both pgVector and ChromaDB backends
"""
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.db import connection
from .clients.chromadb_client import ChromaClient

logger = logging.getLogger(__name__)


class RAGDatabase:
    """Database operations for RAG system"""
    
    def __init__(self):
        # Database configuration from Django settings
        db_config = settings.DATABASES['default']
        self.db_config = {
            'host': db_config.get('HOST', 'localhost'),
            'port': int(db_config.get('PORT', 5432)),
            'database': db_config.get('NAME', 'pdf_rag_db'),
            'user': db_config.get('USER', 'pdf_rag_user'),
            'password': db_config.get('PASSWORD', 'pdf_rag_password')
        }
    
    def get_db_connection(self):
        """Get database connection - Using Django's connection"""
        return connection
    
    async def store_chunks_with_embeddings(
        self, 
        filename: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Store chunks with embeddings in database
        
        Args:
            filename: Name of the document
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            metadata: Additional metadata
            
        Returns:
            Number of chunks stored successfully
        """
        from asgiref.sync import sync_to_async
        
        def _store_chunks_sync():
            """Synchronous database operations wrapped for async context"""
            try:
                # Check if Chroma storage is requested
                storage_backend = metadata.get('storage_backend') if metadata else getattr(settings, 'DEFAULT_STORAGE_BACKEND', 'pgvector')
                
                if storage_backend and storage_backend.lower() == 'chroma':
                    stored = self._store_to_chroma(filename, chunks, embeddings, metadata)
                    if stored > 0:
                        return stored
                    logger.warning("Chroma storage failed, falling back to pgVector")
                
                # Store to pgVector
                return self._store_to_pgvector(filename, chunks, embeddings, metadata)
                
            except Exception as e:
                logger.error(f"Error storing chunks: {e}")
                return 0
        
        # Run the sync function in async context
        return await sync_to_async(_store_chunks_sync)()
    
    def _store_to_chroma(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store chunks to ChromaDB"""
        try:
            chroma_dir = getattr(settings, 'CHROMA_PERSIST_DIR', None)
            chroma = ChromaClient(persist_directory=chroma_dir if chroma_dir else "./chroma_data")
            collection_name = getattr(settings, 'CHROMA_COLLECTION', 'lmforge_collection')
            
            ids = []
            embs = []
            metadatas = []
            docs = []

            base_doc_id = filename.replace('.pdf', '').replace(' ', '_').lower()
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                ids.append(f"{base_doc_id}_{i}")
                embs.append(embedding)
                metadatas.append({
                    'filename': filename,
                    'chunk_index': chunk.get('chunk_index', i),
                    'word_count': chunk.get('word_count', 0)
                })
                docs.append(chunk.get('text', '')[:10000])

            ok = chroma.upsert(collection_name, ids=ids, embeddings=embs, metadatas=metadatas, documents=docs)
            if ok:
                logger.info(f"Stored {len(ids)} chunks to Chroma collection '{collection_name}' for {filename}")
                return len(ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Chroma storage error: {e}")
            return 0
    
    def _store_to_pgvector(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store chunks to pgVector database"""
        try:
            # Create document entry
            document_id = filename.replace('.pdf', '').replace(' ', '_').lower()
            session_id = metadata.get('session_id', str(uuid.uuid4())) if metadata else str(uuid.uuid4())
            
            with connection.cursor() as cursor:
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
                    metadata.get('file_size', 0) if metadata else 0,
                    session_id,
                    True,
                    len(chunks),
                    json.dumps(metadata or {})
                ])
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Failed to create document entry for {filename}")
                    return 0
                
                document_pk = result[0]
                logger.info(f"Document '{filename}' stored with PK: {document_pk}")
                
                # Store chunks with embeddings
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
                        
                        # Insert chunk
                        cursor.execute("""
                            INSERT INTO chunks (
                                document_id, chunk_index, text_content, 
                                embedding, chunk_size, word_count, metadata
                            )
                            VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                        """, [
                            document_pk,
                            chunk.get('chunk_index', i),
                            chunk['text'][:50000],  # Limit text length
                            embedding_str,
                            chunk.get('length', len(chunk['text'])),
                            chunk.get('word_count', len(chunk['text'].split())),
                            json.dumps(chunk_metadata)
                        ])
                        
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing chunk {i}: {e}")
                        failed_chunks.append(i)
                        continue
                
                if failed_chunks:
                    logger.warning(f"Failed to store {len(failed_chunks)} chunks: {failed_chunks[:10]}...")
                
                logger.info(f"Stored {stored_count} chunks for {filename} (failed: {len(failed_chunks)})")
                return stored_count
                
        except Exception as e:
            logger.error(f"Error in pgVector storage: {e}")
            return 0
    
    def search_similar_chunks_sync(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7,
        storage_backend: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar chunks to return
            threshold: Minimum similarity threshold
            storage_backend: 'chroma' or 'pgvector' (default)
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Check if Chroma storage is requested
            if storage_backend and storage_backend.lower() == 'chroma':
                results = self._search_chroma(query_embedding, top_k)
                if results:
                    return results
                logger.warning("Chroma search failed, falling back to pgVector")
            
            # Search pgVector
            return self._search_pgvector(query_embedding, top_k, threshold)
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar chunks"""
        try:
            chroma = ChromaClient(persist_directory=getattr(settings, 'CHROMA_PERSIST_DIR', None))
            collection_name = getattr(settings, 'CHROMA_COLLECTION', 'lmforge_collection')
            result = chroma.query(collection_name, embedding=query_embedding, top_k=top_k)
            
            if not result:
                return []

            ids = result.get('ids', [])
            docs = result.get('documents', [])
            metadatas = result.get('metadatas', [])
            distances = result.get('distances', [])

            similar_chunks = []
            for i, _id in enumerate(ids):
                dist = distances[i] if i < len(distances) else 0.0
                # Convert distance to similarity (approximate)
                try:
                    similarity = 1.0 - float(dist)
                except Exception:
                    similarity = 0.0

                chunk = {
                    "chunk_id": _id,
                    "document_id": metadatas[i].get('filename', 'unknown') if i < len(metadatas) else 'unknown',
                    "content": docs[i] if i < len(docs) else '',
                    "chunk_index": metadatas[i].get('chunk_index', 0) if i < len(metadatas) else 0,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "document_name": metadatas[i].get('filename', 'Unknown') if i < len(metadatas) else 'Unknown',
                    "document_metadata": {},
                    "similarity_score": float(similarity)
                }
                similar_chunks.append(chunk)

            logger.info(f"Chroma: found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            return []
    
    def _search_pgvector(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search pgVector database for similar chunks with optimized query"""
        try:
            # Use optimized query with explicit index usage
            with connection.cursor() as cursor:
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

                cursor.execute(search_query, [embedding_str, embedding_str, threshold, embedding_str, top_k])
                results = cursor.fetchall()

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
            logger.error(f"Error in pgVector search: {e}")
            return []
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        threshold: float = 0.7,
        storage_backend: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper for search_similar_chunks_sync
        """
        return self.search_similar_chunks_sync(query_embedding, top_k, threshold, storage_backend)


# Global instance
rag_database = RAGDatabase()
