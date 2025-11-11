"""
PDF RAG Backend with Ollama Services
FastAPI backend that processes PDFs and creates embeddings using CPU/GPU Ollama containers
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os

# Import services
from clients.ollama_client import ollama_client
from services.embedding_service import embedding_service
# Removed Redis dependency - using PostgreSQL exclusively
from services.chat_service import chat_service
from services.chunking_service import chunking_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Backend with Ollama",
    description="Backend for processing PDFs and creating embeddings using CPU/GPU Ollama services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"  
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize embedding service (already initialized in services.embedding_service)
# embedding_service is imported from services.embedding_service

# Request/Response models
class EmbeddingRequest(BaseModel):
    text: str
    use_gpu: Optional[bool] = True

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    service_used: str
    embedding_dim: int

class ChunkRequest(BaseModel):
    text: str
    method: Optional[str] = "semantic"  # "semantic", "recursive", "document_specific", "semantic_embedding"

class ChunkResponse(BaseModel):
    success: bool
    chunks: List[Dict[str, Any]]
    total_chunks: int
    method_used: str
    error: Optional[str] = None

class EmbedRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    use_gpu: Optional[bool] = True

class EmbedResponse(BaseModel):
    success: bool
    enhanced_chunks: List[Dict[str, Any]]
    total_embeddings: int
    service_used: str
    error: Optional[str] = None

class PDFProcessingRequest(BaseModel):
    use_semantic_chunking: Optional[bool] = True
    perform_cluster_chunking: Optional[bool] = True

class HealthResponse(BaseModel):
    status: str
    ollama_services: Dict[str, Any]
    database: bool = False
    services: Dict[str, bool] = {}

class PDFProcessingResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    total_chunks: int = 0
    chunks_stored: int = 0
    documents_stored: int = 0
    embedding_service: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None
    document_summaries: Optional[List[Dict[str, Any]]] = []  # Per-document details
    chunks: List[str] = []  # Preview chunks for frontend
    error: Optional[str] = None

# Chat-related models
class ChatSessionRequest(BaseModel):
    user_id: Optional[str] = None
    session_name: Optional[str] = None

class ChatSessionResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    session_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None
    use_rag: Optional[bool] = True
    max_context_chunks: Optional[int] = 5

class ChatMessageResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    context_chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    success: bool
    messages: List[Dict[str, Any]] = []
    total_messages: int = 0
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF RAG Backend API with Ollama", 
        "status": "running", 
        "version": "1.0.0",
        "endpoints": [
            "/health", 
            "/ollama/embed",
            "/process-pdfs",
            "/upload",
            "/upload-pdfs", 
            "/api/upload",
            "/api/process-pdfs",
            "/api/chunk",
            "/api/embed",
            "/api/chunkers",
            "/database/status"
        ],
        "new_endpoints": [
            "POST /api/chunk - Chunk text using semantic/recursive/document_specific/semantic_embedding methods",
            "GET /api/chunkers - Get information about available chunking methods",
            "POST /api/embed - Generate embeddings for chunks",
            "POST /process-pdfs - Enhanced PDF processing with modular chunking",
            "POST /api/chat/session - Create new chat session",
            "GET /api/chat/session/{session_id} - Get chat session info",
            "POST /api/chat/message - Send message and get RAG response",
            "GET /api/chat/history/{session_id} - Get chat history",
            "DELETE /api/chat/session/{session_id} - Delete chat session",
            "GET /api/chat/models - Get available chat models"
        ]
    }

@app.post("/api/chunk", response_model=ChunkResponse)
async def chunk_text_endpoint(request: ChunkRequest):
    """
    Chunk text using different methods
    
    Methods:
    - semantic: Sentence-aware semantic chunking (default)
    - recursive: Recursive character text splitting
    - document_specific: Document-aware chunking (academic, technical, legal, article)
    - semantic_embedding: Advanced semantic chunking with embeddings and clustering
    """
    try:
        chunks = await chunk_data(request.text, request.method)
        
        return ChunkResponse(
            success=True,
            chunks=chunks,
            total_chunks=len(chunks),
            method_used=request.method
        )
    except Exception as e:
        logger.error(f"Error in chunk endpoint: {e}")
        return ChunkResponse(
            success=False,
            chunks=[],
            total_chunks=0,
            method_used=request.method,
            error=str(e)
        )

@app.get("/api/chunkers")
async def get_chunkers_info():
    """
    Get information about all available chunking methods
    """
    try:
        chunkers_info = embedding_service.get_available_chunkers()
        return {
            "success": True,
            "chunkers": chunkers_info,
            "default_method": "semantic",
            "message": "Available chunking methods retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting chunkers info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunkers info: {str(e)}")

@app.post("/api/embed", response_model=EmbedResponse)
async def embed_chunks_endpoint(request: EmbedRequest):
    """
    Generate embeddings for chunks using Ollama services
    """
    try:
        enhanced_chunks = await embed_data(request.chunks, request.use_gpu)
        
        service_used = "unknown"
        if enhanced_chunks:
            service_used = enhanced_chunks[0].get("service_used", "unknown")
        
        return EmbedResponse(
            success=True,
            enhanced_chunks=enhanced_chunks,
            total_embeddings=len(enhanced_chunks),
            service_used=service_used
        )
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        return EmbedResponse(
            success=False,
            enhanced_chunks=[],
            total_embeddings=0,
            service_used="error",
            error=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Get Ollama service status
        ollama_status = ollama_client.get_service_status()
        
        # Test database connection
        database_connected = False
        try:
            conn = embedding_service.get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                database_connected = True
                conn.close()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            database_connected = False
        
        # Determine overall health
        ollama_available = ollama_status["service_available"]
        
        status = "healthy" if ollama_available and database_connected else "degraded" if ollama_available else "unhealthy"
        
        return HealthResponse(
            status=status,
            ollama_services=ollama_status,
            database=database_connected,
            services={
                "ollama": ollama_available,
                "database": database_connected
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            ollama_services={
                "cpu_ollama": {"available": False, "url": "http://localhost:11434"},
                "gpu_ollama": {"available": False, "url": "http://localhost:11435"},
                "preferred_service": "none",
                "error": str(e)
            },
            database=False,
            services={
                "cpu_ollama": False,
                "gpu_ollama": False,
                "database": False
            }
        )

@app.post("/ollama/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Create embedding using Ollama services"""
    try:
        logger.info(f"Creating embedding for text: {request.text[:50]}...")
        
        # Generate embedding
        embedding = ollama_client.generate_embedding(
            text=request.text,
            use_gpu=request.use_gpu
        )
        
        if not embedding:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate embedding"
            )
        
        # Determine which service was used
        status = ollama_client.get_service_status()
        if request.use_gpu and status["gpu_ollama"]["available"]:
            service_used = "gpu"
        elif status["cpu_ollama"]["available"]:
            service_used = "cpu"
        else:
            service_used = "unknown"
        
        return EmbeddingResponse(
            embedding=embedding,
            service_used=service_used,
            embedding_dim=len(embedding)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")


@app.post("/process-pdfs", response_model=PDFProcessingResponse)
async def process_pdfs(
    files: List[UploadFile] = File(...),
    use_gpu: bool = Form(True)
):
    """
    Simplified PDF processing pipeline with robust document-based chunking:
    1. Extract text from PDFs with multiple fallback strategies
    2. Apply document-based chunking optimized for academic content
    3. Generate embeddings using Ollama
    4. Store in PostgreSQL database with pgvector
    """
    try:
        logger.info(f"Processing {len(files)} PDF files using {'GPU' if use_gpu else 'CPU'} Ollama")
        
        # Validate files
        pdf_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            pdf_files.append(file)  # Pass the UploadFile object, not just file.file
        
        # Process PDFs using enhanced embedding service with document-based chunking
        result = await embedding_service.process_pdfs_complete(pdf_files)
        
        if result["success"]:
            return PDFProcessingResponse(
                success=True,
                session_id=result.get("session_id"),
                total_chunks=result.get("total_chunks", 0),
                chunks_stored=result.get("chunks_stored", 0),
                documents_stored=result.get("documents_stored", 0),
                embedding_service=result.get("embedding_service"),
                processing_stats=result.get("processing_stats"),
                chunks=result.get("chunks", [])  # Include chunk previews
            )
        else:
            return PDFProcessingResponse(
                success=False,
                error=result.get("error", "Unknown error occurred")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        return PDFProcessingResponse(
            success=False,
            error=str(e)
        )

@app.get("/database/status")
async def database_status():
    """Get database connection status and basic info"""
    try:
        # Try to connect to database
        conn = embedding_service.get_db_connection()
        if not conn:
            return {
                "connected": False,
                "error": "Cannot connect to database"
            }
        
        # Test basic queries
        with conn.cursor() as cur:
            # Check if tables exist
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            # Count records in each table
            counts = {}
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            
            # Check pgvector extension
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            vector_ext = cur.fetchone()
            
        conn.close()
        
        return {
            "connected": True,
            "database": "pdf_rag_db",
            "tables": tables,
            "record_counts": counts,
            "pgvector_extension": {
                "installed": vector_ext is not None,
                "version": vector_ext[1] if vector_ext else None
            }
        }
        
    except Exception as e:
        logger.error(f"Database status check error: {e}")
        return {
            "connected": False,
            "error": str(e)
        }

# Additional endpoints that frontend might be calling
@app.post("/upload")
async def upload_pdfs_alternative(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """Alternative upload endpoint - redirects to process-pdfs"""
    return await process_pdfs(files, use_gpu=use_gpu)

@app.post("/upload-pdfs") 
async def upload_pdfs_alternative2(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """Alternative upload endpoint - redirects to process-pdfs"""
    return await process_pdfs(files, use_gpu=use_gpu)

@app.post("/api/upload")
async def api_upload_pdfs(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """API upload endpoint - redirects to process-pdfs"""
    return await process_pdfs(files, use_gpu=use_gpu)

@app.post("/api/upload-pdfs")
async def api_upload_pdfs_endpoint(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """API upload PDFs endpoint - simplified version"""
    return await process_pdfs(files, use_gpu=use_gpu)

@app.post("/api/process-pdfs")
async def api_process_pdfs(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """API process PDFs endpoint - redirects to process-pdfs"""
    return await process_pdfs(files, use_gpu=use_gpu)

@app.post("/api/chunk-pdfs")
async def api_chunk_pdfs(files: List[UploadFile] = File(...)):
    """
    Chunk PDFs only - Step 1 of two-step process
    Extract text and create chunks but don't generate embeddings
    """
    try:
        logger.info(f"Chunking {len(files)} PDF files (Step 1 of 2-step process)")
        
        # Validate files
        pdf_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            pdf_files.append(file)
        
        # Process PDFs - chunk only, no embeddings
        result = await embedding_service.process_pdfs_chunk_only(pdf_files)
        
        if result["success"]:
            return PDFProcessingResponse(
                success=True,
                session_id=result.get("session_id"),
                total_chunks=result.get("total_chunks", 0),
                chunks_stored=0,  # No chunks stored to DB yet
                documents_stored=result.get("documents_stored", 0),
                embedding_service=None,  # No embeddings generated yet
                processing_stats=result.get("processing_stats"),
                document_summaries=result.get("document_summaries", []),  # Include per-document details
                chunks=result.get("chunks", [])  # Include chunk previews
            )
        else:
            return PDFProcessingResponse(
                success=False,
                error=result.get("error", "Unknown error occurred")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error chunking PDFs: {e}")
        return PDFProcessingResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/chunk-pdfs-only")
async def api_chunk_pdfs_only(files: List[UploadFile] = File(...)):
    """
    NEW WORKFLOW: Chunk PDFs only using dedicated chunking service
    Returns detailed chunk information for user review before embedding
    User can review chunks and decide to proceed with embedding
    """
    try:
        logger.info(f"Chunking {len(files)} PDF(s) using dedicated chunking service")
        
        # Validate files
        pdf_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            pdf_files.append(file)
        
        # Process each PDF through chunking service
        all_results = []
        session_id = str(__import__('uuid').uuid4())
        
        for pdf_file in pdf_files:
            # Read file content first
            file_content = await pdf_file.read()
            result = chunking_service.chunk_pdf_file(file_content, pdf_file.filename)
            
            if result['success']:
                # Store chunks in PostgreSQL pending_chunks table
                import psycopg2
                from psycopg2.extras import Json
                import json
                
                db_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'dbname': os.getenv('DB_NAME', 'pdf_rag_db'),
                    'user': os.getenv('DB_USER', 'pdf_rag_user'),
                    'password': os.getenv('DB_PASSWORD', 'pdf_rag_password')
                }
                
                conn = psycopg2.connect(**db_params)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO pending_chunks (session_id, filename, chunk_data)
                    VALUES (%s, %s, %s)
                """, (session_id, pdf_file.filename, json.dumps(result)))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                all_results.append(result)
                logger.info(f"✅ {pdf_file.filename}: {result['chunk_count']} chunks created")
            else:
                all_results.append(result)
                logger.error(f"❌ {pdf_file.filename}: {result.get('error')}")
        
        # Aggregate statistics
        successful_results = [r for r in all_results if r['success']]
        total_chunks = sum(r['chunk_count'] for r in successful_results)
        total_pages = sum(r['page_count'] for r in successful_results)
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            "success": True,
            "session_id": session_id,
            "total_files": len(pdf_files),
            "successful_files": len(successful_results),
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "results": all_results,
            "aggregate_stats": {
                "avg_processing_time": round(avg_processing_time, 2),
                "avg_chunks_per_file": round(total_chunks / len(successful_results), 1) if successful_results else 0,
                "avg_pages_per_file": round(total_pages / len(successful_results), 1) if successful_results else 0
            },
            "ready_for_embedding": len(successful_results) > 0,
            "workflow_step": "chunking_complete"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chunking service: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/embed-and-store")
async def api_embed_and_store(
    session_id: str = Form(...),
    use_gpu: bool = Form(True),
    selected_files: Optional[str] = Form(None)  # JSON array of filenames
):
    """
    NEW WORKFLOW: Generate embeddings and store in pgVector
    Takes chunks from chunking session and creates vector embeddings
    Stores in database just like the init_rag_storage.py script
    Returns detailed progress information
    """
    try:
        import json
        import time
        start_time = time.time()
        logger.info(f"Embedding and storing chunks for session {session_id}")
        
        # Get selected files or all files
        file_list = json.loads(selected_files) if selected_files else None
        
        # Retrieve chunking results from PostgreSQL
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'dbname': os.getenv('DB_NAME', 'pdf_rag_db'),
            'user': os.getenv('DB_USER', 'pdf_rag_user'),
            'password': os.getenv('DB_PASSWORD', 'pdf_rag_password')
        }
        
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT filename, chunk_data
            FROM pending_chunks
            WHERE session_id = %s AND expires_at > CURRENT_TIMESTAMP
        """, (session_id,))
        
        pending_chunks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not pending_chunks:
            raise HTTPException(status_code=404, detail="Chunking session not found or expired")
        
        # Process each file's chunks
        total_embeddings = 0
        total_chunks = 0
        processed_files = []
        
        for row in pending_chunks:
            filename = row['filename']
            chunk_data = row['chunk_data']
            
            # Skip if not in selected files
            if file_list and filename not in file_list:
                continue
            
            if not chunk_data or not chunk_data.get('success'):
                continue
            
            chunks = chunk_data['chunks']
            logger.info(f"Processing {len(chunks)} chunks from {filename}")
            
            # Generate embeddings using Ollama
            chunk_texts = [c['text'] for c in chunks]
            embeddings = await embedding_service.generate_embeddings_batch(chunk_texts, use_gpu)
            
            if not embeddings:
                logger.error(f"Failed to generate embeddings for {filename}")
                continue
            
            # Store in pgVector database
            stored_count = await embedding_service.store_chunks_with_embeddings(
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
                metadata=chunk_data.get('metadata', {})
            )
            
            total_embeddings += stored_count
            total_chunks += len(chunks)
            processed_files.append({
                'filename': filename,
                'chunks': len(chunks),
                'embeddings_stored': stored_count
            })
            
            logger.info(f"✅ {filename}: {stored_count} embeddings stored in pgVector")
        
        processing_time = time.time() - start_time
        embeddings_per_second = total_embeddings / processing_time if processing_time > 0 else 0
        
        return {
            "success": True,
            "session_id": session_id,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "processed_files": processed_files,
            "processing_time": round(processing_time, 2),
            "embeddings_per_second": round(embeddings_per_second, 2),
            "service_used": "GPU Ollama" if use_gpu else "CPU Ollama",
            "workflow_step": "embedding_complete",
            "message": f"Successfully generated {total_embeddings} embeddings in {processing_time:.2f}s"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embedding and storage: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/chunking-session/{session_id}")
async def api_get_chunking_session(session_id: str):
    """
    Get detailed chunking information for review from PostgreSQL
    Returns all chunks and statistics for user to review before embedding
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'dbname': os.getenv('DB_NAME', 'pdf_rag_db'),
            'user': os.getenv('DB_USER', 'pdf_rag_user'),
            'password': os.getenv('DB_PASSWORD', 'pdf_rag_password')
        }
        
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT filename, chunk_data
            FROM pending_chunks
            WHERE session_id = %s AND expires_at > CURRENT_TIMESTAMP
        """, (session_id,))
        
        pending_chunks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not pending_chunks:
            raise HTTPException(status_code=404, detail="Chunking session not found or expired")
        
        files_data = []
        
        for row in pending_chunks:
            filename = row['filename']
            chunk_data = row['chunk_data']
            
            if chunk_data and chunk_data.get('success'):
                # Get preview of chunks
                chunk_previews = chunking_service.get_chunk_preview(chunk_data['chunks'], count=10)
                
                files_data.append({
                    'filename': filename,
                    'chunk_count': chunk_data['chunk_count'],
                    'page_count': chunk_data['page_count'],
                    'statistics': chunk_data['chunk_statistics'],
                    'quality': chunk_data['quality_assessment'],
                    'processing_time': chunk_data['processing_time'],
                    'chunk_previews': chunk_previews
                })
        
        return {
            "success": True,
            "session_id": session_id,
            "files": files_data,
            "total_files": len(files_data),
            "total_chunks": sum(f['chunk_count'] for f in files_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunking session: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/health")
async def api_health():
    """API health endpoint - redirects to health"""
    return await health_check()

@app.get("/health_concurrent")
async def health_concurrent():
    """Health check endpoint that matches Django's naming"""
    return await health_check()

# Chat endpoints
@app.post("/api/chat/session", response_model=ChatSessionResponse)
async def create_chat_session(request: ChatSessionRequest):
    """Create a new chat session"""
    try:
        session_id = await chat_service.create_chat_session(
            user_id=request.user_id,
            session_name=request.session_name
        )
        
        session_data = await chat_service.get_chat_session(session_id)
        
        return ChatSessionResponse(
            success=True,
            session_id=session_id,
            session_data=session_data
        )
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        return ChatSessionResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/chat/session/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str):
    """Get chat session information"""
    try:
        session_data = await chat_service.get_chat_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return ChatSessionResponse(
            success=True,
            session_id=session_id,
            session_data=session_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        return ChatSessionResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Send a message and get RAG-enhanced response"""
    try:
        # Validate session exists
        session_data = await chat_service.get_chat_session(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Generate RAG response
        response = await chat_service.generate_rag_response(
            user_message=request.message,
            session_id=request.session_id
        )
        
        if response["success"]:
            return ChatMessageResponse(
                success=True,
                response=response["content"],
                context_chunks=response.get("context_chunks", []),
                metadata={
                    "model_used": response.get("model_used"),
                    "tokens_used": response.get("tokens_used"),
                    "generation_time": response.get("generation_time"),
                    "context_chunks_used": len(response.get("context_chunks", []))
                }
            )
        else:
            return ChatMessageResponse(
                success=False,
                error=response.get("error", "Failed to generate response")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return ChatMessageResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, limit: int = 20):
    """Get chat history for a session"""
    try:
        # Validate session exists
        session_data = await chat_service.get_chat_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        messages = await chat_service.get_chat_history(session_id, limit)
        
        return ChatHistoryResponse(
            success=True,
            messages=messages,
            total_messages=len(messages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return ChatHistoryResponse(
            success=False,
            error=str(e)
        )

@app.delete("/api/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and its messages"""
    try:
        success = await chat_service.delete_chat_session(session_id)
        
        if success:
            return {"success": True, "message": "Chat session deleted successfully"}
        else:
            return {"success": False, "message": "Failed to delete chat session"}
        
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/chat/models")
async def get_chat_models():
    """Get available chat models"""
    try:
        models = await chat_service.get_available_models()
        
        return {
            "success": True,
            "models": models,
            "default_model": chat_service.default_model
        }
        
    except Exception as e:
        logger.error(f"Error getting chat models: {e}")
        return {
            "success": False,
            "error": str(e),
            "models": [chat_service.default_model],
            "default_model": chat_service.default_model
        }

@app.get("/api/chat/sessions/{user_id}")
async def get_user_chat_sessions(user_id: str, limit: int = 20):
    """Get all chat sessions for a user"""
    try:
        sessions = await chat_service.get_user_sessions(user_id, limit)
        
        return {
            "success": True,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        return {
            "success": False,
            "error": str(e),
            "sessions": [],
            "total_sessions": 0
        }

# RAG Initialization Endpoints
@app.post("/api/initialize-rag")
async def initialize_rag_storage(force: bool = False, use_gpu: bool = True):
    """
    Initialize RAG vector storage from JSON files
    Args:
        force: Force reinitialization even if storage exists
        use_gpu: Use GPU for embedding generation
    """
    try:
        from init_rag_storage import RAGVectorInitializer
        
        initializer = RAGVectorInitializer()
        result = initializer.initialize(force=force, use_gpu=use_gpu)
        
        return result
        
    except Exception as e:
        logger.error(f"Error initializing RAG storage: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/rag-status")
async def get_rag_status():
    """Get RAG vector storage status"""
    try:
        from init_rag_storage import RAGVectorInitializer
        
        initializer = RAGVectorInitializer()
        
        # Check if storage exists
        storage_exists = initializer.check_vector_storage_exists()
        
        # Get chunk count
        conn = initializer.get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                chunk_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM documents")
                document_count = cur.fetchone()[0]
            conn.close()
        else:
            chunk_count = 0
            document_count = 0
        
        return {
            "success": True,
            "initialized": storage_exists,
            "total_chunks": chunk_count,
            "total_documents": document_count
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
