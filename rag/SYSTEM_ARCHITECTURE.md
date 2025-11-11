# PDF RAG System Architecture

## API Endpoints
- `GET /health` - Health check for backend service and Ollama connectivity
- `POST /api/chunk-pdfs-only` - Upload PDFs and create chunks (returns session_id)
- `GET /api/chunking-session/{id}` - Retrieve chunks for a session from pending_chunks table
- `POST /api/embed-and-store` - Generate embeddings and store in pgVector database
- `POST /api/chat` - Chat with documents using RAG (retrieves context via pgVector similarity search)
- `GET /api/chat-history/{session_id}` - Get conversation history from PostgreSQL
- `POST /api/initialize-rag` - Initialize vector storage from JSON files in /app/jsons/
- `GET /api/rag-status` - Check RAG initialization status and chunk counts

## Database Schema (PostgreSQL + pgVector)
```
documents (stores PDF/JSON metadata)
├── id (PK)
├── document_id (unique identifier)
├── filename (original file name)
├── session_id (UUID reference)
└── metadata (JSONB with chunking info)

chunks (stores text chunks with vector embeddings)
├── id (PK)
├── document_id (FK → documents.id)
├── text_content (actual text)
├── embedding (vector(768) for pgVector similarity search)
├── word_count, chunk_size
└── metadata (JSONB with chunk details)

pending_chunks (temporary storage for chunks before embedding, 2hr TTL)
├── session_id (workflow identifier)
├── filename (which file)
├── chunk_data (JSONB with all chunks)
└── expires_at (auto-cleanup after 2 hours)

sessions (chat session tracking)
└── id (UUID PK), status, timestamps

conversations (chat message history)
├── session_id (FK → sessions.id)
├── message_type (user/assistant)
└── content (message text)
```

## Project Structure
```
rag-backend/
├── main.py (FastAPI app with all API endpoints)
├── init_rag_storage.py (JSON initialization script, runs on startup)
├── clients/
│   └── ollama_client.py (connects to Ollama containers for embeddings/chat)
├── services/
│   ├── chunking_service.py (word-based recursive semantic chunking, 200 words target)
│   ├── embedding_service.py (generates embeddings via Ollama, stores in pgVector)
│   └── chat_service.py (RAG: retrieves context via pgVector, calls qwen2.5:0.5b-instruct)
└── utils/
    └── chunking.py (chunking strategy implementations)

jsons/ (JSON files for auto-initialization, moved to root level)

scripts/
├── docs/ (PDF documents for processing)
├── if_extractor.py (document processing script)
└── ordered_output/ (processing results)

docker/
├── Dockerfile.backend (Python FastAPI)
├── Dockerfile.postgres (PostgreSQL 16 + pgVector extension)
├── Dockerfile.ollama-unified (Unified Ollama with all-minilm:33m + qwen2.5:0.5b-instruct)
└── init-db.sql (creates tables, indexes, pgVector extension)

Note: Frontend now runs in separate workspace
```

## Docker Container Dependencies
```
ollama-embeddings (GPU container - dedicated for embeddings)
├── Model: nomic-embed-text (768-dim embeddings, 274MB)
├── Resources: 1-2 CPU, 2-4GB RAM, GPU shared
└── Port: 11434, Env: OLLAMA_NUM_PARALLEL=4, OLLAMA_MAX_LOADED_MODELS=1

ollama-chat (GPU container - dedicated for chat)
├── Model: llama3.2 (chat generation, 2.0GB)
├── Resources: 2-4 CPU, 3-6GB RAM, GPU shared
└── Port: 11435→11434, Env: OLLAMA_NUM_PARALLEL=2, OLLAMA_MAX_LOADED_MODELS=1

postgres (database)
├── Extension: pgVector for similarity search
├── Resources: 0.5-2 CPU, 512MB-2GB RAM
└── Port: 5433→5432

backend (FastAPI)
├── Depends: postgres (health check), ollama-embeddings, ollama-chat (service_started)
├── Connects: postgres:5432, ollama-embeddings:11434, ollama-chat:11434
├── Env: EMBEDDING_OLLAMA_URL=http://ollama-embeddings:11434, CHAT_OLLAMA_URL=http://ollama-chat:11434
├── Resources: 1-2 CPU, 1-2GB RAM
└── Port: 8000

frontend (Django)
├── Depends: backend (service_started)
├── Connects: backend:8000
├── Resources: 0.5-1 CPU, 512MB-1GB RAM
└── Port: 8081→8000

Network: pdf-rag-network (bridge, connects all containers)
Volumes: postgres_data (persists database)
GPU: NVIDIA GTX 1050 (4GB VRAM shared between embedding and chat containers)
```

## Data Flow
```
1. PDF Upload → frontend:8081/chunk-new/
2. Chunk Creation → backend:8000/api/chunk-pdfs-only → pending_chunks table (temp storage)
3. Chunk Review → frontend retrieves from backend:8000/api/chunking-session/{id}
4. Embedding Generation → backend:8000/api/embed-and-store → ollama-embeddings:11434/api/embeddings → chunks table with vector(768)
5. Chat Query → frontend:8081/chat/ → backend:8000/api/chat → pgVector similarity search → ollama-chat:11434/api/generate → response
```

## Key Technologies
- Backend: FastAPI + uvicorn, psycopg2 for PostgreSQL
- Frontend: Django 5.2.7, Bootstrap 5
- Database: PostgreSQL 16 + pgVector (IVFFlat index for cosine similarity)
- Embeddings: nomic-embed-text (768 dimensions) via dedicated Ollama container
- Chat: llama3.2 via dedicated Ollama container (RAG with context from pgVector)
- Chunking: Word-based recursive semantic (200-300 words, 30 word overlap, 0.08-0.17s per document)
- GPU: NVIDIA GeForce GTX 1050 (4GB VRAM shared between embedding and chat services)

## Resource Management Benefits
- **Embedding Service**: Optimized for parallel batch processing (4 parallel requests)
- **Chat Service**: Optimized for conversational generation (2 parallel requests, higher memory)
- **Independent Scaling**: Each service can be scaled independently based on workload
- **Resource Isolation**: Prevents embedding workload from affecting chat response times
- **GPU Sharing**: Both containers share the same GPU but with different memory allocations
```
1. PDF Upload → frontend:8081/chunk-new/
2. Chunk Creation → backend:8000/api/chunk-pdfs-only → pending_chunks table (temp storage)
3. Chunk Review → frontend retrieves from backend:8000/api/chunking-session/{id}
4. Embedding Generation → backend:8000/api/embed-and-store → ollama:11434/api/embeddings → chunks table with vector(768)
5. Chat Query → frontend:8081/chat/ → backend:8000/api/chat → pgVector similarity search → llama3.2 → response
```

## Key Technologies
- Backend: FastAPI + uvicorn, psycopg2 for PostgreSQL
- Frontend: Django 5.2.7, Bootstrap 5
- Database: PostgreSQL 16 + pgVector (IVFFlat index for cosine similarity)
- Embeddings: nomic-embed-text (768 dimensions) via Ollama GPU
- Chat: llama3.2 via Ollama GPU (RAG with context from pgVector)
- Chunking: Word-based recursive semantic (200-300 words, 30 word overlap)

