# RAG Feature Setup and Usage Guide

## Overview

The RAG (Retrieval-Augmented Generation) feature provides document processing, embedding generation, and AI-powered chat capabilities using PDF documents. The system uses FastAPI as a backend service with Ollama for LLM operations and PostgreSQL with pgvector for vector storage.

---

## Architecture

### Components

1. **Backend (FastAPI)**: 
   - Handles PDF processing, chunking, and embedding generation
   - Integrates with Ollama for LLM services
   - Manages PostgreSQL/pgvector operations
   - Exposes REST API endpoints (see `main.py` for all endpoints)

2. **Ollama Client Container**:
   - Runs 2 models:
     - **Embedding Model**: For generating document embeddings
     - **Chat Model**: For conversational AI responses

3. **PostgreSQL Database with pgvector**:
   - Stores document metadata and chunks
   - Vector embeddings for semantic search (384 dimensions using all-minilm:33m)

4. **Frontend (Django/LMForge)**:
   - User interface for document upload and chat
   - Currently handles some PDF processing (may be moved to backend later)

---

## Resource Requirements

### Default Resource Allocation

| Service | CPU Cores | RAM | Notes |
|---------|-----------|-----|-------|
| **Ollama Client** | 4 cores | 8 GB | Most resource-intensive; reduce if system hangs |
| **PostgreSQL** | 2 cores | 2 GB | Database with pgvector extension |
| **Backend (FastAPI)** | 2 cores | 2 GB | API service |

**⚠️ Important**: These are **caps**, not requirements. If your system is hanging or experiencing performance issues, reduce these values in `docker-compose.yml`.

### Performance Considerations

- **PDF Processing Time**: 5-9 minutes for typical documents
- **Timeout**: Default is 10 minutes (600 seconds)
- **If using lower CPU resources**: You may need to increase the timeout in `rag_chat.py`
  - File: `lmforge/lmforge_core/views/rag_chat.py`
  - Function: `rag_embed_and_store`
  - Line: ~150
  - Current: `timeout=600` (10 minutes)
  - Increase if needed: `timeout=900` (15 minutes) or higher

---

## Setup Instructions

### 1. Configure Environment Variables

Create/update `.env` file in the `lmforge` directory:

```properties
DATABASE_NAME=pdf_rag_db
DATABASE_USER=pdf_rag_user
DATABASE_PASSWORD=pdf_rag_password
DATABASE_HOST=localhost
DATABASE_PORT=5433
BACKEND_URL=http://localhost:8100
WANDB_API_KEY=
HF_API_KEY=
OPENAI_API_KEY=
```

**Note**: Database credentials are currently hardcoded in the container configuration. These should match the values in the Docker container setup.

### 2. Start Docker Services

Navigate to the RAG directory and start the services:

```bash
cd rag
docker-compose up -d --build
```

Or use Docker Compose directly:

```bash
docker compose -f rag/docker-compose.yml up -d --build
```

**GPU Support**: The configuration supports both GPU and CPU modes. GPU will be used automatically if available.

### 3. Wait for Initialization

**Important**: Wait for the following to complete:
- Model downloads (first run only)
- Ollama model setup
- Backend auto-embedding initialization
- Database schema creation

This can take 5-10 minutes on first run. Check logs:

```bash
docker-compose -f rag/docker-compose.yml logs -f
```

### 4. Start Django Frontend

Once the backend is ready:

```bash
cd lmforge
python manage.py runserver
```

### 5. Access the Application

- **Frontend**: http://localhost:8000
- **Backend API**: http://localhost:8100
- **API Documentation**: http://localhost:8100/docs (FastAPI Swagger UI)

---

## Database Schema

The RAG feature uses PostgreSQL with the pgvector extension. Below is the database structure:

### Tables

#### 1. **sessions**
Manages user sessions for document processing and chat.

```sql
sessions
├── id (UUID, PRIMARY KEY)
├── status (VARCHAR(50), DEFAULT 'active')
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

#### 2. **documents**
Stores document metadata and processing status.

```sql
documents
├── id (SERIAL, PRIMARY KEY)
├── document_id (VARCHAR(255), UNIQUE)
├── filename (VARCHAR(255))
├── file_size (INTEGER)
├── page_count (INTEGER)
├── session_id (UUID, FOREIGN KEY → sessions.id)
├── processed (BOOLEAN, DEFAULT FALSE)
├── total_chunks (INTEGER, DEFAULT 0)
├── metadata (JSONB)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

#### 3. **chunks**
Stores document chunks with vector embeddings for semantic search.

```sql
chunks
├── id (SERIAL, PRIMARY KEY)
├── document_id (INTEGER, FOREIGN KEY → documents.id)
├── chunk_index (INTEGER)
├── text_content (TEXT)
├── embedding (vector(384)) -- pgvector type, 384 dimensions
├── start_position (INTEGER)
├── chunk_size (INTEGER)
├── word_count (INTEGER)
├── metadata (JSONB)
└── created_at (TIMESTAMP)
```

#### 4. **pending_chunks**
Temporary storage for chunks before embedding generation.

```sql
pending_chunks
├── id (SERIAL, PRIMARY KEY)
├── session_id (UUID)
├── filename (VARCHAR(255))
├── chunk_data (JSONB) -- Full chunk result from chunking service
├── created_at (TIMESTAMP)
└── expires_at (TIMESTAMP, DEFAULT created_at + 2 hours)
```

#### 5. **conversations**
Stores chat conversation history.

```sql
conversations
├── id (SERIAL, PRIMARY KEY)
├── session_id (UUID, FOREIGN KEY → sessions.id)
├── message_type (VARCHAR(20)) -- 'user' or 'assistant'
├── content (TEXT)
├── message_index (INTEGER)
├── metadata (JSONB)
└── created_at (TIMESTAMP)
```

### Database Indexes

Optimized for performance:

- `idx_documents_session_id`: Fast session-based document lookup
- `idx_chunks_document_id`: Efficient chunk retrieval
- `idx_chunks_embedding`: IVFFlat index for vector similarity search
- `idx_conversations_session_id`: Fast conversation history retrieval
- `idx_pending_chunks_session_id`: Quick pending chunk access
- `idx_pending_chunks_expires_at`: Automatic cleanup of expired chunks

### Entity Relationship Diagram

```
┌─────────────┐         ┌──────────────┐         ┌────────────┐
│  sessions   │◄────────│  documents   │◄────────│   chunks   │
│             │ 1     * │              │ 1     * │            │
│ id (UUID)   │         │ document_id  │         │ embedding  │
│ status      │         │ session_id   │         │ (vector)   │
└──────┬──────┘         │ processed    │         └────────────┘
       │                └──────────────┘
       │ 1
       │
       │ *
┌──────┴──────────┐     ┌──────────────────┐
│ conversations   │     │ pending_chunks   │
│                 │     │                  │
│ session_id      │     │ session_id       │
│ message_type    │     │ chunk_data       │
│ content         │     │ expires_at       │
└─────────────────┘     └──────────────────┘
```

---

## Usage Workflow

### 1. Upload PDF Documents
- Navigate to RAG Chat View
- Upload one or more PDF files
- Backend processes and chunks the documents

### 2. Review Chunks
- Review the generated chunks
- Check chunk statistics (count, size, distribution)

### 3. Generate Embeddings
- Click "Process and Store" to generate embeddings
- Embeddings are stored in pgvector (384 dimensions)
- **Time**: 5-9 minutes depending on document size

### 4. Chat with Documents
- Use the chat interface to ask questions
- System retrieves relevant chunks using vector similarity
- LLM generates contextual responses

---

## API Endpoints

Key backend endpoints (see `main.py` for complete list):

### Document Processing
- `POST /api/chunk-pdfs-only`: Chunk PDF documents
- `POST /api/embed-and-store`: Generate and store embeddings
- `GET /api/chunking-session/{session_id}`: Get chunking results

### Chat Operations
- `POST /api/chat/session`: Create chat session
- `POST /api/chat/message`: Send message and get AI response
- `GET /api/chat/history/{session_id}`: Get conversation history

### Health Checks
- `GET /health`: Backend health status
- `GET /health_concurrent`: Concurrent service health check

---

## Future Improvements

### Database Migration
Currently, the RAG feature uses a separate PostgreSQL database. Future plans include:
- Migrating to unified LMForge database
- Consolidating document storage
- Sharing user authentication and sessions

### PDF Processing
Some PDF processing is currently handled in the Django frontend. Consider moving all PDF operations to the FastAPI backend for:
- Better separation of concerns
- Improved performance
- Easier scaling

---

## Troubleshooting

### Issue: System Hanging or Slow Performance
**Solution**: Reduce resource allocation in `docker-compose.yml`
- Lower CPU cores for Ollama (try 2 cores)
- Reduce memory allocation (try 4 GB for Ollama)

### Issue: Timeout Errors During Embedding
**Solution**: Increase timeout in `rag_chat.py`
```python
# Line ~150 in rag_embed_and_store function
timeout=900  # Increase from 600 to 900 seconds
```

### Issue: Database Connection Errors
**Solution**: Verify Docker services are running
```bash
docker-compose -f rag/docker-compose.yml ps
docker-compose -f rag/docker-compose.yml logs postgres
```

### Issue: Ollama Models Not Loading
**Solution**: Check Ollama container logs
```bash
docker-compose -f rag/docker-compose.yml logs ollama
```

Wait for model download to complete on first run.

---

## Development Notes

### Current Configuration
- Database credentials are hardcoded in container configuration
- Environment variables in `.env` file must match container settings
- Port 5433 is used to avoid conflicts with default PostgreSQL port

### Known Limitations
- Processing time increases with document size
- First run requires model downloads (can take 10+ minutes)
- Resource usage is high for full GPU acceleration

---

## Support

For issues or questions:
1. Check Docker container logs
2. Verify all services are healthy: `GET /health`
3. Review Django terminal for error messages
4. Check browser console for frontend errors

---

**Last Updated**: November 10, 2025
