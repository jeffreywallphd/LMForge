# RAG Chat System Documentation

## Overview

The RAG (Retrieval-Augmented Generation) Chat System is a document-based conversational AI feature that allows users to upload PDFs, process them into semantic chunks, generate embeddings, and chat with an AI that retrieves relevant context from the uploaded documents.

## Architecture

### Technology Stack

- **Backend Framework**: Django 4.2.23
- **Database**: PostgreSQL with pgvector extension (vector similarity search)
- **LLM Service**: Ollama (local inference)
- **Embedding Model**: all-minilm:33m (384-dimensional vectors)
- **Chat Model**: qwen2.5:0.5b-instruct
- **Containerization**: Docker Compose

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Chat Frontend                         │
│              (rag_chat.html - Interactive UI)                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Django Views Layer                          │
│              (rag_chat.py - Request Handlers)                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Services Layer                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ chat_service.py        - Chat session & RAG logic    │  │
│  │ chunking_service.py    - Text chunking algorithms    │  │
│  │ embedding_client.py    - Embedding generation        │  │
│  │ pdf_processor.py       - PDF extraction & processing │  │
│  │ rag_database.py        - Vector storage & retrieval  │  │
│  │ service_health.py      - System health monitoring    │  │
│  │ rag_initializer.py     - System initialization       │  │
│  │ rag_vector_initializer - JSON knowledge base setup   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              External Services (Docker)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PostgreSQL + pgvector  - Vector database             │  │
│  │ Ollama Service         - LLM inference engine        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Services

### 1. Chat Service (`chat_service.py`)

**Purpose**: Manages chat sessions, context retrieval, and LLM communication.

**Key Features**:
- Create and manage chat sessions
- Store chat history in PostgreSQL
- Retrieve relevant document context using vector similarity
- Generate RAG-enhanced responses using Ollama
- Batch embedding generation for queries (parallel processing)

**Main Methods**:
```python
create_chat_session(user_id, session_name) -> session_id
retrieve_relevant_context(query, session_id, top_k=8) -> chunks
generate_rag_response(user_message, session_id, context_chunks) -> response
get_chat_history(session_id, limit=20) -> messages
```

**Configuration**:
- `chat_model`: qwen2.5:0.5b-instruct
- `max_context_chunks`: 8 chunks per query
- `max_chat_history`: 10 previous messages
- `similarity_threshold`: 0.6 (cosine similarity)

---

### 2. Chunking Service (`chunking_service.py`)

**Purpose**: Handles text extraction and semantic chunking of PDF documents.

**Algorithm**: Word-based Recursive Semantic Chunking
1. Split text by paragraphs (natural boundaries)
2. Combine paragraphs to reach target word count
3. Split large paragraphs by sentences
4. Add overlap between chunks for context continuity

**Configuration**:
- `target_words`: 200 words (~1000 characters)
- `max_words`: 300 words (~1500 characters)
- `min_words`: 50 words (~250 characters)
- `overlap_words`: 30 words (~150 characters)
- `batch_size`: 25 paragraphs per batch

**Main Methods**:
```python
extract_text_from_pdf(pdf_file) -> (text, page_count, page_texts)
word_based_recursive_chunking(text, filename) -> chunks
chunk_pdf_file(pdf_file, filename) -> processing_result
```

**Output Format**:
Each chunk includes:
- `text`: Content of the chunk
- `chunk_id`: Unique identifier
- `chunk_index`: Position in document
- `length`: Character count
- `word_count`: Word count
- `sentence_count`: Number of sentences
- `metadata`: Source file, timestamps, etc.

---

### 3. Embedding Client (`embedding_client.py`)

**Purpose**: Generates vector embeddings using Ollama's embedding model.

**Model**: all-minilm:33m
- 384-dimensional vectors
- Optimized for semantic similarity
- 33MB model size

**Main Methods**:
```python
generate_embeddings_batch(texts, use_gpu=True) -> embeddings
generate_embeddings(chunks, use_gpu=True) -> enhanced_chunks
```

**Error Handling**:
- ConnectionError: Ollama service unavailable
- RuntimeError: Embedding generation failure
- Automatic service health checks

---

### 4. RAG Database (`rag_database.py`)

**Purpose**: Handles vector storage and similarity search using PostgreSQL + pgvector.

**Database Schema**:
```sql
-- Documents table
documents (id, document_id, filename, file_size, page_count, 
           session_id, processed, total_chunks, metadata)

-- Chunks table with vectors
chunks (id, document_id, chunk_index, text_content, 
        embedding vector(384), metadata)

-- Sessions table
sessions (id UUID, status, created_at, updated_at)

-- Conversations table
conversations (id, session_id, message_type, content, 
               message_index, metadata)
```

**Main Methods**:
```python
store_chunks_with_embeddings(filename, chunks, embeddings) -> count
search_similar_chunks_sync(query_embedding, top_k=5, threshold=0.7) -> chunks
```

**Performance Optimizations**:
- Bulk inserts (100 rows per batch)
- IVFFlat index for vector similarity (100 lists)
- Query hints for optimal execution plans
- Connection pooling via Django

---

### 5. PDF Processor (`pdf_processor.py`)

**Purpose**: Orchestrates PDF processing pipeline.

**Processing Pipeline**:
1. Extract text from PDF files (PyPDF2)
2. Clean and preprocess text
3. Apply recursive semantic chunking
4. Generate chunk previews
5. Return processing statistics

**Main Methods**:
```python
extract_text_from_pdfs(pdf_files) -> combined_text
recursive_semantic_chunking(text, filename) -> chunks
process_pdfs_complete(pdf_files) -> processing_result
process_pdfs_chunk_only(pdf_files) -> chunking_result
```

---

### 6. Service Health Monitor (`service_health.py`)

**Purpose**: Monitor system health and service availability.

**Monitored Services**:
- PostgreSQL database (connection, pgvector extension, tables)
- Ollama LLM service (availability, models, GPU/CPU mode)
- Knowledge Base (documents, chunks, embeddings)

**Main Methods**:
```python
check_postgres_health() -> status_dict
check_ollama_health() -> status_dict
check_knowledge_base_status() -> kb_status
get_system_status(force_refresh=False) -> full_status
```

**Status Caching**:
- 30-second cache duration
- Force refresh option available
- Reduces overhead on frequent checks

---

### 7. RAG Initializer (`rag_initializer.py`)

**Purpose**: System initialization and setup operations.

**Capabilities**:
- Initialize PostgreSQL tables from init-db.sql
- Setup pgvector extension
- Initialize knowledge base from JSON files
- Handle existing object conflicts gracefully

**Main Methods**:
```python
initialize_database_tables() -> result
initialize_rag_storage() -> result
```

---

### 8. RAG Vector Initializer (`rag_vector_initializer.py`)

**Purpose**: Process JSON files into vector storage for knowledge base.

**Processing Pipeline**:
1. Scan media/JSON directory for files
2. Extract text chunks from JSON structures
3. Generate embeddings in batches (50 chunks/batch)
4. Store in database with bulk inserts (100 rows/batch)

**Supported JSON Structures**:
- Chapter-based (chapters → sections → paragraphs)
- Content-based (simple content field)
- Array-based (list of text items)
- Generic (auto-detect text fields)

**Configuration**:
- `embedding_model`: all-minilm:33m
- `chunk_size`: 800 characters
- `embedding_batch_size`: 50 chunks
- `db_batch_size`: 100 rows

**Main Methods**:
```python
check_vector_storage_exists() -> bool
find_json_files() -> file_paths
extract_chunks_from_json(json_data) -> chunks
generate_embeddings(chunks, use_gpu=True) -> chunks_with_embeddings
store_in_database(json_file, chunks) -> success
initialize(force=False, use_gpu=True) -> result
```

---

## Docker Services

### PostgreSQL + pgvector

**Image**: pgvector/pgvector:0.8.1-pg18-trixie

**Configuration**:
```yaml
ports: "5435:5432"
volumes:
  - postgres_data:/var/lib/postgresql/data
  - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
```

**Health Check**:
```bash
pg_isready -U ${DATABASE_USER} -d ${DATABASE_NAME}
```

**Features**:
- pgvector extension for vector operations
- IVFFlat indexing for fast similarity search
- Automatic initialization from init-db.sql

---

### Ollama Service

**Image**: Custom (Dockerfile.ollama-unified)

**Configuration**:
```yaml
ports: "11434:11434"
environment:
  - OLLAMA_MAX_LOADED_MODELS=2
  - OLLAMA_KEEP_ALIVE=5m
  - OLLAMA_NUM_PARALLEL=4
  - OLLAMA_CONTEXT_LENGTH=1024
  - OLLAMA_SCHED_SPREAD=true
```

**Resource Limits**:
- Memory: 8G reserved, 12G limit
- CPU: 4.0 cores
- GPU: NVIDIA (if available)

**Models**:
1. **all-minilm:33m** (33MB) - Embedding model
2. **qwen2.5:0.5b-instruct** (395MB) - Chat model

**Startup Process**:
1. Auto-detect GPU/CPU availability
2. Pull models in parallel
3. Warm up models with test requests
4. Ready for production use

---

## API Endpoints

### Chat Operations

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rag_chat_view/` | GET | Main chat interface |
| `/rag_send_message/` | POST | Send chat message |
| `/clear-chat/` | POST | Clear all chat history and sessions |

### Document Operations

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rag_upload_pdf/` | POST | Upload and chunk PDF files |
| `/rag_process_json/` | POST | Process JSON files |
| `/rag_embed_and_store/` | POST | Generate embeddings and store |

### System Operations

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/initialize_database/` | POST | Initialize database tables |
| `/initialize_knowledge_base/` | POST | Initialize KB from JSON files |
| `/knowledge-base-details/` | GET | Get KB statistics |
| `/clear-database/` | POST | Clear all database data |
| `/clear-knowledge-base/` | POST | Clear KB only |

---

## User Workflow

### 1. Upload and Process Documents

```
User uploads PDF → PDF Processor extracts text → Chunking Service creates chunks
→ Embedding Client generates vectors → RAG Database stores chunks with embeddings
```

**Steps**:
1. Select PDF files in upload section
2. Choose chunking method (recursive semantic)
3. Click "Start Chunking Process"
4. Review chunk statistics
5. Confirm or proceed to embedding generation

### 2. Chat with Documents

```
User asks question → Chat Service retrieves relevant chunks → LLM generates response
→ Response displayed with source citations
```

**Steps**:
1. Type question in chat input
2. System retrieves top 8 relevant chunks
3. LLM generates contextual response
4. Response shown with document sources
5. Chat history maintained for context

### 3. Clear Chat History

```
User clicks "Clear" → All chat messages deleted → All sessions removed 
→ PostgreSQL tables truncated → Fresh start ready
```

**What Gets Cleared**:
- All chat messages (ChatMessage model)
- All processed documents (ProcessedDocument model)
- All chat sessions (ChatSession model)
- PostgreSQL conversations table (TRUNCATE)
- PostgreSQL sessions table (TRUNCATE)
- Django session data

---

## Performance Optimizations

### Batch Processing

1. **Embedding Generation**: 50 chunks per batch
   - Location: `rag_vector_initializer.py:367`
   - Speedup: ~50x faster than sequential

2. **Database Inserts**: 100 rows per batch
   - Location: `rag_vector_initializer.py:504`
   - Speedup: ~10-20x faster than individual inserts

3. **Chunking**: 1000 paragraphs per batch
   - Location: `chunking_service.py:33`
   - Optimized for memory efficiency

### Query Optimization

- IVFFlat vector index with 100 lists
- Query hints for optimal execution plans
- Connection pooling via Django ORM
- 30-second status caching

### Parallel Operations

- Parallel query embedding generation in context retrieval
- Concurrent model downloads during Ollama startup
- Async-compatible service methods

**Overall Performance**: 30-40x faster for large datasets

---

## Configuration

### Environment Variables

```bash
# Database
DATABASE_USER=pdf_rag_user
DATABASE_PASSWORD=pdf_rag_password
DATABASE_NAME=pdf_rag_db
DATABASE_PORT=5435

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:0.5b-instruct
OLLAMA_EMBEDDING_MODEL=all-minilm:33m

# Django
DEBUG=True
SECRET_KEY=your-secret-key
```

### Tuning Parameters

- Embedding batch size (50)
- Database batch size (100)
- Chunking batch size (1000)
- Context chunks (8)
- Similarity threshold (0.6)

---

## Error Handling

### Common Issues

**1. Ollama Service Unavailable**
- Check: `docker ps` - ensure ollama container running
- Check: `curl http://localhost:11434/api/tags`
- Solution: Restart ollama container

**2. PostgreSQL Connection Failed**
- Check: Port 5435 not in use
- Check: Database credentials in .env
- Solution: `docker-compose restart postgres`

**3. Out of Memory During Embedding**
- Reduce embedding batch size (line 367 in rag_vector_initializer.py)
- Reduce Ollama parallel requests (OLLAMA_NUM_PARALLEL)

**4. Slow Chat Response**
- Reduce context chunks (max_context_chunks in chat_service.py)
- Reduce OLLAMA_CONTEXT_LENGTH in docker-compose

---

## Security Considerations

1. **CSRF Protection**: Enabled for all POST endpoints
2. **Session Management**: Django sessions with secure cookies
3. **SQL Injection**: Parameterized queries throughout
4. **File Upload**: PDF validation and size limits
5. **Access Control**: Session-based isolation

---

## Monitoring and Logging

### System Health Dashboard

Real-time monitoring of:
- PostgreSQL connection status
- Ollama service availability
- Knowledge base statistics
- Model loading status
- GPU/CPU mode detection

### Logging Levels

```python
logger.info()    # Normal operations
logger.warning() # Degraded performance
logger.error()   # Failures requiring attention
```

**Log Locations**:
- Django: Console output / logs directory
- PostgreSQL: Container logs (`docker logs lmforge_postgres`)
- Ollama: Container logs (`docker logs lmforge_ollama`)

---

## Future Enhancements

1. **Multi-user Support**: User-specific knowledge bases
2. **Document Updates**: Incremental updates without full reprocessing
3. **Advanced Chunking**: Sentence-window chunking, metadata filtering
4. **Model Selection**: User-selectable LLM models
5. **Export Functionality**: Export chat sessions, download chunks
6. **Analytics**: Query patterns, response quality metrics

---

## References

- **Django Documentation**: https://docs.djangoproject.com/
- **pgvector**: https://github.com/pgvector/pgvector
- **Ollama**: https://ollama.ai/
- **PyPDF2**: https://pypdf2.readthedocs.io/

---

## Quick Start Commands

```bash
# Start services
docker-compose -f lmforge/docker-compose.services.yaml up -d

# Run Django server
cd lmforge
python manage.py runserver localhost:8000

# Initialize knowledge base
python manage.py init_rag_storage --force

# Stop services
docker-compose -f lmforge/docker-compose.services.yaml down
```

---

## Support

For issues or questions:
1. Check logs: `docker logs lmforge_ollama` or `docker logs lmforge_postgres`
2. Verify service health in the System Health Monitor
3. Review error messages in Django console

---

**Last Updated**: November 14, 2025  
**Version**: 1.0  
**Author**: LMForge Development Team
