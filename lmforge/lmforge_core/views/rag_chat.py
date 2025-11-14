"""
Views for PDF Chat app - Updated to use direct services instead of API calls
"""
import uuid
import time
import asyncio
import json
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.core.management import call_command
from django.db import connection

logger = logging.getLogger(__name__)

# Import forms
from ..forms.forms import (
    PDFUploadForm, JSONDataForm, QuestionForm, 
    ChatMessageForm, ChatSessionForm, EmbeddingProcessForm
)
# Import RAG models from the models folder
from ..models.rag_chat import ChatSession, ProcessedDocument, ChatMessage

# Import RAG services - Direct integration instead of API calls
from ..services.embedding_client import embedding_client
from ..services.chunking_service import chunking_service
from ..services.chat_service import chat_service
from ..services.clients.ollama_client import ollama_client
from ..services.service_health import service_health
from ..services.rag_initializer import rag_initializer

# Backend URL from environment
# Main RAG Chat View
def rag_chat_view(request):
    """Main RAG chat interface with direct service integration - no API calls"""
    session = get_or_create_session(request)
    
    # Handle chat message submission directly through ChatService
    chat_response = None
    if request.method == 'POST' and request.POST.get('message'):
        try:
            message = request.POST.get('message', '').strip()
            use_rag = request.POST.get('use_rag') == 'true'
            storage_backend = request.POST.get('storage_backend', 'pgvector')
            
            if message:
                # Use ChatService directly - no API calls
                if use_rag:
                    chat_response = chat_service.generate_rag_response(
                        user_message=message,
                        session_id=session.session_id,
                        storage_backend=storage_backend
                    )
                else:
                    # Simple response without RAG context
                    chat_response = chat_service.generate_rag_response(
                        user_message=message,
                        session_id=session.session_id,
                        context_chunks=[],  # Empty context for non-RAG
                        storage_backend=storage_backend
                    )
                    
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            chat_response = {
                'success': False,
                'content': 'Sorry, I encountered an error processing your message.',
                'error': str(e)
            }
    
    # Initialize forms
    upload_form = PDFUploadForm()
    json_form = JSONDataForm()
    embedding_form = EmbeddingProcessForm()
    
    # Get service status from direct service integration with error handling
    try:
        service_status_response = get_service_status()
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        service_status_response = {
            'services': {'ollama': 'offline', 'database': 'offline'},
            'response_time': 0
        }
    
    # Extract status information from service check
    services = service_status_response.get('services', {})
    ollama_status = services.get('ollama', 'unknown')
    database_status = services.get('database', 'unknown')
    
    # Get chat history directly from ChatService
    try:
        chat_history = chat_service.get_chat_history(session.session_id, limit=20)
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        chat_history = []
    
    # Get comprehensive system status for status tab with error handling
    try:
        system_status = service_health.get_system_status()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        system_status = {
            'postgres': {'available': False, 'message': 'Connection failed'},
            'ollama': {'available': False, 'message': 'Connection failed'},
            'system_ready': False,
            'timestamp': time.time()
        }
    
    try:
        kb_status = service_health.check_knowledge_base_status()
    except Exception as e:
        logger.error(f"Error checking knowledge base status: {e}")
        kb_status = {
            'initialized': False,
            'json_files_found': 0,
            'message': 'Unable to check status'
        }
    
    context = {
        'session': session,
        'session_id': session.session_id,
        'upload_form': upload_form,
        'json_form': json_form,
        'embedding_form': embedding_form,
        'ollama_status': ollama_status,
        'database_status': database_status,
        'service_status_response': service_status_response,
        'service_type': 'direct',  # Indicate we're using direct service calls
        'health_check_time': service_status_response.get('response_time', 0),
        'chat_history': chat_history,
        'chat_response': chat_response,  # Latest response if any
        'system_status': system_status,  # Full system status for status tab
        'kb_status': kb_status,  # Knowledge base status
        'system_status_json': json.dumps(system_status),  # JSON for JavaScript
        'kb_status_json': json.dumps(kb_status),  # JSON for JavaScript
    }
    
    return render(request, 'rag_chat.html', context)

@csrf_exempt
def rag_upload_pdf(request):
    """Handle PDF upload for RAG system"""
    if request.method == 'POST':
        return chunk_pdfs_new_workflow(request)
    return redirect('rag-chat-view')


def rag_process_json(request):
    """Handle JSON processing for RAG system"""
    if request.method == 'POST':
        return process_json(request)
    return redirect('rag-chat-view')


@csrf_exempt
def rag_send_message(request):
    """Handle sending chat messages"""
    if request.method == 'POST':
        return send_chat_message(request)
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def rag_get_chunks(request, chunking_session_id):
    """Get chunks for a chunking session"""
    return chunk_review(request, chunking_session_id)


@csrf_exempt
def rag_embed_and_store(request):
    """Handle embedding and storing chunks using direct service calls"""
    if request.method == 'POST':
        try:
            # Parse request data
            if 'multipart/form-data' in request.content_type or 'application/x-www-form-urlencoded' in request.content_type:
                session_id = request.POST.get('session_id')
                use_gpu = request.POST.get('use_gpu', 'true').lower() == 'true'
                selected_files_str = request.POST.get('selected_files')
                
                if selected_files_str:
                    try:
                        selected_files = json.loads(selected_files_str)
                    except:
                        selected_files = [selected_files_str]
                else:
                    selected_files = None
            else:
                data = json.loads(request.body)
                session_id = data.get('session_id')
                use_gpu = data.get('use_gpu', True)
                selected_files = data.get('selected_files', None)
            
            if not session_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Session ID is required'
                })
            
            # Get stored chunk data from Django session
            total_embeddings = 0
            total_chunks = 0
            processed_files = []
            
            # Process each file's chunks
            for key in request.session.keys():
                if key.startswith(f'chunks_{session_id}_'):
                    filename = key.replace(f'chunks_{session_id}_', '')
                    
                    # Skip if not in selected files
                    if selected_files and filename not in selected_files:
                        continue
                    
                    file_data = request.session[key]
                    result = file_data['result']
                    
                    if not result.get('success'):
                        continue
                    
                    chunks = result['chunks']
                    
                    # Generate embeddings using service
                    chunk_texts = [c['text'] for c in chunks]
                    
                    # Run async function in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        embeddings = loop.run_until_complete(
                            embedding_client.generate_embeddings_batch(chunk_texts, use_gpu)
                        )
                        
                        if embeddings:
                            # Import rag_database
                            from ..services.rag_database import rag_database
                            
                            # Store chunks with embeddings
                            stored_count = loop.run_until_complete(
                                rag_database.store_chunks_with_embeddings(
                                    filename=filename,
                                    chunks=chunks,
                                    embeddings=embeddings,
                                    metadata={'session_id': session_id, 'file_size': file_data.get('file_data', {}).get('size', 0)}
                                )
                            )
                            
                            total_embeddings += stored_count
                            total_chunks += len(chunks)
                            processed_files.append({
                                'filename': filename,
                                'chunks': len(chunks),
                                'embeddings_stored': stored_count
                            })
                    finally:
                        loop.close()
            
            if total_embeddings > 0:
                return JsonResponse({
                    "success": True,
                    "session_id": session_id,
                    "total_chunks": total_chunks,
                    "total_embeddings": total_embeddings,
                    "processed_files": processed_files,
                    "service_used": "GPU Ollama" if use_gpu else "CPU Ollama",
                    "storage_backend": "pgvector",
                    "workflow_step": "embedding_complete",
                    "message": f"Successfully generated {total_embeddings} embeddings"
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'No chunks found for embedding'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def rag_status(request):
    """Get RAG system status"""
    status = get_service_status()
    return JsonResponse(status)


def rag_chat_history(request, session_id):
    """Get chat history for RAG session"""
    return get_chat_history(request, session_id)


@csrf_exempt
@require_http_methods(["POST"])
def initialize_database_action(request):
    """Initialize database tables"""
    try:
        result = rag_initializer.initialize_database_tables()
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def initialize_knowledge_base_action(request):
    """Initialize knowledge base from JSON files"""
    try:
        from io import StringIO
        
        # Check if force flag is requested
        try:
            data = json.loads(request.body) if request.body else {}
            force = data.get('force', False)
        except:
            force = False
        
        # Capture output
        out = StringIO()
        if force:
            call_command('init_rag_storage', '--force', stdout=out, stderr=out)
        else:
            call_command('init_rag_storage', stdout=out, stderr=out)
        output = out.getvalue()
        
        # Get updated KB status
        kb_status = service_health.check_knowledge_base_status()
        
        # Check if it was already initialized
        already_initialized = 'already initialized' in output.lower()
        
        return JsonResponse({
            'success': True,
            'message': 'Knowledge base already initialized' if already_initialized else 'Knowledge base initialization completed',
            'output': output,
            'kb_status': kb_status,
            'already_initialized': already_initialized
        })
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@require_http_methods(["POST"])
@csrf_exempt
def clear_chat_history(request):
    """Clear ALL chat history and session data from ALL database tables"""
    try:
        # Truncate ALL data from PostgreSQL tables (conversations, sessions)
        with connection.cursor() as cursor:
            # Drop all conversations
            cursor.execute("TRUNCATE TABLE conversations CASCADE")
            
            # Drop all sessions
            cursor.execute("TRUNCATE TABLE sessions CASCADE")
            
            pg_conversations_deleted = cursor.rowcount
        
        # Delete ALL Django ORM chat data
        messages_deleted = ChatMessage.objects.all().delete()[0]
        documents_deleted = ProcessedDocument.objects.all().delete()[0]
        sessions_deleted = ChatSession.objects.all().delete()[0]
        
        # Clear session from request
        if 'chat_session_id' in request.session:
            del request.session['chat_session_id']
        
        logger.info(f"Dropped ALL chat data: {messages_deleted} messages, {documents_deleted} documents, {sessions_deleted} sessions from Django ORM; PostgreSQL tables truncated")
        
        return JsonResponse({
            'success': True,
            'message': 'All chat history and session data cleared from database',
            'messages_deleted': messages_deleted,
            'documents_deleted': documents_deleted,
            'sessions_deleted': sessions_deleted,
            'postgresql_tables_truncated': True
        })
        
    except Exception as e:
        logger.error(f"Error clearing all chat data: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def clear_database(request):
    """Clear all database tables (truncate data only, keep structure)"""
    try:
        with connection.cursor() as cursor:
            # Get count before clearing
            cursor.execute("SELECT COUNT(*) FROM documents")
            docs_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunks_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            sessions_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conversations_count = cursor.fetchone()[0]
            
            # Truncate tables (clears data but keeps table structure)
            cursor.execute("TRUNCATE TABLE documents, chunks, sessions, conversations RESTART IDENTITY CASCADE")
            
        return JsonResponse({
            'success': True,
            'message': f'Database data cleared successfully. Tables remain for new data.',
            'deleted': {
                'documents': docs_count,
                'chunks': chunks_count,
                'sessions': sessions_count,
                'conversations': conversations_count
            },
            'note': 'Table structures preserved - ready for new data'
        })
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Failed to clear database: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def clear_knowledge_base(request):
    """Clear only knowledge base data (documents and chunks tables)"""
    try:
        with connection.cursor() as cursor:
            # Get count before clearing
            cursor.execute("SELECT COUNT(*) FROM documents")
            docs_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunks_count = cursor.fetchone()[0]
            
            # Truncate only knowledge base tables
            cursor.execute("TRUNCATE TABLE documents, chunks RESTART IDENTITY CASCADE")
            
        return JsonResponse({
            'success': True,
            'message': f'Knowledge base cleared successfully',
            'deleted': {
                'documents': docs_count,
                'chunks': chunks_count
            }
        })
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Failed to clear knowledge base: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_knowledge_base_details(request):
    """Get detailed knowledge base information including all documents"""
    try:
        with connection.cursor() as cursor:
            # Get all documents with their details
            cursor.execute("""
                SELECT 
                    d.id,
                    d.filename,
                    d.total_chunks,
                    d.processed,
                    d.created_at,
                    COUNT(c.id) as chunks_with_embeddings,
                    d.session_id
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id AND c.embedding IS NOT NULL
                GROUP BY d.id, d.filename, d.total_chunks, d.processed, d.created_at, d.session_id
                ORDER BY d.created_at DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'total_chunks': row[2],
                    'processed': row[3],
                    'created_at': row[4].isoformat() if row[4] else None,
                    'chunks_with_embeddings': row[5],
                    'session_id': row[6],
                    'embedding_progress': f"{row[5]}/{row[2]}" if row[2] > 0 else "0/0"
                })
            
            # Get overall statistics
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE processed = true")
            processed_documents = cursor.fetchone()[0]
            
        return JsonResponse({
            'success': True,
            'statistics': {
                'total_documents': total_documents,
                'processed_documents': processed_documents,
                'total_chunks': total_chunks,
                'total_embeddings': total_embeddings,
                'embedding_coverage': f"{(total_embeddings/total_chunks*100):.1f}%" if total_chunks > 0 else "0%"
            },
            'documents': documents
        })
    except Exception as e:
        logger.error(f"Error getting knowledge base details: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Failed to get knowledge base details: {str(e)}'
        }, status=500)


# Health endpoint for Docker health checks
def health_check(request):
    """Health check endpoint for load balancers and Docker"""
    return JsonResponse({
        'status': 'healthy',
        'service': 'django-frontend',
        'timestamp': time.time()
    })

def get_or_create_session(request):
    """Get or create a chat session"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'session_id': session_id}
    )
    return session

def get_service_status():
    """Get integrated service status using direct service calls"""
    start_time = time.time()
    
    try:
        # Check Ollama service status
        ollama_status = ollama_client.get_service_status()
        ollama_available = ollama_status.get("service_available", False)
        
        # Check database connection
        database_connected = False
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                database_connected = True
        except Exception as e:
            print(f"Database health check failed: {e}")
            database_connected = False
        
        # Determine overall health
        overall_status = "healthy" if ollama_available and database_connected else "degraded" if ollama_available else "unhealthy"
        
        response_time = round(time.time() - start_time, 3)
        
        return {
            "status": overall_status,
            "response_time": response_time,
            "timestamp": time.time(),
            "service_integration": {
                "type": "direct",
                "status": "healthy"  # Django services are healthy if we reach this point
            },
            "ollama_health": {
                "ollama": ollama_status,
                "ollama_status": "healthy" if ollama_available else "unhealthy"
            },
            "services": {
                "integration": "direct",
                "ollama": "healthy" if ollama_available else "unhealthy",
                "database": "healthy" if database_connected else "unhealthy"
            },
            "raw_service_data": {
                "status": overall_status,
                "integration_type": "direct",
                "ollama_services": ollama_status,
                "database": database_connected,
                "services": {
                    "ollama": ollama_available,
                    "database": database_connected
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "response_time": round(time.time() - start_time, 3),
            "timestamp": time.time(),
            "error": str(e),
            "services": {
                "integration": "error",
                "ollama": "unknown",
                "database": "unknown"
            }
        }

def get_available_chunkers():
    """Get available chunking methods using direct service calls"""
    try:
        chunkers_info = embedding_client.get_available_chunkers()
        return {
            "success": True,
            "chunkers": chunkers_info,
            "default_method": "semantic",
            "message": "Available chunking methods retrieved successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def index(request):
    """Main page view"""
    session = get_or_create_session(request)
    
    # Get backend status (includes Ollama and database)
    backend_status_response = get_service_status()
    
    # Extract status information from health check
    services = backend_status_response.get('services', {})
    backend_status = services.get('backend', 'unknown')
    ollama_status = services.get('ollama', 'unknown')
    database_status = services.get('database', 'unknown')
    
    # Get available chunkers
    chunkers_info = get_available_chunkers()
    
    # Get recent messages for this session
    chat_messages = ChatMessage.objects.filter(session=session).order_by('timestamp')[:20]
    
    # Get processing results from session
    processing_result = request.session.get('last_processing_result', {})
    
    # Get current session info
    current_session = None
    if chat_messages.exists() or processing_result:
        current_session = {
            'document_name': processing_result.get('filename', 'Unknown'),
            'chunking_method': processing_result.get('chunking_method', 'Unknown'),
            'created_at': session.created_at,
            'message_count': chat_messages.count()
        }
    
    # Initialize forms
    upload_form = PDFUploadForm()
    json_form = JSONDataForm()
    question_form = QuestionForm()
    embedding_form = EmbeddingProcessForm()
    
    context = {
        'session': session,
        'backend_status': backend_status,
        'ollama_status': ollama_status,
        'database_status': database_status,
        'backend_status_response': backend_status_response,  # Full response for debugging
        'chunkers_info': chunkers_info,
        'chat_messages': chat_messages,
        'current_session': current_session,
        'processing_result': processing_result,
        'upload_form': upload_form,
        'json_form': json_form,
        'question_form': question_form,
        'embedding_form': embedding_form,
        'health_check_time': backend_status_response.get('response_time', 0),
    }
    
    return render(request, 'pdf_chat/simple_index.html', context)


@require_http_methods(["POST"])
def process_json(request):
    """Process JSON data"""
    form = JSONDataForm(request.POST)
    
    if form.is_valid():
        session = get_or_create_session(request)
        
        # Get form data
        json_data = form.cleaned_data['json_data']
        chunking_method = form.cleaned_data['chunking_method']
        chunk_size = form.cleaned_data['chunk_size']
        chunk_overlap = form.cleaned_data['chunk_overlap']
        
        try:
            # Validate JSON
            import json
            parsed_data = json.loads(json_data)
            
            # Extract text from JSON (support different formats)
            text_content = ""
            if isinstance(parsed_data, dict):
                if 'text' in parsed_data:
                    text_content = parsed_data['text']
                elif 'content' in parsed_data:
                    text_content = parsed_data['content']
                elif 'documents' in parsed_data:
                    # Handle array of documents
                    documents = parsed_data['documents']
                    if isinstance(documents, list):
                        text_content = "\n\n".join([
                            doc.get('text', doc.get('content', str(doc))) 
                            for doc in documents if isinstance(doc, dict)
                        ])
                else:
                    # Convert entire JSON to text
                    text_content = json.dumps(parsed_data, indent=2)
            elif isinstance(parsed_data, list):
                # Handle array of strings or objects
                text_content = "\n\n".join([
                    item.get('text', item.get('content', str(item))) if isinstance(item, dict) else str(item)
                    for item in parsed_data
                ])
            else:
                text_content = str(parsed_data)
            
            if not text_content.strip():
                messages.error(request, "No text content found in JSON data.")
                return redirect('rag-chat-view')
            
            # Use direct chunking service instead of API call
            from ..services.chunking_service import chunking_service
            
            # Create a temporary file-like object for the text
            class TextFile:
                def __init__(self, content, name="json_data.txt"):
                    self.content = content
                    self.name = name
                    self.size = len(content)
                
                def read(self):
                    return self.content.encode('utf-8')
            
            text_file = TextFile(text_content)
            result = chunking_service.chunk_pdf_file(text_file, "JSON_Data")
            
            if result.get('success'):
                chunks_data = result.get('chunks', [])
                
                # Extract text content from chunks for display
                chunks = [chunk.get('text', '') for chunk in chunks_data]
                
                # Calculate enhanced chunk statistics
                total_chunks = len(chunks)
                if total_chunks > 0:
                    chunk_sizes = [len(chunk) for chunk in chunks]
                    avg_chunk_size = sum(chunk_sizes) / total_chunks
                    min_chunk_size = min(chunk_sizes)
                    max_chunk_size = max(chunk_sizes)
                    
                    # Calculate additional metrics
                    word_counts = [len(chunk.split()) for chunk in chunks]
                    avg_words = sum(word_counts) / total_chunks
                    sentence_counts = [chunk.count('.') + chunk.count('!') + chunk.count('?') for chunk in chunks]
                    avg_sentences = sum(sentence_counts) / total_chunks
                else:
                    avg_chunk_size = min_chunk_size = max_chunk_size = 0
                    avg_words = avg_sentences = 0
                
                # Store enhanced processing result in session
                request.session['last_processing_result'] = {
                    'source': 'json',
                    'filename': 'JSON Data',
                    'chunking_method': chunking_method,
                    'chunks': chunks,
                    'total_chunks': total_chunks,
                    'processing_time': result.get('processing_time', 0),
                    'original_length': len(text_content),
                    'upload_success': True,
                    'avg_chunk_size': round(avg_chunk_size, 2),
                    'min_chunk_size': min_chunk_size,
                    'max_chunk_size': max_chunk_size,
                    'total_characters': sum(chunk_sizes) if total_chunks > 0 else 0,
                    'avg_words_per_chunk': round(avg_words, 2),
                    'avg_sentences_per_chunk': round(avg_sentences, 2),
                    'data_format': 'JSON',
                    'backend_session_id': result.get('session_id', '')
                }
                
                # Create ProcessedDocument record
                ProcessedDocument.objects.create(
                    session=session,
                    filename='JSON Data',
                    file_size=len(json_data.encode('utf-8')),
                    chunks_created=total_chunks,
                    chunking_method=chunking_method,
                    backend_session_id=result.get('session_id', '')
                )
                
                messages.success(
                    request,
                    f"Successfully processed JSON data into {total_chunks} chunks using word-based chunking."
                )
            else:
                error_msg = result.get('error', 'Unknown error occurred during chunking')
                messages.error(request, f"Processing failed: {error_msg}")
                
        except json.JSONDecodeError as e:
            messages.error(request, f"Invalid JSON format: {str(e)}")
        except Exception as e:
            messages.error(request, f"Error processing JSON: {str(e)}")
    else:
        messages.error(request, "Please correct the form errors.")
    
    return redirect('rag-chat-view')


@require_http_methods(["GET"])
def get_chunkers(request):
    """Get available chunking methods"""
    chunkers_info = get_available_chunkers()
    return JsonResponse(chunkers_info)


@require_http_methods(["GET"])
def backend_status(request):
    """Get backend status with concurrent health checks"""
    status = get_service_status()
    return JsonResponse(status)


@require_http_methods(["POST"])
def ask_question(request):
    """Ask a question about the documents"""
    form = QuestionForm(request.POST)
    
    if form.is_valid():
        session = get_or_create_session(request)
        question = form.cleaned_data['question']
        
        # Get processing result to provide context-aware responses
        processing_result = request.session.get('last_processing_result', {})
        
        # Create more contextual dummy responses for demonstration
        import random
        
        if processing_result.get('upload_success'):
            filename = processing_result.get('filename', 'document')
            total_chunks = processing_result.get('total_chunks', 0)
            method = processing_result.get('chunking_method', 'unknown')
            
            context_responses = [
                f"Based on the analysis of '{filename}' (processed into {total_chunks} chunks using {method}), I found relevant information about your question: '{question[:50]}...'",
                f"From the {total_chunks} document chunks extracted from '{filename}', here's what I can tell you about '{question[:30]}...': This appears to be an important topic that spans multiple sections of the document.",
                f"After analyzing '{filename}' using {method} chunking method, I found several relevant passages. Your question about '{question[:40]}...' relates to key concepts discussed in the document.",
                f"The document '{filename}' contains {total_chunks} chunks of information. Regarding your question '{question[:35]}...', I found connections across multiple sections that suggest this is a central theme.",
                f"Based on semantic analysis of '{filename}' ({total_chunks} chunks), your question touches on important aspects of the content. The {method} method helped identify relevant passages for your inquiry."
            ]
            response_text = random.choice(context_responses)
        else:
            # Fallback responses when no document is uploaded
            generic_responses = [
                "Please upload a PDF document first so I can answer questions about its content.",
                "I'd be happy to help answer questions about your document. Please upload a PDF file to get started.",
                "To provide accurate answers, I need you to upload a PDF document first. Then I can analyze its content and answer your questions.",
                "Upload a PDF document and I'll be able to answer questions about its specific content using advanced chunking and analysis techniques."
            ]
            response_text = random.choice(generic_responses)
        
        # Save the question and response
        ChatMessage.objects.create(
            session=session,
            message=question,
            response=response_text
        )
        
        messages.success(request, "Question processed successfully!")
    else:
        messages.error(request, "Please enter a valid question.")
    
    return redirect('rag-chat-view')

@csrf_exempt
@require_http_methods(["POST"])
def create_chat_session(request):
    """Create a new chat session"""
    try:
        form = ChatSessionForm(request.POST)
        
        if form.is_valid():
            session_name = form.cleaned_data.get('session_name')
            user_id = form.cleaned_data.get('user_id', 'anonymous')
            
            # Use direct chat service instead of API call
            from ..services.chat_service import chat_service
            
            # Use synchronous method to avoid async context issues
            session_id = chat_service.create_chat_session_sync(
                user_id=user_id,
                session_name=session_name or f"Session_{int(time.time())}"
            )
            
            if session_id:
                request.session['chat_session_id'] = session_id
                
                return JsonResponse({
                    'success': True,
                    'session_id': session_id,
                    'session_data': {
                        'user_id': user_id,
                        'session_name': session_name or f'Session_{session_id[:8]}'
                    }
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Failed to create session'
                })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid form data'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def send_chat_message(request):
    """Send a chat message and get RAG response using direct service calls"""
    try:
        form = ChatMessageForm(request.POST)
        
        if form.is_valid():
            message = form.cleaned_data['message']
            session_id = form.cleaned_data.get('session_id') or request.session.get('chat_session_id')
            model = form.cleaned_data.get('model', 'qwen2.5:0.5b-instruct')  # Use configured model
            use_rag = form.cleaned_data.get('use_rag', True)
            
            if not session_id:
                return JsonResponse({
                    'success': False,
                    'error': 'No chat session found. Please create a new session.'
                })
            
            # Use direct chat service instead of API calls
            from ..services.chat_service import ChatService
            chat_service = ChatService()
            
            # Use synchronous method to avoid async context issues
            if use_rag:
                # Generate RAG-enhanced response
                result = chat_service.generate_rag_response_sync(
                    user_message=message,
                    session_id=session_id
                )
            else:
                # Generate simple response (non-RAG)
                # For now, we'll still use RAG but with minimal context
                result = chat_service.generate_rag_response_sync(
                    user_message=message,
                    session_id=session_id,
                    context_chunks=[]  # Empty context for non-RAG
                )
                
                if result.get('success'):
                    return JsonResponse({
                        'success': True,
                        'response': result.get('content'),  # Service returns 'content' not 'response'
                        'context_chunks': result.get('context_chunks', []),
                        'metadata': {
                            'model_used': result.get('model_used'),
                            'generation_time': result.get('generation_time'),
                            'chat_history_length': result.get('chat_history_length', 0)
                        }
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': result.get('error', 'Failed to generate response')
                    })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid message data'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@require_http_methods(["GET"])
def get_chat_history(request, session_id):
    """Get chat history for a session using direct service calls"""
    try:
        from ..services.chat_service import ChatService
        chat_service = ChatService()
        
        # Get limit parameter
        limit = int(request.GET.get('limit', 50))
        
        # Run async function in sync context
        # Use synchronous method to avoid async context issues
        chat_history = chat_service.get_chat_history_sync(session_id, limit=limit)
        
        return JsonResponse({
            'success': True,
            'history': chat_history,
            'session_id': session_id
        })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_chat_session(request, session_id):
    """Delete a chat session using direct service calls"""
    try:
        from ..services.chat_service import chat_service
        
        # Use synchronous method to avoid async context issues
        deleted = chat_service.delete_chat_session_sync(session_id)
        
        if deleted:
            # Clear session from Django session
            if request.session.get('chat_session_id') == session_id:
                del request.session['chat_session_id']
            
            return JsonResponse({
                'success': True,
                'message': 'Chat session deleted successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to delete session'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@require_http_methods(["GET"])
def get_chat_models(request):
    """Get available chat models using direct service calls"""
    try:
        from ..services.chat_service import chat_service
        
        # Use synchronous method to avoid async context issues
        models = chat_service.get_available_models_sync()
        
        return JsonResponse({
            'success': True,
            'models': models,
            'default_model': chat_service.default_model
        })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'models': ['llama3.2'],
            'default_model': 'llama3.2'
        })


# NEW WORKFLOW: Chunk Review and Embedding
@require_http_methods(["POST"])
def chunk_pdfs_new_workflow(request):
    """
    NEW: Chunk PDFs using direct chunking service
    Returns session ID for reviewing chunks before embedding
    """
    form = PDFUploadForm(request.POST, request.FILES)
    
    if form.is_valid():
        session = get_or_create_session(request)
        pdf_files = request.FILES.getlist('pdf_files')
        
        if not pdf_files:
            messages.error(request, "Please select at least one PDF file.")
            return redirect('rag-chat-view')
        
        try:
            messages.info(request, f"Chunking {len(pdf_files)} PDF file(s) using word-based algorithm...")
            
            # Process each PDF through chunking service directly
            all_results = []
            session_id = str(uuid.uuid4())
            total_chunks = 0
            successful_files = 0
            
            for pdf_file in pdf_files:
                try:
                    # Use chunking service directly
                    result = chunking_service.chunk_pdf_file(pdf_file, pdf_file.name)
                    
                    if result.get('success'):
                        all_results.append(result)
                        total_chunks += result['chunk_count']
                        successful_files += 1
                        
                        # Store chunks in session for review (temporary storage)
                        request.session[f'chunks_{session_id}_{pdf_file.name}'] = {
                            'result': result,
                            'file_data': {
                                'filename': pdf_file.name,
                                'size': pdf_file.size,
                                'chunks': result['chunks']
                            }
                        }
                    else:
                        all_results.append(result)
                        
                except Exception as e:
                    all_results.append({
                        'success': False,
                        'error': str(e),
                        'filename': pdf_file.name
                    })
            
            if successful_files > 0:
                # Store session ID for review page
                request.session['chunking_session_id'] = session_id
                request.session['chunking_result'] = {
                    'success': True,
                    'session_id': session_id,
                    'total_files': len(pdf_files),
                    'successful_files': successful_files,
                    'total_chunks': total_chunks,
                    'results': all_results
                }
                
                messages.success(
                    request,
                    f"✅ Chunking complete! {total_chunks} chunks from {successful_files} file(s). Review before embedding."
                )
                
                # Redirect to review page
                return redirect('rag-get-chunks', chunking_session_id=session_id)
            else:
                messages.error(request, "Failed to chunk any PDF files.")
                
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
    else:
        messages.error(request, "Invalid form data.")
    
    return redirect('rag-chat-view')


@require_http_methods(["GET"])
def chunk_review(request, session_id):
    """
    NEW: Review chunking results before embedding
    Shows detailed chunk information for user approval
    """
    try:
        # Get chunking session data from Django session
        chunking_result = request.session.get('chunking_result', {})
        
        if chunking_result.get('session_id') == session_id:
            # Reconstruct session data from stored results
            files_data = []
            total_chunks = 0
            total_pages = 0
            
            # Get stored chunk data for each file
            for key in request.session.keys():
                if key.startswith(f'chunks_{session_id}_'):
                    file_data = request.session[key]
                    result = file_data['result']
                    
                    # Get preview of chunks
                    chunk_previews = chunking_service.get_chunk_preview(result['chunks'], count=10)
                    
                    files_data.append({
                        'filename': result['filename'],
                        'chunk_count': result['chunk_count'],
                        'page_count': result['page_count'],
                        'statistics': result['chunk_statistics'],
                        'quality': result['quality_assessment'],
                        'processing_time': result['processing_time'],
                        'chunk_previews': chunk_previews
                    })
                    
                    total_chunks += result['chunk_count']
                    total_pages += result['page_count']
            
            session_data = {
                "success": True,
                "session_id": session_id,
                "files": files_data,
                "total_files": len(files_data),
                "total_chunks": total_chunks,
                "total_pages": total_pages
            }
            
            context = {
                'session_id': session_id,
                'session_data': session_data,
                'total_pages': total_pages
            }
            
            return render(request, 'chunk_review.html', context)
        else:
            messages.error(request, "Chunking session not found or expired.")
            
    except Exception as e:
        messages.error(request, f"Error retrieving session: {str(e)}")
    
    return redirect('rag-chat-view')