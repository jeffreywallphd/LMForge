"""
Views for PDF Chat app
"""
import requests
import uuid
import time
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages

# Import forms
from ..forms.forms import (
    PDFUploadForm, JSONDataForm, QuestionForm, 
    ChatMessageForm, ChatSessionForm, EmbeddingProcessForm
)
# Import RAG models from the models folder
from ..models.rag_chat import ChatSession, ProcessedDocument, ChatMessage

# Backend URL from environment
def get_backend_url():
    """Get backend URL from environment variable"""
    import os
    from django.conf import settings
    
    return os.getenv('BACKEND_URL', getattr(settings, 'BACKEND_URL', 'http://localhost:8100'))

# Main RAG Chat View
def rag_chat_view(request):
    """Main RAG chat interface - maps to rag_chat.html template"""
    session = get_or_create_session(request)
    
    # Initialize forms
    upload_form = PDFUploadForm()
    json_form = JSONDataForm()
    embedding_form = EmbeddingProcessForm()
    
    # Get backend status (includes Ollama and database)
    backend_status_response = get_backend_status()
    
    # Extract status information from health check
    services = backend_status_response.get('services', {})
    backend_status = services.get('backend', 'unknown')
    ollama_status = services.get('ollama', 'unknown')
    database_status = services.get('database', 'unknown')
    
    context = {
        'session': session,
        'session_id': session.session_id,
        'upload_form': upload_form,
        'json_form': json_form,
        'embedding_form': embedding_form,
        'backend_status': backend_status,
        'ollama_status': ollama_status,
        'database_status': database_status,
        'backend_status_response': backend_status_response,  # Full response for debugging
        'backend_url': get_backend_url(),  # Use dynamic backend URL
        'health_check_time': backend_status_response.get('response_time', 0),
    }
    
    return render(request, 'rag_chat.html', context)

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
    """Handle embedding and storing chunks"""
    if request.method == 'POST':
        try:
            import json
            import logging
            logger = logging.getLogger(__name__)
            
            # Log raw request body for debugging
            logger.info(f"🔍 Request content-type: {request.content_type}")
            
            # Handle both JSON and FormData
            if 'multipart/form-data' in request.content_type or 'application/x-www-form-urlencoded' in request.content_type:
                # Parse FormData from request.POST
                session_id = request.POST.get('session_id')
                use_gpu = request.POST.get('use_gpu', 'true').lower() == 'true'
                selected_files_str = request.POST.get('selected_files')
                
                # Parse selected_files if it's a JSON string
                if selected_files_str:
                    try:
                        selected_files = json.loads(selected_files_str)
                    except:
                        selected_files = [selected_files_str]
                else:
                    selected_files = None
            else:
                # Parse JSON from request.body
                data = json.loads(request.body)
                session_id = data.get('session_id')
                use_gpu = data.get('use_gpu', True)
                selected_files = data.get('selected_files', None)
            
            # Call backend API with FormData (backend expects Form parameters, not JSON)
            form_data = {
                'session_id': session_id,
                'use_gpu': str(use_gpu).lower()
            }
            if selected_files:
                form_data['selected_files'] = json.dumps(selected_files) if isinstance(selected_files, list) else selected_files
            
            backend_url = get_backend_url()
            full_url = f"{backend_url}/api/embed-and-store"
            logger.info(f"🔍 Calling backend at: {full_url}")
            logger.info(f"🔍 Form data: {form_data}")
            
            response = requests.post(
                full_url,
                data=form_data,
                timeout=600  # 10 minutes for embedding
            )
            
            logger.info(f"🔍 Backend response status: {response.status_code}")
            logger.info(f"🔍 Backend response: {response.text[:500]}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return JsonResponse(result)
                except ValueError as json_error:
                    logger.error(f"❌ Invalid JSON response from backend: {json_error}")
                    logger.error(f"❌ Response text: {response.text}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Backend returned invalid JSON',
                        'details': response.text[:500],
                        'json_error': str(json_error)
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Backend error: {response.status_code}',
                    'details': response.text[:500]
                })
                
        except ValueError as json_error:
            # JSON parsing error from request body
            logger.error(f"❌ JSON parsing error: {json_error}")
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON in request body',
                'details': str(json_error)
            })
        except requests.exceptions.Timeout:
            logger.error(f"❌ Backend request timeout")
            return JsonResponse({
                'success': False,
                'error': 'Backend request timed out after 5 minutes'
            })
        except requests.exceptions.RequestException as req_error:
            logger.error(f"❌ Request exception: {req_error}")
            return JsonResponse({
                'success': False,
                'error': f'Network error: {str(req_error)}'
            })
        except Exception as e:
            logger.error(f"❌ Exception in rag_embed_and_store: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def rag_status(request):
    """Get RAG system status"""
    return backend_status(request)


def rag_chat_history(request, session_id):
    """Get chat history for RAG session"""
    return get_chat_history(request, session_id)


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

def get_backend_status():
    """Get backend service status including Ollama and database from /health endpoint"""
    import time
    
    start_time = time.time()
    
    try:
        response = requests.get(f"{get_backend_url()}/health", timeout=5)
        response_time = round(time.time() - start_time, 3)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Extract status from health response
            # Expected format:
            # {
            #     "status": "healthy",
            #     "ollama_services": {...},
            #     "database": true,
            #     "services": {
            #         "ollama": true,
            #         "database": true
            #     }
            # }
            
            overall_status = health_data.get('status', 'unknown')
            services = health_data.get('services', {})
            
            # Determine individual service statuses
            backend_status = "healthy" if overall_status == "healthy" else "unhealthy"
            ollama_status = "healthy" if services.get('ollama', False) else "unhealthy"
            database_status = "healthy" if services.get('database', False) else "unhealthy"
            
            return {
                "status": overall_status,
                "response_time": response_time,
                "timestamp": time.time(),
                "backend_health": {
                    "backend": health_data,
                    "backend_status": backend_status
                },
                "ollama_health": {
                    "ollama": health_data.get('ollama_services', {}),
                    "ollama_status": ollama_status
                },
                "services": {
                    "backend": backend_status,
                    "ollama": ollama_status,
                    "database": database_status
                },
                "raw_health_data": health_data
            }
        else:
            return {
                "status": "unhealthy",
                "response_time": response_time,
                "timestamp": time.time(),
                "error": f"HTTP {response.status_code}",
                "services": {
                    "backend": "unhealthy",
                    "ollama": "unknown",
                    "database": "unknown"
                }
            }
            
    except Exception as e:
        return {
            "status": "error",
            "response_time": round(time.time() - start_time, 3),
            "timestamp": time.time(),
            "error": str(e),
            "services": {
                "backend": "unreachable",
                "ollama": "unknown",
                "database": "unknown"
            }
        }

def get_available_chunkers():
    """Get available chunking methods from backend"""
    try:
        response = requests.get(f"{get_backend_url()}/api/chunkers", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def index(request):
    """Main page view"""
    session = get_or_create_session(request)
    
    # Get backend status (includes Ollama and database)
    backend_status_response = get_backend_status()
    
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
            
            # Send to backend for processing
            response = requests.post(
                f"{get_backend_url()}/api/test-chunking",
                json={
                    'text': text_content,
                    'chunking_method': chunking_method,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                chunks = result.get('chunks', [])
                
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
                    f"Successfully processed JSON data into {total_chunks} chunks using {chunking_method} method."
                )
            else:
                messages.error(request, f"Processing failed: {response.text}")
                
        except json.JSONDecodeError as e:
            messages.error(request, f"Invalid JSON format: {str(e)}")
        except requests.exceptions.RequestException as e:
            messages.error(request, f"Network error: {str(e)}")
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
    status = get_backend_status()
    return JsonResponse(status)


@require_http_methods(["GET"])
def health_status_concurrent(request):
    """Real-time health check for backend, Ollama, and database"""
    import time
    
    start_time = time.time()
    
    try:
        response = requests.get(f"{get_backend_url()}/health", timeout=5)
        response_time = round(time.time() - start_time, 3)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Parse health response
            # Expected format:
            # {
            #     "status": "healthy",
            #     "ollama_services": {...},
            #     "database": true,
            #     "services": {
            #         "ollama": true,
            #         "database": true
            #     }
            # }
            
            overall_status = health_data.get('status', 'unknown')
            services = health_data.get('services', {})
            
            # Build detailed status for each service
            backend_result = {
                "status": "healthy" if overall_status in ["healthy", "degraded"] else "unhealthy",
                "response_time": response_time,
                "details": health_data,
                "url": get_backend_url()
            }
            
            ollama_result = {
                "status": "healthy" if services.get('ollama', False) else "unhealthy",
                "response_time": response_time,
                "details": health_data.get('ollama_services', {})
            }
            
            database_result = {
                "status": "healthy" if services.get('database', False) else "unhealthy",
                "response_time": response_time,
                "details": {"connected": services.get('database', False)}
            }
            
            # Overall system health
            overall_healthy = (
                backend_result["status"] == "healthy" and
                ollama_result["status"] == "healthy" and
                database_result["status"] == "healthy"
            )
            
            return JsonResponse({
                "overall_status": "healthy" if overall_healthy else "unhealthy",
                "total_response_time": response_time,
                "timestamp": time.time(),
                "services": {
                    "backend": backend_result,
                    "ollama": ollama_result,
                    "database": database_result
                },
                "concurrent_check": False,
                "raw_health_data": health_data
            })
        else:
            return JsonResponse({
                "overall_status": "unhealthy",
                "total_response_time": response_time,
                "error": f"HTTP {response.status_code}",
                "timestamp": time.time(),
                "services": {
                    "backend": {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "url": get_backend_url()
                    },
                    "ollama": {"status": "unknown"},
                    "database": {"status": "unknown"}
                }
            })
            
    except Exception as e:
        return JsonResponse({
            "overall_status": "error",
            "total_response_time": round(time.time() - start_time, 3),
            "error": str(e),
            "timestamp": time.time(),
            "services": {
                "backend": {
                    "status": "unreachable",
                    "error": str(e),
                    "url": get_backend_url()
                },
                "ollama": {"status": "unknown"},
                "database": {"status": "unknown"}
            }
        })


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
            
            # Call backend API
            response = requests.post(
                f"{get_backend_url()}/api/chat/session",
                json={
                    'user_id': user_id,
                    'session_name': session_name
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    session_id = result.get('session_id')
                    request.session['chat_session_id'] = session_id
                    
                    return JsonResponse({
                        'success': True,
                        'session_id': session_id,
                        'session_data': result.get('session_data', {})
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': result.get('error', 'Failed to create session')
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Backend error: {response.status_code}'
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
    """Send a chat message and get RAG response"""
    try:
        form = ChatMessageForm(request.POST)
        
        if form.is_valid():
            message = form.cleaned_data['message']
            session_id = form.cleaned_data.get('session_id') or request.session.get('chat_session_id')
            model = form.cleaned_data.get('model', 'llama3.2')
            use_rag = form.cleaned_data.get('use_rag', True)
            
            if not session_id:
                return JsonResponse({
                    'success': False,
                    'error': 'No chat session found. Please create a new session.'
                })
            
            # Call backend API
            response = requests.post(
                f"{get_backend_url()}/api/chat/message",
                json={
                    'session_id': session_id,
                    'message': message,
                    'model': model,
                    'use_rag': use_rag,
                    'max_context_chunks': 5
                },
                timeout=120  # 2 minutes for LLM generation
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return JsonResponse({
                        'success': True,
                        'response': result.get('response'),
                        'context_chunks': result.get('context_chunks', []),
                        'metadata': result.get('metadata', {})
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': result.get('error', 'Failed to generate response')
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Backend error: {response.status_code}'
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
    """Get chat history for a session"""
    try:
        response = requests.get(
            f"{get_backend_url()}/api/chat/history/{session_id}",
            params={'limit': 50},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'messages': result.get('messages', []),
                    'total_messages': result.get('total_messages', 0)
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Failed to get chat history')
                })
        else:
            return JsonResponse({
                'success': False,
                'error': f'Backend error: {response.status_code}'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_chat_session(request, session_id):
    """Delete a chat session"""
    try:
        response = requests.delete(
            f"{get_backend_url()}/api/chat/session/{session_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
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
                    'error': result.get('error', 'Failed to delete session')
                })
        else:
            return JsonResponse({
                'success': False,
                'error': f'Backend error: {response.status_code}'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@require_http_methods(["GET"])
def get_chat_models(request):
    """Get available chat models"""
    try:
        response = requests.get(
            f"{get_backend_url()}/api/chat/models",
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return JsonResponse(result)
        else:
            return JsonResponse({
                'success': False,
                'error': f'Backend error: {response.status_code}',
                'models': ['llama3.2'],
                'default_model': 'llama3.2'
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
    NEW: Chunk PDFs using dedicated chunking service
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
            
            # Upload to new chunking-only endpoint
            files = [('files', (pdf_file.name, pdf_file.read(), 'application/pdf')) for pdf_file in pdf_files]
            
            response = requests.post(
                f"{get_backend_url()}/api/chunk-pdfs-only",
                files=files,
                timeout=600
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    session_id = result['session_id']
                    
                    # Store session ID for review page
                    request.session['chunking_session_id'] = session_id
                    request.session['chunking_result'] = result
                    
                    messages.success(
                        request,
                        f"✅ Chunking complete! {result['total_chunks']} chunks from {result['successful_files']} file(s). Review before embedding."
                    )
                    
                    # Redirect to review page
                    return redirect('rag-get-chunks', chunking_session_id=session_id)
                else:
                    messages.error(request, f"Chunking failed: {result.get('error')}")
            else:
                messages.error(request, f"Backend error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            messages.error(request, f"Network error: {str(e)}")
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
        # Get chunking session data from backend
        response = requests.get(
            f"{get_backend_url()}/api/chunking-session/{session_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                context = {
                    'session_id': session_id,
                    'session_data': result,
                    'total_pages': sum(f.get('page_count', 0) for f in result.get('files', []))
                }
                return render(request, 'chunk_review.html', context)
            else:
                messages.error(request, f"Session not found: {result.get('error')}")
        else:
            messages.error(request, f"Backend error: {response.status_code}")
            
    except Exception as e:
        messages.error(request, f"Error retrieving session: {str(e)}")
    
    return redirect('rag-chat-view')