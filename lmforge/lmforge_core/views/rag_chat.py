from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
import torch
import numpy as np
from functools import lru_cache
from decouple import config
from django.conf import settings
from django.db import connection
from ..models.rag_chat import DocumentChunk

# Import reusable functions from generate_q_and_a (optional - with fallback)
try:
    from .generate_q_and_a import split_text as split_text_with_tokenizer
    HAS_TOKENIZER_SPLIT = True
except ImportError:
    HAS_TOKENIZER_SPLIT = False

# No longer need psycopg - using Django models instead

# Suppress warnings
warnings.filterwarnings("ignore")


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use a smaller model for CPU/RAM-limited environments
# Options: "microsoft/phi-2" (2.7B - requires ~8GB RAM), "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B - requires ~3GB RAM)
MODEL_NAME = config("RAG_MODEL_NAME", default="microsoft/phi-2")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Global state
_embeddings_model = None
_vectorstore_connection = None
_text_chunks = []
_embeddings_cache = None
_llm_pipeline = None

def get_pdf_text(pdf_files):
    """Extract text from PDF files"""
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_simple(text, chunk_size=256, chunk_overlap=64):
    """Simple text splitter that doesn't require tokenizer"""
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        # If adding this paragraph would exceed chunk size
        if current_size + para_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Handle overlap
            if chunk_overlap > 0:
                # Take last part of current chunk for overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_size = len(current_chunk)
            else:
                current_chunk = para
                current_size = para_size
        else:
            current_chunk += "\n\n" + para if current_chunk else para
            current_size += para_size + 2 if current_chunk else para_size
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def get_text_chunks(text, max_tokens=256):
    """Split text into chunks - tries token-aware splitting first, falls back to simple"""
    # Try token-aware splitting if available and HF token is configured
    if HAS_TOKENIZER_SPLIT:
        try:
            # Check if HF token is available before trying to use tokenizer
            from decouple import config
            hf_token = config("HUGGINGFACE_TOKEN", default="")
            if hf_token:
                chunks = split_text_with_tokenizer(text, max_tokens=max_tokens)
                if chunks:  # Only use if it successfully returned chunks
                    return chunks
        except Exception:
            # Fall through to simple splitting (no warnings needed)
            pass
    
    # Fallback to simple character-based splitting (no HF token needed)
    # Convert max_tokens to approximate characters (rough estimate: 1 token ≈ 4 characters)
    chunk_size = max_tokens * 4 if max_tokens else 1024  # 256 tokens ≈ 1024 chars
    return split_text_simple(text, chunk_size=chunk_size, chunk_overlap=64)

@lru_cache(maxsize=1)
def get_embeddings_model():
    """Cache embeddings model with GPU support"""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )
    return _embeddings_model

def get_connection_params():
    """Get PostgreSQL connection parameters - uses same database as Django"""
    # Use Django's database settings (matches docker-compose setup)
    db_config = settings.DATABASES.get('default', {})
    
    return {
        "host": config("DATABASE_HOST", default=db_config.get("HOST", "localhost")),
        "port": config("DATABASE_PORT", default=db_config.get("PORT", "5432")),
        "database": config("DATABASE_NAME", default=db_config.get("NAME", "")),
        "user": config("DATABASE_USER", default=db_config.get("USER", "postgres")),
        "password": config("DATABASE_PASSWORD", default=db_config.get("PASSWORD", "postgres"))
    }

def setup_vectorstore(text_chunks, collection_name="pdf_embeddings"):
    """Setup vector store using Django models"""
    global _text_chunks, _embeddings_cache
    
    # Generate embeddings
    embeddings_model = get_embeddings_model()
    embeddings = embeddings_model.encode(
        text_chunks,
        normalize_embeddings=True,
        batch_size=128 if DEVICE == "cuda" else 32,
        show_progress_bar=False
    )
    
    # Store chunks and embeddings in memory (for fast retrieval)
    _text_chunks = text_chunks
    _embeddings_cache = embeddings

    # Store in database using Django models (optional - in-memory also works)
    try:
        # Check if table exists using Django's connection
        table_name = DocumentChunk._meta.db_table
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, [table_name])
            table_exists = cursor.fetchone()[0]
        
        if table_exists:
            # Clear existing chunks for this collection
            DocumentChunk.objects.filter(collection_name=collection_name).delete()
            
            # Create new chunk records
            chunk_objects = []
            for chunk, embedding in zip(text_chunks, embeddings):
                chunk_objects.append(
                    DocumentChunk(
                        text=chunk,
                        embedding=embedding.tolist(),  # Convert numpy array to list for JSONField
                        collection_name=collection_name
                    )
                )
            
            # Bulk create for efficiency
            DocumentChunk.objects.bulk_create(chunk_objects, batch_size=100)
        
    except Exception:
        # If database storage fails (table doesn't exist yet), continue with in-memory storage
        # This is fine - in-memory storage works perfectly for RAG
        pass

def retrieve_relevant_chunks(query, k=3):
    """Retrieve relevant chunks using cosine similarity"""
    global _text_chunks, _embeddings_cache
    
    if _text_chunks is None or _embeddings_cache is None:
        return []
    
    # Get query embedding
    embeddings_model = get_embeddings_model()
    query_embedding = embeddings_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]
    
    # Calculate cosine similarity
    similarities = np.dot(_embeddings_cache, query_embedding)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return relevant chunks
    return [_text_chunks[i] for i in top_indices]

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """Load and cache model and tokenizer with memory optimizations"""
    import logging
    logger = logging.getLogger(__name__)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        if DEVICE == "cuda" and torch.cuda.is_available():
            logger.info("Loading model on CUDA with float16...")
            # Use torch_dtype instead of dtype for compatibility
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            )
            model.eval()
        else:
            # CPU mode - use memory optimizations
            logger.info(f"Loading model on CPU: {MODEL_NAME}")
            logger.warning("Note: This may take several minutes and requires significant RAM (8GB+ for phi-2)")
            logger.warning("If you get OOM errors, consider using a smaller model like TinyLlama")
            
            # Load without quantization (8-bit only works on CUDA)
            # Use low_cpu_mem_usage to minimize peak memory
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            model.eval()
        
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_llm_pipeline():
    """Get or create LLM pipeline"""
    global _llm_pipeline
    if _llm_pipeline is None:
        model, tokenizer = load_model_and_tokenizer()
        _llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return _llm_pipeline

def generate_response(question, context_chunks):
    """Generate response using RAG"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Format context
        context = "\n\n".join(context_chunks)
        
        # Create prompt
        prompt = f"""You are a helpful financial assistant. Answer the question based on the provided context.

Context: {context}

Question: {question}

Provide a clear and concise answer:"""
        
        logger.info("Starting model generation...")
        # Generate response
        pipeline = get_llm_pipeline()
        logger.info("Pipeline loaded, generating response...")
        
        result = pipeline(prompt, return_full_text=False)
        logger.info(f"Pipeline result type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
        
        # Extract text from result - handle different response formats
        response = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                response = result[0].get('generated_text', '')
            else:
                response = str(result[0])
        elif isinstance(result, dict):
            response = result.get('generated_text', str(result))
        else:
            response = str(result)
        
        response = response.strip()
        logger.info(f"Generated response length: {len(response)}")
        
        if not response or len(response) < 5:
            logger.warning("Generated response is too short or empty")
            response = "I couldn't generate a complete response. Please try rephrasing your question."
        
        return response
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        raise

def rag_chat_view(request):
    """Main view for RAG chat page"""
    gpu_name = None
    gpu_memory = None
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            pass
    
    gpu_status = {
        "device": DEVICE,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory
    }
    
    db_info = get_connection_params()
    # Don't expose password in template
    db_info_display = {
        "host": db_info["host"],
        "port": db_info["port"],
        "database": db_info["database"],
        "user": db_info["user"]
    }
    
    return render(request, 'rag_chat.html', {
        "gpu_status": gpu_status,
        "db_info": db_info_display
    })

@csrf_exempt
@require_http_methods(["POST"])
def process_pdfs(request):
    """Process uploaded PDF files"""
    global _text_chunks
    
    try:
        if not request.FILES:
            return JsonResponse({"error": "No files uploaded"}, status=400)
        
        pdf_files = []
        if 'files' in request.FILES:
            files_list = request.FILES.getlist('files')
            pdf_files.extend(files_list)
        else:
            for file_key in request.FILES:
                file_obj = request.FILES[file_key]
                if hasattr(file_obj, 'read'):
                    pdf_files.append(file_obj)
        
        if not pdf_files:
            return JsonResponse({"error": "No valid PDF files found"}, status=400)
        
        # Extract text
        raw_text = get_pdf_text(pdf_files)
        
        if not raw_text.strip():
            return JsonResponse({"error": "No text found in PDFs"}, status=400)
        
        # Split into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Setup vectorstore
        setup_vectorstore(text_chunks)
        
        return JsonResponse({
            "success": True,
            "message": f"Ready! {len(pdf_files)} PDF(s) - {len(text_chunks)} chunks",
            "num_files": len(pdf_files),
            "num_chunks": len(text_chunks)
        })
        
    except Exception as e:
        import traceback
        error_details = {
            "error": f"Processing failed: {str(e)}",
            "type": type(e).__name__,
            "traceback": traceback.format_exc() if hasattr(e, '__traceback__') else None
        }
        return JsonResponse(error_details, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def chat_message(request):
    """Handle chat messages"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        data = json.loads(request.body)
        user_question = data.get("question", "").strip()
        
        logger.info(f"Received chat message: {user_question[:50]}...")
        
        if not user_question:
            return JsonResponse({"error": "Question is required"}, status=400)
        
        if _text_chunks is None or len(_text_chunks) == 0:
            return JsonResponse({"error": "Please process PDFs first"}, status=400)
        
        logger.info("Retrieving relevant chunks...")
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(user_question, k=3)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        if not relevant_chunks:
            return JsonResponse({"error": "No relevant context found"}, status=400)
        
        # Generate response - this might take a while (model loading/generation)
        try:
            logger.info("Starting response generation...")
            response = generate_response(user_question, relevant_chunks)
            
            # Clean response
            response = str(response).strip()
            
            # Handle empty or very short responses
            if not response or len(response) < 10:
                response = "I couldn't generate a proper response. Please try rephrasing your question."
            
            # Limit response length if it's too long
            if len(response) > 2000:
                response = response[:2000] + "..."
            
            logger.info(f"Response generated successfully, length: {len(response)}")
            return JsonResponse({
                "success": True,
                "response": response,
                "question": user_question
            })
        except Exception as gen_error:
            # Model generation error
            logger.error(f"Error generating response: {str(gen_error)}", exc_info=True)
            return JsonResponse({
                "error": f"Error generating response: {str(gen_error)}",
                "type": type(gen_error).__name__
            }, status=500)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return JsonResponse({"error": "Invalid JSON in request"}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in chat_message: {str(e)}", exc_info=True)
        error_details = {
            "error": f"Error: {str(e)}",
            "type": type(e).__name__
        }
        return JsonResponse(error_details, status=500)
