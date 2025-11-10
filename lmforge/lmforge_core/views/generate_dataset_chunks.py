from django.shortcuts import render
from ..models.scraped_data import ScrapedData
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.utils import LocalTokenNotFoundError
from huggingface_hub import login
from decouple import config
import torch
import logging
import os
import importlib

# ---------- LOAD SETTINGS ----------
QDRANT_HOST = config("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(config("QDRANT_PORT", default=6333))
LOG_FILE = config("QDRANT_LOG_FILE", default="application.log")
DEFAULT_HF_API_KEY = config("HF_API_KEY", default="")

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- TOKENIZER ----------
_tokenizer = None
def get_tokenizer():
    if not DEFAULT_HF_API_KEY:
        logging.warning("No Hugging Face token found.")
        return None, None, device

    try:
        login(token=DEFAULT_HF_API_KEY)
        os.environ["HF_HOME"] = "D:/huggingface_cache"
        os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"

        model_name = "meta-llama/Llama-3.2-1B-Instruct"  
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        model.to(device)
        return model, _tokenizer, device

    except LocalTokenNotFoundError:
        logging.error("Hugging Face token missing in .env.")
        return None, None, device
    except Exception as e:
        logging.error(f"Tokenizer error: {e}")
        return None, None, device


# ---------- TOKEN COUNT + CHUNKING ----------
def count_tokens(text):
    if not _tokenizer:
        return len(text.split())
    return len(_tokenizer.encode(text))

def split_text(text, max_tokens=1000):
    paragraphs = text.split("\n\n")
    chunks, current_chunk, current_tokens = [], [], 0

    for paragraph in paragraphs:
        para_tokens = count_tokens(paragraph)
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_tokens = [paragraph], para_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


# ---------- QDRANT UTILS ----------
def safe_import_qdrant():
    """Safely import Qdrant without crashing if it's missing."""
    try:
        qdrant_client = importlib.import_module("qdrant_client")
        qmodels = importlib.import_module("qdrant_client.http.models")
        return qdrant_client.QdrantClient, qmodels
    except ImportError:
        logging.warning("Qdrant client not installed. Qdrant features will be disabled.")
        return None, None
    

def get_qdrant_client():
    QdrantClient, _ = safe_import_qdrant()
    if not QdrantClient:
        return None
    try:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    except Exception as e:
        logging.warning(f"Qdrant connection failed: {e}")
        return None
def get_existing_collections():
    client = get_qdrant_client()
    if not client:
        return []
    try:
        return [c.name for c in client.get_collections().collections]
    except Exception as e:
        logging.error(f"Error fetching collections: {e}")
        return []
    
def get_collection_vector_size(client, collection_name):
    try:
        info = client.get_collection(collection_name)
        # Qdrant API 1.7+ stores dimension in vectors_config
        if hasattr(info, "config") and hasattr(info.config, "params"):
            return info.config.params.size
        elif hasattr(info, "vectors_config"):
            if isinstance(info.vectors_config, dict):
                # Single vector config
                return list(info.vectors_config.values())[0].size
            return info.vectors_config.size
    except Exception:
        pass
    return None

def ensure_collection_exists(client, collection_name, vector_size):
    _, qmodels = safe_import_qdrant()
    existing = get_existing_collections()
    if not client:
        return
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )
        logging.info(f"✅ Created new collection '{collection_name}' with dim={vector_size}")
    else:
        try:
            # Optional: validate the dimension
            info = client.get_collection(collection_name)
            existing_dim = info.vectors_count if hasattr(info, 'vectors_count') else None
            if existing_dim and existing_dim != vector_size:
                logging.warning(
                    f"⚠ Dimension mismatch for '{collection_name}'. "
                    f"Existing={existing_dim}, Trying={vector_size}"
                )
                # Recreate the collection with correct dimension
                client.delete_collection(collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
                )
                logging.info(f"✅ Recreated collection '{collection_name}' with dim={vector_size}")
        except Exception as e:
            logging.error(f"Collection validation failed: {e}")



# ---------- STORE CHUNKS ----------
def store_chunks_in_qdrant(chunks, collection_name):
    _, qmodels = safe_import_qdrant()
    client = get_qdrant_client()
    if not client:
        logging.warning("Qdrant client not available.")
        return False
    
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks).tolist()
        vector_size = len(embeddings[0])
        ensure_collection_exists(client, collection_name, vector_size)

        try:
            existing_count = client.count(collection_name=collection_name).count
            start_id = existing_count
        except Exception:
            start_id = 0

        points = [
            qmodels.PointStruct(
                id=start_id + i + 1,
                vector=embeddings[i],
                payload={"text": chunks[i]}
            )
            for i in range(len(chunks))
        ]

        client.upsert(collection_name=collection_name, points=points)
        logging.info(f"✅ Stored {len(chunks)} chunks in '{collection_name}'.")
        return True
    except Exception as e:
        logging.error(f"Error storing chunks: {e}")
        return False

# ---------- Fetch Chunks ----------
def fetch_chunks_from_collection(collection_name, batch_size=100):
    client = get_qdrant_client()
    if not client:
        return []
    all_chunks = []
    offset = None

    try:
        while True:
            result, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                with_payload=True,
                offset=offset
            )
            if not result:
                break
            all_chunks.extend([p.payload.get("text", "") for p in result])
            if offset is None:
                break

        logging.info(f"✅ Fetched {len(all_chunks)} chunks from '{collection_name}'.")
        return all_chunks

    except Exception as e:
        logging.error(f"Error fetching chunks from {collection_name}: {e}")
        return []

# ---------- MAIN VIEW ----------
def database_workflow(request):
    try:
        existing_collections = get_existing_collections()
    except Exception:
        existing_collections = []

    # Warn users if setup not ready
    setup_missing = []
    if not DEFAULT_HF_API_KEY:
        setup_missing.append("Hugging Face API key not configured.")
    if not get_qdrant_client():
        setup_missing.append("Qdrant is not running or unreachable.")

    if setup_missing:
        return render(request, "database_chunks.html", {
            "warning": "⚠ Setup incomplete. " + " ".join(setup_missing),
            "existing_collections": existing_collections
        })

    # Handle AJAX chunk fetch
    if request.method == "GET" and request.GET.get("collection_name"):
        collection_name = request.GET.get("collection_name")
        chunks = fetch_chunks_from_collection(collection_name)
        return JsonResponse({"chunks": chunks})

    # Handle chunk generation and storage
    if request.method == "POST":
        selected_document_ids = request.POST.getlist('selected_documents')

        if not selected_document_ids:
            return render(request, "database_chunks.html", {
                "error": "You must select at least one document to proceed.",
                "existing_collections": existing_collections,
            })

        documents = ScrapedData.objects.filter(id__in=selected_document_ids)
        combined_text = "\n\n".join([doc.content for doc in documents])
        text_chunks = split_text(combined_text, max_tokens=1000)
        total_chunks = len(text_chunks)

        new_collection = request.POST.get("new_collection_name", "").strip()
        selected_collection = request.POST.get("collection_name", "").strip()
        collection_name = new_collection if new_collection else selected_collection

        if not collection_name:
            return render(request, "database_chunks.html", {
                "documents": documents,
                "total_chunks": total_chunks,
                "existing_collections": get_existing_collections(),
                "error": "Please select or enter a collection name.",
                "selected_document_ids": selected_document_ids
            })

        success = store_chunks_in_qdrant(text_chunks, collection_name)

        if not success:
            return render(request, "database_chunks.html", {
                "error": f"⚠ Failed to store chunks in '{collection_name}'. Ensure Qdrant is running.",
                "existing_collections": existing_collections,
            })

        return render(request, "database_chunks.html", {
            "documents": documents,
            "total_chunks": total_chunks,
            "existing_collections": get_existing_collections(),
            "selected_document_ids": selected_document_ids,
            "success": f"Stored {total_chunks} chunks in Qdrant collection '{collection_name}'."
        })

    return render(request, "database_chunks.html", {
        "existing_collections": get_existing_collections()
    })
