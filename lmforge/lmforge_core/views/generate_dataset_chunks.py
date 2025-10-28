from django.shortcuts import render, redirect
from ..forms.forms import DocumentForm
from ..models.scraped_data import ScrapedData
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.utils import LocalTokenNotFoundError
from huggingface_hub import login
from decouple import config
import torch
import logging
import os

# ---------- LOAD SETTINGS FROM .env ----------
QDRANT_HOST = config("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(config("QDRANT_PORT", default=6333))
LOG_FILE = config("QDRANT_LOG_FILE", default="application.log")

# ---------- LOGGING SETUP ----------
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
DEFAULT_HF_API_KEY = config("HF_API_KEY", default="")

def get_tokenizer():
    if not DEFAULT_HF_API_KEY:
        logging.warning("No Hugging Face token found. Skipping login and model loading.")
        return None, None, device

    try:
        login(token=DEFAULT_HF_API_KEY)

        os.environ["HF_HOME"] = "D:/huggingface_cache" 
        os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"
        logging.info("Hugging Face environment variables set.")

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        model.to(device)
        return model, tokenizer, device

    except LocalTokenNotFoundError:
        logging.error("Hugging Face token missing. Please configure HUGGINGFACE_TOKEN in .env.")
        return None, None, device
    except Exception as e:
        logging.error(f"Unexpected error while loading model: {e}")
        return None, None, device


def count_tokens(text):
    model, tokenizer, _ = get_tokenizer()
    if not tokenizer:
        logging.warning("No tokenizer available. Returning approximate token count.")
        return len(text.split())  # fallback: rough word count
    return len(tokenizer.encode(text))

def split_text(text, max_tokens=1000):
    """Split text into chunks based on token size."""
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
def get_qdrant_client():
    """Connect to local Qdrant instance."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_existing_collections():
    """Return list of existing Qdrant collections."""
    client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]

def store_chunks_in_qdrant(chunks, collection_name):
    """Store generated chunks in Qdrant (append if collection exists)."""
    client = get_qdrant_client()
    existing_collections = get_existing_collections()

    # Create new collection if not exists
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)
        )
        logging.info(f"Created new Qdrant collection: {collection_name}")
    else:
        logging.info(f"Appending to existing Qdrant collection: {collection_name}")

    # Generate embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks).tolist()

    # Get last ID offset (to avoid overwriting)
    existing_points = client.scroll(collection_name=collection_name, limit=1, with_payload=False)
    start_id = len(existing_points[1]) if existing_points[1] else 0

    points = [
        qmodels.PointStruct(
            id=start_id + i + 1,
            vector=embeddings[i],
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)
    logging.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")


# ---------- MAIN VIEW ----------
def database_workflow(request):
    """Extract text from selected documents, chunk, and store in user-selected Qdrant collection."""
    selected_document_ids = request.POST.getlist('selected_documents')

    if not selected_document_ids:
        request.session["redirect_message"] = "You must select at least one document to proceed."
        return redirect('dataset-workflow')

    documents = ScrapedData.objects.filter(id__in=selected_document_ids)
    combined_text = "\n\n".join([doc.content for doc in documents])
    text_chunks = split_text(combined_text, max_tokens=1000)
    total_chunks = len(text_chunks)

    client = get_qdrant_client()
    existing_collections = get_existing_collections()

    if request.method == "POST":
        collection_name = request.POST.get("collection_name", "").strip()
        if not collection_name:
            return render(request, "document_detail.html", {
                "documents": documents,
                "total_chunks": total_chunks,
                "existing_collections": existing_collections,
                "error": "Please enter or select a collection name."
            })

        store_chunks_in_qdrant(text_chunks, collection_name)
        logging.info(f"✅ Successfully stored {total_chunks} chunks in collection: {collection_name}")

        return render(request, "document_detail.html", {
            "documents": documents,
            "total_chunks": total_chunks,
            "qdrant_collection": collection_name,
            "existing_collections": existing_collections,
            "success": f"Stored {total_chunks} chunks in Qdrant collection '{collection_name}'."
        })

    return render(request, "document_detail.html", {
        "documents": documents,
        "total_chunks": total_chunks,
        "existing_collections": existing_collections
    })
