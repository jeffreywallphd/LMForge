from django.shortcuts import render
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

# ---------- LOAD SETTINGS ----------
QDRANT_HOST = config("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(config("QDRANT_PORT", default=6333))
LOG_FILE = config("QDRANT_LOG_FILE", default="application.log")

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
DEFAULT_HF_API_KEY = config("HF_API_KEY", default="")


# ---------- TOKENIZER ----------
def get_tokenizer():
    if not DEFAULT_HF_API_KEY:
        logging.warning("No Hugging Face token found.")
        return None, None, device

    try:
        login(token=DEFAULT_HF_API_KEY)
        os.environ["HF_HOME"] = "D:/huggingface_cache"
        os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"

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
        logging.error("Hugging Face token missing in .env.")
        return None, None, device
    except Exception as e:
        logging.error(f"Tokenizer error: {e}")
        return None, None, device


# ---------- TOKEN COUNT + CHUNKING ----------
def count_tokens(text):
    model, tokenizer, _ = get_tokenizer()
    if not tokenizer:
        return len(text.split())
    return len(tokenizer.encode(text))

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
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_existing_collections():
    client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]

def ensure_collection_exists(client, collection_name, vector_size):
    """Create the collection if it doesn't exist or validate its dimension."""
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )
        logging.info(f"✅ Created new collection '{collection_name}' with dim={vector_size}")
    else:
        # Optional: validate the dimension
        info = client.get_collection(collection_name)
        existing_dim = info.vectors_count if hasattr(info, 'vectors_count') else None
        if existing_dim and existing_dim != vector_size:
            logging.warning(
                f"⚠ Dimension mismatch for '{collection_name}'. "
                f"Existing={existing_dim}, Trying={vector_size}"
            )


def store_chunks_in_qdrant(chunks, collection_name):
    client = get_qdrant_client()
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


# ---------- MAIN VIEW ----------
def database_workflow(request):
    """Handle chunk generation and storage in Qdrant."""
    if request.method == "POST":
        selected_document_ids = request.POST.getlist('selected_documents')

        if not selected_document_ids:
            return render(request, "database_chunks.html", {
                "error": "You must select at least one document to proceed.",
                "existing_collections": get_existing_collections(),
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

        store_chunks_in_qdrant(text_chunks, collection_name)

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
