# app_name/utils/qdrant_utils.py
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
from typing import List, Dict

logger = logging.getLogger(__name__)

# choose model — all-MiniLM-L6 is small and fast
_EMBEDDING_MODEL_NAME = os.environ.get("QA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "qa_chunks")

# Lazy singletons
_embedding_model = None
_qdrant_client = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {_EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_qdrant_client(host: str = "localhost", port: int = 6333, prefer_grpc: bool = False):
    global _qdrant_client
    if _qdrant_client is None:
        url = f"http://{host}:{port}"
        logger.info(f"Connecting to Qdrant at {url}")
        # QdrantClient autodetects; use http
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client

def ensure_collection(collection_name: str = _DEFAULT_COLLECTION, vector_size: int = 384, distance: str = "Cosine"):
    """
    Create collection if not exists.
    vector_size defaults to 384 (for all-MiniLM-L6-v2).
    distance can be "Cosine" | "Euclid" | "Dot".
    """
    client = get_qdrant_client()
    existing = client.get_collections()
    if any(col['name'] == collection_name for col in existing['collections']):
        logger.info(f"Collection '{collection_name}' already exists.")
        return

    logger.info(f"Creating collection '{collection_name}' vector_size={vector_size}")
    metric = qmodels.Distance.COSINE if distance.lower() == "cosine" else (
        qmodels.Distance.EUCLID if distance.lower() == "euclid" else qmodels.Distance.DOT
    )
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=metric)
    )
    logger.info("Collection created.")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for a list of texts and return list of vectors (python lists).
    """
    model = get_embedding_model()
    # model.encode returns numpy arrays
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize? Qdrant distance=Cosine works fine with non-normalized vectors; optional normalization:
    # norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # embeddings = embeddings / (norms + 1e-12)
    return embeddings.tolist()

def upsert_qa_items(items: List[Dict], collection_name: str = _DEFAULT_COLLECTION):
    """
    items: list of dicts, each must contain at least 'question' and 'answer'.
    Optional metadata: 'source', 'doc_id', etc.
    This will upsert points with uuid ids and metadata.
    """
    if not items:
        logger.info("No items to upsert.")
        return {"upserted": 0}

    client = get_qdrant_client()
    # ensure collection exists with appropriate vector size
    # compute one embedding to get size if needed
    sample = items[0]
    concat_texts = [f"Q: {it.get('question','')} A: {it.get('answer','')}" for it in items]
    vectors = embed_texts(concat_texts)
    vector_size = len(vectors[0])
    ensure_collection(collection_name=collection_name, vector_size=vector_size, distance="Cosine")

    # prepare payload
    point_ids = []
    points = []
    for idx, (vec, it) in enumerate(zip(vectors, items)):
        point_id = str(uuid.uuid4())
        metadata = it.copy()
        # remove vector-sized fields if present
        metadata.pop("vector", None)
        points.append(qmodels.PointStruct(id=point_id, vector=vec, payload=metadata))
        point_ids.append(point_id)

    # Upsert
    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Upserted {len(points)} points to collection '{collection_name}'")
    return {"upserted": len(points), "point_ids": point_ids}

def search_similar(query: str, top_k: int = 5, collection_name: str = _DEFAULT_COLLECTION):
    """
    Return results as list of dicts with score and payload.
    """
    client = get_qdrant_client()
    q_vector = embed_texts([query])[0]
    hits = client.search(collection_name=collection_name, query_vector=q_vector, limit=top_k)
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        })
    return results
