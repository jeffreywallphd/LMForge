import json
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
import logging
from huggingface_hub import HfApi
from decouple import config

# ---------- Safe Import for Optional Qdrant ----------
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ModuleNotFoundError:
    QdrantClient = None
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not installed. Qdrant features will be disabled.")

# ---------- Configuration ----------
DEFAULT_HF_KEY = config("HF_API_KEY", default="")
DEFAULT_HF_ACCOUNT = config("HF_ACCOUNT_NAME", default=None)
QDRANT_HOST = config("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(config("QDRANT_PORT", default=6333))

logger = logging.getLogger(__name__)

def home_view(request):
    models=[]
    datasets=[]
    messages=[]
    collections=[]

# ---------- HuggingFace Integration ----------

    if DEFAULT_HF_ACCOUNT and DEFAULT_HF_KEY:
        try:
            hf = HfApi(token=DEFAULT_HF_KEY)

            HFmodels = hf.list_models(author=DEFAULT_HF_ACCOUNT)
            for model in HFmodels:
                models.append(model)

            HFdatasets = hf.list_datasets(author=DEFAULT_HF_ACCOUNT)
            for dataset in HFdatasets:
                datasets.append(dataset)

        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            messages.append(
                "⚠ Failed to fetch HuggingFace data. Please verify your API key."
            )
    else:
        messages.append(
            "ℹ HuggingFace is not configured. "
            "Add your HuggingFace account name and API key in Settings."
        )

# ---------- Handle Collection Deletion ----------
    if request.method == "POST":
        if not QDRANT_AVAILABLE:
            return JsonResponse({"error": "Qdrant client not installed."}, status=400)
        try:
            data = json.loads(request.body.decode())
            collection_name = data.get("collection_name")
            if collection_name:
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                client.delete_collection(collection_name)
                return JsonResponse({"success": True})
        except Exception as e:
            logging.error(f"Failed to delete collection: {e}")
            return JsonResponse({"error": str(e)}, status=500)
    
# ----- Qdrant collections -----
    if QDRANT_AVAILABLE:
            try:
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                all_collections = client.get_collections().collections

                for col in all_collections:
                    name = col.name
                    try:
                        info = client.get_collection(name)
                        count = getattr(info, "points_count", 0)
                    except Exception:
                        count = 0

                    collections.append({
                        "name": name,
                        "count": count
                    })

            except Exception as e:
                logging.error(f"Failed to connect to Qdrant: {e}")
                messages.append("⚠ Could not fetch Qdrant collections. Make sure Qdrant is running.")
    else:
        messages.append("ℹ Qdrant not installed. Vector database features are disabled.")

    # ---------- Render ----------
    return render(request, 'home.html', {
        "messages": messages,
        "models": models,
        "datasets": datasets,
        "collections": collections
    })