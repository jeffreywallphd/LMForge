import os
import logging

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaClient:
    """Simple wrapper around chromadb.Client for basic upsert/query operations.

    This wrapper keeps usage minimal and tolerant if chromadb isn't installed.
    """

    def __init__(self, persist_directory: str = None):
        if chromadb is None:
            logger.warning("chromadb package not available. Chroma operations will fail.")
            self._client = None
            self._collection = None
            return

        # Allow overriding the persist directory via env var
        persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR")

        try:
            if persist_directory:
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
                self._client = chromadb.Client(settings)
            else:
                # Default in-memory client
                self._client = chromadb.Client()
        except Exception as e:
            logger.error(f"Failed to create chromadb client: {e}")
            self._client = None

    def get_collection(self, name: str = "lmforge_collection", create_if_missing: bool = True):
        if not self._client:
            raise RuntimeError("Chromadb client is not initialized")

        try:
            if create_if_missing:
                return self._client.get_or_create_collection(name)
            return self._client.get_collection(name)
        except Exception as e:
            logger.error(f"Error getting chroma collection '{name}': {e}")
            return None

    def upsert(self, collection_name: str, ids, embeddings, metadatas=None, documents=None):
        """Upsert documents into a Chroma collection.

        ids: list of ids
        embeddings: list of embedding vectors
        metadatas: list of metadata dicts
        documents: list of text documents
        """
        if not self._client:
            raise RuntimeError("Chromadb client is not initialized")

        coll = self.get_collection(collection_name)
        if coll is None:
            raise RuntimeError("Could not get/create collection")

        try:
            coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            return True
        except Exception as e:
            logger.error(f"Chroma upsert error: {e}")
            return False

    def query(self, collection_name: str, embedding, top_k: int = 5):
        """Query the collection and return a normalized result list.

        Returns a dict with 'ids', 'documents', 'metadatas', 'distances' lists (as returned by chroma).
        """
        if not self._client:
            logger.error("Chromadb client is not initialized")
            return None

        coll = self.get_collection(collection_name, create_if_missing=False)
        if coll is None:
            logger.error(f"Chroma collection {collection_name} not found")
            return None

        try:
            result = coll.query(embedding=embedding, n_results=top_k, include=['metadatas', 'documents', 'distances', 'ids'])
            return result
        except Exception as e:
            logger.error(f"Chroma query error: {e}")
            return None
