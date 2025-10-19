import os
import uuid
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from chromadb import PersistentClient


try:
    from llama_cpp import Llama
except Exception as _e:  # Defer import errors until runtime usage
    Llama = None  # type: ignore


# Defaults can be overridden via environment variables
HF_REPO_ID = os.environ.get(
    "BGE_GGUF_REPO", "CompendiumLabs/bge-base-en-v1.5-gguf"
)
# Pick a common quant filename by default; can be changed via env
HF_FILENAME = os.environ.get(
    "BGE_GGUF_FILENAME", "bge-base-en-v1.5-q8_0.gguf"
)
HF_CACHE = os.environ.get("HF_HOME", "/app/models")

VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "/app/data/vectordb")
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "documents")


class GGUFEmbedder:
    """Embeds text using a local GGUF model via llama.cpp."""

    def __init__(
        self,
        repo_id: str = HF_REPO_ID,
        filename: str = HF_FILENAME,
        cache_dir: str = HF_CACHE,
        n_ctx: int = 0,
        n_gpu_layers: int = 0,
    ) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python not available. Install it and rebuild the image."
            )

        self.model_path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=cache_dir
        )
        # embedding=True enables the embedding API
        self.engine = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Try the OpenAI-like interface first
        if hasattr(self.engine, "create_embedding"):
            try:
                res = self.engine.create_embedding(input=texts)  # type: ignore
                data = res["data"] if isinstance(res, dict) else getattr(res, "data", [])
                if data:
                    return [row["embedding"] for row in data]
            except Exception:
                pass

        # Fallback: call per-input using the same interface
        vectors: List[List[float]] = []
        for t in texts:
            vector = None
            if hasattr(self.engine, "create_embedding"):
                try:
                    res = self.engine.create_embedding(input=[t])  # type: ignore
                    data = res["data"] if isinstance(res, dict) else getattr(res, "data", [])
                    if data:
                        vector = data[0]["embedding"]
                except Exception:
                    vector = None

            # Some versions expose a direct embed() helper
            if vector is None and hasattr(self.engine, "embed"):
                try:
                    vector = self.engine.embed(t)  # type: ignore
                except Exception:
                    vector = None

            if vector is None:
                raise RuntimeError("Failed to create embeddings via llama.cpp API")
            vectors.append(vector)

        return vectors


_embedder: Optional[GGUFEmbedder] = None
_client: Optional[PersistentClient] = None
_collection = None


def _get_embedder() -> GGUFEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = GGUFEmbedder()
    return _embedder


def _get_collection():
    global _client, _collection
    if _client is None:
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        _client = PersistentClient(path=VECTOR_DB_PATH)
    if _collection is None:
        # Use cosine space for standard text embeddings
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def fill(data: str, *, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> str:
    """Embed and store a single data chunk in the vector DB.

    Stores a row with fields: {id, data (as document), embedding, metadata}.
    Returns the stored id.
    """
    if not isinstance(data, str) or not data.strip():
        raise ValueError("data must be a non-empty string")

    embedder = _get_embedder()
    collection = _get_collection()

    vector = embedder.embed([data])[0]
    doc_id = id or f"doc-{uuid.uuid4()}"

    collection.add(
        ids=[doc_id],
        documents=[data],
        embeddings=[vector],
        metadatas=[metadata or {}],
    )
    return doc_id


def retrieve(query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed the query and return the top_k most similar rows.

    Similarity uses cosine distance via Chroma. Each result contains:
    {id, document, metadata, distance, similarity}
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    embedder = _get_embedder()
    collection = _get_collection()

    qvec = embedder.embed([query])[0]
    res = collection.query(
        query_embeddings=[qvec], n_results=int(max(1, top_k)), include=["documents", "metadatas", "distances", "embeddings"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else None # Best distance cosine is near to 0
        sim = (1.0 - dist) if dist is not None else None # Transform it to a realistic value (ex: distance = 0, convert it to 1)
        out.append(
            {
                "id": ids[i],
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else None,
                "distance": dist,
                "similarity": sim,
            }
        )

    return out
