import os
import uuid
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from chromadb import PersistentClient

try:  # chromadb <0.5 exposes HttpClient for remote servers
    from chromadb import HttpClient
except ImportError:  # Fallback if HttpClient unavailable
    HttpClient = None  # type: ignore


try:
    from llama_cpp import Llama
except Exception as _e:  # Defer import errors until runtime usage
    Llama = None  # type: ignore


# Defaults can be overridden via environment variables
HF_REPO_ID = os.environ.get("BGE_GGUF_REPO", "CompendiumLabs/bge-base-en-v1.5-gguf")
# Pick a common quant filename by default; can be changed via env
HF_FILENAME = os.environ.get("BGE_GGUF_FILENAME", "bge-base-en-v1.5-q8_0.gguf")
HF_CACHE = os.environ.get("HF_HOME", "/app/models")

VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "/app/data/vectordb")
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "documents")
CHROMA_SERVER_HOST = os.environ.get("CHROMA_SERVER_HOST")
CHROMA_SERVER_PORT = int(os.environ.get("CHROMA_SERVER_PORT", "8000"))
CHROMA_SERVER_SSL = os.environ.get("CHROMA_SERVER_SSL", "false").lower() in {
    "1",
    "true",
    "yes",
}


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
                data = (
                    res["data"] if isinstance(res, dict) else getattr(res, "data", [])
                )
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
                    data = (
                        res["data"]
                        if isinstance(res, dict)
                        else getattr(res, "data", [])
                    )
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
_client: Optional[Any] = None
_collection = None


def _get_embedder() -> GGUFEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = GGUFEmbedder()
    return _embedder


def _get_collection():
    global _client, _collection
    if _client is None:
        if CHROMA_SERVER_HOST:
            if HttpClient is None:
                raise RuntimeError(
                    "chromadb.HttpClient not available; ensure chromadb>=0.4.22 is installed"
                )
            _client = HttpClient(
                host=CHROMA_SERVER_HOST,
                port=CHROMA_SERVER_PORT,
                ssl=CHROMA_SERVER_SSL,
            )
        else:
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            _client = PersistentClient(path=VECTOR_DB_PATH)
    if _collection is None:
        # Use cosine space for standard text embeddings
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def fill(
    data: str,
    *,
    business_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
) -> str:
    """Embed and store a single data chunk in the vector DB.

    Required metadata field: business_id (string). It is injected into the row's
    metadata and enforced to match the provided value.

    If an explicit id is provided and an existing record is found, updates its
    document, embedding, and metadata (after validating business ownership).
    When metadata includes a "field" string, the embedding text is augmented
    with that field name ("field\ndata") so downstream retrieval can reason on
    both the business_id and the structured field label. If no id is provided
    but a document already exists with the same business_id and field, that
    record is updated in-place. Otherwise inserts a new row, allowing multiple
    distinct fields per business. Returns the stored or updated id.
    """
    if not isinstance(data, str) or not data.strip():
        raise ValueError("data must be a non-empty string")
    if not isinstance(business_id, str) or not business_id.strip():
        raise ValueError("business_id must be a non-empty string")

    embedder = _get_embedder()
    collection = _get_collection()

    # Prepare metadata and enforce business_id
    meta: Dict[str, Any] = dict(metadata or {})
    if "business_id" in meta and meta.get("business_id") != business_id:
        raise ValueError("metadata.business_id does not match provided business_id")
    meta["business_id"] = business_id

    field_name = meta.get("field") if isinstance(meta, dict) else None
    embed_text = data
    normalized_field: Optional[str] = None
    if isinstance(field_name, str) and field_name.strip():
        normalized_field = field_name.strip()
        meta["field"] = normalized_field
        embed_text = f"{normalized_field}\n{data}"
    else:
        normalized_field = None
        meta.pop("field", None)

    vector = embedder.embed([embed_text])[0]
    document_text = embed_text

    doc_id = id

    if doc_id:
        try:
            existing = collection.get(ids=[doc_id], include=["metadatas"])  # type: ignore
        except Exception:
            existing = {"ids": [], "metadatas": []}

        existing_ids: List[str] = list(existing.get("ids", []) or [])
        if existing_ids:
            metas_list = existing.get("metadatas", []) or []
            base_meta = (
                metas_list[0]
                if len(metas_list) > 0 and isinstance(metas_list[0], dict)
                else {}
            )
            existing_business_id = (
                base_meta.get("business_id") if isinstance(base_meta, dict) else None
            )
            if existing_business_id and existing_business_id != business_id:
                raise ValueError(
                    "existing record business_id does not match provided business_id"
                )

            updated_meta: Dict[str, Any] = dict(base_meta)
            updated_meta.update(meta)
            updated_meta["business_id"] = business_id

            collection.update(
                ids=[doc_id],
                documents=[document_text],
                embeddings=[vector],
                metadatas=[updated_meta],
            )
            return doc_id

    if doc_id is None and normalized_field:
        try:
            existing = collection.get(  # type: ignore
                where={"business_id": business_id, "field": normalized_field},
                include=["metadatas"],
            )
        except Exception:
            existing = {"ids": [], "metadatas": []}

        existing_ids = list(existing.get("ids", []) or [])
        if existing_ids:
            doc_id = existing_ids[0]
            metas_list = existing.get("metadatas", []) or []
            base_meta = (
                metas_list[0]
                if len(metas_list) > 0 and isinstance(metas_list[0], dict)
                else {}
            )

            updated_meta = dict(base_meta)
            updated_meta.update(meta)
            updated_meta["business_id"] = business_id

            collection.update(
                ids=[doc_id],
                documents=[document_text],
                embeddings=[vector],
                metadatas=[updated_meta],
            )
            return doc_id

    doc_id = doc_id or f"doc-{uuid.uuid4()}"
    collection.add(
        ids=[doc_id],
        documents=[document_text],
        embeddings=[vector],
        metadatas=[meta],
    )
    return doc_id


def retrieve(
    query: str,
    *,
    business_id: str,
    top_k: int = 5,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Embed the query and return the top_k most similar rows for a business.

    Required argument: business_id (string). Results are filtered by metadata.business_id.
    Optional fields list allows further filtering by metadata.field values.

    Similarity uses cosine distance via Chroma. Each result contains:
    {id, document, metadata, distance, similarity}
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(business_id, str) or not business_id.strip():
        raise ValueError("business_id must be a non-empty string")

    embedder = _get_embedder()
    collection = _get_collection()

    qvec = embedder.embed([query])[0]
    where: Dict[str, Any]
    base_filter: Dict[str, Any] = {"business_id": business_id}
    field_filter: Optional[Any] = None
    if fields:
        normalized_fields = []
        seen: set[str] = set()
        for f in fields:
            if isinstance(f, str):
                stripped = f.strip()
                if stripped and stripped not in seen:
                    normalized_fields.append(stripped)
                    seen.add(stripped)
        if normalized_fields:
            if len(normalized_fields) == 1:
                field_filter = {"field": normalized_fields[0]}
            else:
                field_filter = {"field": {"$in": normalized_fields}}

    if field_filter:
        where = {"$and": [base_filter, field_filter]}
    else:
        where = base_filter

    res = collection.query(
        query_embeddings=[qvec],
        n_results=int(max(1, top_k)),
        include=["documents", "metadatas", "distances", "embeddings"],
        where=where,
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        dist = (
            float(dists[i]) if i < len(dists) and dists[i] is not None else None
        )  # Best distance cosine is near to 0
        sim = (
            (1.0 - dist) if dist is not None else None
        )  # Transform it to a realistic value (ex: distance = 0, convert it to 1)
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


def list_all(*, include_embeddings: bool = False) -> List[Dict[str, Any]]:
    """Return every stored row in the collection."""
    collection = _get_collection()
    include_fields: List[str] = ["documents", "metadatas"]
    if include_embeddings:
        include_fields.append("embeddings")

    res = collection.get(include=include_fields)  # type: ignore
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    embeds = res.get("embeddings", []) if include_embeddings else []

    out: List[Dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        item: Dict[str, Any] = {
            "id": doc_id,
            "document": docs[i] if i < len(docs) else None,
            "metadata": metas[i] if i < len(metas) else None,
        }
        if include_embeddings and i < len(embeds):
            item["embedding"] = embeds[i]
        out.append(item)
    return out


def list_business_records(
    business_id: str, *, include_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """Return all vector rows for a specific business."""
    if not isinstance(business_id, str) or not business_id.strip():
        raise ValueError("business_id must be a non-empty string")

    collection = _get_collection()
    include_fields: List[str] = ["documents", "metadatas"]
    if include_embeddings:
        include_fields.append("embeddings")

    res = collection.get(  # type: ignore
        where={"business_id": business_id},
        include=include_fields,
    )

    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    embeds = res.get("embeddings", []) if include_embeddings else []

    out: List[Dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        item: Dict[str, Any] = {
            "id": doc_id,
            "document": docs[i] if i < len(docs) else None,
            "metadata": metas[i] if i < len(metas) else None,
        }
        if include_embeddings and i < len(embeds):
            item["embedding"] = embeds[i]
        out.append(item)
    return out


def delete(business_id: str) -> bool:
    """Delete vectorDB records for a specific biz
    Required argument: business_id (string).
    """
    collection = _get_collection()

    try:
        collection.delete(where={"business_id": business_id})
        return True
    except Exception as e:
        print("[vectodb] Deletion failed for business_id %s", business_id, e)
        return False


