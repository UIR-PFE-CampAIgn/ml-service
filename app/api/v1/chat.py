import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple

from app.ml.intent import IntentPredictor
from app.ml.rag.retrieval import retrieve, fill
from huggingface_hub import hf_hub_download
from app.core.logging import api_logger

try:
    from llama_cpp import Llama  # optional local GGUF generator
except Exception:
    Llama = None  # type: ignore


router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., description="User chat message/question")
    context_limit: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    min_confidence: float = Field(0.4, ge=0.0, le=1.0, description="Confidence threshold for intent gating")
    chat_id: Optional[str] = Field(default=None, description="Optional chat identifier")
    timestamp: Optional[str] = Field(default=None, description="Optional ISO timestamp of the message")


# --- Optional local LLM via llama.cpp (GGUF) ----------------------------------------------------
_llm_engine = None


def _get_llm_engine() -> Tuple[Optional[object], str]:
    """Return cached llama.cpp engine if configured via env.

    Requires:
    - LLM_GGUF_REPO (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
    - LLM_GGUF_FILENAME (e.g., "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    Optional:
    - HF_HOME (/app/models default)
    - LLM_N_CTX (default 4096), LLM_N_GPU_LAYERS (default 0)
    """
    global _llm_engine
    if _llm_engine is not None:
        api_logger.debug("chat.llm: reusing cached engine")
        return _llm_engine, "ok"
    if Llama is None:
        api_logger.error("chat.llm: llama_cpp import missing")
        return None, "llama_cpp_unavailable"

    repo = os.environ.get("LLM_GGUF_REPO")
    fn = os.environ.get("LLM_GGUF_FILENAME")
    cache_dir = os.environ.get("HF_HOME", "/app/models")
    override_path = os.environ.get("LLM_MODEL_PATH")  # bypass HF download if set
    api_logger.info(
        "chat.llm: init with repo=%s file=%s cache_dir=%s override=%s",
        repo, fn, cache_dir, bool(override_path),
    )
    if not repo or not fn:
        api_logger.warning("chat.llm: env not configured (LLM_GGUF_REPO/LLM_GGUF_FILENAME)")
        return None, "llm_not_configured"

    try:
        if override_path and os.path.isfile(override_path):
            model_path = override_path
            api_logger.info("chat.llm: using override model path: %s", model_path)
        else:
            model_path = hf_hub_download(
                repo_id=repo, filename=fn, local_dir=cache_dir
            )
            api_logger.info("chat.llm: downloaded/cached model at: %s", model_path)

        try:
            size = os.path.getsize(model_path)
        except Exception:
            size = -1
        api_logger.info("chat.llm: model file exists=%s size=%s", os.path.isfile(model_path), size)

        n_ctx = int(os.environ.get("LLM_N_CTX", "4096"))
        n_gpu_layers = int(os.environ.get("LLM_N_GPU_LAYERS", "0"))
        api_logger.info("chat.llm: creating engine n_ctx=%d n_gpu_layers=%d", n_ctx, n_gpu_layers)
        engine = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        _llm_engine = engine
        api_logger.info("chat.llm: engine initialized successfully")
        return engine, "ok"
    except Exception as e:
        api_logger.exception("chat.llm: init failed: %s", e)
        return None, "llm_init_failed"


def _format_context(snippets: List[Dict[str, Any]], limit: int) -> str:
    take = max(0, min(limit, len(snippets)))
    parts: List[str] = []
    for i in range(take):
        s = snippets[i]
        meta = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
        src = meta.get("source")
        parts.append(f"[#{i+1}{' '+str(src) if src else ''}]\n{s.get('document','')}")
    return "\n\n".join(parts)


class ChatChunk(BaseModel):
    event: str  # intent_info | token | complete | error
    content: str = ""
    intent: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None
    low_confidence: Optional[bool] = None
    is_complete: bool = False
    chat_id: Optional[str] = None


class FeedRequest(BaseModel):
    content: str = Field(..., description="Text content to store in the vector DB")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the content")
    id: Optional[str] = Field(default=None, description="Optional custom ID for the stored record")


class FeedResponse(BaseModel):
    id: str
    status: str = "stored"


@router.post("/intent_answer")
async def intent_answer(request: ChatRequest) -> StreamingResponse:
    """
    Predict intent, retrieve relevant chunks, optionally call a local LLM, and send final answer.

    Stream events:
    - intent_info: includes intent, confidence, and top_predictions
    - complete: includes final answer and marks end of stream
    - error: emitted if any failure occurs
    """

    async def generate() -> AsyncGenerator[str, None]:
        userQuery = request.query
        
        try:
            predictor = IntentPredictor()
            pred = await predictor.predict(userQuery)
            api_logger.info(
                "chat.intent: predicted intent=%s conf=%.3f chat_id=%s",
                pred.get("intent"), float(pred.get("confidence", 0.0)), request.chat_id,
            )

            low_conf = float(pred.get("confidence", 0.0)) < request.min_confidence

            # Emit intent info first
            intent_event = ChatChunk(
                event="intent_info",
                intent=pred.get("intent"),
                confidence=float(pred.get("confidence", 0.0)),
                top_predictions=pred.get("top_predictions", []),
                low_confidence=low_conf,
                chat_id=request.chat_id,
            )
            yield f"data: {intent_event.model_dump_json()}\n\n"
            
            api_logger.info(
                "chat.retrieve: querying vectordb top_k=%d", request.context_limit
            )
            relevantData = retrieve(userQuery, top_k=request.context_limit)
            api_logger.info(
                "chat.retrieve: got %d hits; first_id=%s",
                len(relevantData),
                (relevantData[0]["id"] if relevantData else None),
            )

            context_text = _format_context(relevantData, request.context_limit)

            # Generate answer
            answer: Optional[str] = None
            engine, status = _get_llm_engine()
            api_logger.info("chat.llm: engine_status=%s", status)
            if status == "llm_not_configured":
                api_logger.warning(
                    "chat.llm: not configured; set LLM_GGUF_REPO and LLM_GGUF_FILENAME env vars"
                )
            elif status == "llama_cpp_unavailable":
                api_logger.error(
                    "chat.llm: llama-cpp-python not installed or failed to import"
                )
            elif status == "llm_init_failed":
                api_logger.error("chat.llm: initialization failed (download/init)")
            if engine and status == "ok":
                prompt = (
                    "You are a helpful assistant. Use the provided context to answer the question.\n"
                    "If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {userQuery}\nAnswer:"
                )
                try:
                    if hasattr(engine, "create_chat_completion"):
                        comp = engine.create_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are a concise assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=float(request.temperature),
                            max_tokens=512,
                        )
                        answer = comp["choices"][0]["message"]["content"].strip()
                    else:
                        comp = engine.create_completion(
                            prompt=prompt,
                            temperature=float(request.temperature),
                            max_tokens=512,
                        )
                        answer = comp["choices"][0]["text"].strip()
                    api_logger.info(
                        "chat.llm: generated %d chars of answer", len(answer)
                    )
                except Exception as e:
                    api_logger.exception("chat.llm: generation failed: %s", e)
                    answer = None

            if not answer:
                preview = context_text[:800] + ("..." if len(context_text) > 800 else "")
                answer = (
                    "No LLM configured; returning top relevant snippets:\n\n" + preview
                )

            final = ChatChunk(event="complete", content=answer, is_complete=True, chat_id=request.chat_id)
            yield f"data: {final.model_dump_json()}\n\n"

        except Exception as e:
            api_logger.exception("chat.intent_answer: error: %s", e)
            err = ChatChunk(event="error", content=str(e))
            yield f"data: {err.model_dump_json()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/feed_vector", response_model=FeedResponse)
async def feed_vector(request: FeedRequest) -> FeedResponse:
    """Embed and store a single text in the vector database.

    Returns the stored document ID and status.
    """
    try:
        api_logger.info("feed_vector: storing content len=%d", len(request.content or ""))
        doc_id = fill(request.content, metadata=request.metadata, id=request.id)
        api_logger.info("feed_vector: stored id=%s", doc_id)
        return FeedResponse(id=doc_id, status="stored")
    except Exception as e:
        api_logger.exception("feed_vector: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Vector feed failed: {e}")
