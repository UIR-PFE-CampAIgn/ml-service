import os
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from app.ml.score import ScorePredictor
from app.ml.intent import IntentPredictor
from app.ml.rag.retrieval import retrieve, fill
from huggingface_hub import hf_hub_download
from app.core.logging import api_logger
from app.clients.gateway import GatewayClient

try:
    from llama_cpp import Llama  # optional local GGUF generator
except Exception:
    Llama = None  # type: ignore


router = APIRouter()
gw = GatewayClient()

class ChatRequest(BaseModel):
    query: str = Field(..., description="User chat message/question")
    business_id: str = Field(..., description="Business identifier to filter context")
    context_limit: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    min_confidence: float = Field(0.4, ge=0.0, le=1.0, description="Confidence threshold for intent gating")
    chat_id: Optional[str] = Field(default=None, description="Optional chat identifier")
    lead_id:Optional[str]=Field(default=None,description="lead identifier")
    timestamp: Optional[str] = Field(default=None, description="Optional ISO timestamp of the message")
    messages_in_session: Optional[int] = Field(default=1, ge=1, description="Messages in current session")
    conversation_duration_minutes: Optional[float] = Field(default=1.0, ge=0.0, description="Duration in minutes")
    user_response_time_avg_seconds: Optional[float] = Field(default=60.0, ge=0.0, description="Avg response time")
    user_initiated_conversation: Optional[bool] = Field(default=False, description="Did user start chat?")
    is_returning_customer: Optional[bool] = Field(default=False, description="Is user returning?")
    time_of_day: Optional[str] = Field(default="business_hours", description="business_hours/off_hours")


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


def _build_enhanced_prompt(
    intent: str,
    lead_category: str,
    userQuery: str,
    context_text: str
) -> str:
    """Build an enhanced, adaptive prompt based on intent and lead score."""
    
    # === INTENT-SPECIFIC INSTRUCTIONS ===
    intent_templates = {
        "greeting": "Respond warmly and professionally in 1-2 sentences.",
        "question": "Answer directly with specific facts from context.",
        "complaint": "Acknowledge empathetically and provide solution from context.",
        "pricing": "State exact pricing from context with [#N] citations.",
        "feature_inquiry": "List features from context with citations.",
        "technical_support": "Provide step-by-step from context only.",
    }
    
    # === LEAD-BASED PRIORITY & LENGTH ===
    if lead_category == "hot":
        priority_instruction = "HIGH-VALUE LEAD: Be helpful and thorough, but cite every claim."
        max_tokens_guidance = "Maximum 4 sentences"
    elif lead_category == "warm":
        priority_instruction = "ENGAGED USER: Be informative with clear citations."
        max_tokens_guidance = "Maximum 3 sentences"
    else:
        priority_instruction = "Be concise and cite sources."
        max_tokens_guidance = "Maximum 2 sentences"
    
    intent_instruction = intent_templates.get(intent, "Answer factually from context only.")
    
    # === UNIFIED PROMPT WITH STRONGER CONSTRAINTS ===
    prompt = (
        f"You are a precise assistant that ONLY uses provided context.\n\n"
        
        f"=== CRITICAL RULES (VIOLATING = INCORRECT RESPONSE) ===\n"
        f"1. Read CONTEXT below carefully\n"
        f"2. Answer ONLY with facts from CONTEXT - add [#N] after each fact\n"
        f"3. If information is NOT in CONTEXT, respond ONLY: 'I don't have that information. Let me connect you with our sales team.'\n"
        f"4. NEVER add information from your training - CONTEXT ONLY\n"
        f"5. LENGTH: {max_tokens_guidance} maximum\n"
        f"6. Intent: {intent} - {intent_instruction}\n\n"
        
        f"=== CONTEXT (ONLY SOURCE) ===\n"
        f"{context_text if context_text.strip() else '[EMPTY - Use rule 3]'}\n\n"
        
        f"=== EXAMPLE (correct citation) ===\n"
        f"Q: Where is the company located?\n"
        f"A: San Francisco, CA [#1]\n\n"
        f"=== EXAMPLE (missing info) ===\n"
        f"Q: What are shipping times?\n"
        f"A: I don't have that information. Let me connect you with our sales team.\n\n"
        
        f"USER QUESTION: {userQuery}\n"
        f"YOUR ANSWER:"
    )
    
    return prompt


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
    business_id: str = Field(..., description="Business identifier to attach to the record")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the content")
    id: Optional[str] = Field(default=None, description="Optional custom ID for the stored record")


class FeedResponse(BaseModel):
    id: str
    status: str = "stored"


class ChatAnswerResponse(BaseModel):
    answer: str = ""
    model: Optional[str] = None
    confidence: Optional[float] = None
    


@router.post("/chat/answer", response_model=ChatAnswerResponse)
async def chat_answer(request: ChatRequest) -> ChatAnswerResponse:
    """
    Predict intent, retrieve context, generate answer, 
    AND compute lead score (cold/warm/hot).
    """
    userQuery = request.query

    try:
        # === 1. INTENT PREDICTION ===
        intent_predictor = IntentPredictor()
        pred = await intent_predictor.predict(userQuery)
        intent = pred.get("intent", "unknown")
        intent_conf = float(pred.get("confidence", 0.0))
        
        api_logger.info(
            "chat.intent: predicted intent=%s conf=%.3f chat_id=%s",
            intent, intent_conf, request.chat_id,
        )

        low_conf = intent_conf < request.min_confidence

        # === 2. RETRIEVAL ===
        api_logger.info("chat.retrieve: querying vectordb top_k=%d", request.context_limit)
        relevantData = retrieve(userQuery, business_id=request.business_id, top_k=request.context_limit)
        api_logger.info(
            "chat.retrieve: got %d hits; first_id=%s",
            len(relevantData),
            (relevantData[0]["id"] if relevantData else None),
        )
        context_text = _format_context(relevantData, request.context_limit)

        # === 3. LEAD SCORING ===
        score_result = {}
        try:
            # Extract behavioral features
            score_features = {
                "messages_in_session": request.messages_in_session or 1,
                "user_msg": userQuery,
                "conversation_duration_minutes": request.conversation_duration_minutes or 1.0,
                "user_response_time_avg_seconds": request.user_response_time_avg_seconds or 60.0,
                "user_initiated_conversation": request.user_initiated_conversation or False,
                "is_returning_customer": request.is_returning_customer or False,
                "time_of_day": request.time_of_day or "business_hours"
            }

            # Predict score
            score_predictor = ScorePredictor(model_type="xgboost")
            score_result = await score_predictor.predict(score_features)
            
            api_logger.info(
                "chat.score: category=%s score=%.3f chat_id=%s",
                score_result.get("category"), score_result.get("score"), request.chat_id
            )
             # ✅ Fire the webhook to NestJS
            lead_id = request.lead_id
            lead_category = score_result.get("category", "cold")
            asyncio.create_task(gw.update_lead_score(lead_id, lead_category))

        except Exception as e:
            api_logger.warning("chat.score: failed: %s", e)
            score_result = {"category": "unknown", "score": 0.0, "confidence": 0.0}

        # === 4. LLM ANSWER WITH ENHANCED PROMPT ===
        answer: str = ""
        model_name: str = "unknown"
        engine, status = _get_llm_engine()
        api_logger.info("chat.llm: engine_status=%s", status)

        if engine and status == "ok":
            # Build enhanced prompt with intent and lead awareness
            lead_category = score_result.get("category", "cold")
            prompt = _build_enhanced_prompt(
                intent=intent,
                lead_category=lead_category,
                userQuery=userQuery,
                context_text=context_text
            )
            
            api_logger.info(
                "chat.llm: using enhanced prompt for intent=%s lead=%s",
                intent, lead_category
            )
            
            try:
                if hasattr(engine, "create_chat_completion"):
                    comp = engine.create_chat_completion(
                        messages=[
                            {"role": "system", "content": "You are an expert support assistant."},
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
                api_logger.info("chat.llm: generated %d chars", len(answer))
            except Exception as e:
                api_logger.exception("chat.llm: generation failed: %s", e)
                answer = "Sorry, I couldn't generate an answer."

        # === 5. WEBHOOK (fire-and-forget) ===
        try:
            asyncio.create_task(
                gw.send_chat_webhook(
                    intent=intent,
                    reply=answer,
                    chat_id=request.chat_id,
                    metadata={
                        "low_confidence": low_conf,
                        "retrieved_count": len(relevantData),

                    },
                )
            )
            api_logger.info("chat.gateway: webhook scheduled")
        except Exception as e:
            api_logger.exception("chat.gateway: failed: %s", e)

        # === 6. RESPONSE ===
        model_name = os.environ.get("LLM_GGUF_FILENAME") or os.environ.get("LLM_GGUF_REPO") or "local-llm"

        return ChatAnswerResponse(
            answer=answer,
            model=model_name,
            confidence=intent_conf,
        )

    except Exception as e:
        api_logger.exception("chat.chat_answer: error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/feed_vector", response_model=FeedResponse)
async def feed_vector(request: FeedRequest) -> FeedResponse:
    """Embed and store a single text in the vector database.

    Returns the stored document ID and status.
    """
    try:
        api_logger.info("feed_vector: storing content len=%d", len(request.content or ""))
        doc_id = fill(request.content, business_id=request.business_id, metadata=request.metadata, id=request.id)
        api_logger.info("feed_vector: stored id=%s", doc_id)
        return FeedResponse(id=doc_id, status="stored")
    except Exception as e:
        api_logger.exception("feed_vector: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Vector feed failed: {e}")
