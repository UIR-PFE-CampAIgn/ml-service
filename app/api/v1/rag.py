from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import json

from app.ml.rag import RAGChain

router = APIRouter()


class RAGRequest(BaseModel):
    query: str
    context_limit: int = 5
    temperature: float = 0.7


class RAGChunk(BaseModel):
    content: str
    is_complete: bool = False


@router.post("/rag_answer")
async def rag_answer(request: RAGRequest) -> StreamingResponse:
    """
    Generate streaming RAG answer using LangChain RAG chain.
    
    Args:
        request: RAG request with query, context limit, and temperature
        
    Returns:
        Streaming response with answer chunks
    """
    
    async def generate_response() -> AsyncGenerator[str, None]:
        rag_chain = RAGChain()
        
        async for chunk in rag_chain.stream_answer(
            query=request.query,
            context_limit=request.context_limit,
            temperature=request.temperature
        ):
            response_chunk = RAGChunk(
                content=chunk,
                is_complete=False
            )
            yield f"data: {response_chunk.model_dump_json()}\n\n"
        
        # Send completion marker
        final_chunk = RAGChunk(content="", is_complete=True)
        yield f"data: {final_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )