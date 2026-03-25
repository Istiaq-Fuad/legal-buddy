from fastapi import APIRouter, HTTPException
from api.api.models import LegalChatRequest, LegalChatResponse, SourceItem
from api.core.config import config
import logging
from api.agents.retrieval_generation import (
    retrieve_sources,
    build_grounded_prompt,
    run_llm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@rag_router.post("/legal/chat", response_model=LegalChatResponse)
async def legal_chat(payload: LegalChatRequest) -> LegalChatResponse:
    try:
        max_tokens = payload.max_tokens or config.ANSWER_MAX_TOKENS
        top_k = payload.top_k or config.RETRIEVAL_TOP_K

        sources = retrieve_sources(payload.question, top_k=top_k)
        if not sources:
            return LegalChatResponse(
                answer="I could not find relevant legal sources in the vector store for this question.",
                sources=[],
            )

        messages = build_grounded_prompt(payload.question, sources)
        answer = run_llm(
            messages=messages,
            max_tokens=max_tokens,
        )
        return LegalChatResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("Error in /legal/chat")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
