from fastapi import APIRouter, HTTPException
from api.api.models import LegalChatRequest, LegalChatResponse
import logging
from api.agents.legal_chat.pipeline import legal_chat_pipeline

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
        return legal_chat_pipeline(
            payload.question,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
        )
    except Exception as exc:
        logger.exception("Error in /legal/chat")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
