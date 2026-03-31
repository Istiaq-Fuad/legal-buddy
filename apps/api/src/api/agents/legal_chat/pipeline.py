from api.api.models import LegalChatResponse
from api.core.config import config
from api.core.observability import get_langfuse_client

from api.agents.legal_chat.generation import run_llm
from api.agents.legal_chat.prompting import build_grounded_prompt
from api.agents.legal_chat.retrieval import retrieve_sources


def legal_chat_pipeline(
    question: str,
    *,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> LegalChatResponse:
    resolved_top_k = top_k or config.RETRIEVAL_TOP_K
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else config.ANSWER_MAX_TOKENS
    )

    langfuse = get_langfuse_client()
    if langfuse is None:
        sources = retrieve_sources(question, top_k=resolved_top_k)
        if not sources:
            return LegalChatResponse(
                answer="I could not find relevant legal sources in the vector store for this question.",
                sources=[],
            )

        messages = build_grounded_prompt(question, sources)
        answer = run_llm(
            messages=messages,
            sources=sources,
            max_tokens=resolved_max_tokens,
        )
        return LegalChatResponse(answer=answer, sources=sources)

    with langfuse.start_as_current_observation(
        as_type="span",
        name="legal-chat-request",
        input={
            "question": question,
            "top_k": resolved_top_k,
            "max_tokens": resolved_max_tokens,
        },
        metadata={"endpoint": "/rag/legal/chat"},
    ) as request_span:
        sources = retrieve_sources(question, top_k=resolved_top_k)
        if not sources:
            response = LegalChatResponse(
                answer="I could not find relevant legal sources in the vector store for this question.",
                sources=[],
            )
            request_span.update(output=response.model_dump())
            return response

        messages = build_grounded_prompt(question, sources)
        answer = run_llm(
            messages=messages,
            sources=sources,
            max_tokens=resolved_max_tokens,
        )
        response = LegalChatResponse(answer=answer, sources=sources)
        request_span.update(
            output={
                "answer_preview": answer[:200],
                "source_count": len(sources),
            }
        )
        return response
