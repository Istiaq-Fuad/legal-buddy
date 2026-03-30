from qdrant_client import QdrantClient
from api.core.config import config
from api.core.observability import get_langfuse_client
import boto3
from botocore.config import Config
import json
from api.api.models import LegalChatResponse, SourceItem
from google import genai
from google.genai import types

_qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=30)


def _extract_gemini_usage(response: types.GenerateContentResponse) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    usage_details: dict[str, int] = {}
    if input_tokens is not None:
        usage_details["input"] = int(input_tokens)
    if output_tokens is not None:
        usage_details["output"] = int(output_tokens)
    if total_tokens is not None:
        usage_details["total"] = int(total_tokens)

    return usage_details


def get_bedrock_client() -> boto3.client:
    return boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_DEFAULT_REGION,
        config=Config(retries={"mode": "adaptive"}),
    )


def embed_text_query(text: str, *, max_input_chars: int = 2048) -> list[float]:
    query_text = text[:max_input_chars]
    body = {
        "input_type": "search_query",
        "embedding_types": ["float"],
        "texts": [query_text],
    }
    response = get_bedrock_client().invoke_model(
        modelId=config.EMBEDDING_MODEL,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(response["body"].read())

    embeddings = result.get("embeddings")
    if isinstance(embeddings, dict) and "float" in embeddings:
        return embeddings["float"][0]
    if isinstance(embeddings, list):
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings
    if "embedding" in result:
        return result["embedding"]
    raise ValueError(f"Embedding missing in Bedrock response: {result}")


def _embed_text_query_with_trace(
    text: str,
    *,
    max_input_chars: int,
    langfuse,
) -> list[float]:
    if langfuse is None:
        return embed_text_query(text, max_input_chars=max_input_chars)

    query_text = text[:max_input_chars]
    with langfuse.start_as_current_observation(
        as_type="embedding",
        name="embed-query",
        model=config.EMBEDDING_MODEL,
        input={
            "input_chars": len(query_text),
            "max_input_chars": max_input_chars,
        },
        metadata={"provider": "bedrock"},
    ) as embedding_span:
        vector = embed_text_query(text, max_input_chars=max_input_chars)
        embedding_span.update(output={"embedding_dimensions": len(vector)})
        return vector


def retrieve_sources(question: str, top_k: int) -> list[SourceItem]:
    langfuse = get_langfuse_client()
    if langfuse is None:
        vector = _embed_text_query_with_trace(
            question, max_input_chars=2048, langfuse=None
        )
        hits = _qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

        sources: list[SourceItem] = []
        for idx, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            excerpt = str(payload.get("section_content_clean") or "").strip()
            if not excerpt:
                excerpt = "No excerpt available."
            sources.append(
                SourceItem(
                    citation_id=idx,
                    act_title=payload.get("act_title"),
                    act_year=payload.get("act_year"),
                    section_index=(
                        str(payload.get("section_index"))
                        if payload.get("section_index") is not None
                        else None
                    ),
                    source_url=payload.get("source_url"),
                    excerpt=excerpt,
                    score=float(hit.score or 0.0),
                )
            )
        return sources

    with langfuse.start_as_current_observation(
        as_type="span",
        name="retrieve-sources",
        input={"question": question, "top_k": top_k},
    ) as retrieval_span:
        vector = _embed_text_query_with_trace(
            question, max_input_chars=2048, langfuse=langfuse
        )

        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="vector-search",
            input={
                "collection": config.QDRANT_COLLECTION,
                "top_k": top_k,
            },
            metadata={"provider": "qdrant"},
        ) as search_span:
            hits = _qdrant_client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
            search_span.update(output={"hit_count": len(hits)})

        sources: list[SourceItem] = []
        for idx, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            excerpt = str(payload.get("section_content_clean") or "").strip()
            if not excerpt:
                excerpt = "No excerpt available."
            sources.append(
                SourceItem(
                    citation_id=idx,
                    act_title=payload.get("act_title"),
                    act_year=payload.get("act_year"),
                    section_index=(
                        str(payload.get("section_index"))
                        if payload.get("section_index") is not None
                        else None
                    ),
                    source_url=payload.get("source_url"),
                    excerpt=excerpt,
                    score=float(hit.score or 0.0),
                )
            )

        retrieval_span.update(
            output={
                "source_count": len(sources),
                "top_score": max((source.score for source in sources), default=0.0),
            }
        )
        return sources


def build_grounded_prompt(question: str, sources: list[SourceItem]) -> list[dict]:
    context_blocks = []
    for source in sources:
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {source.citation_id}]",
                    f"Act: {source.act_title or 'Unknown'}",
                    f"Year: {source.act_year if source.act_year is not None else 'Unknown'}",
                    f"Section: {source.section_index or 'Unknown'}",
                    f"Text: {source.excerpt}",
                    f"URL: {source.source_url or 'N/A'}",
                ]
            )
        )

    system_prompt = (
        "You are a legal research assistant for Bangladesh law. "
        "Answer only from the supplied legal sources. "
        "If sources are insufficient, explicitly say what is missing. "
        "Every substantive claim must include source citations like [Source 1]. "
        "Do not fabricate statutes, sections, or outcomes."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Legal sources:\n{chr(10).join(context_blocks)}\n\n"
        "Provide:\n"
        "1) Direct answer in plain language.\n"
        "2) Brief legal basis with citation tags.\n"
        "3) If uncertain, mention limits clearly."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_llm(messages: list[dict], max_tokens: int | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    langfuse = get_langfuse_client()
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    generation_config_kwargs: dict[str, int | float] = {
        "temperature": 0.2,
        "top_p": 0.9,
    }
    if max_tokens is not None:
        generation_config_kwargs["max_output_tokens"] = max_tokens
    generation_config = types.GenerateContentConfig(**generation_config_kwargs)

    if langfuse is None:
        response = client.models.generate_content(
            model=config.DEFAULT_MODEL_NAME,
            contents=[message["content"] for message in messages],
            config=generation_config,
        )
        return response.text or "No response generated."

    model_parameters: dict[str, int | float] = {
        "temperature": 0.2,
        "top_p": 0.9,
    }
    if max_tokens is not None:
        model_parameters["max_output_tokens"] = max_tokens

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="answer-generation",
        model=config.DEFAULT_MODEL_NAME,
        input=messages,
        model_parameters=model_parameters,
    ) as generation:
        response = client.models.generate_content(
            model=config.DEFAULT_MODEL_NAME,
            contents=[message["content"] for message in messages],
            config=generation_config,
        )
        answer = response.text or "No response generated."
        usage_details = _extract_gemini_usage(response)
        if usage_details:
            generation.update(output=answer, usage_details=usage_details)
        else:
            generation.update(output=answer)
        return answer


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
        answer = run_llm(messages=messages, max_tokens=resolved_max_tokens)
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
        answer = run_llm(messages=messages, max_tokens=resolved_max_tokens)
        response = LegalChatResponse(answer=answer, sources=sources)
        request_span.update(
            output={
                "answer_preview": answer[:200],
                "source_count": len(sources),
            }
        )
        return response
