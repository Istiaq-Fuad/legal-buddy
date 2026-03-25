from qdrant_client import QdrantClient
from api.core.config import config
import boto3
from botocore.config import Config
import json
from api.api.models import SourceItem
from google import genai
from google.genai import types

_qdrant_client = QdrantClient(url=config.QDRANT_URL, timeout=30)


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


def retrieve_sources(question: str, top_k: int) -> list[SourceItem]:
    vector = embed_text_query(question)
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


def run_llm(messages: list[dict], max_tokens: int) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    generation_config = types.GenerateContentConfig(
        # max_output_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
    )

    response = client.models.generate_content(
        model=config.DEFAULT_MODEL_NAME,
        contents=[message["content"] for message in messages],
        config=generation_config,
    )
    return response.text or "No response generated."
