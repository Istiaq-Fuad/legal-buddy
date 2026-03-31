import json

import boto3
from botocore.config import Config

from api.core.config import config


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


def embed_text_query_with_trace(
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
