import instructor
from google import genai
from google.genai import types

from api.api.models import SourceItem
from api.core.config import config
from api.core.observability import get_langfuse_client

from api.agents.legal_chat.structured_models import StructuredLegalAnswer


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


def _source_tag_list(source_ids: list[int]) -> str:
    return " ".join([f"[Source {source_id}]" for source_id in source_ids])


def _normalize_source_ids(source_ids: list[int], max_source_id: int) -> list[int]:
    valid_ids = {
        int(source_id)
        for source_id in source_ids
        if isinstance(source_id, int) and 1 <= source_id <= max_source_id
    }
    return sorted(valid_ids)


def _render_structured_answer(answer: StructuredLegalAnswer, max_source_id: int) -> str:
    lines: list[str] = []
    answer_text = answer.answer.strip()
    citation_ids = _normalize_source_ids(answer.citations, max_source_id)
    source_tags = _source_tag_list(citation_ids)
    if source_tags:
        lines.append(f"{answer_text} {source_tags}")
    else:
        lines.append(answer_text)

    if answer.limitations:
        lines.append(f"\nLimitations: {answer.limitations.strip()}")

    return "\n".join(lines).strip()


def _build_structured_messages(messages: list[dict]) -> list[dict]:
    if len(messages) < 2:
        return messages

    system_message = messages[0]
    user_message = messages[1]
    structured_instruction = (
        "\n\nReturn JSON for this schema:\n"
        "- answer: string\n"
        "- citations: int[]\n"
        "- limitations: string | null\n"
        "Rules:\n"
        "- Use only source ids from provided [Source n] context.\n"
        "- Never invent citations.\n"
        "- Keep answer concise, natural, and grounded."
    )
    return [
        system_message,
        {
            "role": "user",
            "content": f"{user_message['content']}{structured_instruction}",
        },
    ]


def _run_llm_text(messages: list[dict], max_tokens: int | None = None) -> str:
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


def run_llm(
    messages: list[dict], sources: list[SourceItem], max_tokens: int | None = None
) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    langfuse = get_langfuse_client()
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    structured_client = instructor.from_genai(
        client,
        model=config.DEFAULT_MODEL_NAME,
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    generation_config_kwargs: dict[str, int | float] = {
        "temperature": 0.2,
        "top_p": 0.9,
    }
    if max_tokens is not None:
        generation_config_kwargs["max_output_tokens"] = max_tokens
    generation_config = types.GenerateContentConfig(**generation_config_kwargs)

    structured_messages = _build_structured_messages(messages)
    max_source_id = len(sources)

    if langfuse is None:
        try:
            structured_answer = structured_client.create(
                response_model=StructuredLegalAnswer,
                messages=structured_messages,
                config=generation_config,
            )
            return _render_structured_answer(structured_answer, max_source_id)
        except Exception:
            return _run_llm_text(messages=messages, max_tokens=max_tokens)

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
        input=structured_messages,
        model_parameters=model_parameters,
    ) as generation:
        try:
            structured_answer = structured_client.create(
                response_model=StructuredLegalAnswer,
                messages=structured_messages,
                config=generation_config,
            )
            answer_text = _render_structured_answer(structured_answer, max_source_id)
            generation.update(output=answer_text)
            return answer_text
        except Exception:
            fallback_answer = _run_llm_text(messages=messages, max_tokens=max_tokens)
            generation.update(
                output=fallback_answer,
                metadata={"fallback": "plain-genai-response"},
            )
            return fallback_answer
