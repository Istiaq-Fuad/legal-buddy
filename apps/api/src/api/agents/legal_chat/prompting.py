from api.api.models import SourceItem


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
        "You are an expert legal assistant focused on Bangladesh law. "
        "Write clear, helpful, and professional responses in natural language. "
        "Use only the supplied legal sources and do not fabricate statutes, sections, facts, or outcomes. "
        "If evidence is insufficient, state the limitation clearly and suggest what legal text is needed. "
        "Ground the answer with source citations like [Source 1]."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Legal sources:\n{chr(10).join(context_blocks)}\n\n"
        "Provide:\n"
        "1) A concise and easy-to-read legal answer.\n"
        "2) Support with relevant source citations.\n"
        "3) If uncertain, mention limits clearly."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
