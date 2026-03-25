from pydantic import BaseModel, Field


class LegalChatRequest(BaseModel):
    question: str = Field(min_length=3)
    max_tokens: int | None = None
    top_k: int | None = None


class SourceItem(BaseModel):
    citation_id: int
    act_title: str | None = None
    act_year: int | None = None
    section_index: str | None = None
    source_url: str | None = None
    excerpt: str
    score: float


class LegalChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
