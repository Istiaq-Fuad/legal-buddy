from pydantic import BaseModel, Field


class StructuredLegalAnswer(BaseModel):
    answer: str = Field(description="Final answer to the user")
    citations: list[int] = Field(
        default_factory=list,
        description="List of supporting source ids, e.g. [1, 2]",
    )
    limitations: str | None = Field(
        default=None,
        description="Optional uncertainty or missing context",
    )
