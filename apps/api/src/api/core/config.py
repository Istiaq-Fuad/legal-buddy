from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OPENAI_API_KEY: str | None = None
    # GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # LANGFUSE_ENABLED: bool = True
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_BASE_URL: str | None = None
    # LANGFUSE_HOST: str | None = None
    # LANGFUSE_TRACING_ENVIRONMENT: str = "development"

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str
    EMBEDDING_MODEL: str

    QDRANT_URL: str = "http://213.136.80.53:6333"
    QDRANT_COLLECTION: str = "legal_acts_event_rag_full"

    DEFAULT_MODEL_NAME: str = "gemini-2.5-flash"
    RETRIEVAL_TOP_K: int = 6
    ANSWER_MAX_TOKENS: int | None = None


config = Config()
