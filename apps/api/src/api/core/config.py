from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str
    EMBEDDING_MODEL: str

    QDRANT_URL: str = "http://213.136.80.53:6333"
    QDRANT_COLLECTION: str = "legal_acts_event_rag_full"

    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL_NAME: str = "openai/gpt-5.4-nano"
    RETRIEVAL_TOP_K: int = 6
    ANSWER_MAX_TOKENS: int = 700


config = Config()
