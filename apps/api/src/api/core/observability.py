import logging

from langfuse import Langfuse

from api.core.config import config

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None


def _langfuse_host() -> str | None:
    if config.LANGFUSE_HOST:
        return config.LANGFUSE_HOST
    return config.LANGFUSE_BASE_URL


def get_langfuse_client():
    global _langfuse_client

    if not config.LANGFUSE_ENABLED:
        return None

    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    _langfuse_client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        base_url=_langfuse_host(),
    )
    return _langfuse_client


def validate_langfuse_auth() -> None:
    client = get_langfuse_client()
    if client is None:
        logger.info(
            "Langfuse disabled or not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
        )
        return

    try:
        if client.auth_check():
            logger.info("Langfuse auth check passed.")
        else:
            logger.warning("Langfuse auth check failed. Traces may not be ingested.")
    except Exception:
        logger.exception("Langfuse auth check failed with an exception.")


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception:
        logger.exception("Failed to flush Langfuse events.")
