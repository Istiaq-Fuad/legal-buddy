from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.api.endpoints import api_router
from api.core.observability import flush_langfuse, validate_langfuse_auth


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_langfuse_auth()
    yield
    flush_langfuse()


app = FastAPI(title="Legal Acts RAG API", lifespan=lifespan)

app.include_router(api_router)
