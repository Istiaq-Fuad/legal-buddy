from fastapi import FastAPI
from api.api.endpoints import api_router

app = FastAPI(title="Legal Acts RAG API")

app.include_router(api_router)
