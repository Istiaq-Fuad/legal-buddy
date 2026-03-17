from fastapi import FastAPI, Request
from pydantic import BaseModel

from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_llm(provider, model_name, messages, max_tokens=500):
    if provider == "openai":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENAI_API_KEY,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif provider == "groq":
        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif provider == "gemini":
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages],
        )
        return response.text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list
    max_tokens: int = 500


class ChatResponse(BaseModel):
    response: str


app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    try:
        response = run_llm(
            provider=payload.provider,
            model_name=payload.model_name,
            messages=payload.messages,
            max_tokens=payload.max_tokens,
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(response=f"Error: {str(e)}")
