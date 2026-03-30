# Legal Acts RAG Monorepo

Production-style Retrieval Augmented Generation (RAG) project for Bangladesh legal acts.

This workspace contains:

- A FastAPI backend that performs retrieval from Qdrant and answer generation with Gemini
- A Streamlit frontend chat UI for end users
- Shared workspace tooling via `uv`, Docker, and Docker Compose

## What This Project Does

For each user question:

1. The API creates an embedding with AWS Bedrock
2. It retrieves top matching legal sections from Qdrant
3. It builds a grounded prompt with citation tags
4. It generates an answer with Gemini
5. It returns answer + source metadata to the UI

The UI displays the answer and expandable source cards with similarity scores.

## Repository Layout

```text
llm_engineering/
├── apps/
│   ├── api/                # FastAPI RAG service
│   └── chatbot_ui/         # Streamlit chat frontend
├── data/acts/              # Legal act JSON corpus
├── notebooks/
│   └── qdrant_ingestion.ipynb
├── docker-compose.yml
├── Makefile
├── pyproject.toml          # uv workspace root
└── README.md
```

## Prerequisites

- Python 3.13+
- `uv` installed: https://docs.astral.sh/uv/
- Docker + Docker Compose (recommended for full stack)
- Access to:
  - AWS Bedrock embedding model
  - Gemini API key
  - Reachable Qdrant instance

## Quick Start (Recommended)

1. Create your local env file from the template:

```bash
cp .env.example .env
```

Then fill values in `.env`:

```env
# API generation model
GEMINI_API_KEY=your_gemini_key
DEFAULT_MODEL_NAME=gemini-2.5-flash

# AWS Bedrock embeddings
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
EMBEDDING_MODEL=cohere.embed-v4

# Qdrant
QDRANT_URL=http://your-qdrant-host:6333
QDRANT_COLLECTION=legal_acts_event_rag_full

# Retrieval + generation defaults
RETRIEVAL_TOP_K=6
# Optional. Leave empty/unset to let model default behavior apply.
ANSWER_MAX_TOKENS=

# Frontend -> API URL (inside docker network default is fine)
API_URL=http://api:8000

# Optional Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
LANGFUSE_HOST=
LANGFUSE_TRACING_ENVIRONMENT=development
```

2. Install workspace deps:

```bash
make sync
```

3. Start both services:

```bash
docker compose up --build
```

4. Open apps:

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/rag/health`

## Local Development Without Docker

Install all workspace packages once:

```bash
uv sync --all-packages --all-extras --all-groups
```

Run API:

```bash
uv run --package api uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Run UI in another terminal:

```bash
API_URL=http://localhost:8000 uv run --package chatbot-ui streamlit run apps/chatbot_ui/src/chatbot_ui/app.py
```

## API Contract Summary

`POST /rag/legal/chat`

Request body:

```json
{
  "question": "What is the punishment for theft?",
  "top_k": 6,
  "max_tokens": null
}
```

Notes:

- `question` is required
- `top_k` is optional and falls back to `RETRIEVAL_TOP_K`
- `max_tokens` is optional. If omitted/null and `ANSWER_MAX_TOKENS` is also unset, the model default token limit behavior is used.
- The Streamlit UI currently does not expose `max_tokens` to end users.

## Observability

Langfuse instrumentation is integrated in the API for retrieval/generation spans. Configure Langfuse env vars to enable ingestion. If keys are not configured, pipeline still runs without tracing.

## Data and Ingestion

- Legal act JSON files live under `data/acts/`
- The ingestion workflow is documented in `notebooks/qdrant_ingestion.ipynb`

## Useful Commands

```bash
make sync                # install/update workspace deps
make run-docker-compose  # sync + docker compose up --build
```

## Additional Docs

- API details: `apps/api/README.md`
- UI details: `apps/chatbot_ui/README.md`
