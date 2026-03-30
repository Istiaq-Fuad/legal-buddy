# Chatbot UI (Streamlit)

Frontend chat application for interacting with the legal RAG API.

## Features

- Chat interface for legal questions
- Adjustable retrieval depth (`top_k`) in sidebar
- Source citations with score, excerpt, and URL
- Conversation history in session state
- Basic error handling for connectivity and HTTP failures

## Configuration

The UI reads settings from env vars via `chatbot_ui.core.config.Config`.

```env
API_URL=http://api:8000
```

When running outside Docker, set:

```env
API_URL=http://localhost:8000
```

## Run With Docker

From repository root:

```bash
docker compose up --build streamlit-app
```

Then open `http://localhost:8501`.

## Run Locally (No Docker)

From repository root:

```bash
uv sync --all-packages --all-extras --all-groups
API_URL=http://localhost:8000 uv run --package chatbot-ui streamlit run apps/chatbot_ui/src/chatbot_ui/app.py
```

## API Interaction

The UI sends:

```json
{
  "question": "...",
  "top_k": 6
}
```

Notes:

- UI intentionally does not expose `max_tokens`.
- Token limit behavior is controlled by API defaults/config.

## Code Map

- `apps/chatbot_ui/src/chatbot_ui/app.py`: Streamlit page and request flow
- `apps/chatbot_ui/src/chatbot_ui/core/config.py`: env-based settings

## Troubleshooting

- Cannot connect to API: verify `API_URL` and that API is running
- Empty/failed responses: inspect API logs and `/docs` endpoint
- Slow responses: reduce `top_k` or inspect upstream model/retrieval latency
