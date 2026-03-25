# API Service

This FastAPI service powers the legal RAG endpoint used by the Streamlit UI.

## Langfuse Observability (Self-Hosted)

The API is instrumented with Langfuse spans and generations for:

- request-level tracing (`legal-chat-request`)
- retrieval tracing (`retrieve-sources`)
- generation tracing (`answer-generation`)

### 1) Set environment variables

Add these to your root `.env` file (used by Docker Compose):

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://<your-langfuse-host>:3000

# Optional: used if LANGFUSE_HOST is not set
LANGFUSE_BASE_URL=http://<your-langfuse-host>:3000

# Optional: tag traces by environment in Langfuse UI
LANGFUSE_TRACING_ENVIRONMENT=production
```

`LANGFUSE_HOST` should point to your self-hosted Langfuse web URL.

### 2) Run the stack

```bash
docker compose up --build
```

### 3) Verify ingestion

- On API startup, the app performs `auth_check()` and logs whether Langfuse credentials are valid.
- Execute a request to `POST /rag/legal/chat`.
- Confirm new traces appear in your Langfuse project.

## Notes

- If `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are missing, observability is skipped and API behavior stays unchanged.
- The service flushes Langfuse events on shutdown.
