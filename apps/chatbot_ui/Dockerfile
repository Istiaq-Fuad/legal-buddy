FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Copy workspace root files for dependency resolution
COPY pyproject.toml uv.lock ./

# Copy package files and source
COPY apps/chatbot_ui ./apps/chatbot_ui

ENV UV_COMPILE_BYTECODE=1

# Install dependencies including workspace packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package chatbot-ui

# Enable bytecode compilation and Python optimization
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure Streamlit and Python import path
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV HOME=/home/app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/chatbot_ui/src:$PYTHONPATH"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app/.streamlit && \
    chown -R app:app /home/app

# Switch to non-root user
USER app

# Expose the Streamlit port
EXPOSE 8501

WORKDIR /app/apps/chatbot_ui/src

# Command to run the Streamlit application
CMD ["streamlit", "run", "chatbot_ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
