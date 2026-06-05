# Use a slim Python 3.11 base image
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (extremely fast Python package installer and resolver)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv (creates a .venv)
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source code
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# Final runtime stage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment and source code from the builder stage
COPY --from=builder /app /app

# Expose ports: 8000 for Ingestion API, 8080 for LangGraph dev server
EXPOSE 8000 8080

# Default command runs the ingestion API. Can be overridden in docker-compose
CMD ["python", "-m", "ingestion.api.run"]
