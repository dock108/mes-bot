# Multi-stage Dockerfile for MES 0DTE Lotto-Grid Options Bot

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY logs/ ./logs/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose ports
EXPOSE 8501

# Default command (can be overridden)
CMD ["poetry", "run", "python", "-m", "app.bot"]

# --- Development stage ---
FROM base as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN poetry install && rm -rf $POETRY_CACHE_DIR

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# --- Production stage ---
FROM base as production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/healthz', timeout=10)" || exit 1

# Labels for metadata
LABEL maintainer="MES Bot Team"
LABEL version="1.0.0"
LABEL description="MES 0DTE Lotto-Grid Options Trading Bot"