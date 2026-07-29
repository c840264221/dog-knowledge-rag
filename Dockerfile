# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    API_ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    API_RELOAD=false \
    HF_HOME=/app/models_cache/huggingface

WORKDIR /app

# Torch, ONNX Runtime and scikit-learn may require the OpenMP runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source code so this expensive layer can
# be reused when only application code changes.
COPY requirements-torch-cpu.txt requirements-prod.txt ./
RUN python -m pip install --requirement requirements-torch-cpu.txt \
    && python -m pip install --requirement requirements-prod.txt

# Run the API as a non-root user with a stable uid/gid for mounted volumes.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app \
        --home-dir /app --shell /usr/sbin/nologin app

COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/api_run.py ./scripts/api_run.py
COPY --chown=app:app data/ ./data/

# These paths hold runtime state and will be mounted by Docker Compose.
RUN mkdir --parents \
        /app/chroma_db \
        /app/chroma_memory_db \
        /app/models_cache \
        /app/logs/report \
        /app/data/checkpoints_db \
        /app/data/memory_db \
        /app/data/user \
    && chown --recursive app:app \
        /app/chroma_db \
        /app/chroma_memory_db \
        /app/models_cache \
        /app/logs \
        /app/data/checkpoints_db \
        /app/data/memory_db \
        /app/data/user

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('API_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/ready', timeout=5).close()"]

CMD ["python", "-m", "scripts.api_run"]
