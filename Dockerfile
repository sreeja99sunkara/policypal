# syntax=docker/dockerfile:1
# PolicyPal images. Targets:
#   api  - FastAPI app (uvicorn, no --reload), embedded Chroma, CPU-only torch
#   ui   - Streamlit front end
# No secrets and no data/ in the images: .env is passed at run time (compose
# env_file); the Chroma index and the Hugging Face model cache live in volumes.

ARG PYTHON_VERSION=3.13.5

# ---------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 1000 app

# ---------------------------------------------------------------- wheels
# requirements.txt (the dev venv's freeze) is used only as a constraints file so
# every installed package matches the tested version; Windows-only entries in it
# (e.g. pywinpty) are never required here, so they are ignored.
# The pip cache is a BuildKit cache mount (not an image layer), and downloads
# resume/retry, so a dropped connection does not throw away finished downloads.
FROM base AS builder-api
WORKDIR /build
COPY requirements.txt docker/requirements-api.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels \
        --retries 10 --timeout 120 --resume-retries 10 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements-api.txt -c requirements.txt

FROM base AS builder-ui
WORKDIR /build
COPY requirements.txt docker/requirements-ui.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels \
        --retries 10 --timeout 120 --resume-retries 10 \
        -r requirements-ui.txt -c requirements.txt

# ---------------------------------------------------------------- api
FROM base AS api
ARG PRELOAD_MODELS=false
ENV HF_HOME=/models/huggingface
WORKDIR /app
# Wheels are bind-mounted from the builder, so they never become an image layer.
RUN --mount=type=bind,from=builder-api,source=/wheels,target=/wheels \
    --mount=type=bind,source=docker/requirements-api.txt,target=/tmp/requirements-api.txt \
    pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements-api.txt
# Writable dirs for the non-root user; data/processed/chroma and HF_HOME are
# volume mount points (a fresh named volume inherits this ownership).
RUN mkdir -p /app/data/raw /app/data/processed/chroma "$HF_HOME" \
    && chown -R app:app /app /models
COPY --chown=app:app app/ ./app/
USER app
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base'); print('models cached')"; \
    fi
EXPOSE 8000
# Long start period: the first start downloads the embedding model into HF_HOME.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- ui
FROM base AS ui
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true
WORKDIR /app
RUN --mount=type=bind,from=builder-ui,source=/wheels,target=/wheels \
    --mount=type=bind,source=docker/requirements-ui.txt,target=/tmp/requirements-ui.txt \
    pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements-ui.txt
COPY --chown=app:app streamlit_app.py ./
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
