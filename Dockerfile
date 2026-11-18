# syntax=docker/dockerfile:1
# PolicyPal images. Targets:
#   api  - FastAPI app (uvicorn, no --reload), embedded Chroma, CPU-only torch
#   ui   - Streamlit front end
#   demo - both in one container for a public demo: uvicorn on 127.0.0.1 and
#          Streamlit on $PORT (default 7860); DEMO_MODE on, fixed policy PDF
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
USER app
# Models before the app code, so a code change does not re-download them.
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base'); print('models cached')"; \
    fi
COPY --chown=app:app app/ ./app/
EXPOSE 8000
# Long start period: on an empty HF_HOME the first start downloads the embedding
# model and the reranker (~1.2 GB) before /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
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

# ---------------------------------------------------------------- demo
# One container, one public port: what free/small hosts (Render, Railway,
# Hugging Face Spaces) expect. The API listens on 127.0.0.1 only; Streamlit
# calls it on localhost. The policy PDF is indexed on first start (the Chroma
# store is empty), not at build time; see README "Deploying a demo".
FROM base AS demo
ARG PRELOAD_MODELS=false
ENV HF_HOME=/models/huggingface \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    DEMO_MODE=true \
    DEFAULT_POLICY_PDF=/app/demo/policy.pdf \
    PORT=7860 \
    API_PORT=8000
WORKDIR /app
RUN --mount=type=bind,from=builder-api,source=/wheels,target=/wheels-api \
    --mount=type=bind,from=builder-ui,source=/wheels,target=/wheels-ui \
    --mount=type=bind,source=docker/requirements-api.txt,target=/tmp/requirements-api.txt \
    --mount=type=bind,source=docker/requirements-ui.txt,target=/tmp/requirements-ui.txt \
    pip install --no-cache-dir --no-index --find-links /wheels-api --find-links /wheels-ui \
        -r /tmp/requirements-api.txt -r /tmp/requirements-ui.txt
RUN mkdir -p /app/data/raw /app/data/processed/chroma /app/demo "$HF_HOME" \
    && chown -R app:app /app /models
USER app
# Models before the app code, so a code change does not re-download ~1.2 GB.
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base'); print('models cached')"; \
    fi
COPY --chown=app:app data/raw/StarHealthAssureInsurancePolicy-Policy-wording.pdf /app/demo/policy.pdf
COPY --chown=app:app --chmod=755 docker/start-demo.sh ./start-demo.sh
COPY --chown=app:app app/ ./app/
COPY --chown=app:app streamlit_app.py ./
EXPOSE 7860
# Healthy once the API has loaded both models and indexed the policy (it only
# answers /health after startup) and Streamlit is serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import os,urllib.request,sys; ok=lambda u: urllib.request.urlopen(u, timeout=4).status == 200; sys.exit(0 if ok('http://127.0.0.1:%s/health' % os.environ['API_PORT']) and ok('http://127.0.0.1:%s/_stcore/health' % os.environ['PORT']) else 1)"
CMD ["./start-demo.sh"]
