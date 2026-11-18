import os

from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


def ingest_default_policy() -> None:
    """Index DEFAULT_POLICY_PDF on startup if the vector store is empty.

    Used by the public demo, which serves one fixed policy: a fresh container
    (or a host without a persistent disk) indexes it once at boot; an existing
    index is left alone.
    """
    import time
    from app.core import settings
    from app.core.logger import get_logger, log_with_fields
    from app.services.vectorstore import has_documents, ingest_documents
    from app.services.document_loader import load_and_chunk

    logger = get_logger("app.startup")
    path = settings.default_policy_pdf()
    if not path:
        return
    if has_documents():
        log_with_fields(logger, "info", "Default policy skipped: index already populated", pdf=path)
        return
    if not os.path.isfile(path):
        log_with_fields(logger, "warning", "DEFAULT_POLICY_PDF not found; index stays empty", pdf=path)
        return
    start = time.time()
    chunks = load_and_chunk(path)
    ingest_documents(chunks)
    log_with_fields(
        logger, "info", "Default policy indexed",
        pdf=path, chunk_count=len(chunks), seconds=round(time.time() - start, 1),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs once when server starts
    # Initialize heavy objects here so they're ready before first request
    print("Starting up PolicyPal...")
    from app.services.embedder import get_embeddings
    from app.services.vectorstore import get_vectorstore
    from app.services.rag_chain import get_rag_chain
    from app.services.hybrid import build_bm25
    from app.services.reranker import get_reranker
    get_embeddings()
    # Load the cross-encoder now (downloads it on a fresh model cache) and run one
    # predict, so the first user query does not pay for the download or warm-up.
    get_reranker().predict([("warm-up query", "warm-up passage")])
    get_vectorstore()
    ingest_default_policy()  # DEFAULT_POLICY_PDF, only when the store is empty
    build_bm25()  # keyword index for hybrid search; rebuilt lazily after each ingestion
    get_rag_chain()
    print("All services ready.")
    yield
    # Shutdown — runs when server stops
    print("Shutting down PolicyPal...")


app = FastAPI(
    title="PolicyPal",
    description="RAG-powered insurance policy question answering",
    version="0.1.0",
    lifespan=lifespan
)

# Register routers
from app.routers import query, ingestion                   
app.include_router(query.router, prefix="/api") 
app.include_router(ingestion.router, prefix="/api")

@app.get("/health")
def health():
    from app.core import settings
    return {"status": "ok", "demo_mode": settings.demo_mode()}