from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.logger import get_logger, log_with_fields
from app.services.embedder import get_embeddings

CHROMA_DIR = "data/processed/chroma"

_vectorstore = None
logger = get_logger(__name__)


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings()
        )
    return _vectorstore


def has_documents() -> bool:
    try:
        return get_vectorstore()._collection.count() > 0
    except Exception:
        return False


def ingest_documents(docs: List[Document]) -> None:
    vs = get_vectorstore()

    # Replace, don't append: empty the collection via Chroma's API
    # (no file deletion, so no Windows file lock), then add the new chunks.
    vs.reset_collection()
    vs.add_documents(docs)

    stored = vs._collection.count()
    log_with_fields(logger, "info", "Ingestion stored", chunks_in=len(docs), chunks_in_store=stored)
    if stored != len(docs):
        logger.warning(f"Store has {stored} chunks but {len(docs)} were ingested")

    # The BM25 index is built lazily from the collection; drop the stale one.
    # (Local import: hybrid imports this module.)
    try:
        from app.services.hybrid import invalidate_bm25
    except ImportError:  # hybrid search optional (rank_bm25 not installed)
        return
    invalidate_bm25()