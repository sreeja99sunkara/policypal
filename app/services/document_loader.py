from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

from app.core.logger import get_logger
from app.services.table_loader import load_and_chunk_tables

logger = get_logger(__name__)


def load_and_chunk(pdf_path: str) -> List[Document]:
    """Table-aware loading (see table_loader.py); falls back to the legacy
    text-only path if table extraction fails for any reason."""
    try:
        return load_and_chunk_tables(pdf_path)
    except Exception:
        logger.exception(f"Table-aware loading failed for {pdf_path}; using legacy loader")
        return load_and_chunk_legacy(pdf_path)


def load_and_chunk_legacy(pdf_path: str) -> List[Document]:
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Chunk
    # 500/80 chosen because policy clauses average ~400 chars
    # 80 overlap prevents key terms from being split across boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    return chunks
