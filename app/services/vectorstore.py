import shutil
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from app.services.embedder import get_embeddings

CHROMA_DIR = "data/processed/chroma"

_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings()
        )
    return _vectorstore


def ingest_documents(docs: List[Document]) -> None:
    global _vectorstore

    # Release any existing Chroma client connection first
    if _vectorstore is not None:
        try:
            _vectorstore._client.reset()
        except Exception:
            pass
        _vectorstore = None

    # Now safe to wipe existing data
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    _vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR
    )
    print(f"Ingested {len(docs)} chunks into ChromaDB")

def has_documents() -> bool:
    vs = get_vectorstore()
    try:
        return vs._collection.count() > 0
    except Exception:
        return False