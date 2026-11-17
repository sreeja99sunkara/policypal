import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from app.services.vectorstore import get_vectorstore

# (BM25Okapi, docs) swapped as one tuple so a concurrent query never pairs
# a new document list with an old index.
_index: tuple[BM25Okapi, list[Document]] | None = None

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25() -> None:
    global _index
    data = get_vectorstore().get(include=["documents", "metadatas"])
    docs = [Document(page_content=c, metadata=m or {})
            for c, m in zip(data["documents"], data["metadatas"])]
    # BM25Okapi raises on an empty corpus; stay unbuilt until something is ingested.
    _index = (BM25Okapi([_tokenize(d.page_content) for d in docs]), docs) if docs else None

def invalidate_bm25() -> None:
    global _index
    _index = None

def bm25_search(question: str, k: int = 20) -> list[Document]:
    if _index is None:
        build_bm25()
    index = _index
    if index is None:
        return []
    bm25, docs = index
    scores = bm25.get_scores(_tokenize(question))
    top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return [docs[i] for i in top if scores[i] > 0]

def hybrid_search(question: str, k: int = 20,
                  w_vector: float = 0.5, w_bm25: float = 0.5,
                  rrf_k: int = 60) -> list[Document]:
    # A zero weight drops that source entirely; otherwise its hits would still be
    # appended with fused score 0 and fill candidate slots.
    vector_hits = get_vectorstore().similarity_search(question, k=k) if w_vector > 0 else []
    keyword_hits = bm25_search(question, k=k) if w_bm25 > 0 else []

    fused: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for weight, hits in ((w_vector, vector_hits), (w_bm25, keyword_hits)):
        for rank, doc in enumerate(hits):
            key = doc.page_content
            by_key[key] = doc
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank + 1)

    ordered = sorted(fused, key=fused.get, reverse=True)
    return [by_key[key] for key in ordered]