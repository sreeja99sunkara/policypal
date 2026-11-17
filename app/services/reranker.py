from functools import lru_cache
from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)

def rerank(question: str, docs: list, top_n: int = 5) -> list:
    if not docs:
        return docs
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]