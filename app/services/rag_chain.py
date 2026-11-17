from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from typing import List
import os

from app.prompts.qa_prompt import get_qa_prompt
from app.core.logger import get_logger, log_with_fields

from langchain_core.runnables import RunnableLambda
from app.services.reranker import rerank
from app.services.hybrid import hybrid_search
from app.services.vectorstore import get_vectorstore
from operator import itemgetter

_chain = None
logger = get_logger(__name__)

def format_docs(docs: List[Document]) -> str:
    log_with_fields(
        logger, "info", "Chunks retrieved",
        chunk_count=len(docs),
        pages=[doc.metadata.get("page") for doc in docs]
    )
    return "\n\n".join(
        f"[Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    )

MAX_CONTEXT_CHUNKS = 6


def retrieve_and_rerank(question: str) -> List[Document]:
    """Retrieval used by the chain.

    1. Hybrid candidates: vector top-20 + BM25 top-20 fused with RRF, first 25.
    2. Cross-encoder rerank -> top 5.
    3. Header attachment: a table row chunk ("S.No: 29 | Benefit: Co-payment | ...")
       only makes sense with its table's header chunk ("Table SCHEDULE OF BENEFITS
       gives Benefit for each Sum Insured (INR): ..."). For every table that has a
       row in the top 5, its header chunk is inserted just before that table's
       first row: once per table (rows share metadata "table_key"), skipped if the
       header is already retrieved, in rank order, total capped at
       MAX_CONTEXT_CHUNKS (so at most one header is added to a top 5).
    """
    candidates = hybrid_search(question, k=20)[:25]
    top = rerank(question, candidates, top_n=5)
    return attach_table_headers(top, MAX_CONTEXT_CHUNKS)


def attach_table_headers(docs: List[Document], max_total: int) -> List[Document]:
    present = {d.metadata.get("table_key") for d in docs if d.metadata.get("table_part") == "header"}
    wanted = []  # table keys in rank order of their first retrieved row
    for d in docs:
        key = d.metadata.get("table_key")
        if d.metadata.get("table_part") == "rows" and key and key not in present and key not in wanted:
            wanted.append(key)
    wanted = wanted[: max(0, max_total - len(docs))]
    if not wanted:
        return docs

    found = get_vectorstore().get(
        where={"$and": [{"table_part": "header"}, {"table_key": {"$in": wanted}}]},
        include=["documents", "metadatas"],
    )
    headers = {m["table_key"]: Document(page_content=c, metadata=m)
               for c, m in zip(found["documents"], found["metadatas"])}

    out, inserted = [], set()
    for d in docs:
        key = d.metadata.get("table_key")
        if key in headers and key not in inserted:
            out.append(headers[key])
            inserted.add(key)
        out.append(d)
    return out

def get_rag_chain():
    global _chain
    if _chain is not None:
        return _chain
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # Initialize components
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = RunnableLambda(retrieve_and_rerank)
    prompt = get_qa_prompt()
    parser = StrOutputParser()

    # Retrieve once, format once, then fan out to answer + sources
    _chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnableParallel(
            answer=prompt | llm | parser,
            sources=itemgetter("context"),
        )
    )

    return _chain