import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_chain import get_rag_chain
from app.services.vectorstore import has_documents
from app.services.citations import rewrite_citations
from app.services.rate_limit import RateLimited, client_ip, get_query_limiter
from app.core import settings
from app.core.logger import get_logger, log_with_fields

router = APIRouter()
logger = get_logger(__name__)


def get_chain():
    return get_rag_chain()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {"description": "Question longer than MAX_QUESTION_CHARS"},
        429: {"description": "Per-IP rate limit reached (see Retry-After)"},
    },
)
def query(request: QueryRequest, http_request: Request, chain=Depends(get_chain)):
    log_with_fields(logger, "info", "Query received", question_length=len(request.question))

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    max_chars = settings.max_question_chars()
    if len(request.question) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Your question is {len(request.question)} characters long; "
                   f"please keep it under {max_chars} characters.",
        )

    ip = client_ip(
        http_request.headers.get("x-forwarded-for"),
        http_request.client.host if http_request.client else None,
    )
    try:
        get_query_limiter().hit(ip)
    except RateLimited as e:
        log_with_fields(logger, "warning", "Rate limited", retry_after=e.retry_after)
        return JSONResponse(
            status_code=429,
            content={"detail": e.message},
            headers={"Retry-After": str(e.retry_after)},
        )

    if not has_documents():
        raise HTTPException(status_code=404, detail="No documents have been uploaded yet. Please upload a policy PDF first.")

    start_time = time.time()
    try:
        result = chain.invoke(request.question)
        latency = round(time.time() - start_time, 2)
        log_with_fields(
            logger, "info", "Query completed",
            latency_seconds=latency,
            answer_length=len(result["answer"])
        )
    except Exception:
        latency = round(time.time() - start_time, 2)
        logger.exception(f"RAG chain failed | latency={latency}s | question_length={len(request.question)}")
        raise HTTPException(status_code=500, detail="Failed to process your question. Please try again.")

    # The LLM cites the 0-indexed pages from its context; convert to viewer pages here.
    answer, cited_pages = rewrite_citations(result["answer"])
    return QueryResponse(
        answer=answer,
        sources=result["sources"],
        cited_pages=cited_pages,
    )