import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_chain import get_rag_chain
from app.services.vectorstore import has_documents
from app.services.citations import rewrite_citations
from app.services.rate_limit import (
    Busy, RateLimited, client_ip, get_daily_cap, get_query_limiter, get_query_slots,
)
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
        429: {"description": "Per-IP rate limit or daily demo cap reached (see Retry-After)"},
        503: {"description": "All query slots busy for QUERY_WAIT_SECONDS"},
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
    limiter, cap, slots = get_query_limiter(), get_daily_cap(), get_query_slots()
    # Cheap checks first, so a limited client is turned away without waiting.
    try:
        limiter.check(ip)
        cap.check()
    except RateLimited as e:
        return _too_many(e)

    if not has_documents():
        raise HTTPException(status_code=404, detail="No documents have been uploaded yet. Please upload a policy PDF first.")

    try:
        slots.acquire()
    except Busy as e:
        log_with_fields(logger, "warning", "Query rejected: busy", wait_seconds=slots.wait_seconds)
        return JSONResponse(status_code=503, content={"detail": str(e)}, headers={"Retry-After": "10"})
    try:
        # Count the query only now that it will run, so 4xx/503 rejections never
        # use up a client's quota or the daily cap. Re-checked here because other
        # requests may have been counted while this one waited for a slot.
        try:
            limiter.hit(ip)
        except RateLimited as e:
            return _too_many(e)
        try:
            cap.hit()
        except RateLimited as e:
            limiter.unhit(ip)  # rejected, so it must not use the client's quota
            return _too_many(e)
        return _answer(request, chain)
    finally:
        slots.release()


def _too_many(e: RateLimited) -> JSONResponse:
    log_with_fields(logger, "warning", "Rate limited", retry_after=e.retry_after)
    return JSONResponse(
        status_code=429,
        content={"detail": e.message},
        headers={"Retry-After": str(e.retry_after)},
    )


def _answer(request: QueryRequest, chain) -> QueryResponse:
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