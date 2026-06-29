from fastapi import APIRouter, Depends, HTTPException
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_chain import get_rag_chain

router = APIRouter()


def get_chain():
    return get_rag_chain()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, chain=Depends(get_chain)):
    try:
        result = chain.invoke(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))