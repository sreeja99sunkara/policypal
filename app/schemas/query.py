from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask about the insurance policy",
        examples=["What is the waiting period for pre-existing diseases?"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is the waiting period for pre-existing diseases?"}
            ]
        }
    }


class Source(BaseModel):
    page: int = Field(..., description="Viewer page number (1-indexed) of the chunk")
    pages: List[int] = Field(
        ..., description="All viewer pages the chunk appears on (identical chunks are stored once)"
    )
    chunk_type: Literal["text", "table"] = Field(..., description="Plain text or a serialized table chunk")
    title: Optional[str] = Field(None, description="Table title for table chunks, else null")
    snippet: str = Field(..., description="First ~300 characters of the chunk")
    text: str = Field(..., description="Full chunk text")


class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The answer generated from the policy document; page citations use viewer page numbers"
    )
    sources: List[Source] = Field(
        ...,
        description="The retrieved chunks used to generate the answer, in retrieval order"
    )
    cited_pages: List[int] = Field(
        default_factory=list,
        description="Sorted viewer pages cited in the answer (empty if the answer cites none)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Pre-existing diseases are covered after 36 months of continuous coverage (Page 12).",
                    "sources": [
                        {
                            "page": 12,
                            "pages": [12],
                            "chunk_type": "text",
                            "title": None,
                            "snippet": "1. Pre-Existing Diseases - Code Excl 01: Expenses related to the treatment of a pre-existing Disease ...",
                            "text": "1. Pre-Existing Diseases - Code Excl 01: Expenses related to the treatment of a pre-existing Disease ...",
                        }
                    ],
                    "cited_pages": [12],
                }
            ]
        }
    }
