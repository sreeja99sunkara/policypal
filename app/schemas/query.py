from pydantic import BaseModel, Field
from typing import List


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


class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The answer generated from the policy document"
    )
    sources: str = Field(
        ...,
        description="The source chunks used to generate the answer with page numbers"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The waiting period for pre-existing diseases is 48 months. (Page 11)",
                    "sources": "[Page 11]\nwaiting period of 48 months from inception..."
                }
            ]
        }
    }