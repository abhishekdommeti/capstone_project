"""Pydantic models for the /ask endpoint.

AskResponse is the JSON output schema enforced on every final answer (Task 4):
- answer: the natural-language answer text
- sources: list of chunk/document ids the answer was grounded in (empty for general_question)
- confidence: float in [0, 1]
"""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question for the Zepto support assistant.")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
