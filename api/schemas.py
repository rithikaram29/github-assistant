from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=20)
    conversation_id: Optional[str] = Field(default=None, min_length=1, max_length=128)

class Source(BaseModel):
    title: str
    url: Optional[str] = None
    repo: Optional[str] = None
    path: Optional[str] = None
    extra: Optional[str | dict] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    conversation_id: Optional[str] = None
