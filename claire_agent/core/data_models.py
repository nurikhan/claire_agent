# claire_agent/core/data_models.py
from typing import List, Optional
from pydantic import BaseModel, Field

class StoredMemoryEntry(BaseModel):
    id: Optional[int] = None
    vector_id: Optional[str] = None
    session_id: str
    summary: str
    keywords: List[str] = Field(default_factory=list)
    full_conversation_snippet: Optional[str] = None
    creation_time: str
    last_accessed_time: str
    access_count: int = 0
    user_importance_score: float = 0.5