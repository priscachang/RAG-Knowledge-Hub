from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.6
    use_hybrid: bool = True
    intent: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    evidence_score: float
    query_type: str
    processing_time: float

class IngestionResponse(BaseModel):
    status: str
    ingested_chunks: int
    files_processed: List[str]
    total_chunks: int
