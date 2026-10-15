"""
Pydantic Schemas for CIT Knowledge AI Backend
Validates input requests and structures API responses for Multi-Source RAG.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Citation(BaseModel):
    chunk_id: str
    document_title: str
    source_url: str
    category: str
    academic_year: Optional[str] = None
    similarity_score: Optional[float] = None
    text_snippet: str
    source_type: Optional[str] = "dataset"  # "dataset", "official_cit_website", "public_web"
    verified_date: Optional[str] = None

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Student question about CIT")
    model_choice: Optional[str] = Field("hybrid", description="Choice of retrieval model: baseline, finetuned, hybrid")
    top_k: Optional[int] = Field(4, ge=1, le=10, description="Number of source passages to retrieve")
    category_filter: Optional[str] = Field(None, description="Optional category filter")

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    source_type: str = "dataset"  # "dataset", "official_cit_website", "public_web"
    confidence: float = 1.0
    retrieval_level: int = 1  # Level 1 (Local), Level 2 (Official CIT Website), Level 3 (Public Web)
    verified_at: Optional[str] = None
    retrieved_evidence: List[Dict[str, Any]] = []
    category: Optional[str] = None
    model_used: str = "hybrid"
    latency_ms: float = 0.0
    has_conflicts: bool = False
    conflict_notes: List[str] = []
    is_refusal: bool = False

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    model_choice: Optional[str] = Field("hybrid", description="baseline, finetuned, hybrid")
    top_k: Optional[int] = Field(5, ge=1, le=15)
    category_filter: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    model_used: str
    total_found: int
    latency_ms: float
    results: List[Dict[str, Any]]

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str

class DocumentSummary(BaseModel):
    document_id: str
    title: str
    category: str
    format: str
    chunks_count: int
    source_url: str
    academic_year: Optional[str] = None

class ModelInfoResponse(BaseModel):
    baseline_model: str
    finetuned_model: str
    vector_database: str
    total_chunks_indexed: int
    default_strategy: str
    supported_strategies: List[str]
    training_loss: float
    fine_tuning_epochs: int
    leakage_resistance: str
