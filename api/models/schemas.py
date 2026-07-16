"""
schemas.py
----------
Pydantic v2 request/response models for the FastAPI API.

Why strict Pydantic models:
  - Automatic validation: invalid input returns clear 422 errors
  - Auto-generated OpenAPI docs (FastAPI uses these for /docs)
  - Type safety throughout the codebase
  - Serialization control (exclude_none, field aliases, etc.)

Naming convention:
  {Resource}{Action}Request  →  e.g., IngestURLRequest
  {Resource}{Action}Response →  e.g., IngestResponse
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, AnyHttpUrl


# ============================================================
# Ingestion Schemas
# ============================================================

class IngestURLRequest(BaseModel):
    """Request body for ingesting a web page."""
    url: str = Field(..., description="URL to ingest", examples=["https://company.com/policy"])
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata to attach to the document",
    )

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    success: bool
    document_id: Optional[str] = None
    document_name: str
    source_type: str
    chunks_created: int
    chunks_skipped: int
    was_duplicate: bool
    error: Optional[str] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentListItem(BaseModel):
    """Summary of an ingested document."""
    id: str
    name: str
    source_type: str
    source_path: str
    total_chunks: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """List of ingested documents."""
    documents: list[DocumentListItem]
    total: int


# ============================================================
# Query Schemas
# ============================================================

class QueryRequest(BaseModel):
    """Request body for a knowledge base query."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to ask the knowledge base",
        examples=["What is the employee leave policy?"],
    )
    filter_doc_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Restrict search to specific document IDs. "
            "Null = search all documents."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of chunks to retrieve (default from settings)",
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to pass to LLM after reranking",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results (set to False to force fresh retrieval)",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be blank")
        return v.strip()


class SourceReference(BaseModel):
    """A source citation for a chunk used in the answer."""
    document_id: str
    document_name: str
    chunk_index: int
    citation_label: str
    doc_source_type: str
    page_number: Optional[int] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response to a knowledge base query."""
    answer: str
    sources: list[SourceReference]
    citations: list[str]           # Inline citations extracted from answer
    was_cached: bool
    was_blocked: bool
    block_reason: Optional[str] = None
    model: str
    prompt_tokens: int
    completion_tokens: int
    grounding_score: float         # Fraction of sentences with citations
    faithfulness_score: Optional[float] = None
    has_refusal: bool              # True if model couldn't find answer
    latency_ms: int
    query_log_id: Optional[str] = None


# ============================================================
# Admin Schemas
# ============================================================

class CacheStatsResponse(BaseModel):
    """Cache layer statistics."""
    cache_enabled: bool
    redis_available: bool
    local_cache_size: int
    ttl_seconds: int
    redis_url: str


class SystemHealthResponse(BaseModel):
    """System health check response."""
    status: str                  # "healthy" | "degraded" | "unhealthy"
    database: str
    cache: str
    embedder: str
    details: dict[str, Any] = Field(default_factory=dict)


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document."""
    success: bool
    document_id: str
    chunks_deleted: int
    message: str
