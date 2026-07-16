"""
ingest.py
---------
FastAPI routes for document ingestion.

Endpoints:
  POST /ingest/file    — Upload a PDF or DOCX file
  POST /ingest/url     — Ingest a web page by URL
  GET  /documents      — List all ingested documents
  DELETE /documents/{id} — Delete a document and its chunks
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from loguru import logger

from api.models.schemas import (
    IngestResponse,
    IngestURLRequest,
    DocumentListResponse,
    DocumentListItem,
    DeleteDocumentResponse,
)
from src.ingestion.pipeline import IngestionPipeline
from src.cache.query_cache import QueryCache
from src.config.settings import get_settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Allowed MIME types for file upload
ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
}

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def get_pipeline() -> IngestionPipeline:
    """Dependency: return a shared ingestion pipeline instance."""
    return IngestionPipeline()


def get_cache() -> QueryCache:
    return QueryCache()


@router.post("/file", response_model=IngestResponse, summary="Upload a PDF or DOCX file")
async def ingest_file(
    file: UploadFile = File(..., description="PDF or DOCX file to ingest"),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    cache: QueryCache = Depends(get_cache),
):
    """
    Ingest a document file into the knowledge base.

    Accepts PDF and DOCX files. Processes the file:
      1. Extract text
      2. Chunk into segments
      3. Embed with Voyage AI
      4. Store in Supabase

    Duplicate files (same content hash) are detected and skipped.
    """
    # Validate content type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Supported types: PDF, DOCX"
            ),
        )

    # Read file bytes
    data = await file.read()

    # Size check
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data) / 1e6:.1f} MB. Maximum: 50 MB",
        )

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info(f"File upload: {file.filename} ({len(data) / 1024:.1f} KB)")

    # Ingest
    result = pipeline.ingest_bytes(data, file.filename)

    # Invalidate cache after new content (prevents stale answers)
    if result.success and not result.was_duplicate:
        cache.invalidate_all()
        logger.info("Cache invalidated after new document ingestion")

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Ingestion failed")

    return IngestResponse(
        success=result.success,
        document_id=result.document_id,
        document_name=result.source_name,
        source_type=result.source_type,
        chunks_created=result.chunks_created,
        chunks_skipped=result.chunks_skipped,
        was_duplicate=result.was_duplicate,
        error=result.error,
    )


@router.post("/url", response_model=IngestResponse, summary="Ingest a web page")
async def ingest_url(
    request: IngestURLRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    cache: QueryCache = Depends(get_cache),
):
    """
    Ingest a web page into the knowledge base by URL.

    Fetches and extracts the main content from the page using
    trafilatura (removes navigation, ads, boilerplate).
    """
    logger.info(f"URL ingestion request: {request.url}")

    result = pipeline.ingest_url(request.url)

    if result.success and not result.was_duplicate:
        cache.invalidate_all()

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error or f"Failed to ingest URL: {request.url}",
        )

    return IngestResponse(
        success=result.success,
        document_id=result.document_id,
        document_name=result.source_name,
        source_type=result.source_type,
        chunks_created=result.chunks_created,
        chunks_skipped=result.chunks_skipped,
        was_duplicate=result.was_duplicate,
        error=result.error,
    )


@router.get("/documents", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(
    source_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all documents in the knowledge base.

    Optionally filter by source_type: 'pdf' | 'docx' | 'web'.
    """
    from supabase import create_client
    settings = get_settings()
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    query = (
        supabase.table("documents")
        .select("id, name, source_type, source_path, total_chunks, created_at, metadata")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    if source_type:
        query = query.eq("source_type", source_type)

    response = query.execute()
    docs = response.data or []

    total_response = supabase.table("documents").select("id", count="exact").execute()
    total = total_response.count or len(docs)

    return DocumentListResponse(
        documents=[
            DocumentListItem(
                id=d["id"],
                name=d["name"],
                source_type=d["source_type"],
                source_path=d["source_path"],
                total_chunks=d["total_chunks"] or 0,
                created_at=d["created_at"],
                metadata=d.get("metadata") or {},
            )
            for d in docs
        ],
        total=total,
    )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    cache: QueryCache = Depends(get_cache),
):
    """
    Delete a document and all its chunks from the knowledge base.

    The CASCADE constraint in the schema ensures chunks are automatically
    deleted when the parent document is removed.
    """
    from supabase import create_client
    settings = get_settings()
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    # Get chunk count before deletion (for response)
    chunks_response = (
        supabase.table("chunks")
        .select("id", count="exact")
        .eq("document_id", document_id)
        .execute()
    )
    chunk_count = chunks_response.count or 0

    # Delete document (chunks cascade)
    response = (
        supabase.table("documents")
        .delete()
        .eq("id", document_id)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    # Invalidate cache (answers may reference deleted document)
    cache.invalidate_all()

    logger.info(f"Document deleted: {document_id} ({chunk_count} chunks)")

    return DeleteDocumentResponse(
        success=True,
        document_id=document_id,
        chunks_deleted=chunk_count,
        message=f"Document and {chunk_count} chunks deleted successfully",
    )
