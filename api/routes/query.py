"""
query.py
--------
FastAPI route for knowledge base queries.

The full RAG pipeline in one endpoint:
  1. Guardrails (input): injection → PII → topic
  2. Cache check
  3. Hybrid retrieval (semantic + keyword)
  4. Cross-encoder reranking
  5. Grounded generation
  6. Guardrails (output): faithfulness → PII
  7. Cache store
  8. Audit log
  9. Return structured response
"""

from __future__ import annotations

import time
import uuid
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from api.models.schemas import QueryRequest, QueryResponse, SourceReference
from src.cache.query_cache import QueryCache, make_cache_key
from src.generation.generator import GroundedGenerator
from src.guardrails.pipeline import GuardrailPipeline
from src.retrieval.hybrid_search import HybridSearcher, RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.prompts import format_context
from src.config.settings import get_settings

router = APIRouter(tags=["Query"])


def get_searcher() -> HybridSearcher:
    return HybridSearcher()


def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def get_generator() -> GroundedGenerator:
    return GroundedGenerator()


def get_guardrails() -> GuardrailPipeline:
    return GuardrailPipeline()


def get_cache() -> QueryCache:
    return QueryCache()


@router.post("/query", response_model=QueryResponse, summary="Query the knowledge base")
async def query_knowledge_base(
    request: QueryRequest,
    searcher: HybridSearcher = Depends(get_searcher),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    generator: GroundedGenerator = Depends(get_generator),
    guardrails: GuardrailPipeline = Depends(get_guardrails),
    cache: QueryCache = Depends(get_cache),
):
    """
    Query the enterprise knowledge base.

    Full pipeline: guardrails → cache → retrieval → reranking → generation → guardrails.
    Returns a grounded answer with source citations.
    """
    start_time = time.perf_counter()
    settings = get_settings()
    query_log_id = str(uuid.uuid4())

    logger.info(f"Query received: '{request.query[:80]}...' | log_id={query_log_id}")

    # ================================================================
    # 1. Input Guardrails
    # ================================================================
    guard_result = guardrails.check_input(request.query)

    if not guard_result.input_decision.passed:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            f"Query blocked by {guard_result.input_decision.blocked_by} | "
            f"latency={latency_ms}ms"
        )
        # Log the violation
        await _log_guardrail_violation(
            query_log_id=query_log_id,
            query=request.query,
            violation_type=guard_result.input_decision.blocked_by,
            flags=guard_result.to_audit_dict(),
        )
        return QueryResponse(
            answer=guard_result.input_decision.safe_response or "Request blocked.",
            sources=[],
            citations=[],
            was_cached=False,
            was_blocked=True,
            block_reason=guard_result.input_decision.reason,
            model="blocked",
            prompt_tokens=0,
            completion_tokens=0,
            grounding_score=0.0,
            has_refusal=False,
            latency_ms=latency_ms,
            query_log_id=query_log_id,
        )

    # Use redacted query if PII was found (prevents PII embedding)
    safe_query = guard_result.input_decision.redacted_query or request.query

    # ================================================================
    # 2. Cache Check
    # ================================================================
    cache_hit = False
    if request.use_cache:
        cached = cache.get(safe_query, request.filter_doc_ids)
        if cached:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"Cache hit | latency={latency_ms}ms")
            cached["was_cached"] = True
            cached["latency_ms"] = latency_ms
            cached["query_log_id"] = query_log_id
            return QueryResponse(**cached)

    # ================================================================
    # 3. Hybrid Retrieval
    # ================================================================
    top_k = request.top_k or settings.RETRIEVAL_TOP_K
    chunks = searcher.search(
        query=safe_query,
        top_k=top_k,
        filter_doc_ids=request.filter_doc_ids,
    )
    logger.debug(f"Retrieved {len(chunks)} candidates")

    # ================================================================
    # 4. Reranking
    # ================================================================
    top_n = request.top_n or settings.RERANK_TOP_N
    if chunks:
        reranked_chunks = reranker.rerank(safe_query, chunks, top_n=top_n)
        logger.debug(f"Reranked to top {len(reranked_chunks)}")
    else:
        reranked_chunks = []

    # ================================================================
    # 5. Grounded Generation
    # ================================================================
    generation = generator.generate(query=safe_query, chunks=reranked_chunks)

    # Build context string for faithfulness check
    context_text = format_context(reranked_chunks) if reranked_chunks else ""

    # ================================================================
    # 6. Output Guardrails
    # ================================================================
    guard_result = guardrails.check_output(
        result=guard_result,
        generation=generation,
        context=context_text,
    )

    final_answer = guard_result.final_answer or generation.answer

    # ================================================================
    # 7. Build Response
    # ================================================================
    sources = [
        SourceReference(
            document_id=c.document_id,
            document_name=c.doc_name,
            chunk_index=c.chunk_index,
            citation_label=c.citation_label(),
            doc_source_type=c.doc_source_type,
            page_number=c.page_number,
            rerank_score=c.rerank_score,
        )
        for c in reranked_chunks
    ]

    faithfulness_score = (
        guard_result.faithfulness.faithfulness_score
        if guard_result.faithfulness
        else None
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    response_data = {
        "answer": final_answer,
        "sources": [s.model_dump() for s in sources],
        "citations": generation.citations,
        "was_cached": False,
        "was_blocked": False,
        "block_reason": None,
        "model": generation.model,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "grounding_score": generation.grounding_score,
        "faithfulness_score": faithfulness_score,
        "has_refusal": generation.has_refusal,
        "latency_ms": latency_ms,
        "query_log_id": query_log_id,
    }

    # ================================================================
    # 8. Cache Store
    # ================================================================
    if request.use_cache and not generation.has_refusal:
        # Don't cache refusals: if KB is updated, the answer might change
        cache.set(safe_query, response_data, request.filter_doc_ids)

    # ================================================================
    # 9. Audit Log
    # ================================================================
    await _log_query(
        query_log_id=query_log_id,
        query=safe_query,
        retrieved_ids=[c.chunk_id for c in chunks],
        reranked_ids=[c.chunk_id for c in reranked_chunks],
        response=final_answer,
        guardrail_flags=guard_result.to_audit_dict(),
        latency_ms=latency_ms,
        cache_hit=cache_hit,
    )

    logger.info(
        f"Query complete | latency={latency_ms}ms | "
        f"chunks={len(reranked_chunks)} | "
        f"faithfulness={f'{faithfulness_score:.2f}' if faithfulness_score is not None else 'N/A'} | ..."
        f"tokens={generation.total_tokens()}"
    )

    return QueryResponse(**response_data)


# ============================================================
# Audit logging helpers (fire-and-forget, non-blocking)
# ============================================================

async def _log_query(
    query_log_id: str,
    query: str,
    retrieved_ids: list[str],
    reranked_ids: list[str],
    response: str,
    guardrail_flags: dict,
    latency_ms: int,
    cache_hit: bool,
) -> None:
    """Insert a row into query_logs for audit and analytics."""
    try:
        from supabase import create_client
        import hashlib
        settings = get_settings()
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()

        supabase.table("query_logs").insert({
            "id": query_log_id,
            "query_text": query[:2000],  # Cap length
            "query_hash": query_hash,
            "retrieved_chunk_ids": retrieved_ids,
            "reranked_chunk_ids": reranked_ids,
            "response_text": response[:5000],
            "guardrail_flags": guardrail_flags,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
        }).execute()
    except Exception as e:
        logger.warning(f"Audit log failed (non-fatal): {e}")


async def _log_guardrail_violation(
    query_log_id: str,
    query: str,
    violation_type: str,
    flags: dict,
) -> None:
    """Insert a guardrail violation for security auditing."""
    try:
        from supabase import create_client
        settings = get_settings()
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

        type_mapping = {
            "injection_guard": "prompt_injection",
            "pii_filter_input": "pii_input",
            "pii_filter_output": "pii_output",
            "topic_guard": "off_topic",
        }
        vtype = type_mapping.get(violation_type, "other")

        supabase.table("guardrail_violations").insert({
            "violation_type": vtype,
            "severity": flags.get("injection_severity", "medium"),
            "details": {**flags, "query_preview": query[:200]},
        }).execute()
    except Exception as e:
        logger.warning(f"Violation log failed (non-fatal): {e}")
