"""
main.py
-------
FastAPI application entry point.

Startup sequence (lifespan):
  1. Validate settings (fail fast before serving any traffic)
  2. Test Supabase connection
  3. Test Redis connection (non-fatal if unavailable)
  4. Warm up cross-encoder model (avoids cold start on first query)
  5. Compute domain embedding for topic guard

These warm-up steps mean the first query pays ~0 extra latency vs cold start.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from api.models.schemas import SystemHealthResponse, CacheStatsResponse
from src.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context: runs startup tasks before serving,
    cleanup tasks after shutdown.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # --- Startup ---
    try:
        # 1. Test Supabase connection
        from supabase import create_client
        sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        sb.table("documents").select("id").limit(1).execute()
        logger.info("Supabase connection: OK")
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        # Don't exit — let health check surface this

    try:
        # 2. Warm up reranker (loads model into memory)
        from src.retrieval.reranker import CrossEncoderReranker
        _ = CrossEncoderReranker().model
        logger.info("Cross-encoder warmed up.")
    except Exception as e:
        logger.warning(f"Reranker warm-up failed: {e}")

    try:
        # 3. Warm up faithfulness guard
        from src.guardrails.faithfulness_guard import FaithfulnessGuard
        _ = FaithfulnessGuard().model
        logger.info("Faithfulness guard warmed up.")
    except Exception as e:
        logger.warning(f"Faithfulness guard warm-up failed: {e}")

    try:
        # 4. Pre-compute domain embedding for topic guard
        from src.guardrails.topic_guard import TopicGuard
        _ = TopicGuard().domain_embedding
        logger.info("Topic guard domain embedding cached.")
    except Exception as e:
        logger.warning(f"Topic guard warm-up failed: {e}")

    logger.info(f"Server ready on http://{settings.API_HOST}:{settings.API_PORT}")

    yield  # Serve requests

    # --- Shutdown ---
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise knowledge assistant with hybrid search, "
            "cross-encoder reranking, and strict grounding guardrails."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS ---
    # Allow Streamlit frontend (localhost:8501) and any configured origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(ingest_router)
    app.include_router(query_router)

    # --- Health & Admin ---
    @app.get("/health", response_model=SystemHealthResponse, tags=["System"])
    async def health_check():
        """System health check. Used by load balancers and monitoring."""
        status = "healthy"
        db_status = "unknown"
        cache_status = "unknown"
        embedder_status = "unknown"

        try:
            from supabase import create_client
            sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            sb.table("documents").select("id").limit(1).execute()
            db_status = "healthy"
        except Exception as e:
            db_status = f"unhealthy: {e}"
            status = "degraded"

        try:
            from src.cache.query_cache import QueryCache
            cache = QueryCache()
            cache_stats = cache.stats()
            cache_status = "redis" if cache_stats["redis_available"] else "local_only"
        except Exception as e:
            cache_status = f"unhealthy: {e}"
            status = "degraded"

        try:
            from src.ingestion.embedder import VoyageEmbedder
            emb = VoyageEmbedder()
            test_emb = emb.embed_query("test")
            embedder_status = "healthy" if len(test_emb) == settings.EMBEDDING_DIM else "dim_mismatch"
        except Exception as e:
            embedder_status = f"unhealthy: {e}"
            status = "degraded"

        return SystemHealthResponse(
            status=status,
            database=db_status,
            cache=cache_status,
            embedder=embedder_status,
            details={
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "llm_provider": settings.LLM_PROVIDER,
                "voyage_model": settings.VOYAGE_MODEL,
            },
        )

    @app.get("/cache/stats", response_model=CacheStatsResponse, tags=["System"])
    async def cache_stats():
        """Cache layer statistics."""
        from src.cache.query_cache import QueryCache
        return CacheStatsResponse(**QueryCache().stats())

    @app.post("/cache/clear", tags=["System"])
    async def clear_cache():
        """Manually clear the query cache."""
        from src.cache.query_cache import QueryCache
        QueryCache().invalidate_all()
        return {"message": "Cache cleared"}

    # --- Global exception handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.value.lower(),
    )
