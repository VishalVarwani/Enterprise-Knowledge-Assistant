"""
settings.py
-----------
Single source of truth for all runtime configuration.

Why Pydantic Settings?
  - Validates types at startup (fail fast, not at 2 AM during a query)
  - Reads from .env, environment variables, or direct override
  - IDE-completable; no magic string lookups scattered across the codebase

Design rule: never import os.environ directly anywhere else.
Always import `settings` from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from functools import lru_cache

from pydantic import Field, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """
    All configuration for the Enterprise Knowledge Assistant.

    Precedence (highest → lowest):
      1. Environment variables
      2. .env file
      3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars (safe for CI/CD)
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    APP_NAME: str = "Enterprise Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: LogLevel = LogLevel.INFO
    ENVIRONMENT: str = "development"  # development | staging | production

    # ------------------------------------------------------------------
    # Supabase / Postgres
    # Why Supabase: pgvector + standard SQL in one managed service.
    # No extra infra for the vector layer; JOIN vector results with
    # relational tables natively.
    # ------------------------------------------------------------------
    SUPABASE_URL: str = Field(..., description="Your Supabase project URL")
    SUPABASE_SERVICE_KEY: str = Field(
        ..., description="Service-role key (bypasses RLS for server-side ops)"
    )
    DATABASE_URL: str = Field(
        ...,
        description=(
            "Direct Postgres connection string for raw SQL/psycopg2. "
            "Format: postgresql://user:password@host:port/db"
        ),
    )

    # ------------------------------------------------------------------
    # Voyage AI
    # Why voyage-3-lite: matches or beats OpenAI text-embedding-3-large
    # on BEIR retrieval benchmarks; 512 dimensions keeps pgvector index
    # size ~50% smaller than 1536-dim embeddings.
    # ------------------------------------------------------------------
    VOYAGE_API_KEY: str = Field(..., description="Voyage AI API key")
    VOYAGE_MODEL: str = "voyage-3-lite"
    VOYAGE_BATCH_SIZE: int = Field(
        default=128,
        description=(
            "Max texts per Voyage API call. "
            "128 balances throughput and request size limits."
        ),
    )
    EMBEDDING_DIM: int = Field(
        default=512,
        description="Must match your Voyage model's output dimension.",
    )

    # ------------------------------------------------------------------
    # LLM (Generation)
    # Default: Groq + llama-3.3-70b-versatile
    # Why Groq: ~350 tokens/s; enterprise users notice >2s latency.
    # ------------------------------------------------------------------
    LLM_PROVIDER: LLMProvider = LLMProvider.GROQ
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        description=(
            "0.0 for grounded Q&A: we want deterministic, fact-derived answers. "
            "Higher temps introduce more 'creativity' = more hallucination risk."
        ),
    )
    LLM_MAX_TOKENS: int = 1024

    # ------------------------------------------------------------------
    # Chunking
    # Why these numbers:
    #   chunk_size=512 tokens ≈ ~350 words; fits one dense paragraph.
    #   chunk_overlap=50 tokens: prevents hard cutoffs splitting a sentence
    #   that bridges two chunks.
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = Field(
        default=512,
        description="Token target per chunk. Match to Voyage context window.",
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Overlap in tokens between consecutive chunks.",
    )
    CHUNK_MIN_LENGTH: int = Field(
        default=50,
        description="Discard chunks shorter than this (headers, page numbers, etc.).",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    RETRIEVAL_TOP_K: int = Field(
        default=20,
        description=(
            "Number of candidates fetched per retrieval pass (before reranking). "
            "20 gives reranker enough coverage without excessive latency."
        ),
    )
    RERANK_TOP_N: int = Field(
        default=5,
        description=(
            "Final context chunks passed to LLM after reranking. "
            "5 × 512 tokens ≈ 2560 tokens of context; fits most LLM windows "
            "while keeping prompt cost low."
        ),
    )
    # Hybrid search weights before RRF fusion
    # RRF is rank-based so these weights are for the retrieval candidates,
    # not raw scores. Semantic gets more weight for conceptual queries.
    HYBRID_SEMANTIC_WEIGHT: float = Field(default=0.7, ge=0.0, le=1.0)
    HYBRID_KEYWORD_WEIGHT: float = Field(default=0.3, ge=0.0, le=1.0)
    RRF_K: int = Field(
        default=60,
        description=(
            "RRF constant. 60 is the community-standard value from the "
            "original Cormack et al. paper; reduces sensitivity to high-rank outliers."
        ),
    )

    # Reranker model: MiniLM L-6 v2 trained on MS MARCO passage ranking.
    # 22M params; runs on CPU in ~50ms for 20 candidates.
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    # Off-topic guard: embed the domain description once, compare cosine
    # similarity of every query against it. Below threshold → refuse.
    DOMAIN_DESCRIPTION: str = Field(
        default=(
            "enterprise knowledge base questions about company policies, "
            "procedures, products, technical documentation, and internal processes"
        ),
        description=(
            "Plain-English description of valid query domain. "
            "Used by topic guard for cosine similarity threshold."
        ),
    )
    TOPIC_SIMILARITY_THRESHOLD: float = Field(
        default=0.10,
        description=(
            "Cosine similarity below this → off-topic refusal. "
            "0.10 empirically chosen; tune on your eval set."
        ),
    )
    # Faithfulness guard: NLI model checks if answer is entailed by context.
    # Using DeBERTa-v3 NLI: SOTA on NLI benchmarks, 86M params, runs CPU.
    FAITHFULNESS_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    FAITHFULNESS_THRESHOLD: float = Field(
        default=0.5,
        description=(
            "NLI entailment score below this → flag potential hallucination. "
            "0.5 balances false positive rate vs catch rate."
        ),
    )
    # Prompt injection guard patterns file
    INJECTION_PATTERNS_PATH: str = "src/guardrails/injection_patterns.json"

    # ------------------------------------------------------------------
    # Cache (Redis)
    # Why Redis over in-process dict: survives restarts, works with multiple
    # Gunicorn workers, native TTL, ~0.1ms get latency.
    # ------------------------------------------------------------------
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )
    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        description=(
            "1-hour TTL: enterprise KB updates at most a few times per day; "
            "stale cache risk is low, latency savings are high."
        ),
    )
    CACHE_ENABLED: bool = True

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = Field(
        default=False,
        description="Enable hot-reload (dev only; disable in production).",
    )
    API_CORS_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("API_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Accept both comma-separated string and JSON array in .env"""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["http://localhost:8501"]
            if v.startswith("["):
                import json
                return json.loads(v)
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("HYBRID_SEMANTIC_WEIGHT", "HYBRID_KEYWORD_WEIGHT")
    @classmethod
    def weights_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Hybrid search weights must be non-negative")
        return v

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("LLM temperature must be between 0.0 and 2.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton accessor.

    Use this everywhere instead of instantiating Settings() directly.
    The @lru_cache ensures .env is read only once at startup.

    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()
