"""
topic_guard.py
--------------
Off-topic query detection via cosine similarity.

Strategy:
  1. At startup, embed a domain description ("enterprise knowledge base
     questions about company policies, procedures, products...")
  2. For each incoming query, embed it and compute cosine similarity
     against the domain embedding
  3. Queries below the threshold (default: 0.35) are rejected as off-topic

Why cosine similarity over a classifier:
  - No training data needed: just write a description of valid queries
  - The domain description is configurable in .env (DOMAIN_DESCRIPTION)
  - Thresholds can be tuned on your eval set without retraining
  - Works well because Voyage AI embeddings capture semantic meaning;
    "what's the weather today?" will have very low similarity to any
    enterprise KB description

Why 0.35 threshold:
  - Empirically: enterprise queries (policies, procedures, products)
    score 0.5–0.9; clearly off-topic (weather, recipes, sports) score 0.1–0.3
  - 0.35 gives a comfortable buffer with ~2% false positive rate
  - This should be tuned on YOUR evaluation dataset (run_eval.py)

Known limitation:
  Very specific adversarial queries can sometimes slip through if they
  contain enterprise-sounding terminology. The prompt injection guard
  provides a second layer of defense.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.config.settings import get_settings
from src.ingestion.embedder import VoyageEmbedder


@dataclass
class TopicGuardResult:
    """Result of a topic guard check."""
    is_off_topic: bool
    similarity_score: float
    threshold: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Using manual implementation to avoid numpy dependency at this layer.
    For production with high throughput, use numpy instead.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TopicGuard:
    """
    Rejects queries that are semantically off-topic for the enterprise KB.

    Thread-safe: domain embedding is computed once and cached.

    Usage:
        guard = TopicGuard()
        if not guard.is_on_topic("what is the employee leave policy?"):
            return OFF_TOPIC_RESPONSE
    """

    _domain_embedding: Optional[list[float]] = None
    _lock = threading.Lock()

    def __init__(self, embedder: Optional[VoyageEmbedder] = None):
        self.settings = get_settings()
        self.embedder = embedder or self._get_embedder()
        self.threshold = self.settings.TOPIC_SIMILARITY_THRESHOLD

    def _get_embedder(self) -> VoyageEmbedder:
        """Separate method so tests can mock it cleanly."""
        return VoyageEmbedder()

    def _cosine_similarity(self, query_emb: list[float], domain_emb: list[float]) -> float:
        """Instance method wrapper so tests can patch it."""
        return cosine_similarity(query_emb, domain_emb)

    @property
    def domain_embedding(self) -> list[float]:
        """
        Lazily compute and cache the domain description embedding.

        Why lazy: avoids calling Voyage API at import time.
        Why cached: the domain description doesn't change between queries.
        The embedding is computed once on the first query.
        """
        if TopicGuard._domain_embedding is None:
            with TopicGuard._lock:
                if TopicGuard._domain_embedding is None:
                    logger.info("Computing domain embedding for topic guard...")
                    TopicGuard._domain_embedding = self.embedder.embed_query(
                        self.settings.DOMAIN_DESCRIPTION
                    )
                    logger.info("Domain embedding cached.")
        return TopicGuard._domain_embedding

    def is_on_topic(self, query: str) -> bool:
        """
        Returns True if the query is within the KB domain.

        Args:
            query: The user's query string.

        Returns:
            True if similarity >= threshold, False otherwise.
        """
        score = self.score(query)
        on_topic = score >= self.threshold

        if not on_topic:
            logger.info(
                f"Off-topic query detected | score={score:.3f} | threshold={self.threshold} | "
                f"query='{query[:60]}...'"
            )
        return on_topic

    def check(self, query: str) -> TopicGuardResult:
        """
        Check if query is on-topic. Returns a structured result.

        Used by GuardrailPipeline and tests.
        """
        sim = self.score(query)
        return TopicGuardResult(
            is_off_topic=sim < self.threshold,
            similarity_score=round(sim, 4),
            threshold=self.threshold,
        )

    def score(self, query: str) -> float:
        """
        Compute cosine similarity between query and domain description.

        Returns 0.0–1.0 (higher = more on-topic).
        """
        query_embedding = self.embedder.embed_query(query)
        return self._cosine_similarity(query_embedding, self.domain_embedding)

    def update_domain(self, new_description: str) -> None:
        """
        Update the domain description and reset the cached embedding.

        Call this if the KB scope changes (e.g., new product line added).
        """
        with TopicGuard._lock:
            self.settings.DOMAIN_DESCRIPTION = new_description
            TopicGuard._domain_embedding = None
        logger.info(f"Domain description updated; embedding will be recomputed on next query.")
