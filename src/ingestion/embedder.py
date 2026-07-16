"""
embedder.py
-----------
Voyage AI embedding client for the ingestion and retrieval pipelines.

Why Voyage AI:
  - voyage-3-lite outperforms OpenAI text-embedding-3-large on BEIR
    retrieval benchmarks while being 40% cheaper per token
  - 512-dimension output: keeps pgvector HNSW index ~50% smaller
    than 1536-dim embeddings with minimal recall loss for enterprise KB
  - First-class support for RAG use case: separate input_type for
    "document" vs "query" (asymmetric embedding — they use different
    model heads optimized for each role)

Asymmetric embedding (critical for RAG precision):
  - input_type="document": used during INGESTION.
    Optimizes the embedding for "what content does this text contain?"
  - input_type="query": used during RETRIEVAL.
    Optimizes the embedding for "what am I looking for?"
  Embedding a query with the document head (or vice versa) measurably
  degrades retrieval performance.

Batching:
  - Voyage API allows 128 texts per request.
  - We batch to maximize throughput and minimize API overhead.
  - tqdm progress bar for large ingestion jobs.

Retry:
  - tenacity retries on rate limit (429) and transient errors (5xx)
  - Exponential backoff: 1s, 2s, 4s, 8s, up to 5 attempts
"""

from __future__ import annotations

from typing import Literal

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.config.settings import get_settings

try:
    import voyageai
except ImportError:
    raise ImportError("voyageai not installed. Run: pip install voyageai")

InputType = Literal["document", "query"]


class VoyageEmbedder:
    """
    Voyage AI embedding wrapper.

    Usage:
        embedder = VoyageEmbedder()

        # During ingestion:
        embeddings = embedder.embed_documents(["text1", "text2"])

        # During retrieval:
        query_emb = embedder.embed_query("what is the refund policy?")
    """

    def __init__(self):
        settings = get_settings()
        self.client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
        self.model = settings.VOYAGE_MODEL
        self.batch_size = settings.VOYAGE_BATCH_SIZE
        self.embedding_dim = settings.EMBEDDING_DIM
        logger.info(f"Voyage embedder initialized: model={self.model}, dim={self.embedding_dim}")

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embed a list of document texts.

        Uses input_type="document" — the encoder head optimized for
        representing content that will be retrieved.

        Args:
            texts        : List of text strings to embed.
            show_progress: Show tqdm progress bar for large batches.

        Returns:
            List of embedding vectors (one per text), each of length
            EMBEDDING_DIM.
        """
        return self._embed_batched(texts, input_type="document", show_progress=show_progress)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Uses input_type="query" — the encoder head optimized for
        representing what the user is searching for.

        Returns:
            Single embedding vector of length EMBEDDING_DIM.
        """
        results = self._embed_batched([text], input_type="query", show_progress=False)
        return results[0]

    def _embed_batched(
        self,
        texts: list[str],
        input_type: InputType,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Split texts into batches, embed each batch, collect results.

        Args:
            texts      : All texts to embed.
            input_type : "document" or "query" (Voyage asymmetric heads).
            show_progress: Show tqdm bar.

        Returns:
            Flat list of embeddings in original text order.
        """
        if not texts:
            return []

        # Filter out empty strings (would error on Voyage API)
        valid_texts = [t if t and t.strip() else " " for t in texts]

        all_embeddings: list[list[float]] = []
        batches = [
            valid_texts[i : i + self.batch_size]
            for i in range(0, len(valid_texts), self.batch_size)
        ]

        iterator = tqdm(batches, desc=f"Embedding ({input_type})", unit="batch") \
            if show_progress and len(batches) > 1 else batches

        for batch in iterator:
            embeddings = self._embed_batch_with_retry(batch, input_type)
            all_embeddings.extend(embeddings)

        assert len(all_embeddings) == len(texts), (
            f"Embedding count mismatch: got {len(all_embeddings)}, "
            f"expected {len(texts)}"
        )
        return all_embeddings

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch_with_retry(
        self,
        texts: list[str],
        input_type: InputType,
    ) -> list[list[float]]:
        """
        Call Voyage API for a single batch, with exponential backoff retry.

        tenacity config:
          - Retries on ANY exception (catches network errors and 429s)
          - wait_exponential: 1s → 2s → 4s → 8s → 16s (capped at 30s)
          - 5 attempts max; raises original exception if all fail
        """
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            truncation=True,  # Truncate instead of error on overlong text
        )
        return result.embeddings

    def verify_dimensions(self, embedding: list[float]) -> bool:
        """
        Check that an embedding has the expected number of dimensions.
        Called once during startup to catch model/config mismatches.
        """
        if len(embedding) != self.embedding_dim:
            logger.error(
                f"Dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(embedding)}. Check VOYAGE_MODEL and EMBEDDING_DIM in .env"
            )
            return False
        return True
