"""
hybrid_search.py
----------------
Hybrid retrieval: semantic similarity + keyword full-text search, fused via RRF.

Why hybrid over pure semantic:
  - Semantic search: excels at conceptual/paraphrase queries
    e.g. "employee compensation" matches text about "salary" and "wages"
  - Keyword search: excels at exact-match queries
    e.g. "Form W-2", "Article 3.2(b)", product model numbers
  - Enterprise KBs have BOTH types of queries. A pure semantic system
    misses exact-match lookups. A pure keyword system misses synonyms.
  - RRF fusion costs near-zero extra latency (all SQL, one DB round trip)
    and consistently outperforms either method alone on BEIR benchmarks.

Reciprocal Rank Fusion (RRF):
  score = Σ 1/(k + rank_i)  for each result list i

  k=60: the standard value from the original paper (Cormack et al. 2009).
  Intuition: ranks near the top matter more than absolute scores.
  This normalizes across score scales (cosine vs ts_rank) automatically.

The actual SQL fusion runs inside Supabase as a stored function (hybrid_search)
for efficiency — one RPC call, no Python-side merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from supabase import Client, create_client

from src.config.settings import get_settings
from src.ingestion.embedder import VoyageEmbedder


@dataclass
class RetrievedChunk:
    """
    A search result chunk, ready for reranking and generation.

    Fields map directly to the hybrid_search SQL function output.
    Direct fields are used for everything so callers never have to
    dig into a metadata dict for common attributes.
    """
    chunk_id: str
    document_id: str
    content: str
    doc_name: str
    doc_source_type: str
    chunk_index: int
    rrf_score: float
    # Page number from PDF/DOCX (None for web content)
    page_number: Optional[int] = None
    # Score from cross-encoder reranker (set after reranking, None before)
    rerank_score: Optional[float] = None
    # Semantic similarity score from embedding search
    semantic_score: Optional[float] = None
    # BM25/FTS keyword score
    keyword_score: Optional[float] = None
    # Ranking positions in the two sorted lists (pre-fusion)
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    # Raw metadata from document (source_path, title, etc.)
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def citation_label(self) -> str:
        """
        Build a citation string for this chunk.

        Format: "Document Name, p. 7" or "Document Name, section 3"
        Used in generated responses for source attribution.
        """
        if self.page_number:
            return f"{self.doc_name}, p. {self.page_number}"
        return f"{self.doc_name}, section {self.chunk_index + 1}"


class HybridSearcher:
    """
    Executes hybrid semantic + keyword search against the Supabase KB.

    Usage:
        searcher = HybridSearcher()
        chunks = searcher.search("what is the refund policy?", top_k=20)
    """

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        embedder: Optional[VoyageEmbedder] = None,
    ):
        self.settings = get_settings()
        self.supabase = supabase_client or create_client(
            self.settings.SUPABASE_URL,
            self.settings.SUPABASE_SERVICE_KEY,
        )
        self.embedder = embedder or VoyageEmbedder()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid search for a query.

        Args:
            query          : The user's query string.
            top_k          : Number of results (default: RETRIEVAL_TOP_K from settings).
            filter_doc_ids : Restrict search to specific documents (None = all docs).

        Returns:
            List of RetrievedChunk objects sorted by descending RRF score.

        Flow:
            1. Embed query with Voyage (input_type="query")
            2. Call hybrid_search() Supabase RPC
            3. Parse results into RetrievedChunk objects
        """
        top_k = top_k or self.settings.RETRIEVAL_TOP_K

        # 1. Embed query
        logger.debug(f"Embedding query: {query[:80]}...")
        query_embedding = self.embedder.embed_query(query)

        # 2. Call stored hybrid search function via Supabase RPC
        # Why RPC instead of two separate calls + Python merge:
        #   - One network round trip (critical for latency)
        #   - RRF fusion happens in Postgres where both ranked lists coexist
        #   - Avoids sending 40 rows of vector data to Python just to merge
        rpc_params: dict = {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": top_k,
            "rrf_k": self.settings.RRF_K,
        }
        if filter_doc_ids:
            rpc_params["filter_doc_ids"] = filter_doc_ids

        logger.debug(f"Calling hybrid_search RPC | top_k={top_k}")
        response = self.supabase.rpc("hybrid_search", rpc_params).execute()

        if not response.data:
            logger.debug("Hybrid search returned no results")
            return []

        # 3. Parse results
        chunks = [self._parse_row(row) for row in response.data]
        logger.debug(
            f"Hybrid search: {len(chunks)} results | "
            f"top score={chunks[0].rrf_score:.4f}" if chunks else "no results"
        )
        return chunks

    def _parse_row(self, row: dict) -> RetrievedChunk:
        """Parse a single row from hybrid_search RPC response."""
        raw_meta = row.get("metadata") or {}
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            doc_name=row["doc_name"],
            doc_source_type=row["doc_source_type"],
            chunk_index=row["chunk_index"],
            rrf_score=float(row["rrf_score"]),
            page_number=raw_meta.get("page_number"),
            semantic_score=row.get("semantic_score"),
            keyword_score=row.get("keyword_score"),
            semantic_rank=row.get("semantic_rank"),
            keyword_rank=row.get("keyword_rank"),
            metadata=raw_meta,
        )
