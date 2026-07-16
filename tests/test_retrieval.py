"""
test_retrieval.py
-----------------
Tests for hybrid search and cross-encoder reranking using correct field names.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestHybridSearcher:

    @pytest.mark.unit
    @patch("src.retrieval.hybrid_search.create_client")
    def test_search_returns_retrieved_chunks(self, mock_create_client, mock_embedder):
        from src.retrieval.hybrid_search import HybridSearcher, RetrievedChunk

        mock_supabase = MagicMock()
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "chunk_id": "abc-123",
                "document_id": "doc-001",
                "doc_name": "HR Manual",
                "content": "Employees get 25 days annual leave.",
                "doc_source_type": "pdf",
                "chunk_index": 0,
                "rrf_score": 0.033,
                "semantic_score": 0.91,
                "keyword_score": 0.80,
                "semantic_rank": 1,
                "keyword_rank": 2,
                "metadata": {"page_number": 5},
            }
        ]
        mock_create_client.return_value = mock_supabase

        searcher = HybridSearcher()
        searcher.embedder = mock_embedder
        results = searcher.search("annual leave entitlement")

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].doc_name == "HR Manual"
        assert results[0].rrf_score == pytest.approx(0.033)
        assert results[0].page_number == 5  # direct field now
        assert results[0].semantic_score == pytest.approx(0.91)

    @pytest.mark.unit
    @patch("src.retrieval.hybrid_search.create_client")
    def test_empty_result_returns_empty_list(self, mock_create_client, mock_embedder):
        mock_supabase = MagicMock()
        mock_supabase.rpc.return_value.execute.return_value.data = []
        mock_create_client.return_value = mock_supabase

        from src.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        searcher.embedder = mock_embedder
        results = searcher.search("nonexistent topic xyz")
        assert results == []

    @pytest.mark.unit
    @patch("src.retrieval.hybrid_search.create_client")
    def test_search_calls_rpc_with_correct_function_name(self, mock_create_client, mock_embedder):
        mock_supabase = MagicMock()
        mock_supabase.rpc.return_value.execute.return_value.data = []
        mock_create_client.return_value = mock_supabase

        from src.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        searcher.embedder = mock_embedder
        searcher.search("test query", top_k=10)

        call_args = mock_supabase.rpc.call_args
        assert call_args[0][0] == "hybrid_search"


class TestRetrievedChunkCitation:

    @pytest.mark.unit
    def test_citation_label_with_page(self, sample_chunks):
        chunk = sample_chunks[0]  # page_number=5
        label = chunk.citation_label()
        assert "HR Policy Manual" in label
        assert "5" in label

    @pytest.mark.unit
    def test_citation_label_without_page(self, sample_chunks):
        chunk = sample_chunks[2]  # page_number=None
        label = chunk.citation_label()
        assert "Employee Handbook" in label

    @pytest.mark.unit
    def test_rerank_score_is_direct_field(self, sample_chunks):
        """rerank_score should be a direct field, not in metadata."""
        chunk = sample_chunks[0]
        assert chunk.rerank_score == pytest.approx(0.95)
        assert chunk.page_number == 5
        assert chunk.semantic_score == pytest.approx(0.91)


class TestCrossEncoderReranker:

    @pytest.mark.unit
    def test_reranker_returns_at_most_top_n(self, sample_chunks):
        import numpy as np
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.6, 0.3])

        # Inject model via class variable (thread-safe singleton pattern)
        CrossEncoderReranker._model = mock_model
        try:
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker.settings = MagicMock(RERANK_TOP_N=5, RERANKER_MODEL="test")
            reranker.model_name = "test"
            reranker.top_n = 5
            results = reranker.rerank("What is the leave policy?", sample_chunks, top_n=2)
            assert len(results) <= 2
        finally:
            CrossEncoderReranker._model = None

    @pytest.mark.unit
    def test_reranker_sorts_descending(self, sample_chunks):
        import numpy as np
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.7])

        CrossEncoderReranker._model = mock_model
        try:
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker.top_n = 5
            results = reranker.rerank("leave policy", sample_chunks, top_n=3)
            scores = [r.rerank_score for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            CrossEncoderReranker._model = None

    @pytest.mark.unit
    def test_reranker_empty_input_returns_empty(self):
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        results = reranker.rerank("leave policy", [], top_n=5)
        assert results == []

    @pytest.mark.unit
    def test_rerank_score_set_as_direct_field(self, sample_chunks):
        """After reranking, rerank_score should be a direct field, not in metadata."""
        import numpy as np
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.4])

        CrossEncoderReranker._model = mock_model
        try:
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker.top_n = 3
            results = reranker.rerank("leave", sample_chunks[:3], top_n=3)
            for chunk in results:
                assert chunk.rerank_score is not None
                assert isinstance(chunk.rerank_score, float)
        finally:
            CrossEncoderReranker._model = None


class TestContextPrecisionMetric:

    @pytest.mark.unit
    def test_perfect_precision(self):
        from src.evaluation.metrics import context_precision_score
        result = context_precision_score(
            ["HR Policy Manual", "HR Policy Manual", "Employee Handbook"],
            ["HR Policy Manual", "Employee Handbook"],
            at_k=3,
        )
        assert result.score == pytest.approx(1.0)

    @pytest.mark.unit
    def test_zero_precision(self):
        from src.evaluation.metrics import context_precision_score
        result = context_precision_score(
            ["Unrelated Doc", "Another Doc"],
            ["HR Policy Manual"],
            at_k=2,
        )
        assert result.score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_partial_precision(self):
        from src.evaluation.metrics import context_precision_score
        result = context_precision_score(
            ["HR Policy Manual", "Unrelated Doc"],
            ["HR Policy Manual"],
            at_k=2,
        )
        assert result.score == pytest.approx(0.5)

    @pytest.mark.unit
    def test_empty_retrieved(self):
        from src.evaluation.metrics import context_precision_score
        result = context_precision_score([], ["HR Policy Manual"], at_k=5)
        assert result.score == pytest.approx(0.0)
        assert result.passed is False
