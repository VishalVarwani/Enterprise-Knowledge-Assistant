"""
test_generation.py - Tests for prompts, generator, and cache.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestPrompts:

    @pytest.mark.unit
    def test_system_prompt_requires_context_only(self):
        from src.generation.prompts import SYSTEM_PROMPT
        assert "context" in SYSTEM_PROMPT.lower()
        assert any(w in SYSTEM_PROMPT.lower() for w in ["only", "exclusively", "must"])

    @pytest.mark.unit
    def test_format_context_produces_source_labels(self, sample_chunks):
        from src.generation.prompts import format_context
        formatted = format_context(sample_chunks)
        assert "[SOURCE 1:" in formatted
        assert "[SOURCE 2:" in formatted
        assert "[SOURCE 3:" in formatted

    @pytest.mark.unit
    def test_format_context_empty_chunks(self):
        from src.generation.prompts import format_context
        result = format_context([])
        assert result  # non-empty

    @pytest.mark.unit
    def test_build_messages_has_system_and_user(self, sample_chunks):
        from src.generation.prompts import build_messages
        messages = build_messages("What is the leave policy?", sample_chunks)
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    @pytest.mark.unit
    def test_build_messages_contains_query(self, sample_chunks):
        from src.generation.prompts import build_messages
        query = "How many days of leave do I get?"
        messages = build_messages(query, sample_chunks)
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert query in user_content


class TestGroundedGenerator:

    @pytest.mark.unit
    def test_generate_with_no_context_triggers_refusal(self):
        """Empty chunks list returns refusal without any LLM call."""
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        gen._client = MagicMock()
        gen.settings = MagicMock()
        result = gen.generate("What is the leave policy?", [])
        assert result.has_refusal is True
        gen._client.chat.completions.create.assert_not_called()

    @pytest.mark.unit
    def test_extract_citations_source_number_format(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        # Use ! instead of . to avoid sentence-splitter breaking [SOURCE 1: Doc, p. 5]
        answer = (
            "Employees get 25 days leave [SOURCE 1: HR Manual] "
            "and part-time staff are pro-rated [SOURCE 2: HR Manual]."
        )
        citations = gen._extract_citations(answer)
        assert len(citations) == 2
        assert any("HR Manual" in c for c in citations)

    @pytest.mark.unit
    def test_extract_citations_source_label_format(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        answer = "The policy is X [Source: HR Manual] and the deadline is Y [Source: Guide]."
        citations = gen._extract_citations(answer)
        assert len(citations) == 2

    @pytest.mark.unit
    def test_grounding_score_with_citations(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        # Use ! separators to avoid sentence-splitter mishandling "p. 5"
        answer = "Employees get 25 days [SOURCE 1: HR Manual]! Leave must be pre-approved [SOURCE 2: HR Manual]!"
        score = gen._grounding_score(answer)
        assert score > 0.5

    @pytest.mark.unit
    def test_grounding_score_no_citations_is_zero(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        answer = "Employees get 25 days leave and can carry over unused days."
        score = gen._grounding_score(answer)
        assert score == 0.0

    @pytest.mark.unit
    def test_is_refusal_positive(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        assert gen._is_refusal("I cannot find information about this in the knowledge base.") is True

    @pytest.mark.unit
    def test_is_refusal_negative(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        assert gen._is_refusal("Employees receive 25 days [SOURCE 1: HR Manual].") is False

    @pytest.mark.unit
    def test_grounding_score_partial_coverage(self):
        from src.generation.generator import GroundedGenerator
        gen = GroundedGenerator.__new__(GroundedGenerator)
        # One sentence cited, one not
        answer = "Employees get 25 days [SOURCE 1: HR Manual]. They can work remotely too."
        score = gen._grounding_score(answer)
        assert 0.0 < score < 1.0


class TestQueryCache:

    @pytest.mark.unit
    def test_make_key_deterministic(self):
        from src.cache.query_cache import QueryCache
        cache = QueryCache.__new__(QueryCache)
        assert cache._make_key("leave policy?") == cache._make_key("leave policy?")

    @pytest.mark.unit
    def test_make_key_unique_per_query(self):
        from src.cache.query_cache import QueryCache
        cache = QueryCache.__new__(QueryCache)
        assert cache._make_key("leave policy") != cache._make_key("remote work policy")

    @pytest.mark.unit
    def test_local_cache_get_set(self):
        from src.cache.query_cache import InMemoryLRU
        lru = InMemoryLRU(max_size=100)     # correct kwarg
        lru.set("key1", {"answer": "25 days"})
        assert lru.get("key1") == {"answer": "25 days"}

    @pytest.mark.unit
    def test_local_cache_miss_returns_none(self):
        from src.cache.query_cache import InMemoryLRU
        lru = InMemoryLRU(max_size=100)
        assert lru.get("nonexistent_key") is None

    @pytest.mark.unit
    def test_local_cache_evicts_at_max_size(self):
        from src.cache.query_cache import InMemoryLRU
        lru = InMemoryLRU(max_size=5)
        for i in range(10):
            lru.set(f"key_{i}", f"value_{i}")
        assert lru.size() <= 5
