"""
generator.py
------------
LLM-based grounded answer generator.

Responsibilities:
  1. Format retrieved chunks + query into a strict grounding prompt
  2. Call the configured LLM (Groq / OpenAI / Anthropic)
  3. Extract and validate inline citations from the response
  4. Return a structured GenerationResult with answer + source refs

Multi-provider design:
  - LLM_PROVIDER setting controls which client is used
  - All providers share the same messages format (OpenAI-compatible)
  - Groq is default: ~350 tokens/sec; enterprise users feel 2s+ latency
  - Provider switching requires only .env change, no code change

Citation extraction:
  - Parses [Source: X] patterns from LLM response
  - Cross-references with actually-retrieved sources
  - Flags any citations that reference non-retrieved sources
    (sign of hallucination: model cited a document that wasn't in context)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings, LLMProvider
from src.retrieval.hybrid_search import RetrievedChunk
from .prompts import (
    build_messages,
    NO_CONTEXT_RESPONSE,
    format_context,
)


@dataclass
class GenerationResult:
    """
    Structured output from the generator.

    Fields:
        answer          : The generated answer text.
        citations       : List of source citations extracted from the answer.
        retrieved_chunks: The context chunks that were provided to the LLM.
        model           : Which LLM model was used.
        prompt_tokens   : Input token count (for cost tracking).
        completion_tokens: Output token count.
        grounding_score : Fraction of sentences that contain a citation (0.0–1.0).
                          Proxy for answer groundedness. 1.0 = every claim cited.
        has_refusal     : True if the model said it couldn't find the answer.
    """
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    grounding_score: float = 0.0
    has_refusal: bool = False

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def sources_used(self) -> list[str]:
        """Unique document names actually cited in the answer."""
        return list(dict.fromkeys(self.citations))  # Deduplicated, order preserved


# Match both [Source: Doc, p. 3] (model output) and [SOURCE 1: Doc] (context labels)
CITATION_PATTERN = re.compile(
    r"\[SOURCE\s*\d*:?\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# Refusal signal phrases (model declining to answer)
REFUSAL_SIGNALS = [
    "cannot find information",
    "does not contain",
    "not in the knowledge base",
    "not available in the context",
    "no information about",
    "outside the scope",
]


class GroundedGenerator:
    """
    Generates answers grounded strictly in retrieved context.

    Usage:
        generator = GroundedGenerator()
        result = generator.generate(query="...", chunks=[...])
        print(result.answer)
        print(result.citations)
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = self._build_client()

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            query  : The user's original query.
            chunks : Top-N reranked context chunks.

        Returns:
            GenerationResult with answer, citations, token counts.

        If no chunks are provided, returns a refusal response without
        calling the LLM (saves tokens, avoids hallucination risk).
        """
        if not chunks:
            logger.info("No context available; returning no-context refusal")
            return GenerationResult(
                answer=NO_CONTEXT_RESPONSE,
                has_refusal=True,
            )

        messages = build_messages(query, chunks)

        logger.debug(
            f"Calling {self.settings.LLM_PROVIDER} | "
            f"model={self._model_name()} | "
            f"context_chunks={len(chunks)}"
        )

        try:
            raw_answer, prompt_tokens, completion_tokens = self._call_llm(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        # Post-process response
        citations = self._extract_citations(raw_answer)
        grounding_score = self._grounding_score(raw_answer)
        has_refusal = self._is_refusal(raw_answer)

        # Validate: warn if model cited a source not in the retrieved chunks
        retrieved_names = {c.doc_name for c in chunks}
        for citation in citations:
            # Check if any retrieved doc name is a substring of the citation
            if not any(name.lower() in citation.lower() for name in retrieved_names):
                logger.warning(
                    f"Possible hallucinated citation: '{citation}' not in "
                    f"retrieved sources: {retrieved_names}"
                )

        return GenerationResult(
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=chunks,
            model=self._model_name(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            grounding_score=grounding_score,
            has_refusal=has_refusal,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_llm(self, messages: list[dict]) -> tuple[str, int, int]:
        """
        Call the configured LLM provider.

        Returns (answer_text, prompt_tokens, completion_tokens).

        tenacity retries handle transient rate limits and network blips.
        3 attempts with 1s/2s/4s backoff.
        """
        provider = self.settings.LLM_PROVIDER
        common_params = {
            "model": self._model_name(),
            "messages": messages,
            "temperature": self.settings.LLM_TEMPERATURE,
            "max_tokens": self.settings.LLM_MAX_TOKENS,
        }

        if provider == LLMProvider.GROQ:
            response = self._client.chat.completions.create(**common_params)
            answer = response.choices[0].message.content
            usage = response.usage
            return answer, usage.prompt_tokens, usage.completion_tokens

        elif provider == LLMProvider.OPENAI:
            response = self._client.chat.completions.create(**common_params)
            answer = response.choices[0].message.content
            usage = response.usage
            return answer, usage.prompt_tokens, usage.completion_tokens

        elif provider == LLMProvider.ANTHROPIC:
            # Anthropic uses system as a separate param, not in messages list
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            response = self._client.messages.create(
                **{**common_params, "system": system_msg},
                messages=user_messages,
            )
            answer = response.content[0].text
            return answer, response.usage.input_tokens, response.usage.output_tokens

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _build_client(self):
        """Instantiate the LLM client based on LLM_PROVIDER setting."""
        provider = self.settings.LLM_PROVIDER

        if provider == LLMProvider.GROQ:
            from groq import Groq
            return Groq(api_key=self.settings.GROQ_API_KEY)

        elif provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=self.settings.OPENAI_API_KEY)

        elif provider == LLMProvider.ANTHROPIC:
            import anthropic
            return anthropic.Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _model_name(self) -> str:
        """Return the model name for the current provider."""
        provider = self.settings.LLM_PROVIDER
        if provider == LLMProvider.GROQ:
            return self.settings.GROQ_MODEL
        elif provider == LLMProvider.OPENAI:
            return self.settings.OPENAI_MODEL
        elif provider == LLMProvider.ANTHROPIC:
            return self.settings.ANTHROPIC_MODEL
        return "unknown"

    def _extract_citations(self, text: str) -> list[str]:
        """
        Extract all [Source: ...] citations from the answer text.

        Returns a list of citation strings (may contain duplicates;
        deduplicate with sources_used() on the result object).
        """
        matches = CITATION_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _grounding_score(self, text: str) -> float:
        """
        Estimate what fraction of the answer is cited.

        Split into sentences; count those containing [Source: ...].
        This is a proxy metric, not ground truth. The faithfulness
        guardrail does proper NLI-based checking.

        Returns 0.0–1.0.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return 0.0
        cited = sum(1 for s in sentences if CITATION_PATTERN.search(s))
        return round(cited / len(sentences), 2)

    def _is_refusal(self, text: str) -> bool:
        """Check if the model refused to answer (couldn't find in context)."""
        lower = text.lower()
        return any(signal in lower for signal in REFUSAL_SIGNALS)
