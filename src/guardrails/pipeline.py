"""
guardrail_pipeline.py
---------------------
Orchestrates all guardrails in the correct execution order.

Order matters:
  INPUT guardrails (before retrieval + generation):
    1. Injection guard  — fastest; blocks obvious attacks immediately
    2. PII filter       — redact/block personal data in queries
    3. Topic guard      — reject off-topic queries (avoid retrieving and calling LLM)

  OUTPUT guardrails (after generation):
    4. Faithfulness guard — detect hallucinated content
    5. PII filter (output) — scrub any PII that leaked into the answer

Why this order:
  - Injection check is fastest (regex, ~0ms); fail fast on malicious input
  - PII check before topic guard: don't embed PII into the query embedding
  - Topic guard before retrieval: avoid DB + embedding cost for off-topic
  - Faithfulness after generation: can only check the actual response
  - Output PII last: scrub anything that slipped through

Design: GuardrailResult is a rich object that captures every guardrail
decision so it can be stored in query_logs.guardrail_flags for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.retrieval.hybrid_search import RetrievedChunk
from src.generation.generator import GenerationResult
from src.generation.prompts import (
    OFF_TOPIC_RESPONSE,
    INJECTION_DETECTED_RESPONSE,
    PII_DETECTED_RESPONSE,
)
from .injection_guard import InjectionGuard, InjectionResult
from .pii_filter import PIIFilter, PIIDetectionResult
from .topic_guard import TopicGuard
from .faithfulness_guard import FaithfulnessGuard, FaithfulnessResult


@dataclass
class GuardrailDecision:
    """The outcome of running guardrails on a query or response."""
    passed: bool                               # True = continue, False = block
    blocked_by: Optional[str] = None           # Which guardrail blocked
    reason: str = ""                           # Human-readable reason
    safe_response: Optional[str] = None        # Response to send to user if blocked
    flags: dict = field(default_factory=dict)  # Detailed flags for audit logging
    redacted_query: Optional[str] = None       # Query with PII redacted


@dataclass
class QueryGuardrailResult:
    """Full guardrail report for input → output pipeline."""
    input_decision: GuardrailDecision
    output_decision: Optional[GuardrailDecision] = None
    injection: Optional[InjectionResult] = None
    input_pii: Optional[PIIDetectionResult] = None
    output_pii: Optional[PIIDetectionResult] = None
    faithfulness: Optional[FaithfulnessResult] = None
    topic_score: Optional[float] = None
    final_answer: Optional[str] = None
    was_blocked: bool = False

    def to_audit_dict(self) -> dict:
        """Serialize for storage in query_logs.guardrail_flags."""
        return {
            "was_blocked": self.was_blocked,
            "blocked_by": self.input_decision.blocked_by
                         or (self.output_decision.blocked_by if self.output_decision else None),
            "injection_detected": self.injection.is_injection if self.injection else False,
            "injection_method": self.injection.detection_method if self.injection else None,
            "input_pii_detected": self.input_pii.has_pii if self.input_pii else False,
            "input_pii_types": self.input_pii.entity_types if self.input_pii else [],
            "output_pii_detected": self.output_pii.has_pii if self.output_pii else False,
            "topic_score": self.topic_score,
            "faithfulness_score": self.faithfulness.faithfulness_score if self.faithfulness else None,
            "faithfulness_violations": len(self.faithfulness.unfaithful_sentences) if self.faithfulness else 0,
        }


class GuardrailPipeline:
    """
    Runs all guardrails in sequence for a query + response pair.

    Usage:
        pipeline = GuardrailPipeline()

        # Input check (before retrieval)
        result = pipeline.check_input("what is the leave policy?")
        if not result.input_decision.passed:
            return result.input_decision.safe_response

        # ... run retrieval + generation ...

        # Output check (after generation)
        result = pipeline.check_output(result, answer="...", chunks=[...])
        return result.final_answer
    """

    def __init__(self):
        self.injection_guard = InjectionGuard()
        self.pii_filter = PIIFilter()
        self.topic_guard = TopicGuard()
        self.faithfulness_guard = FaithfulnessGuard()

    def check_input(self, query: str) -> QueryGuardrailResult:
        """
        Run all input guardrails against a user query.

        Returns a QueryGuardrailResult; if input_decision.passed is False,
        the caller should return input_decision.safe_response immediately
        without proceeding to retrieval or generation.
        """
        # 1. Injection check (fastest; always first)
        injection = self.injection_guard.check(query)
        if injection.is_injection:
            return QueryGuardrailResult(
                input_decision=GuardrailDecision(
                    passed=False,
                    blocked_by="injection_guard",
                    reason=f"Prompt injection detected ({injection.detection_method})",
                    safe_response=INJECTION_DETECTED_RESPONSE,
                    flags={"injection_severity": injection.severity},
                ),
                injection=injection,
                was_blocked=True,
            )

        # 2. PII check on input
        input_pii = self.pii_filter.analyze_input(query)
        if input_pii.has_pii and input_pii.severity() == "high":
            # HIGH severity (SSN, credit cards): block entirely
            return QueryGuardrailResult(
                input_decision=GuardrailDecision(
                    passed=False,
                    blocked_by="pii_filter_input",
                    reason=f"High-risk PII in query: {input_pii.entity_types}",
                    safe_response=PII_DETECTED_RESPONSE,
                    flags={"pii_types": input_pii.entity_types},
                    redacted_query=input_pii.redacted_text,
                ),
                injection=injection,
                input_pii=input_pii,
                was_blocked=True,
            )
        # MEDIUM/LOW severity: allow but use redacted query for retrieval
        safe_query = input_pii.redacted_text if input_pii.has_pii else query

        # 3. Topic guard
        topic_score = self.topic_guard.score(query)
        if topic_score < self.topic_guard.threshold:
            return QueryGuardrailResult(
                input_decision=GuardrailDecision(
                    passed=False,
                    blocked_by="topic_guard",
                    reason=f"Off-topic query (score={topic_score:.3f})",
                    safe_response=OFF_TOPIC_RESPONSE,
                    flags={"topic_score": topic_score},
                ),
                injection=injection,
                input_pii=input_pii,
                topic_score=topic_score,
                was_blocked=True,
            )

        return QueryGuardrailResult(
            input_decision=GuardrailDecision(
                passed=True,
                redacted_query=safe_query,
                flags={
                    "input_pii_detected": input_pii.has_pii,
                    "input_pii_severity": input_pii.severity() if input_pii.has_pii else None,
                    "topic_score": round(topic_score, 3),
                },
            ),
            injection=injection,
            input_pii=input_pii,
            topic_score=topic_score,
            was_blocked=False,
        )

    def check_output(
        self,
        result: QueryGuardrailResult,
        generation: GenerationResult,
        context: str,
    ) -> QueryGuardrailResult:
        """
        Run output guardrails on the generated answer.

        Args:
            result    : The QueryGuardrailResult from check_input().
            generation: The GenerationResult from GroundedGenerator.
            context   : Raw context text passed to the LLM (for faithfulness check).

        Returns:
            Updated QueryGuardrailResult with output decision and final_answer.
        """
        answer = generation.answer

        # 4. Faithfulness check
        faithfulness = self.faithfulness_guard.check(answer, context)
        if not faithfulness.is_faithful:
            logger.warning(
                f"Faithfulness guard triggered | score={faithfulness.faithfulness_score:.3f}"
            )
            # Append a disclaimer rather than blocking (less disruptive)
            answer = (
                answer + "\n\n⚠️ *Note: Some parts of this answer may not be fully "
                "supported by the retrieved documents. Please verify with the source.*"
            )

        # 5. Output PII check
        output_pii = self.pii_filter.analyze_output(answer)
        if output_pii.has_pii:
            logger.info(f"Output PII found: {output_pii.entity_types} — redacting")
            answer = self.pii_filter.redact(answer)

        result.faithfulness = faithfulness
        result.output_pii = output_pii
        result.final_answer = answer
        result.output_decision = GuardrailDecision(
            passed=True,
            flags={
                "faithfulness_score": faithfulness.faithfulness_score,
                "faithfulness_violations": len(faithfulness.unfaithful_sentences),
                "output_pii_detected": output_pii.has_pii,
                "output_pii_types": output_pii.entity_types if output_pii.has_pii else [],
            },
        )
        return result
