"""
evaluator.py
------------
High-level evaluator: wraps all metrics into a single callable
used by both run_eval.py and the API's per-query quality logging.

Usage:
    from src.evaluation.evaluator import Evaluator, EvalOutput

    evaluator = Evaluator()
    output = evaluator.evaluate(
        question="What is the leave policy?",
        answer="Employees get 25 days [SOURCE 1: HR Manual].",
        context="HR Manual states 25 days annual leave.",
        retrieved_chunks=chunks,
        relevant_doc_names=["HR Manual"],
        expected_answer="25 days",
    )
    print(output.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.evaluation.metrics import (
    MetricResult,
    answer_relevancy_score,
    context_precision_score,
    context_recall_score,
    exact_match_score,
    faithfulness_score,
)


@dataclass
class EvalOutput:
    """All metric results for a single query."""
    question: str
    answer: str
    faithfulness: Optional[MetricResult] = None
    answer_relevancy: Optional[MetricResult] = None
    context_precision: Optional[MetricResult] = None
    context_recall: Optional[MetricResult] = None
    exact_match: Optional[MetricResult] = None
    errors: list[str] = field(default_factory=list)

    def overall_pass(self) -> bool:
        """True only if all computed metrics pass their thresholds."""
        metrics = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        computed = [m for m in metrics if m is not None]
        return all(m.passed for m in computed) if computed else False

    def to_dict(self) -> dict:
        result = {"question": self.question[:100], "answer": self.answer[:200], "errors": self.errors}
        for attr in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "exact_match"):
            metric = getattr(self, attr)
            if metric:
                result[attr] = {"score": metric.score, "passed": metric.passed, **metric.details}
        result["overall_pass"] = self.overall_pass()
        return result

    def summary(self) -> str:
        lines = [f"Q: {self.question[:80]}"]
        for attr in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            metric = getattr(self, attr)
            if metric:
                icon = "✓" if metric.passed else "✗"
                lines.append(f"  {icon} {attr}: {metric.score:.3f}")
        lines.append(f"  Overall: {'PASS' if self.overall_pass() else 'FAIL'}")
        return "\n".join(lines)


class Evaluator:
    """
    Runs all RAG evaluation metrics for a single query-answer pair.

    Designed for two use cases:
      1. Offline evaluation: run_eval.py calls this for every QA pair in the dataset
      2. Online monitoring: the query route calls this async to log per-query quality

    Thread-safe: all underlying models are singletons loaded on first use.
    """

    def __init__(self, run_faithfulness: bool = True, run_relevancy: bool = True):
        self.run_faithfulness = run_faithfulness
        self.run_relevancy = run_relevancy

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str = "",
        retrieved_chunks: list | None = None,
        relevant_doc_names: list[str] | None = None,
        expected_answer: str | None = None,
    ) -> EvalOutput:
        """
        Compute all applicable metrics.

        Args:
            question          : The user query
            answer            : The generated answer
            context           : Formatted context string (from format_context())
            retrieved_chunks  : List of RetrievedChunk objects
            relevant_doc_names: Ground truth relevant doc names (for precision)
            expected_answer   : Ground truth answer (for exact match + recall)
        """
        output = EvalOutput(question=question, answer=answer)
        chunks = retrieved_chunks or []
        retrieved_names = [c.doc_name for c in chunks]

        # 1. Faithfulness — requires context
        if self.run_faithfulness and context:
            try:
                output.faithfulness = faithfulness_score(answer, context)
            except Exception as e:
                logger.warning(f"Faithfulness metric failed: {e}")
                output.errors.append(f"faithfulness: {e}")

        # 2. Answer relevancy — always applicable
        if self.run_relevancy:
            try:
                output.answer_relevancy = answer_relevancy_score(question, answer)
            except Exception as e:
                logger.warning(f"Answer relevancy metric failed: {e}")
                output.errors.append(f"answer_relevancy: {e}")

        # 3. Context precision — requires retrieved chunks and relevance labels
        if chunks and relevant_doc_names:
            try:
                output.context_precision = context_precision_score(
                    retrieved_names, relevant_doc_names
                )
            except Exception as e:
                logger.warning(f"Context precision failed: {e}")
                output.errors.append(f"context_precision: {e}")

        # 4. Context recall — requires expected answer and context
        if context and expected_answer:
            try:
                output.context_recall = context_recall_score(context, expected_answer)
            except Exception as e:
                logger.warning(f"Context recall failed: {e}")
                output.errors.append(f"context_recall: {e}")

        # 5. Exact match — requires expected answer
        if expected_answer:
            try:
                output.exact_match = exact_match_score(answer, expected_answer)
            except Exception as e:
                logger.warning(f"Exact match failed: {e}")
                output.errors.append(f"exact_match: {e}")

        return output
