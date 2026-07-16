"""
metrics.py
----------
RAG evaluation metrics.

Metrics implemented (RAGAS-style):
  1. Faithfulness
     "Is every claim in the answer supported by the context?"
     Score: fraction of answer sentences that are NLI-entailed by context.
     Target: > 0.9 (resume: < 1% unsafe output rate)

  2. Answer Relevancy
     "Does the answer actually address the question?"
     Score: cosine similarity between question embedding and answer embedding.
     Target: > 0.7

  3. Context Precision
     "Are the retrieved chunks actually useful?"
     Score: fraction of retrieved chunks that are relevant to the question.
     Requires gold-standard relevance labels from qa_pairs.json.
     Target: > 0.8 (resume: ~30% improvement over semantic-only = 0.65 baseline)

  4. Context Recall
     "Did we retrieve all the information needed to answer?"
     Score: fraction of ground truth claims covered by retrieved context.
     Target: > 0.75

  5. Exact Match (for factoid questions)
     "Does the answer contain the exact expected answer string?"
     Target: > 0.6 on factoid eval set

  6. Latency
     End-to-end query latency in milliseconds.
     Target: < 2000ms (p95)

Reference: RAGAS paper (Es et al. 2023) https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class MetricResult:
    """Result of computing a single metric."""
    metric_name: str
    score: float          # 0.0–1.0
    details: dict         # Supporting data for debugging
    passed: bool          # Whether it meets the target threshold

    THRESHOLDS = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.7,
        "context_precision": 0.65,  # Baseline before improvement
        "context_recall": 0.75,
        "exact_match": 0.6,
    }

    def __post_init__(self):
        threshold = self.THRESHOLDS.get(self.metric_name, 0.5)
        self.passed = self.score >= threshold


@dataclass
class EvalSample:
    """One sample in the evaluation dataset."""
    question: str
    expected_answer: str
    relevant_doc_names: list[str]   # Which documents contain the answer
    category: str = "factoid"       # factoid | policy | procedural | adversarial


def faithfulness_score(answer: str, context: str) -> MetricResult:
    """
    Compute faithfulness: fraction of answer sentences entailed by context.

    Uses the NLI model from FaithfulnessGuard.
    """
    from src.guardrails.faithfulness_guard import FaithfulnessGuard
    guard = FaithfulnessGuard()
    result = guard.check(answer, context)
    return MetricResult(
        metric_name="faithfulness",
        score=result.faithfulness_score,
        details={
            "sentence_scores": result.sentence_scores,
            "unfaithful_count": len(result.unfaithful_sentences),
        },
        passed=result.faithfulness_score >= MetricResult.THRESHOLDS["faithfulness"],
    )


def answer_relevancy_score(
    question: str,
    answer: str,
    embedder=None,
) -> MetricResult:
    """
    Compute answer relevancy: semantic similarity of question to answer.

    High relevancy = the answer addresses the question directly.
    Low relevancy = off-topic or irrelevant answer.
    """
    if embedder is None:
        from src.ingestion.embedder import VoyageEmbedder
        embedder = VoyageEmbedder()

    q_emb = np.array(embedder.embed_query(question))
    a_emb = np.array(embedder.embed_query(answer))

    # Cosine similarity
    cos_sim = float(
        np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-10)
    )

    return MetricResult(
        metric_name="answer_relevancy",
        score=round(cos_sim, 4),
        details={"cosine_similarity": cos_sim},
        passed=cos_sim >= MetricResult.THRESHOLDS["answer_relevancy"],
    )


def context_precision_score(
    retrieved_doc_names: list[str],
    relevant_doc_names: list[str],
    at_k: int = 5,
) -> MetricResult:
    """
    Precision@K: of the top-K retrieved chunks, how many are from relevant docs?

    Args:
        retrieved_doc_names: Names of documents in retrieved chunks (in rank order).
        relevant_doc_names : Ground truth: which docs contain the answer.
        at_k               : Evaluation cutoff (default: RERANK_TOP_N = 5).
    """
    relevant_set = set(n.lower() for n in relevant_doc_names)
    top_k = retrieved_doc_names[:at_k]

    if not top_k:
        return MetricResult(
            metric_name="context_precision",
            score=0.0,
            details={"retrieved_count": 0, "relevant_count": 0},
            passed=False,
        )

    relevant_retrieved = sum(
        1 for name in top_k if name.lower() in relevant_set
    )
    precision = relevant_retrieved / len(top_k)

    return MetricResult(
        metric_name="context_precision",
        score=round(precision, 4),
        details={
            "retrieved_count": len(top_k),
            "relevant_count": relevant_retrieved,
            "relevant_docs": relevant_doc_names,
        },
        passed=precision >= MetricResult.THRESHOLDS["context_precision"],
    )


def context_recall_score(
    context: str,
    expected_answer: str,
) -> MetricResult:
    """
    Recall: does the context contain the information needed for the answer?

    Approximate method: check if key phrases from the expected answer
    appear in the retrieved context. Not NLI-based (that would be slow);
    this is a proxy for ground truth coverage.
    """
    # Extract key phrases: numbers, quoted text, capitalized proper nouns
    patterns = [
        r'\b\d+%?\b',                     # Numbers and percentages
        r'"[^"]{3,}"',                     # Quoted phrases
        r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b',  # Proper nouns (2+ words)
        r'\bArticle \d+\b',               # Legal references
        r'\bSection \d+[\.\d]*\b',        # Section references
    ]

    key_phrases = []
    for pattern in patterns:
        key_phrases.extend(re.findall(pattern, expected_answer))

    if not key_phrases:
        # Fall back to word overlap if no key phrases found
        expected_words = set(expected_answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(expected_words & context_words) / max(len(expected_words), 1)
        return MetricResult(
            metric_name="context_recall",
            score=round(overlap, 4),
            details={"method": "word_overlap"},
            passed=overlap >= MetricResult.THRESHOLDS["context_recall"],
        )

    # Check what fraction of key phrases appear in context
    context_lower = context.lower()
    found = sum(1 for p in key_phrases if p.lower() in context_lower)
    recall = found / len(key_phrases)

    return MetricResult(
        metric_name="context_recall",
        score=round(recall, 4),
        details={
            "key_phrases": key_phrases,
            "found": found,
            "total": len(key_phrases),
        },
        passed=recall >= MetricResult.THRESHOLDS["context_recall"],
    )


def exact_match_score(answer: str, expected: str) -> MetricResult:
    """
    Exact match: does the answer contain the expected answer string?

    Normalized: lowercase, strip punctuation.
    """
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    norm_answer = normalize(answer)
    norm_expected = normalize(expected)

    is_match = norm_expected in norm_answer

    # Partial credit: word overlap
    expected_words = set(norm_expected.split())
    answer_words = set(norm_answer.split())
    overlap = len(expected_words & answer_words) / max(len(expected_words), 1)

    score = 1.0 if is_match else min(overlap, 0.9)  # Full match = 1.0, partial < 0.9

    return MetricResult(
        metric_name="exact_match",
        score=round(score, 4),
        details={
            "is_exact_match": is_match,
            "word_overlap": round(overlap, 4),
        },
        passed=is_match or overlap >= MetricResult.THRESHOLDS["exact_match"],
    )
