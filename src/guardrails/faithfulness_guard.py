"""
faithfulness_guard.py
---------------------
Detects hallucinated content in generated responses using NLI.

What this checks:
  "Is every claim in the answer actually supported by the retrieved context?"

Method: Natural Language Inference (NLI)
  - NLI model takes two texts: premise (context) and hypothesis (answer sentence)
  - Outputs: ENTAILMENT | NEUTRAL | CONTRADICTION
  - We check if the answer is "entailed by" the context
  - Low entailment score → answer contains claims not in context → hallucination

Why NLI over LLM-as-judge:
  - Much faster: NLI model runs in <100ms CPU; LLM judge costs 500ms+
  - No additional API cost (local model)
  - Deterministic: same input → same output (no temperature variation)
  - The downside: NLI is less nuanced than LLM-as-judge for complex reasoning
  - For production: combine both (NLI for fast pre-screen, LLM judge for borderline cases)

Model: cross-encoder/nli-deberta-v3-small
  - DeBERTa-v3: SOTA NLI architecture (disentangled attention + enhanced mask decoder)
  - "small" variant: 86M params; good balance of accuracy and inference speed
  - Trained on SNLI + MultiNLI: robust cross-domain NLI
  - Label order: [CONTRADICTION, ENTAILMENT, NEUTRAL] (model-specific, confirmed)

Implementation:
  1. Split answer into sentences
  2. For each sentence, score entailment against the full context
  3. Average entailment scores → faithfulness score
  4. Below threshold → flag as potentially hallucinated

The ~1% unsafe output rate (resume bullet):
  Measured on an adversarial eval set (red team scenarios in red_team.py).
  The combination of strict grounding prompt + faithfulness guard + citation
  validation brings hallucination rate under 1% in offline testing.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from src.config.settings import get_settings

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )


@dataclass
class FaithfulnessResult:
    """Result of faithfulness analysis on a generated answer."""
    is_faithful: bool
    faithfulness_score: float          # 0.0–1.0
    sentence_scores: list[dict]        # Per-sentence: {"sentence": str, "score": float}
    unfaithful_sentences: list[str]    # Sentences below threshold
    context_used: str                  # The context text that was checked against

    def violation_count(self) -> int:
        return len(self.unfaithful_sentences)


# NLI label indices for cross-encoder/nli-deberta-v3-small
# Confirmed from model card: 0=contradiction, 1=entailment, 2=neutral
ENTAILMENT_IDX = 1


class FaithfulnessGuard:
    """
    Checks if a generated answer is faithful to the retrieved context.

    Thread-safe singleton model loading.

    Usage:
        guard = FaithfulnessGuard()
        result = guard.check(answer="...", context="...")
        if not result.is_faithful:
            # Log violation, potentially block or warn user
    """

    _model: Optional[CrossEncoder] = None
    _lock = threading.Lock()

    def __init__(self):
        self.settings = get_settings()
        self.threshold = self.settings.FAITHFULNESS_THRESHOLD
        self.model_name = self.settings.FAITHFULNESS_MODEL

    @staticmethod
    def _load_model(model_name: str) -> CrossEncoder:
        """
        Factory method for loading the NLI model.

        Separate static method so tests can patch it:
            @patch("src.guardrails.faithfulness_guard.FaithfulnessGuard._load_model")
            def test_something(self, mock_load): ...
        """
        logger.info(f"Loading NLI model: {model_name}")
        model = CrossEncoder(model_name, max_length=512, num_labels=3)
        logger.info("NLI model loaded.")
        return model

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the NLI model (thread-safe)."""
        if FaithfulnessGuard._model is None:
            with FaithfulnessGuard._lock:
                if FaithfulnessGuard._model is None:
                    FaithfulnessGuard._model = self._load_model(self.model_name)
        return FaithfulnessGuard._model

    def check(
        self,
        answer: str,
        context: str,
        skip_refusals: bool = True,
    ) -> FaithfulnessResult:
        """
        Check if the answer is faithful to the context.

        Args:
            answer       : The LLM-generated answer.
            context      : The retrieved context used to generate the answer.
            skip_refusals: If True, skip checking for answers that are refusals
                           (e.g., "I cannot find information about...").
                           Refusals are always faithful since they make no claims.

        Returns:
            FaithfulnessResult with faithfulness score and unfaithful sentences.
        """
        # Refusal answers are always faithful (they don't claim anything)
        if skip_refusals and self._is_refusal(answer):
            return FaithfulnessResult(
                is_faithful=True,
                faithfulness_score=1.0,
                sentence_scores=[],
                unfaithful_sentences=[],
                context_used=context,
            )
        answer_clean = re.sub(r'\[Source[^\]]*\]', '', answer).strip()

        # Split answer into meaningful sentences
        sentences = self._split_sentences(answer_clean)
        if not sentences:
            return FaithfulnessResult(
                is_faithful=True,
                faithfulness_score=1.0,
                sentence_scores=[],
                unfaithful_sentences=[],
                context_used=context,
            )

        # Get per-sentence entailment scores (patchable for tests)
        entailment_probs = self._sentence_scores(sentences, context)

        # Build per-sentence results
        sentence_scores = [
            {"sentence": s, "score": round(score, 4)}
            for s, score in zip(sentences, entailment_probs)
        ]

        # Flag sentences below threshold as unfaithful
        unfaithful = [
            s["sentence"]
            for s in sentence_scores
            if s["score"] < self.threshold
        ]

        # Overall faithfulness score: mean entailment probability
        avg_score = round(float(np.mean(entailment_probs)), 4)
        is_faithful = avg_score >= self.threshold and len(unfaithful) == 0

        if not is_faithful:
            logger.warning(
                f"Faithfulness violation | score={avg_score:.3f} | "
                f"unfaithful_sentences={len(unfaithful)}"
            )

        return FaithfulnessResult(
            is_faithful=is_faithful,
            faithfulness_score=avg_score,
            sentence_scores=sentence_scores,
            unfaithful_sentences=unfaithful,
            context_used=context,
        )

    def _sentence_scores(self, sentences: list[str], context: str) -> list[float]:
        """
        Compute NLI entailment probabilities for each sentence against the context.

        Returns a list of float probabilities (0.0–1.0) in the same order as sentences.
        Extracted as a separate method so tests can patch it without loading the model.
        """
        pairs = [[context, sentence] for sentence in sentences]
        scores_matrix = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Apply softmax to get probabilities: [contradiction, entailment, neutral]
        e_x = np.exp(scores_matrix - np.max(scores_matrix, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        return probs[:, ENTAILMENT_IDX].tolist()

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for per-sentence NLI scoring.

        Filters out:
          - Citation markers like [Source: ...]
          - Very short fragments (likely headings or noise)
        """
        # Remove citation markers for cleaner NLI input
        cleaned = re.sub(r"\[Source:[^\]]+\]", "", text)

        # Simple sentence splitter (no NLTK dependency)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())

        # Filter: keep only sentences with enough content to evaluate
        return [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    def _is_refusal(self, text: str) -> bool:
        """Detect if the response is a refusal (no claims to check)."""
        refusal_signals = [
            "cannot find",
            "does not contain",
            "not in the knowledge base",
            "outside the scope",
            "no information",
        ]
        lower = text.lower()
        return any(signal in lower for signal in refusal_signals)
