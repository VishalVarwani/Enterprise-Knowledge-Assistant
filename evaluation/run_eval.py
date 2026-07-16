"""
run_eval.py
-----------
Full evaluation pipeline: loads QA dataset, runs queries, scores all metrics.

Usage:
    python evaluation/run_eval.py --mode full
    python evaluation/run_eval.py --mode metrics-only --category policy
    python evaluation/run_eval.py --mode red-team

Output:
    evaluation/results/eval_report_{timestamp}.json
    evaluation/results/eval_report_{timestamp}.csv
    Printed summary to stdout

What gets measured:
  - Faithfulness (per answer)
  - Answer relevancy (per answer)
  - Context precision (per query)
  - Context recall (per query)
  - Exact match (for factoid/policy questions with known answers)
  - End-to-end latency (p50, p95, p99)
  - Guardrail block rate (legitimate vs adversarial)
  - Cache hit rate simulation
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config.settings import get_settings
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import GroundedGenerator, GenerationResult
from src.generation.prompts import format_context
from src.guardrails.pipeline import GuardrailPipeline
from src.evaluation.metrics import (
    faithfulness_score,
    answer_relevancy_score,
    context_precision_score,
    context_recall_score,
    exact_match_score,
)
from src.evaluation.red_team import run_red_team, RED_TEAM_SCENARIOS


DATASET_PATH = Path(__file__).parent / "dataset" / "qa_pairs.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(category_filter: str | None = None) -> list[dict]:
    """Load evaluation QA pairs from JSON."""
    with open(DATASET_PATH) as f:
        pairs = json.load(f)

    if category_filter:
        pairs = [p for p in pairs if p.get("category") == category_filter]

    # Only evaluate questions with known expected answers
    evaluable = [p for p in pairs if p.get("expected_answer")]
    logger.info(f"Loaded {len(evaluable)} evaluable QA pairs (filtered from {len(pairs)})")
    return evaluable


def run_single_query(
    question: str,
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    generator: GroundedGenerator,
    guardrails: GuardrailPipeline,
) -> tuple[GenerationResult | None, list, float]:
    """
    Run the full RAG pipeline for one question.

    Returns (generation_result, reranked_chunks, latency_ms).
    Returns (None, [], latency_ms) if guardrails block the query.
    """
    start = time.perf_counter()

    guard = guardrails.check_input(question)
    if not guard.input_decision.passed:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, [], latency_ms

    chunks = searcher.search(question)
    reranked = reranker.rerank(question, chunks) if chunks else []
    generation = generator.generate(question, reranked)
    latency_ms = (time.perf_counter() - start) * 1000

    return generation, reranked, latency_ms


def run_full_eval(category_filter: str | None = None) -> dict:
    """
    Run the full evaluation on the QA dataset.

    Returns a summary dict with all metric averages.
    """
    settings = get_settings()
    dataset = load_dataset(category_filter)

    searcher = HybridSearcher()
    reranker = CrossEncoderReranker()
    generator = GroundedGenerator()
    guardrails = GuardrailPipeline()

    rows = []
    latencies = []

    for sample in tqdm(dataset, desc="Evaluating"):
        question = sample["question"]
        expected = sample["expected_answer"]
        relevant_docs = sample.get("relevant_documents", [])

        generation, reranked_chunks, latency_ms = run_single_query(
            question, searcher, reranker, generator, guardrails
        )

        latencies.append(latency_ms)

        if generation is None:
            # Query was blocked (adversarial queries are expected to be blocked)
            rows.append({
                "id": sample["id"],
                "category": sample["category"],
                "question": question,
                "was_blocked": True,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "exact_match": None,
                "latency_ms": latency_ms,
            })
            continue

        answer = generation.answer
        context = format_context(reranked_chunks) if reranked_chunks else ""
        retrieved_doc_names = [c.doc_name for c in reranked_chunks]

        # Compute all metrics
        faith = faithfulness_score(answer, context) if context else None
        relevancy = answer_relevancy_score(question, answer)
        precision = context_precision_score(retrieved_doc_names, relevant_docs)
        recall = context_recall_score(context, expected) if context else None
        em = exact_match_score(answer, expected) if expected else None

        rows.append({
            "id": sample["id"],
            "category": sample["category"],
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer[:300],
            "was_blocked": False,
            "faithfulness": faith.score if faith else None,
            "answer_relevancy": relevancy.score,
            "context_precision": precision.score,
            "context_recall": recall.score if recall else None,
            "exact_match": em.score if em else None,
            "grounding_score": generation.grounding_score,
            "has_refusal": generation.has_refusal,
            "retrieved_chunks": len(reranked_chunks),
            "model": generation.model,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "latency_ms": latency_ms,
        })

    df = pd.DataFrame(rows)
    non_blocked = df[~df["was_blocked"]]

    # Compute summary statistics
    latency_arr = sorted(latencies)
    n = len(latency_arr)

    summary = {
        "eval_timestamp": datetime.utcnow().isoformat(),
        "total_samples": len(dataset),
        "evaluated": len(non_blocked),
        "blocked": int(df["was_blocked"].sum()),
        "metrics": {
            "faithfulness": {
                "mean": round(non_blocked["faithfulness"].dropna().mean(), 4),
                "target": 0.90,
            },
            "answer_relevancy": {
                "mean": round(non_blocked["answer_relevancy"].dropna().mean(), 4),
                "target": 0.70,
            },
            "context_precision": {
                "mean": round(non_blocked["context_precision"].dropna().mean(), 4),
                "target": 0.65,
            },
            "context_recall": {
                "mean": round(non_blocked["context_recall"].dropna().mean(), 4),
                "target": 0.75,
            },
            "exact_match": {
                "mean": round(non_blocked["exact_match"].dropna().mean(), 4),
                "target": 0.60,
            },
        },
        "latency_ms": {
            "p50": latency_arr[int(n * 0.50)],
            "p95": latency_arr[int(n * 0.95)],
            "p99": latency_arr[int(n * 0.99)] if n >= 100 else latency_arr[-1],
            "mean": sum(latency_arr) / n,
        },
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"eval_report_{ts}.json"
    csv_path = RESULTS_DIR / f"eval_report_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump({**summary, "rows": rows}, f, indent=2, default=str)
    df.to_csv(csv_path, index=False)

    _print_summary(summary)
    logger.info(f"Results saved: {json_path}")
    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"  Timestamp: {summary['eval_timestamp']}")
    print(f"  Samples:   {summary['total_samples']} total | {summary['evaluated']} evaluated | {summary['blocked']} blocked")
    print()
    print("  METRICS (mean | target)")
    for name, data in summary["metrics"].items():
        mean = data["mean"]
        target = data["target"]
        status = "✓" if mean >= target else "✗"
        print(f"  {status} {name:<22} {mean:.4f}  (target ≥ {target})")
    print()
    print("  LATENCY")
    lat = summary["latency_ms"]
    print(f"    p50: {lat['p50']:.0f}ms | p95: {lat['p95']:.0f}ms | p99: {lat['p99']:.0f}ms | mean: {lat['mean']:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EKA evaluation")
    parser.add_argument(
        "--mode",
        choices=["full", "metrics-only", "red-team"],
        default="full",
        help="Evaluation mode",
    )
    parser.add_argument("--category", help="Filter QA pairs by category")
    args = parser.parse_args()

    if args.mode == "red-team":
        report = run_red_team()
        report.print_summary()
    elif args.mode in ("full", "metrics-only"):
        run_full_eval(category_filter=args.category)
