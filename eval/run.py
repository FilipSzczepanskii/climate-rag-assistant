"""Evaluate the RAG system against the golden Q&A set and log results to MLflow.

Modes:
  retrieval  retrieval metrics only (hit rate, MRR). No LLM key required, so
             this is what CI runs as a quality gate.
  full       also generates answers and scores keyword recall and citation
             accuracy. Needs a configured LLM provider.

Each run is logged as an MLflow experiment, so different chunking, retrieval
and model configurations can be compared side by side.

Run:
    python -m eval.run --mode retrieval
    python -m eval.run --mode full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import mlflow

from app.config import settings
from app.rag import RagPipeline
from eval.metrics import citation_match, hit_rate, keyword_recall, reciprocal_rank

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


def load_golden() -> list[dict]:
    rows: list[dict] = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(mode: str) -> dict[str, float]:
    golden = load_golden()
    pipeline = RagPipeline(settings)

    hit, mrr, kw_recall, citations = [], [], [], []

    for item in golden:
        question = item["question"]
        expected = item["expected_sources"]

        hits = pipeline.retriever.retrieve(question)
        retrieved_titles = [h.title for h in hits]
        hit.append(hit_rate(retrieved_titles, expected))
        mrr.append(reciprocal_rank(retrieved_titles, expected))

        if mode == "full":
            result = pipeline.answer(question)
            kw_recall.append(keyword_recall(result.answer, item.get("keywords", [])))
            citations.append(citation_match(result.answer, expected))

    results = {"hit_rate": mean(hit), "mrr": mean(mrr)}
    if mode == "full":
        results["keyword_recall"] = mean(kw_recall)
        results["citation_match"] = mean(citations)
    results["_n_questions"] = float(len(golden))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.7,
        help="Fail with a non-zero exit code if hit rate falls below this.",
    )
    args = parser.parse_args()

    results = evaluate(args.mode)
    n_questions = int(results.pop("_n_questions"))

    mlflow.set_experiment("rag-eval")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "mode": args.mode,
                "n_questions": n_questions,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": settings.top_k,
                "use_reranker": settings.use_reranker,
                "llm_model": settings.resolved_llm_model() if args.mode == "full" else "n/a",
            }
        )
        mlflow.log_metrics(results)

    print(f"\nEvaluation ({args.mode} mode, {n_questions} questions)")
    for name, value in results.items():
        print(f"  {name:16s} {value:.3f}")

    if results["hit_rate"] < args.min_hit_rate:
        print(
            f"\nFAILED: hit_rate {results['hit_rate']:.3f} "
            f"is below the {args.min_hit_rate:.2f} threshold."
        )
        sys.exit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
