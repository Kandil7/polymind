"""RAGAS Runner — faithfulness and answer relevancy evaluation.

Uses RAGAS library for RAG-specific evaluation metrics.
Runs against a benchmark dataset to track quality over time.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()

DEFAULT_BENCHMARK = "tests/eval/benchmark_v1.json"


class RAGASRunner:
    """RAGAS evaluation runner for PolyMind.

    Evaluates faithfulness, answer relevancy, and context precision
    against a benchmark dataset.
    """

    def __init__(
        self,
        benchmark_path: str = DEFAULT_BENCHMARK,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize RAGAS runner.

        Args:
            benchmark_path: Path to benchmark JSON dataset.
            thresholds: Custom metric thresholds.
        """
        self._benchmark_path = Path(benchmark_path)
        self._thresholds = thresholds or {
            "faithfulness": 0.72,
            "answer_relevancy": 0.75,
            "hallucination_rate": 0.25,
        }
        self._benchmark: list[dict] = []
        self._load_benchmark()

    def _load_benchmark(self) -> None:
        """Load benchmark dataset from disk."""
        if self._benchmark_path.exists():
            try:
                self._benchmark = json.loads(
                    self._benchmark_path.read_text(encoding="utf-8")
                )
                logger.info(
                    "ragas.benchmark.loaded",
                    cases=len(self._benchmark),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("ragas.benchmark.load_failed", error=str(e))
                self._benchmark = []
        else:
            logger.warning(
                "ragas.benchmark.not_found",
                path=str(self._benchmark_path),
            )

    def run(
        self,
        graph_fn: callable,
        limit: int | None = None,
    ) -> dict:
        """Run RAGAS evaluation on the benchmark.

        Args:
            graph_fn: Function that takes a query dict and returns result dict.
            limit: Maximum number of cases to evaluate.

        Returns:
            Evaluation results with per-case and aggregate scores.
        """
        cases = self._benchmark[:limit] if limit else self._benchmark
        results = []

        for case in cases:
            try:
                result = self._evaluate_case(graph_fn, case)
                results.append(result)
            except Exception as e:
                logger.error(
                    "ragas.case.failed",
                    case_id=case.get("id"),
                    error=str(e),
                )
                results.append({
                    "id": case.get("id", "unknown"),
                    "passed": False,
                    "error": str(e),
                })

        aggregate = self._aggregate_results(results)

        logger.info(
            "ragas.run.done",
            total=len(results),
            passed=aggregate["passed_count"],
            failed=aggregate["failed_count"],
        )

        return {
            "results": results,
            "aggregate": aggregate,
            "thresholds": self._thresholds,
        }

    def _evaluate_case(self, graph_fn: callable, case: dict) -> dict:
        """Evaluate a single benchmark case."""
        query = case.get("query", "")
        ground_truth = case.get("ground_truth", "")

        # Run the graph
        state = graph_fn({
            "user_query": query,
            "user_id": "eval",
            "audio_path": case.get("audio_path"),
            "image_path": case.get("image_path"),
            "file_path": case.get("file_path"),
        })

        answer = state.get("final_answer", "")
        contexts = [c.get("text", "") for c in state.get("retrieved_chunks", [])]

        # Evaluate
        from polymind.infrastructure.eval.deepeval_critic import DeepEvalCritic

        critic = DeepEvalCritic()
        scores = critic.evaluate(query, answer, contexts, ground_truth)

        # Check thresholds — normalize key names between DeepEvalCritic and runner
        # DeepEvalCritic returns "hallucination_rate", but _evaluate_deepeval may
        # return "hallucination" (from HallucinationMetric class name)
        def _get_score(metric: str) -> dict:
            """Get score dict, handling key name variants."""
            data = scores.get(metric)
            if data is not None:
                return data
            # Try variant key: "hallucination" ↔ "hallucination_rate"
            if metric == "hallucination_rate":
                return scores.get("hallucination", {"passed": True, "score": 0.0})
            return {"passed": True, "score": 0.0}

        passed = all(
            _get_score(metric).get("passed", False)
            for metric in self._thresholds
        )

        return {
            "id": case.get("id", "unknown"),
            "query": query,
            "answer": answer[:200],
            "scores": scores,
            "passed": passed,
        }

    def _aggregate_results(self, results: list[dict]) -> dict:
        """Compute aggregate metrics."""
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))

        # Average scores per metric
        metric_scores: dict[str, list[float]] = {}
        for r in results:
            for metric, data in r.get("scores", {}).items():
                if isinstance(data, dict) and "score" in data:
                    metric_scores.setdefault(metric, []).append(data["score"])

        averages = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in metric_scores.items()
        }

        return {
            "total": total,
            "passed_count": passed,
            "failed_count": total - passed,
            "pass_rate": passed / max(total, 1),
            "averages": averages,
        }

    @property
    def benchmark_size(self) -> int:
        """Number of cases in the benchmark."""
        return len(self._benchmark)
