"""Eval runner script — run evaluation locally."""

from __future__ import annotations

import sys
import structlog
from pathlib import Path

logger = structlog.get_logger()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_eval() -> None:
    """Run the evaluation harness."""
    from polymind.infrastructure.eval.ragas_runner import RAGASRunner
    from polymind.application.graph import build_graph

    logger.info("eval.start")

    # Build graph
    graph = build_graph()

    def graph_fn(state: dict) -> dict:
        return graph.invoke(state)

    # Run evaluation
    runner = RAGASRunner()
    results = runner.run(graph_fn, limit=10)  # Limit to 10 for quick check

    # Print summary
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"Eval Results: {agg['passed_count']}/{agg['total']} passed")
    print(f"Pass Rate: {agg['pass_rate']:.1%}")
    print(f"{'='*60}")

    for metric, avg in agg.get("averages", {}).items():
        print(f"  {metric}: {avg:.3f}")

    # Check thresholds
    thresholds = results["thresholds"]
    all_pass = True
    for metric, threshold in thresholds.items():
        avg = agg.get("averages", {}).get(metric, 0.0)
        if metric == "hallucination_rate":
            ok = avg <= threshold
        else:
            ok = avg >= threshold
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {metric}: {avg:.3f} (threshold: {threshold}) [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_eval()
