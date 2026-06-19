"""Eval endpoint — run evaluation harness."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter

from polymind.api.schemas.eval import EvalRequest, EvalResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=EvalResponse)
async def eval_endpoint(request: EvalRequest) -> EvalResponse:
    """Run the evaluation harness.

    Evaluates PolyMind against the benchmark dataset.
    Returns pass/fail status with per-metric scores.
    """
    start_time = time.time()

    try:
        from polymind.application.graph import build_graph
        from polymind.infrastructure.eval.ragas_runner import RAGASRunner

        graph = build_graph()

        def graph_fn(state: dict) -> dict:
            return graph.invoke(state)

        runner = RAGASRunner()
        results = runner.run(graph_fn, limit=request.limit or 10)

        agg = results["aggregate"]
        processing_time = (time.time() - start_time) * 1000

        return EvalResponse(
            status="completed",
            total=agg["total"],
            passed=agg["passed_count"],
            failed=agg["failed_count"],
            pass_rate=agg["pass_rate"],
            averages=agg["averages"],
            thresholds=results["thresholds"],
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("eval.failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return EvalResponse(
            status="error",
            total=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            averages={},
            thresholds={},
            processing_time_ms=round(processing_time, 2),
        )
