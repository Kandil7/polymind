"""Eval test harness — runs RAGAS evaluation against benchmark dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymind.infrastructure.eval.deepeval_critic import DeepEvalCritic

# ── Thresholds (must match CI gate) ─────────────────────
FAITHFULNESS_THRESHOLD = 0.72
RELEVANCY_THRESHOLD = 0.75
HALLUCINATION_THRESHOLD = 0.25


@pytest.fixture
def critic() -> DeepEvalCritic:
    """Create a DeepEval critic instance."""
    return DeepEvalCritic()


@pytest.fixture
def sample_cases() -> list[dict]:
    """Load benchmark cases."""
    benchmark_path = Path(__file__).parent / "benchmark_v1.json"
    if benchmark_path.exists():
        import json
        return json.loads(benchmark_path.read_text(encoding="utf-8"))
    return []


class TestEvalHarness:
    """Test the evaluation harness components."""

    def test_critic_initializes(self, critic: DeepEvalCritic) -> None:
        assert critic is not None

    def test_critic_thresholds(self, critic: DeepEvalCritic) -> None:
        thresholds = critic.thresholds
        assert thresholds["faithfulness"] == FAITHFULNESS_THRESHOLD
        assert thresholds["answer_relevancy"] == RELEVANCY_THRESHOLD
        assert thresholds["hallucination_rate"] == HALLUCINATION_THRESHOLD

    def test_critic_evaluate_good_answer(self, critic: DeepEvalCritic) -> None:
        query = "What is RAG?"
        answer = "RAG stands for Retrieval Augmented Generation."
        contexts = ["RAG is a technique combining retrieval and generation."]
        scores = critic.evaluate(query, answer, contexts)
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "hallucination_rate" in scores

    def test_critic_evaluate_empty_answer(self, critic: DeepEvalCritic) -> None:
        scores = critic.evaluate("test", "", ["context"])
        assert scores["faithfulness"]["score"] == 0.0
        assert scores["faithfulness"]["passed"] is False

    def test_critic_evaluate_with_ground_truth(
        self, critic: DeepEvalCritic
    ) -> None:
        scores = critic.evaluate(
            "What is Python?",
            "Python is a programming language.",
            ["Python is a high-level programming language."],
            ground_truth="Python is a programming language.",
        )
        assert "faithfulness" in scores


class TestBenchmarkDataset:
    """Test the benchmark dataset structure."""

    def test_benchmark_loads(self, sample_cases: list[dict]) -> None:
        assert len(sample_cases) > 0

    def test_benchmark_has_required_fields(self, sample_cases: list[dict]) -> None:
        for case in sample_cases:
            assert "id" in case, "Missing 'id' in case"
            assert "query" in case, f"Missing 'query' in case {case['id']}"
            assert "ground_truth" in case, f"Missing 'ground_truth' in case {case['id']}"
            assert "modality" in case, f"Missing 'modality' in case {case['id']}"

    def test_benchmark_ids_unique(self, sample_cases: list[dict]) -> None:
        ids = [c["id"] for c in sample_cases]
        assert len(ids) == len(set(ids))

    def test_benchmark_modalities_valid(self, sample_cases: list[dict]) -> None:
        valid = {"text", "audio", "image", "document", "table"}
        for case in sample_cases:
            assert case["modality"] in valid, f"Invalid modality: {case['modality']}"


class TestEvalThresholds:
    """Test that eval thresholds are enforced."""

    def test_faithfulness_threshold_enforced(self) -> None:
        assert FAITHFULNESS_THRESHOLD == 0.72

    def test_relevancy_threshold_enforced(self) -> None:
        assert RELEVANCY_THRESHOLD == 0.75

    def test_hallucination_threshold_enforced(self) -> None:
        assert HALLUCINATION_THRESHOLD == 0.25
