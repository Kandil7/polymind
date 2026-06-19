"""Comprehensive tests for Critic node — LLM-as-Judge and heuristic evaluation."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from polymind.application.agents.critic import (
    FAITHFULNESS_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    MAX_RETRIES,
    RELEVANCY_THRESHOLD,
    _evaluate_heuristic,
    _parse_scores,
    decide,
    run,
)
from polymind.application.state import PolyMindState


class TestCriticThresholds:
    def test_faithfulness_threshold(self) -> None:
        assert FAITHFULNESS_THRESHOLD == 0.72

    def test_relevancy_threshold(self) -> None:
        assert RELEVANCY_THRESHOLD == 0.75

    def test_hallucination_threshold(self) -> None:
        assert HALLUCINATION_THRESHOLD == 0.25

    def test_max_retries(self) -> None:
        assert MAX_RETRIES == 2


class TestHeuristicEvaluation:
    def test_good_answer_high_scores(self) -> None:
        query = "What is RAG?"
        answer = "RAG is Retrieval Augmented Generation, a technique that combines retrieval with generation for better LLM answers."
        contexts = ["RAG stands for Retrieval Augmented Generation. It improves LLM outputs by retrieving relevant documents and combining retrieval with generation."]
        scores = _evaluate_heuristic(query, answer, contexts)
        assert scores["answer_relevancy"] > 0.3  # query words overlap with answer
        assert scores["faithfulness"] > 0.3  # context words overlap with answer
        assert scores["hallucination_rate"] < 0.7

    def test_empty_answer_fails(self) -> None:
        scores = _evaluate_heuristic("test", "", ["context"])
        assert scores["faithfulness"] == 0.0
        assert scores["hallucination_rate"] == 1.0

    def test_off_topic_low_relevancy(self) -> None:
        scores = _evaluate_heuristic(
            "What is RAG?",
            "The weather today is sunny with clear blue skies and warm temperatures.",
            ["RAG is a retrieval technique."],
        )
        assert scores["answer_relevancy"] < 0.5  # Low overlap with query

    def test_context_dependent_high_faithfulness(self) -> None:
        scores = _evaluate_heuristic(
            "What is Python?",
            "Python is a programming language.",
            ["Python is a high-level programming language created by Guido van Rossum."],
        )
        assert scores["faithfulness"] > 0.5


class TestScoreParsing:
    def test_parse_valid_json(self) -> None:
        text = '{"faithfulness": 0.9, "answer_relevancy": 0.8, "hallucination_rate": 0.1}'
        scores = _parse_scores(text)
        assert scores["faithfulness"] == 0.9
        assert scores["answer_relevancy"] == 0.8
        assert scores["hallucination_rate"] == 0.1

    def test_parse_with_surrounding_text(self) -> None:
        text = 'Here are the scores: {"faithfulness": 0.85, "answer_relevancy": 0.7, "hallucination_rate": 0.15} done.'
        scores = _parse_scores(text)
        assert scores["faithfulness"] == 0.85

    def test_parse_invalid_returns_defaults(self) -> None:
        scores = _parse_scores("no json here")
        assert scores["faithfulness"] == 0.5
        assert scores["answer_relevancy"] == 0.5
        assert scores["hallucination_rate"] == 0.5

    def test_parse_partial_json(self) -> None:
        text = '{"faithfulness": 0.9}'
        scores = _parse_scores(text)
        # Missing keys should get defaults
        assert scores["faithfulness"] == 0.9
        assert scores["answer_relevancy"] == 0.5  # default


class TestCriticDecide:
    def test_pass_when_not_retrying(self) -> None:
        state: PolyMindState = {
            "should_retry": False,
            "retry_count": 0,
        }
        assert decide(state) == "pass"

    def test_retry_when_should_retry(self) -> None:
        state: PolyMindState = {
            "should_retry": True,
            "retry_count": 0,
        }
        assert decide(state) == "retry"

    def test_fail_max_when_max_retries(self) -> None:
        # When should_retry is False (as run() sets it when retries exhausted)
        # but retry_count >= MAX_RETRIES, decide returns "fail_max"
        state: PolyMindState = {
            "should_retry": False,
            "retry_count": MAX_RETRIES,
        }
        assert decide(state) == "fail_max"

    def test_retry_when_should_retry_and_below_max(self) -> None:
        state: PolyMindState = {
            "should_retry": True,
            "retry_count": 0,
        }
        assert decide(state) == "retry"


class TestCriticRun:
    @patch("polymind.application.agents.critic._evaluate_with_llm")
    def test_run_passes_good_answer(self, mock_eval) -> None:
        mock_eval.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "hallucination_rate": 0.1,
        }
        state: PolyMindState = {
            "user_query": "What is RAG?",
            "final_answer": "RAG is Retrieval Augmented Generation.",
            "retrieved_chunks": [{"text": "RAG combines retrieval and generation."}],
            "retry_count": 0,
        }
        result = run(state)
        assert result["passed_critic"] is True
        assert result["should_retry"] is False
        assert result["retry_count"] == 0

    @patch("polymind.application.agents.critic._evaluate_with_llm")
    def test_run_retries_bad_answer(self, mock_eval) -> None:
        mock_eval.return_value = {
            "faithfulness": 0.3,  # Below threshold
            "answer_relevancy": 0.4,
            "hallucination_rate": 0.6,
        }
        state: PolyMindState = {
            "user_query": "test",
            "final_answer": "bad answer",
            "retrieved_chunks": [{"text": "context"}],
            "retry_count": 0,
        }
        result = run(state)
        assert result["passed_critic"] is False
        assert result["should_retry"] is True
        assert result["retry_count"] == 1

    @patch("polymind.application.agents.critic._evaluate_with_llm")
    def test_run_stops_retrying_at_max(self, mock_eval) -> None:
        mock_eval.return_value = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.4,
            "hallucination_rate": 0.6,
        }
        state: PolyMindState = {
            "user_query": "test",
            "final_answer": "bad answer",
            "retrieved_chunks": [],
            "retry_count": MAX_RETRIES,
        }
        result = run(state)
        assert result["passed_critic"] is False
        assert result["should_retry"] is False
        assert result["retry_count"] == MAX_RETRIES
