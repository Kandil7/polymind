"""Tests for Critic node."""

from __future__ import annotations

from polymind.application.agents.critic import (
    _evaluate_heuristic,
    _parse_scores,
    decide,
    run,
)


class TestCriticHeuristic:
    def test_good_answer_scores_high(self) -> None:
        query = "What is RAG?"
        answer = "RAG stands for Retrieval Augmented Generation."
        contexts = ["RAG is Retrieval Augmented Generation for LLMs."]
        scores = _evaluate_heuristic(query, answer, contexts)
        assert scores["faithfulness"] > 0.3

    def test_empty_answer_scores_low(self) -> None:
        scores = _evaluate_heuristic("test", "", ["context"])
        assert scores["faithfulness"] == 0.0
        assert scores["hallucination_rate"] == 1.0

    def test_off_topic_answer(self) -> None:
        scores = _evaluate_heuristic("What is Python?", "The weather is nice.", ["Python is a language."])
        # Off-topic: answer words don't overlap with context
        assert scores["faithfulness"] < 0.5


class TestCriticParsing:
    def test_parse_valid_json(self) -> None:
        text = '{"faithfulness": 0.85, "answer_relevancy": 0.9, "hallucination_rate": 0.1}'
        scores = _parse_scores(text)
        assert scores["faithfulness"] == 0.85
        assert scores["answer_relevancy"] == 0.9
        assert scores["hallucination_rate"] == 0.1

    def test_parse_with_surrounding_text(self) -> None:
        text = 'Here are the scores: {"faithfulness": 0.7, "answer_relevancy": 0.8, "hallucination_rate": 0.2} done'
        scores = _parse_scores(text)
        assert scores["faithfulness"] == 0.7

    def test_parse_invalid_returns_defaults(self) -> None:
        scores = _parse_scores("no json here")
        assert scores["faithfulness"] == 0.5


class TestCriticDecide:
    def test_pass_when_should_retry_false(self) -> None:
        state = {"should_retry": False, "retry_count": 0}
        assert decide(state) == "pass"

    def test_retry_when_should_retry_true(self) -> None:
        state = {"should_retry": True, "retry_count": 1}
        assert decide(state) == "retry"

    def test_fail_max_when_max_retries(self) -> None:
        state = {"should_retry": False, "retry_count": 2}
        assert decide(state) == "fail_max"


class TestCriticRun:
    def test_run_with_heuristic(self) -> None:
        state = {
            "user_query": "What is RAG?",
            "final_answer": "RAG is Retrieval Augmented Generation.",
            "retrieved_chunks": [{"text": "RAG is a technique for LLMs.", "source": "test", "score": 0.8}],
            "retry_count": 0,
        }
        result = run(state)
        assert "critic_scores" in result
        assert "passed_critic" in result
        assert "should_retry" in result
