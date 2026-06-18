"""Tests for Synthesizer node."""

from __future__ import annotations

from polymind.application.agents.synthesizer import (
    _calculate_confidence,
    _format_answer,
    run,
)


class TestConfidence:
    def test_high_faithfulness_high_confidence(self) -> None:
        conf = _calculate_confidence(0.9, 0)
        assert conf > 0.8

    def test_low_faithfulness_low_confidence(self) -> None:
        conf = _calculate_confidence(0.3, 0)
        assert conf < 0.5

    def test_retries_reduce_confidence(self) -> None:
        conf_no_retry = _calculate_confidence(0.9, 0)
        conf_one_retry = _calculate_confidence(0.9, 1)
        assert conf_one_retry < conf_no_retry

    def test_confidence_bounded(self) -> None:
        assert _calculate_confidence(0.0, 0) >= 0.0
        assert _calculate_confidence(1.0, 0) <= 1.0


class TestFormatAnswer:
    def test_answer_only(self) -> None:
        result = _format_answer("Hello", [], 0.9)
        assert "Hello" in result
        assert "Confidence: 90%" in result

    def test_answer_with_citations(self) -> None:
        citations = [{"source": "doc.pdf", "score": 0.85}]
        result = _format_answer("Hello", citations, 0.9)
        assert "Sources:" in result
        assert "doc.pdf" in result

    def test_multiple_citations(self) -> None:
        citations = [
            {"source": "a.pdf", "score": 0.9},
            {"source": "b.pdf", "score": 0.8},
        ]
        result = _format_answer("Hello", citations, 0.9)
        assert "a.pdf" in result
        assert "b.pdf" in result


class TestSynthesizerRun:
    def test_run_enhances_answer(self) -> None:
        state = {
            "final_answer": "RAG is Retrieval Augmented Generation.",
            "citations": [{"source": "doc.pdf", "score": 0.85}],
            "critic_scores": {"faithfulness": 0.9},
            "retry_count": 0,
        }
        result = run(state)
        assert "Confidence:" in result["final_answer"]
        assert "Sources:" in result["final_answer"]
