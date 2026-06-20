"""Tests for Cross-Encoder Reranker."""

from __future__ import annotations

import pytest

from polymind.infrastructure.rag.reranker import CrossEncoderReranker


class TestRerankerStructure:
    def test_default_model(self) -> None:
        from polymind.infrastructure.rag.reranker import DEFAULT_MODEL
        assert DEFAULT_MODEL == "BAAI/bge-reranker-v2-m3"

    def test_init_with_defaults(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_id = "test/model"
        reranker._top_k = 5
        reranker._model = None
        assert reranker._top_k == 5

    def test_is_available_when_model_loaded(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = "fake_model"
        assert reranker.is_available is True

    def test_is_available_when_model_not_loaded(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        assert reranker.is_available is False


class TestRerankerFallback:
    def test_rerank_empty_documents(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._top_k = 5
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_fallback_returns_original_order(self) -> None:
        """Without model, returns original order with declining scores."""
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._top_k = 3

        docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = reranker.rerank("query", docs)

        assert len(result) == 3
        # Indices should be 0, 1, 2 (original order)
        assert result[0][0] == 0
        assert result[1][0] == 1
        assert result[2][0] == 2
        # Scores should decline
        assert result[0][1] > result[1][1] > result[2][1]

    def test_rerank_respects_top_k(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._top_k = 2

        docs = ["a", "b", "c"]
        result = reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2


class TestRerankerWithMockModel:
    def test_rerank_with_mock_model(self) -> None:
        """Test reranking with a mocked cross-encoder model."""
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_id = "test/model"
        reranker._top_k = 2

        # Mock model that scores based on exact word presence
        class MockCrossEncoder:
            def predict(self, pairs):
                scores = []
                for query, doc in pairs:
                    # Higher score if doc contains "relevant" as a word
                    score = 1.0 if " relevant " in f" {doc.lower()} " else 0.1
                    scores.append(score)
                return scores

        reranker._model = MockCrossEncoder()

        docs = [
            "This is irrelevant content",
            "This is relevant information",
            "Another irrelevant document",
            "Very relevant answer here",
        ]

        result = reranker.rerank("what is relevant?", docs)

        assert len(result) == 2
        # Top results should be the "relevant" docs
        top_indices = {idx for idx, _ in result}
        assert 1 in top_indices  # "relevant information"
        assert 3 in top_indices  # "relevant answer"
        # Scores should be high
        assert all(score > 0.5 for _, score in result)
