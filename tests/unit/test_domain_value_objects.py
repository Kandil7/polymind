"""Tests for domain value objects."""

from polymind.domain.value_objects.modality import Modality
from polymind.domain.value_objects.retrieval_strategy import RetrievalStrategy
from polymind.domain.value_objects.score import Score


class TestModality:
    def test_all_values(self) -> None:
        assert Modality.TEXT == "text"
        assert Modality.AUDIO == "audio"
        assert Modality.IMAGE == "image"
        assert Modality.DOCUMENT == "document"
        assert Modality.TABLE == "table"
        assert Modality.MULTI == "multi"

    def test_string_comparison(self) -> None:
        assert Modality.TEXT == "text"


class TestRetrievalStrategy:
    def test_all_values(self) -> None:
        assert RetrievalStrategy.SKIP == "skip"
        assert RetrievalStrategy.STANDARD == "standard"
        assert RetrievalStrategy.HIPPORAG == "hipporag"
        assert RetrievalStrategy.SPECULATIVE == "speculative"

    def test_count(self) -> None:
        assert len(RetrievalStrategy) == 4


class TestScore:
    def test_passed_when_above_threshold(self) -> None:
        s = Score(name="faithfulness", value=0.85, threshold=0.72)
        assert s.passed is True

    def test_passed_when_at_threshold(self) -> None:
        s = Score(name="faithfulness", value=0.72, threshold=0.72)
        assert s.passed is True

    def test_failed_when_below_threshold(self) -> None:
        s = Score(name="faithfulness", value=0.50, threshold=0.72)
        assert s.passed is False

    def test_frozen(self) -> None:
        s = Score(name="test", value=0.5, threshold=0.5)
        try:
            s.value = 0.9  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass
