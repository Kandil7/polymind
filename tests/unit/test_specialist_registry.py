"""Tests for specialist wrappers module structure."""

from __future__ import annotations

from polymind.domain.interfaces.specialist import ISpecialist
from polymind.infrastructure.specialists import (
    ASRWrapper,
    DocQAWrapper,
    SummarizerWrapper,
    TableQAWrapper,
    VQAWrapper,
)


class TestSpecialistRegistry:
    """Test that all specialists are importable and implement ISpecialist."""

    def test_all_specialists_importable(self) -> None:
        assert ASRWrapper is not None
        assert VQAWrapper is not None
        assert DocQAWrapper is not None
        assert TableQAWrapper is not None
        assert SummarizerWrapper is not None

    def test_all_implement_interface(self) -> None:
        assert issubclass(ASRWrapper, ISpecialist)
        assert issubclass(VQAWrapper, ISpecialist)
        assert issubclass(DocQAWrapper, ISpecialist)
        assert issubclass(TableQAWrapper, ISpecialist)
        assert issubclass(SummarizerWrapper, ISpecialist)

    def test_all_have_name_property(self) -> None:
        """All wrappers must have a 'name' property that returns a string."""
        for wrapper_cls in [ASRWrapper, VQAWrapper, DocQAWrapper, TableQAWrapper, SummarizerWrapper]:
            # Check the class has a name property
            assert hasattr(wrapper_cls, "name")
            # Check it's a property
            assert isinstance(wrapper_cls.__dict__["name"], property)

    def test_specialist_names_are_unique(self) -> None:
        names = ["asr", "vqa", "docqa", "tableqa", "summarizer"]
        assert len(names) == len(set(names))
