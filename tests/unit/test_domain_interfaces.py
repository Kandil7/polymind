"""Tests for domain interfaces."""

import pytest

from polymind.domain.interfaces.evaluator import IEvaluator
from polymind.domain.interfaces.generator import IGenerator
from polymind.domain.interfaces.memory_store import IMemoryStore
from polymind.domain.interfaces.retriever import IRetriever
from polymind.domain.interfaces.specialist import ISpecialist


class TestInterfacesCannotBeInstantiated:
    def test_retriever_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            IRetriever()  # type: ignore[abstract]

    def test_specialist_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ISpecialist()  # type: ignore[abstract]

    def test_generator_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            IGenerator()  # type: ignore[abstract]

    def test_evaluator_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            IEvaluator()  # type: ignore[abstract]

    def test_memory_store_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            IMemoryStore()  # type: ignore[abstract]
