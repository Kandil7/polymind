"""ISpecialist interface — contract for all specialist model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ISpecialist(ABC):
    """Contract for specialist model wrappers (ASR, VQA, DocQA, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the specialist's name."""
        ...

    @abstractmethod
    async def process(self, input_data: Any, **kwargs: object) -> dict[str, Any]:
        """Process input and return structured results."""
        ...
