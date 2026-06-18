"""Domain exceptions — one per failure mode, never caught silently."""


class DomainError(Exception):
    """Base exception for all PolyMind domain errors."""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class RetrievalError(DomainError):
    """Raised when the retrieval pipeline fails."""


class CriticFailedError(DomainError):
    """Raised when the Critic agent rejects an answer after max retries."""


class IngestionError(DomainError):
    """Raised when document ingestion fails."""


class SpecialistError(DomainError):
    """Raised when a specialist model wrapper fails."""
