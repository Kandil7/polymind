"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_query_text() -> str:
    """A sample text query for testing."""
    return "What is the refund policy for digital goods?"
