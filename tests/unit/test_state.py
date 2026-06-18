"""Tests for PolyMindState schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymind.application.state import PolyMindState


class TestPolyMindState:
    def test_minimal_state(self) -> None:
        state: PolyMindState = {
            "user_query": "hello",
            "user_id": "u1",
        }
        assert state["user_query"] == "hello"

    def test_state_accepts_optional_fields(self) -> None:
        state: PolyMindState = {
            "user_query": "hello",
            "user_id": "u1",
            "audio_path": "/tmp/audio.mp3",
            "modality": "audio",
            "passed_critic": True,
            "retry_count": 2,
        }
        assert state["audio_path"] == "/tmp/audio.mp3"
        assert state["retry_count"] == 2
