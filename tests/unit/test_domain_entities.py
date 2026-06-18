"""Tests for domain entities."""

from uuid import UUID

from polymind.domain.entities.answer import Answer
from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.domain.entities.episode import ConversationEpisode
from polymind.domain.entities.query import Query, ScoreResult


class TestQuery:
    def test_query_has_uuid(self) -> None:
        q = Query(text="hello", user_id="u1")
        assert isinstance(q.id, UUID)

    def test_query_is_frozen(self) -> None:
        q = Query(text="hello", user_id="u1")
        try:
            q.text = "modified"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_query_defaults(self) -> None:
        q = Query(text="test", user_id="u1")
        assert q.audio_path is None
        assert q.image_path is None
        assert q.file_path is None
        assert q.metadata == {}


class TestAnswer:
    def test_answer_confidence_bounds(self) -> None:
        a = Answer(text="yes", confidence=0.9)
        assert a.confidence == 0.9

    def test_answer_is_frozen(self) -> None:
        a = Answer(text="yes", confidence=0.5)
        try:
            a.text = "no"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass


class TestDocumentChunk:
    def test_chunk_metadata(self) -> None:
        meta = ChunkMetadata(source="doc.pdf", file_type="pdf", page=1)
        assert meta.source == "doc.pdf"
        assert meta.modality == "text"

    def test_chunk_has_uuid(self) -> None:
        c = DocumentChunk(text="hello", metadata=ChunkMetadata(source="x", file_type="txt"))
        assert isinstance(c.id, UUID)


class TestScoreResult:
    def test_score_passed(self) -> None:
        s = ScoreResult(name="faithfulness", value=0.85, threshold=0.72, passed=True)
        assert s.passed is True

    def test_score_failed(self) -> None:
        s = ScoreResult(name="faithfulness", value=0.5, threshold=0.72, passed=False)
        assert s.passed is False


class TestConversationEpisode:
    def test_episode_defaults(self) -> None:
        e = ConversationEpisode(query="q", answer="a")
        assert e.modality == "text"
        assert e.faithfulness is None
