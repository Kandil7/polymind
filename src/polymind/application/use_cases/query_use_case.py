"""QueryUseCase — orchestrates the full query pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polymind.application.graph import build_graph
from polymind.domain.entities.answer import Answer
from polymind.domain.entities.query import Query, QueryResult

if TYPE_CHECKING:
    from polymind.application.state import PolyMindState


class QueryUseCase:
    """Orchestrates the PolyMind agent graph for a single query."""

    def __init__(self) -> None:
        self._graph = build_graph()

    async def execute(self, query: Query) -> QueryResult:
        """Run the full agent graph and return a QueryResult."""
        initial_state: PolyMindState = {
            "user_query": query.text,
            "user_id": query.user_id,
            "audio_path": query.audio_path,
            "image_path": query.image_path,
            "file_path": query.file_path,
            "modality": "text",
            "intent": "",
            "retrieval_strategy": "standard",
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
            "past_episodes": [],
            "semantic_facts": [],
            "planning_context": {},
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "draft_answers": [],
            "final_answer": None,
            "citations": [],
            "critic_scores": {},
            "passed_critic": False,
            "retry_count": 0,
            "should_retry": False,
        }

        result_state = await self._graph.ainvoke(initial_state)

        return QueryResult(
            query_id=query.id,
            answer=Answer(
                text=result_state.get("final_answer", ""),
                confidence=1.0 if result_state.get("passed_critic") else 0.5,
            ),
            modality=result_state.get("modality", "text"),
            retry_count=result_state.get("retry_count", 0),
        )
