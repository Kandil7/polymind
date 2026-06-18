"""4-Layer Memory System — orchestrates Episodic, Semantic, and Procedural stores.

Architecture (inspired by cognitive science):
- Working Memory → LangGraph state (managed by graph)
- Episodic Memory → Mem0 (conversation history)
- Semantic Memory → Qdrant (extracted facts)
- Procedural Memory → JSON (successful patterns)
"""

from __future__ import annotations

import structlog

from polymind.infrastructure.memory.episodic_store import EpisodicStore
from polymind.infrastructure.memory.procedural_store import ProceduralStore

logger = structlog.get_logger()


class FourLayerMemory:
    """Orchestrates all memory layers for the PolyMind agent.

    Provides a unified interface for:
    - Storing episodes after each query
    - Recalling relevant context from all layers
    - Consolidating episodic patterns into semantic facts
    """

    def __init__(
        self,
        user_id: str = "default",
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        """Initialize the 4-layer memory system.

        Args:
            user_id: User identifier for memory isolation.
            qdrant_url: Qdrant server URL.
        """
        self._user_id = user_id

        # Layer 2: Episodic (Mem0)
        self._episodic = EpisodicStore(user_id=user_id)

        # Layer 3: Semantic (Qdrant) — lazy loaded
        self._semantic = None
        self._qdrant_url = qdrant_url

        # Layer 4: Procedural (JSON)
        self._procedural = ProceduralStore()

        logger.info("memory.initialized", user_id=user_id)

    def _ensure_semantic(self) -> None:
        """Lazy-load semantic store."""
        if self._semantic is not None:
            return

        try:
            from qdrant_client import QdrantClient

            from polymind.infrastructure.memory.semantic_store import SemanticStore
            from polymind.infrastructure.rag.embedder import Embedder

            client = QdrantClient(url=self._qdrant_url)
            embedder = Embedder()
            self._semantic = SemanticStore(client, embedder)
        except Exception as e:
            logger.warning("memory.semantic.load_failed", error=str(e))

    def store_episode(
        self,
        query: str,
        answer: str,
        faithfulness: float | None = None,
        modality: str = "text",
    ) -> None:
        """Store a complete interaction episode.

        Args:
            query: User's original query.
            answer: Generated answer.
            faithfulness: Critic faithfulness score.
            modality: Input modality.
        """
        self._episodic.store(query, answer, faithfulness, modality)

    def recall_episodes(self, query: str, top_k: int = 3) -> list[dict]:
        """Recall similar past episodes.

        Args:
            query: Search query.
            top_k: Number of episodes to recall.

        Returns:
            List of recalled episodes.
        """
        return self._episodic.recall(query, top_k)

    def recall_semantic(self, query: str, top_k: int = 5) -> list[str]:
        """Recall relevant semantic facts.

        Args:
            query: Search query.
            top_k: Number of facts to recall.

        Returns:
            List of relevant facts.
        """
        self._ensure_semantic()
        if self._semantic is None:
            return []
        return self._semantic.recall(query, top_k)

    def recall_procedure(self, task_type: str) -> list[str] | None:
        """Recall a procedure for a task type.

        Args:
            task_type: Category of task.

        Returns:
            List of steps if found, None otherwise.
        """
        return self._procedural.recall(task_type)

    def store_procedure(
        self,
        task_type: str,
        steps: list[str],
        success_score: float,
    ) -> None:
        """Store a successful procedure.

        Args:
            task_type: Category of task.
            steps: Steps that were taken.
            success_score: How successful (0-1).
        """
        self._procedural.store(task_type, steps, success_score)

    def consolidate(
        self,
        query: str,
        answer: str,
        llm: object | None = None,
    ) -> None:
        """Consolidate episodic patterns into semantic facts.

        Called after each query to check if patterns should be extracted.

        Args:
            query: Original query.
            answer: Generated answer.
            llm: Optional LLM for fact extraction.
        """
        self._ensure_semantic()
        if self._semantic is None:
            return

        # Check if we have enough episodes to consolidate
        episodes = self._episodic.recall(query, top_k=5)
        if len(episodes) < 3:
            return

        # Extract a semantic fact if LLM available
        if llm is not None:
            try:
                from langchain_core.messages import HumanMessage

                prompt = f"""From these similar conversation episodes, extract ONE
reusable semantic fact that would help answer future queries.

Episodes:
{chr(10).join(str(e) for e in episodes[:3])}

Reply with a single factual sentence (no preamble):"""

                response = llm.invoke([HumanMessage(content=prompt)])
                fact = response.content.strip()

                if fact:
                    self._semantic.store(fact, source_query=query)
                    logger.info("memory.consolidated", fact_length=len(fact))
            except Exception as e:
                logger.error("memory.consolidation.failed", error=str(e))

    def get_context(self, query: str) -> dict:
        """Get all memory context for the planner.

        Args:
            query: Current query.

        Returns:
            Dict with episodes, semantic_facts, and procedure.
        """
        episodes = self.recall_episodes(query, top_k=3)
        semantic_facts = self.recall_semantic(query, top_k=5)

        return {
            "episodes": episodes,
            "semantic_facts": semantic_facts,
        }
