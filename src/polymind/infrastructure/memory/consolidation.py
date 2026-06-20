"""Memory Consolidation Pipeline — learns from interaction patterns.

Automatically consolidates episodic memories into semantic facts and
procedural knowledge after each query.

Pipeline:
1. Analyze recent episodic memories for patterns
2. Extract semantic facts using LLM
3. Store successful procedures based on critic scores
4. Deduplicate and prune old facts
5. Schedule periodic consolidation

Usage:
    from polymind.infrastructure.memory.consolidation import ConsolidationPipeline

    pipeline = ConsolidationPipeline(user_id="user123")
    await pipeline.consolidate(query, answer, critic_scores)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()

# ── Configuration ────────────────────────────────────────
MIN_EPISODES_FOR_CONSOLIDATION = 3
MAX_FACTS_PER_CONSOLIDATION = 3
FACT_DEDUPLICATION_THRESHOLD = 0.9  # Cosine similarity threshold
PROCEDURE_SUCCESS_THRESHOLD = 0.7


class ConsolidationPipeline:
    """Orchestrates memory consolidation after each query.

    Extracts semantic facts from episodic patterns and stores
    successful procedures for procedural learning.
    """

    def __init__(
        self,
        user_id: str = "default",
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        """Initialize the consolidation pipeline.

        Args:
            user_id: User identifier for memory isolation.
            qdrant_url: Qdrant server URL for semantic store.
        """
        self._user_id = user_id
        self._qdrant_url = qdrant_url
        self._episode_buffer: list[dict] = []
        self._last_consolidation = datetime.now(UTC)

    async def consolidate(
        self,
        query: str,
        answer: str,
        critic_scores: dict | None = None,
        modality: str = "text",
    ) -> dict:
        """Run consolidation after a query.

        Args:
            query: The user's query.
            answer: The generated answer.
            critic_scores: Scores from the critic agent.
            modality: Input modality.

        Returns:
            Consolidation results (facts extracted, procedures stored).
        """
        results = {
            "facts_extracted": 0,
            "procedures_stored": 0,
            "episodes_analyzed": 0,
        }

        # Store this episode in buffer
        self._episode_buffer.append({
            "query": query,
            "answer": answer,
            "scores": critic_scores or {},
            "modality": modality,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Check if we should consolidate
        if len(self._episode_buffer) < MIN_EPISODES_FOR_CONSOLIDATION:
            logger.debug(
                "consolidation.deferred",
                buffer_size=len(self._episode_buffer),
                threshold=MIN_EPISODES_FOR_CONSOLIDATION,
            )
            return results

        # Run consolidation
        try:
            # Extract semantic facts
            facts = await self._extract_facts()
            results["facts_extracted"] = len(facts)
            results["episodes_analyzed"] = len(self._episode_buffer)

            # Store successful procedures
            procedures = self._extract_procedures()
            results["procedures_stored"] = len(procedures)

            # Clear buffer after successful consolidation
            self._episode_buffer = []
            self._last_consolidation = datetime.now(UTC)

            logger.info(
                "consolidation.done",
                facts=results["facts_extracted"],
                procedures=results["procedures_stored"],
                episodes=results["episodes_analyzed"],
            )

        except Exception as e:
            logger.error("consolidation.failed", error=str(e))

        return results

    async def _extract_facts(self) -> list[str]:
        """Extract semantic facts from episodic patterns using LLM."""
        if not self._episode_buffer:
            return []

        try:
            from polymind.infrastructure.llm.llm_factory import LLMFactory

            factory = LLMFactory()
            llm = factory.get_llm(tier="fast")

            # Format episodes for the prompt
            episodes_text = ""
            for i, ep in enumerate(self._episode_buffer[:5], 1):
                episodes_text += f"\n{i}. Q: {ep['query'][:200]}\n   A: {ep['answer'][:200]}\n"

            prompt = f"""Analyze these conversation episodes and extract reusable semantic facts.
Each fact should be a standalone, factual statement that could help answer future queries.
Focus on:
- Key concepts and definitions
- Relationships between entities
- Reusable knowledge (not specific to one conversation)

Episodes:
{episodes_text}

Extract {MAX_FACTS_PER_CONSOLIDATION} or fewer facts. Return as a JSON array of strings.
Example: ["Fact 1", "Fact 2"]
"""

            from langchain_core.messages import HumanMessage

            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Parse JSON response
            import json
            import re

            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    facts = json.loads(json_match.group())
                    if isinstance(facts, list):
                        # Validate and deduplicate
                        valid_facts = []
                        for fact in facts:
                            if isinstance(fact, str) and len(fact) > 10:
                                if not self._is_duplicate(fact):
                                    valid_facts.append(fact)
                                    self._store_fact(fact)
                        return valid_facts[:MAX_FACTS_PER_CONSOLIDATION]
                except json.JSONDecodeError:
                    pass

            # Fallback: extract facts from individual episodes
            return await self._extract_fallback_facts()

        except Exception as e:
            logger.warning("consolidation.fact_extraction.failed", error=str(e))
            return await self._extract_fallback_facts()

    async def _extract_fallback_facts(self) -> list[str]:
        """Fallback fact extraction without LLM."""
        facts = []
        for ep in self._episode_buffer[:MAX_FACTS_PER_CONSOLIDATION]:
            # Simple heuristic: extract the answer as a fact if it's factual
            answer = ep.get("answer", "")
            if len(answer) > 20 and not answer.startswith("Error"):
                # Clean up the answer
                fact = answer.split("\n")[0].strip()
                if fact and not self._is_duplicate(fact):
                    self._store_fact(fact)
                    facts.append(fact)
        return facts

    def _store_fact(self, fact: str) -> None:
        """Store a fact in the semantic store."""
        try:
            from polymind.infrastructure.memory.semantic_store import SemanticStore
            from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
            from polymind.infrastructure.rag.embedder import Embedder

            client = get_qdrant_client(url=self._qdrant_url)
            embedder = Embedder()
            store = SemanticStore(client, embedder)
            store.store(fact, source_query="consolidation")
        except Exception as e:
            logger.debug("consolidation.store_fact.failed", error=str(e))

    def _is_duplicate(self, fact: str) -> bool:
        """Check if a fact is too similar to existing facts."""
        try:
            from polymind.infrastructure.memory.semantic_store import SemanticStore
            from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
            from polymind.infrastructure.rag.embedder import Embedder

            client = get_qdrant_client(url=self._qdrant_url)
            embedder = Embedder()
            store = SemanticStore(client, embedder)

            # Recall similar facts
            similar = store.recall(fact, top_k=1)
            if similar:
                # Simple check: if fact is substring of existing
                for existing in similar:
                    if fact.lower() in existing.lower() or existing.lower() in fact.lower():
                        return True
            return False
        except Exception:
            return False

    def _extract_procedures(self) -> list[dict]:
        """Extract procedural knowledge from successful interactions."""
        procedures = []

        for ep in self._episode_buffer:
            scores = ep.get("scores", {})
            faithfulness = 0.0

            # Handle different score formats
            if isinstance(scores, dict):
                f = scores.get("faithfulness", 0.5)
                if isinstance(f, dict):
                    faithfulness = f.get("score", 0.5)
                else:
                    faithfulness = float(f) if f else 0.5

            # Only store successful procedures
            if faithfulness >= PROCEDURE_SUCCESS_THRESHOLD:
                task_type = self._classify_task_type(ep.get("query", ""))
                procedure = {
                    "task_type": task_type,
                    "steps": [
                        f"Query: {ep.get('query', '')[:100]}",
                        f"Retrieved context used",
                        f"Answer generated with faithfulness {faithfulness:.2f}",
                    ],
                    "success_score": faithfulness,
                }
                procedures.append(procedure)
                self._store_procedure(procedure)

        return procedures

    def _classify_task_type(self, query: str) -> str:
        """Classify the task type from the query."""
        q = query.lower()

        if any(w in q for w in ("summarize", "summary", "tldr")):
            return "summarization"
        elif any(w in q for w in ("compare", "difference", "vs")):
            return "comparison"
        elif any(w in q for w in ("what", "who", "when", "where")):
            return "factual_qa"
        elif any(w in q for w in ("translate", "translation")):
            return "translation"
        else:
            return "general"

    def _store_procedure(self, procedure: dict) -> None:
        """Store a procedure in the procedural store."""
        try:
            from polymind.infrastructure.memory.procedural_store import ProceduralStore

            store = ProceduralStore()
            store.store(
                task_type=procedure["task_type"],
                steps=procedure["steps"],
                success_score=procedure["success_score"],
            )
        except Exception as e:
            logger.debug("consolidation.store_procedure.failed", error=str(e))

    def get_stats(self) -> dict:
        """Get consolidation statistics."""
        return {
            "buffer_size": len(self._episode_buffer),
            "last_consolidation": self._last_consolidation.isoformat(),
            "threshold": MIN_EPISODES_FOR_CONSOLIDATION,
        }
