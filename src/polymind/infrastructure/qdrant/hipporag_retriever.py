"""HippoRAG Retriever — Knowledge Graph + Personalized PageRank.

Implements HippoRAG v2 pattern:
1. Extract subject-predicate-object triples from passages
2. Build a Knowledge Graph (NetworkX DiGraph)
3. Add synonymy edges between similar entities
4. Retrieve using Personalized PageRank from query entities

Based on: https://arxiv.org/abs/2405.14831
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import structlog

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.domain.interfaces.retriever import IRetriever

if TYPE_CHECKING:
    from polymind.infrastructure.rag.embedder import Embedder

logger = structlog.get_logger()

DEFAULT_SYNONYM_THRESHOLD = 0.85
DEFAULT_PPR_ALPHA = 0.85
DEFAULT_PPR_MAX_ITER = 100


class HippoRAGRetriever(IRetriever):
    """HippoRAG v2 retriever with Knowledge Graph and Personalized PageRank.

    Builds an associative memory graph from document passages and uses
    PageRank to navigate multi-hop reasoning chains.
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: Any | None = None,
        synonym_threshold: float = DEFAULT_SYNONYM_THRESHOLD,
        ppr_alpha: float = DEFAULT_PPR_ALPHA,
    ) -> None:
        """Initialize HippoRAG retriever.

        Args:
            embedder: Embedding model for entity matching.
            llm: LLM for triple extraction (optional, can be added later).
            synonym_threshold: Cosine similarity threshold for synonymy edges.
            ppr_alpha: PageRank damping factor.
        """
        self._embedder = embedder
        self._llm = llm
        self._synonym_threshold = synonym_threshold
        self._ppr_alpha = ppr_alpha

        self._graph = nx.DiGraph()
        self._passages: dict[str, str] = {}
        self._node_embeddings: dict[str, list[float]] = {}

    @property
    def graph(self) -> nx.DiGraph:
        """Access the internal Knowledge Graph."""
        return self._graph

    @property
    def passage_count(self) -> int:
        """Number of indexed passages."""
        return len(self._passages)

    async def retrieve(
        self, query: str, top_k: int = 5, **kwargs: object
    ) -> list[DocumentChunk]:
        """Retrieve using Personalized PageRank.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Ranked list of DocumentChunks.
        """
        if not self._passages:
            return []

        # 1. Extract query entities (simple word-based for now)
        query_entities = self._extract_query_entities(query)

        # 2. Match query entities to graph nodes
        query_nodes = self._match_query_to_nodes(query_entities)

        if not query_nodes:
            return self._fallback_dense_search(query, top_k)

        # 3. Run Personalized PageRank
        personalization = {
            n: 1.0 / len(query_nodes) for n in query_nodes if n in self._graph
        }

        if not personalization:
            return self._fallback_dense_search(query, top_k)

        ppr_scores = nx.pagerank(
            self._graph,
            personalization=personalization,
            alpha=self._ppr_alpha,
            max_iter=DEFAULT_PPR_MAX_ITER,
            weight="weight",
        )

        # 4. Aggregate scores at passage level
        passage_scores: dict[str, float] = {}
        for node, score in ppr_scores.items():
            node_data = self._graph.nodes.get(node, {})
            for pid in node_data.get("passages", []):
                passage_scores[pid] = passage_scores.get(pid, 0) + score

        # 5. Rank and return top_k
        ranked = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        chunks = []
        for pid, score in ranked:
            chunk = DocumentChunk(
                text=self._passages[pid],
                metadata=ChunkMetadata(
                    source=f"passage_{pid}",
                    file_type="text",
                    chunk_index=int(pid.lstrip("p")),
                    modality="text",
                ),
                score=score,
            )
            chunks.append(chunk)

        logger.info("hipporag.retrieve.done", results=len(chunks))
        return chunks

    async def index(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks into the Knowledge Graph.

        Uses LLM-based triple extraction when available, falls back
        to pattern matching for speed.

        Args:
            chunks: List of DocumentChunks to index.
        """
        for chunk in chunks:
            pid = f"p{len(self._passages)}"
            self._passages[pid] = chunk.text

            # Try LLM extraction first, fall back to pattern matching
            triples = self._extract_triples(chunk.text)

            for subj, rel, obj in triples:
                for node in [subj, obj]:
                    if node not in self._graph:
                        self._graph.add_node(node, passages=[])
                    if pid not in self._graph.nodes[node]["passages"]:
                        self._graph.nodes[node]["passages"].append(pid)

                self._graph.add_edge(
                    subj, obj, relation=rel, passage_id=pid, weight=1.0
                )

        # Add synonymy edges
        self._add_synonymy_edges()

        logger.info(
            "hipporag.index.done",
            passages=self.passage_count,
            nodes=len(self._graph.nodes),
            edges=len(self._graph.edges),
        )

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Extract triples using LLM if available, else pattern matching."""
        try:
            return self._extract_triples_llm(text)
        except Exception as e:
            logger.debug("hipporag.llm_triple_failed", error=str(e))
            return self._extract_triples_simple(text)

    def _extract_triples_llm(self, text: str) -> list[tuple[str, str, str]]:
        """Extract subject-predicate-object triples using Groq LLM.

        Uses Llama 3.1 8B for fast, structured extraction.
        """
        import json
        import re

        from langchain_core.messages import HumanMessage

        from polymind.infrastructure.llm.llm_factory import LLMFactory

        factory = LLMFactory()
        llm = factory.get_llm(tier="fast")

        # Truncate long texts to fit context window
        truncated = text[:2000] if len(text) > 2000 else text

        prompt = f"""Extract subject-predicate-object triples from this text.
Return a JSON array of triples. Each triple is [subject, predicate, object].
Use concise, normalized entity names (lowercase, no articles).
Extract 3-10 triples. Focus on factual relationships.

Text: {truncated}

Reply with ONLY a JSON array:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Parse JSON response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                raw_triples = json.loads(json_match.group())
                triples = []
                for t in raw_triples:
                    if isinstance(t, list) and len(t) == 3:
                        subj, pred, obj = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                        if subj and pred and obj:
                            triples.append((subj.lower(), pred.lower(), obj.lower()))
                if triples:
                    logger.debug("hipporag.llm_triples", count=len(triples))
                    return triples
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError("Failed to parse LLM triple extraction response")

    def _extract_triples_simple(self, text: str) -> list[tuple[str, str, str]]:
        """Extract simple triples from text using pattern matching.

        This is a fallback when no LLM is available.
        For production, use LLM-based triple extraction.
        """
        triples = []
        sentences = text.split(". ")

        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) < 3:
                continue

            # Simple pattern: first noun phrase - "is/are/was" - rest
            for i, word in enumerate(words):
                if word.lower() in ("is", "are", "was", "were", "has", "have"):
                    subj = " ".join(words[:i]).strip()
                    obj = " ".join(words[i + 1:]).strip().rstrip(".")
                    if subj and obj:
                        triples.append((subj, "related_to", obj))
                    break

        return triples

    def _add_synonymy_edges(self) -> None:
        """Add bidirectional synonymy edges between similar entities.

        Also populates the node embeddings cache for all nodes
        (used by _match_query_to_nodes for semantic matching).
        """
        nodes = list(self._graph.nodes())
        if len(nodes) < 2:
            return

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        node_embs = self._embedder.embed(nodes)
        node_embs_np = np.array(node_embs)
        sims = cosine_similarity(node_embs_np)

        # Cache ALL node embeddings for semantic matching
        for i, node in enumerate(nodes):
            self._node_embeddings[node] = node_embs_np[i].tolist()

        # Add synonymy edges where similarity exceeds threshold
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sims[i][j] >= self._synonym_threshold and nodes[i] != nodes[j]:
                    self._graph.add_edge(
                        nodes[i], nodes[j],
                        relation="synonymy", weight=float(sims[i][j]),
                    )
                    self._graph.add_edge(
                        nodes[j], nodes[i],
                        relation="synonymy", weight=float(sims[i][j]),
                    )

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract key entities from a query."""
        # Simple approach: use significant words as entities
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "who",
            "how", "when", "where", "why", "which", "this", "that", "it",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "but", "not", "can", "do", "does", "did",
        }
        words = query.lower().split()
        entities = [
            w.strip("?.!,;:")
            for w in words
            if w not in stop_words and len(w) > 2
        ]
        return entities[:5]  # Top 5 entities

    def _match_query_to_nodes(
        self, entities: list[str]
    ) -> list[str]:
        """Match query entities to graph nodes using embedding similarity."""
        if not entities or not self._node_embeddings:
            # Fallback: direct text matching
            matched = set()
            graph_nodes = set(self._graph.nodes())
            for entity in entities:
                for node in graph_nodes:
                    if entity.lower() in node.lower():
                        matched.add(node)
            return list(matched)

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        entity_embs = self._embedder.embed(entities)
        nodes = list(self._node_embeddings.keys())
        node_embs = np.array(list(self._node_embeddings.values()))

        sims = cosine_similarity(entity_embs, node_embs)
        matched = set()

        for row in sims:
            top_indices = np.argsort(row)[-3:][::-1]
            for idx in top_indices:
                if row[idx] >= 0.5:
                    matched.add(nodes[idx])

        return list(matched)

    def _fallback_dense_search(
        self, query: str, top_k: int
    ) -> list[DocumentChunk]:
        """Fallback to dense vector search when PPR fails."""
        if not self._passages:
            return []

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_emb = self._embedder.embed_single(query)
        passage_texts = list(self._passages.values())
        passage_embs = self._embedder.embed(passage_texts)

        sims = cosine_similarity([query_emb], passage_embs)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]

        pids = list(self._passages.keys())
        chunks = []
        for idx in top_indices:
            chunk = DocumentChunk(
                text=self._passages[pids[idx]],
                metadata=ChunkMetadata(
                    source=f"passage_{pids[idx]}",
                    file_type="text",
                    chunk_index=idx,
                    modality="text",
                ),
                score=float(sims[idx]),
            )
            chunks.append(chunk)

        return chunks
