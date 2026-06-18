"""Tests for HippoRAG Retriever."""

from __future__ import annotations

from polymind.infrastructure.qdrant.hipporag_retriever import HippoRAGRetriever


class TestHippoRAGStructure:
    def test_implements_retriever_interface(self) -> None:
        from polymind.domain.interfaces.retriever import IRetriever
        assert issubclass(HippoRAGRetriever, IRetriever)

    def test_graph_starts_empty(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)
        assert retriever.passage_count == 0
        assert len(retriever.graph.nodes) == 0


class TestHippoRAGTripleExtraction:
    def test_simple_triple_extraction(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        triples = retriever._extract_triples_simple(
            "PolyMind is a knowledge assistant. It uses RAG."
        )
        assert len(triples) > 0
        assert all(len(t) == 3 for t in triples)

    def test_empty_text_returns_no_triples(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        triples = retriever._extract_triples_simple("")
        assert triples == []


class TestHippoRAGQueryEntityExtraction:
    def test_extracts_entities(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        entities = retriever._extract_query_entities(
            "What is the connection between Qdrant and RAG?"
        )
        assert "qdrant" in entities
        assert "rag" in entities

    def test_filters_stop_words(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        entities = retriever._extract_query_entities(
            "what is the difference between a and b"
        )
        assert "what" not in entities
        assert "the" not in entities


class TestHippoRAGFallback:
    def test_fallback_returns_empty_when_no_passages(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        chunks = retriever._fallback_dense_search("test query", 5)
        assert chunks == []

    def test_fallback_returns_empty_when_no_nodes(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        # No passages → returns empty
        chunks = retriever._fallback_dense_search("test", 5)
        assert chunks == []


class TestHippoRAGGraphBuilding:
    def test_graph_has_nodes_after_simple_index(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        retriever = HippoRAGRetriever(embedder=embedder)

        # Manually add passages and triples
        retriever._passages = {"p0": "PolyMind is a knowledge assistant."}
        triples = retriever._extract_triples_simple(
            "PolyMind is a knowledge assistant."
        )
        for subj, rel, obj in triples:
            retriever._graph.add_edge(subj, obj, relation=rel, passage_id="p0")
            retriever._graph.add_node(subj, passages=["p0"])
            retriever._graph.add_node(obj, passages=["p0"])

        assert len(retriever._graph.nodes) > 0
        assert len(retriever._graph.edges) > 0


# Need to import for type checking
from polymind.infrastructure.rag.embedder import Embedder  # noqa: E402
