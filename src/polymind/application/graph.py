"""LangGraph graph factory — builds the full PolyMind agent graph.

Architecture:
    Planner → Router → [Specialist] → RAG → Generator → Critic → Synthesizer
                                          ↑              |
                                          └── retry ←────┘

Phase 4: Full agent graph with all nodes wired.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledGraph

from polymind.application.agents import (
    critic,
    generator,
    planner,
    rag_node,
    router,
    specialist_nodes,
    synthesizer,
)
from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def build_graph() -> CompiledGraph:
    """Build and compile the PolyMind agent graph.

    Returns:
        CompiledGraph ready for invocation via invoke() or ainvoke().
    """
    graph = StateGraph(PolyMindState)

    # ── Add nodes ────────────────────────────────────────
    graph.add_node("planner", planner.run)
    graph.add_node("router", router.run)
    graph.add_node("asr", specialist_nodes.asr_node)
    graph.add_node("vqa", specialist_nodes.vqa_node)
    graph.add_node("docqa", specialist_nodes.docqa_node)
    graph.add_node("tableqa", specialist_nodes.tableqa_node)
    graph.add_node("rag", rag_node.run)
    graph.add_node("generator", generator.run)
    graph.add_node("critic", critic.run)
    graph.add_node("synthesizer", synthesizer.run)

    # ── Wire edges ───────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "router")

    # Router → Specialist (conditional)
    # "multi" modality routes to RAG (text-based retrieval with specialist context)
    graph.add_conditional_edges(
        "router",
        router.decide,
        {
            "asr": "asr",
            "vqa": "vqa",
            "docqa": "docqa",
            "tableqa": "tableqa",
            "rag": "rag",
            "multi": "rag",
        },
    )

    # All specialists → RAG (enrich context)
    for node in ["asr", "vqa", "docqa", "tableqa"]:
        graph.add_edge(node, "rag")

    # RAG → Generator → Critic
    graph.add_edge("rag", "generator")
    graph.add_edge("generator", "critic")

    # Critic → retry loop or synthesize
    graph.add_conditional_edges(
        "critic",
        critic.decide,
        {
            "retry": "rag",           # Re-retrieve with expanded query
            "pass": "synthesizer",    # Answer passed evaluation
            "fail_max": "synthesizer",  # Max retries hit, pass anyway
        },
    )

    # Synthesizer → END
    graph.add_edge("synthesizer", END)

    logger.info("graph.built")
    return graph.compile()
