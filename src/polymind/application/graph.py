"""LangGraph graph factory — builds the PolyMind agent graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from polymind.application.state import PolyMindState


def build_graph() -> StateGraph:
    """Build and compile the PolyMind agent graph.

    Phase 1: Stub graph — planner → END.
    Will be expanded in Phase 4 with all agent nodes.
    """
    graph = StateGraph(PolyMindState)

    graph.add_node("planner", _planner_stub)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)

    return graph.compile()


def _planner_stub(state: PolyMindState) -> PolyMindState:
    """Placeholder planner — will be replaced in Phase 4."""
    return {
        **state,
        "modality": "text",
        "intent": "general",
        "final_answer": "Stub: agent graph not yet implemented.",
        "passed_critic": True,
        "retry_count": 0,
    }
