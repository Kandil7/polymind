# Phase 4 — LangGraph Agent Graph

**Goal:** Wire all components into a full agent graph with retry loop and self-evaluation.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `application/agents/planner.py` | Modality + intent detection | Application |
| `application/agents/router.py` | Conditional routing | Application |
| `application/agents/specialist_nodes.py` | ASR/VQA/DocQA/TableQA nodes | Application |
| `application/agents/rag_node.py` | Context retrieval | Application |
| `application/agents/generator.py` | LLM answer synthesis | Application |
| `application/agents/critic.py` | Self-evaluation (LLM-as-Judge) | Application |
| `application/agents/synthesizer.py` | Final formatting + memory | Application |
| `application/graph.py` | Graph factory (10 nodes) | Application |

## Architecture

```
Planner → Router → [ASR|VQA|DocQA|TableQA] → RAG → Generator → Critic → Synthesizer
                                             ↑              |
                                             └── retry ←────┘
```

## Key Design Decisions

1. **ScoreResult import at runtime** — LangGraph needs it to resolve TypedDict type hints
2. **Heuristic fallback in Critic** — Works without LLM using keyword overlap
3. **Max 2 retries** — Prevents infinite loops in retry logic
4. **Synthesizer stores episodes** — Memory integration at the end of pipeline

## Test Results

```
47 new tests (147 total), all passing
```
