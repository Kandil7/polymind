# Phase 5 — 4-Layer Memory System

**Goal:** Implement Episodic, Semantic, and Procedural memory stores with Planner/Synthesizer integration and automatic consolidation.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `infrastructure/memory/episodic_store.py` | Mem0 conversation history | Infrastructure |
| `infrastructure/memory/semantic_store.py` | Qdrant extracted facts | Infrastructure |
| `infrastructure/memory/procedural_store.py` | JSON successful patterns | Infrastructure |
| `infrastructure/memory/four_layer_memory.py` | Memory orchestrator | Infrastructure |
| `infrastructure/memory/consolidation.py` | ConsolidationPipeline — auto-learning | Infrastructure |

## Memory Architecture

```
Working Memory   → LangGraph state (managed by graph)
Episodic Memory  → Mem0 (conversation history with semantic recall)
Semantic Memory  → Qdrant (extracted facts from repeated patterns)
Procedural Memory → JSON (successful task procedures with usage tracking)
```

## Consolidation Pipeline

The `ConsolidationPipeline` (`infrastructure/memory/consolidation.py`) automates learning from interactions:

1. **Buffer episodes** — Stores query/answer pairs in a buffer
2. **Extract facts** — Uses LLM to extract reusable semantic facts (triggers at 3+ episodes)
3. **Store procedures** — Saves successful task procedures (faithfulness ≥ 0.7)
4. **Deduplicate** — Checks cosine similarity against existing facts (threshold: 0.9)
5. **Prune** — Limits to `MAX_FACTS_PER_CONSOLIDATION` (3) per cycle

**Key configuration:**
- `MIN_EPISODES_FOR_CONSOLIDATION = 3` — Minimum episodes before consolidation
- `MAX_FACTS_PER_CONSOLIDATION = 3` — Max facts extracted per cycle
- `FACT_DEDUPLICATION_THRESHOLD = 0.9` — Cosine similarity for dedup
- `PROCEDURE_SUCCESS_THRESHOLD = 0.7` — Min faithfulness to store procedure

**Usage:**
```python
from polymind.infrastructure.memory.consolidation import ConsolidationPipeline

pipeline = ConsolidationPipeline(user_id="user123")
await pipeline.consolidate(query, answer, critic_scores, modality)
```

## Integration Points

- **Planner**: Recalls episodes + semantic facts before routing
- **Synthesizer**: Stores episodes after answer generation
- **Consolidation**: Extracts semantic facts when patterns repeat 3+ times

## Test Results

```
9 new tests (156 total), all passing
```
