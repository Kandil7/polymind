# LangGraph Deep Dive

## What is LangGraph?

LangGraph is a framework for building **stateful, multi-agent systems** using graph-based workflows. It's built on top of LangChain and provides:

- **State management** via TypedDict
- **Conditional routing** between nodes
- **Checkpointing** for conversation persistence
- **Human-in-the-loop** support

## How PolyMind Uses It

### State Definition

```python
class PolyMindState(TypedDict, total=False):
    user_query: str
    modality: Literal["text", "audio", "image", "document", "table"]
    intent: str
    retrieved_chunks: list[dict]
    final_answer: str | None
    critic_scores: dict[str, ScoreResult]
    retry_count: int
    should_retry: bool
```

### Graph Construction

```python
graph = StateGraph(PolyMindState)

# Add nodes
graph.add_node("planner", planner.run)
graph.add_node("router", router.run)
graph.add_node("critic", critic.run)

# Wire edges
graph.set_entry_point("planner")
graph.add_edge("planner", "router")

# Conditional routing
graph.add_conditional_edges(
    "router",
    router.decide,  # Returns node name
    {"asr": "asr", "vqa": "vqa", "rag": "rag"}
)

# Compile
compiled = graph.compile()
```

### Retry Loop

```python
graph.add_conditional_edges(
    "critic",
    critic.decide,
    {
        "retry": "rag",           # Re-retrieve
        "pass": "synthesizer",    # Accept
        "fail_max": "synthesizer" # Max retries
    }
)
```

## Key Concepts

1. **Nodes** — Functions that read/write state
2. **Edges** — Connections between nodes
3. **Conditional edges** — Routing based on state
4. **State** — TypedDict shared across all nodes
5. **Compilation** — Optimizes graph for execution

## Common Patterns

### Fan-out (Parallel)
```python
for node in ["asr", "vqa", "docqa"]:
    graph.add_edge(node, "rag")
```

### Fan-in (Merge)
```python
graph.add_conditional_edges("router", route_fn, targets)
```

### Loop
```python
graph.add_conditional_edges("critic", should_retry, {"retry": "rag", ...})
```

## Best Practices

1. **Keep state minimal** — Only essential data in TypedDict
2. **Use type hints** — LangGraph validates state types
3. **Handle errors gracefully** — Nodes should never crash
4. **Log everything** — Use structlog in each node
5. **Test nodes independently** — Unit test each node function
