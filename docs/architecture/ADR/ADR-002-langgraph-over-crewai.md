# ADR-002: LangGraph Over CrewAI

**Date:** 2026-06-18  
**Status:** Accepted

## Context

PolyMind requires a multi-agent orchestration framework to coordinate 5+ specialist agents with conditional routing, state management, and retry loops.

## Decision

Use **LangGraph** as the sole agent orchestration framework.

## Rationale

- **Stateful graphs**: Native TypedDict state management across agent nodes
- **Conditional routing**: Built-in support for dynamic edge selection (Critic retry loop)
- **Checkpointer**: MemorySaver enables conversation persistence out of the box
- **LangChain integration**: Seamless connection to LangChain tools, chains, and LLMs
- **Community**: Largest multi-agent community in 2026, extensive documentation

## Alternatives Considered

| Option | Pros | Cons | Reason Rejected |
|--------|------|------|-----------------|
| CrewAI | Simple API, role-based agents | Limited state management, no conditional routing, less mature | State + routing needs |
| AutoGen | Multi-agent conversations | Microsoft-centric, heavier abstraction, less flexible | Vendor lock-in |
| LangGraph + custom | Maximum flexibility | More boilerplate, harder to maintain | Use LangGraph directly |

## Consequences

- All agent logic lives as LangGraph nodes in `application/agents/`
- State is defined as a single `PolyMindState` TypedDict
- Conditional edges handle Critic retry logic and modality routing
- Graph is compiled once at startup, reused across requests
