# ADR-003: Clean Architecture

**Date:** 2026-06-18  
**Status:** Accepted

## Context

PolyMind is a complex system with multiple external dependencies (Qdrant, HuggingFace models, n8n, Mem0). Without strict layer boundaries, the codebase will quickly become unmaintainable.

## Decision

Enforce **Clean Architecture** with 4 strict layers:

```
Layer 4 — API / Delivery      (FastAPI routes, schemas, middleware)
Layer 3 — Application         (Agent graph, use cases, orchestration)
Layer 2 — Domain              (Entities, value objects, interfaces)
Layer 1 — Infrastructure      (Qdrant, HuggingFace, n8n, Mem0)
```

## Rationale

- **Testability**: Domain logic can be unit-tested without any I/O
- **Swappability**: Infrastructure implementations can be swapped (e.g., Qdrant → Weaviate) by changing only Layer 1
- **Dependency direction**: Inner layers never import from outer layers
- **Interview signal**: Demonstrates production-grade architecture understanding

## Rules

- Domain layer has ZERO external dependencies
- All infrastructure access via ABC interfaces defined in `domain/interfaces/`
- Dependency Injection everywhere
- No business logic in API routes
- No direct DB calls in agents — always via Repository pattern

## Alternatives Considered

| Option | Pros | Cons | Reason Rejected |
|--------|------|------|-----------------|
| Flat structure | Fast to build, no abstraction overhead | Unmaintainable at scale, hard to test | Scale + testing |
| Hexagonal (Ports & Adapters) | Similar to Clean Arch | More terminology overhead | Clean Arch is simpler |
| Feature-based | Good for UI-heavy apps | Doesn't map well to AI pipelines | Architecture mismatch |

## Consequences

- Every new feature must define its interface in `domain/interfaces/` before implementation
- Import linter (ruff rule TCH) enforces layer boundaries
- Integration tests live outside `src/` and test infrastructure implementations directly
