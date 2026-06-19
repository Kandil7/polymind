# ADR-004: HippoRAG v2 for Multi-Hop Retrieval

**Date:** 2026-06-19
**Status:** Accepted

## Context

PolyMind needs to answer complex multi-hop questions like "Who founded the company that created the technology used in this contract?" Standard vector RAG achieves only 79% accuracy on multi-hop QA benchmarks.

## Decision

Implement **HippoRAG v2** with Knowledge Graph and Personalized PageRank for multi-hop retrieval.

## Rationale

- **86% accuracy** on multi-hop QA vs 79% for standard vector RAG
- **10-30x faster** than iterative retrieval methods (IRCoT)
- **Neuroscience-inspired** — associative memory indexing from hippocampal theory
- **Graceful fallback** — dense search when graph traversal fails

## Alternatives Considered

| Option | Pros | Cons | Reason Rejected |
|--------|------|------|-----------------|
| Standard Vector RAG | Simple, fast | 79% on multi-hop | Insufficient accuracy |
| IRCoT | High accuracy | 10-30x slower | Performance too slow |
| GraphRAG (Microsoft) | Strong | Heavy infrastructure | Overkill for portfolio |
| HippoRAG v2 | 86% accuracy, fast | Requires triple extraction | Best balance |

## Consequences

- Requires LLM for triple extraction (can be done offline)
- NetworkX graph must be seeded before queries
- Fallback to dense search needed for edge cases
