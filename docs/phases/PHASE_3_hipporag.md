# Phase 3 — HippoRAG Retriever

**Goal:** Implement Knowledge Graph-based retrieval with Personalized PageRank for multi-hop reasoning.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `infrastructure/qdrant/client_factory.py` | Qdrant client singleton | Infrastructure |
| `infrastructure/qdrant/chunk_repository.py` | IRetriever implementation | Infrastructure |
| `infrastructure/qdrant/hipporag_retriever.py` | Knowledge Graph + PPR | Infrastructure |
| `infrastructure/qdrant/adaptive_retriever.py` | Strategy classifier | Infrastructure |
| `infrastructure/rag/embedder.py` | BAAI/bge-m3 wrapper | Infrastructure |
| `infrastructure/rag/chunker.py` | Recursive/Semantic/Table chunking | Infrastructure |
| `infrastructure/rag/ingestion.py` | Document ingestion pipeline | Infrastructure |
| `scripts/seed_qdrant.py` | Test data seeder | Script |

## Key Design Decisions

1. **NetworkX for Knowledge Graph** — Simple, well-tested, sufficient for portfolio scale
2. **Lazy loading for embeddings** — Avoids GPU OOM at import time
3. **Deferred numpy/sklearn imports** — Prevents Windows crash on module load
4. **RecursiveChunker with infinite loop fix** — `prev_end` tracking prevents stuck loops
5. **Fallback to dense search** — When PPR fails (no graph nodes), falls back to vector similarity

## Test Results

```
28 new tests (89 total), all passing
```
