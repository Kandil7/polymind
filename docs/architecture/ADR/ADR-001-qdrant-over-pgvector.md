# ADR-001: Qdrant Over pgvector

**Date:** 2026-06-18  
**Status:** Accepted

## Context

PolyMind needs a vector database for storing document embeddings and performing similarity search. The two main candidates are Qdrant (dedicated vector DB) and pgvector (PostgreSQL extension).

## Decision

Use **Qdrant** as the sole vector database.

## Rationale

- **Performance**: Qdrant is purpose-built for vector search with HNSW indexing, delivering sub-10ms query latency at scale
- **Hybrid search**: Native support for sparse + dense vectors (BM25 + embeddings) out of the box
- **Filtering**: Rich payload filtering for modality, source, and metadata-based queries
- **Scalability**: Built-in sharding and replication for horizontal scaling
- **Existing stack**: Team already has Qdrant experience from prior projects (Athar)

## Alternatives Considered

| Option | Pros | Cons | Reason Rejected |
|--------|------|------|-----------------|
| pgvector | Familiar SQL interface, ACID transactions | Slower at scale, limited hybrid search, no native filtering | Performance at scale |
| Pinecone | Managed, zero ops | Vendor lock-in, expensive, no self-hosted option | Cost + control |
| Weaviate | GraphQL API, modules | Heavier resource footprint, less mature | Complexity |

## Consequences

- All vector operations must go through the `IRetriever` interface (no direct Qdrant calls in domain layer)
- Collection schema must be versioned (e.g., `polymind_v1`)
- Backup strategy needed for Qdrant storage volume
