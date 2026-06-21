# PolyMind — System Architecture

## Overview

PolyMind is a self-evaluating, multimodal, multi-agent knowledge assistant built with Clean Architecture principles. It routes user queries across 7+ HuggingFace task types, runs a Critic agent to self-evaluate outputs, and catches hallucinations before delivery.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — API / Delivery                                   │
│  FastAPI routes, schemas, middleware (auth, rate-limit,     │
│  structlog, Prometheus, OpenTelemetry)                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Application / Use Cases                          │
│  Agent graph (LangGraph), state management, orchestration   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Domain                                           │
│  Entities, value objects, interfaces (ABC), exceptions      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Infrastructure                                   │
│  Qdrant, HF Specialists, Groq, Mem0, Modal                 │
│  + Cross-cutting: circuit breaker, cache, degradation,      │
│    HyDE, MoA, reranker, CLIP, feedback, tracing            │
└─────────────────────────────────────────────────────────────┘
```

**Rules:**
- Inner layers know NOTHING about outer layers
- Domain layer has ZERO external dependencies
- All infrastructure access via Interfaces (ABC)
- Use Dependency Injection everywhere
- No business logic inside API routes
- No direct DB calls inside agents — always via Repository pattern

## Agent Graph Architecture

```
User Input (text / audio / image / PDF / CSV)
    │
    ▼
[Planner Agent] ← 4-Layer Memory
    │  • Modality detection
    │  • Intent classification (LLM + keyword fallback)
    │  • Memory recall (episodic + semantic)
    │  • Circuit breaker check (graceful degradation)
    │
    ▼
[Router Agent]
    │  Retrieval strategy classification (LLM + keyword fallback)
    │  Conditional routing by modality + strategy
    │
    ├──→ [ASR Agent] (audio → transcript)
    ├──→ [VQA Agent] (image → answer via BLIP + CLIP)
    ├──→ [DocQA Agent] (PDF → answer via LayoutLM)
    ├──→ [TableQA Agent] (CSV → answer via TAPAS)
    └──→ [RAG Agent] (text → retrieval)
            │
            ▼
    ┌─ [HyDE Query Expansion] (complex queries only)
    │
    ├──→ [HippoRAG Retriever] (Knowledge Graph + PPR)
    ├──→ [Standard Retriever] (dense vector search)
    └──→ [MultiModal Retriever] (CLIP cross-modal search)
            │
            ▼
    [Cross-Encoder Reranker] (BAAI/bge-reranker-v2-m3)
    │  Bi-encoder retrieves 20 → cross-encoder reranks → top 5
    │
    ▼
[Generator] ← Groq Llama 3.3 70B (280 t/s)
    │  └── [Mixture-of-Agents] (3 temp-varied agents + merger)
    │
    ▼
[Critic Agent] ← LLM-as-Judge (DeepEval, 3 metrics)
    │  Faithfulness ≥ 0.72
    │  Answer Relevancy ≥ 0.75
    │  Hallucination Rate ≤ 0.25
    │
    ├──→ [retry] if rejected (max 2 retries)
    └──→ [pass] if accepted
            │
            ▼
    [Synthesizer]
    │  Format answer + citations + confidence
    │  Store episode → Memory consolidation
    │  Record feedback entry
    │
    ▼
    Final Answer + Confidence Score
```

## Data Flow

1. **User Input** → FastAPI gateway accepts text/audio/image/PDF/CSV
2. **Auth** → API key validation (Bearer token, dev mode bypass)
3. **Rate Limit** → Per-IP sliding window (30/min default, 10/min query)
4. **Planner** → Detects modality + intent, recalls memory (circuit breaker gated)
5. **Router** → Dispatches to specialist agent(s) via conditional edges
6. **Specialist** → Processes modality-specific input
7. **RAG** → Retrieves relevant context (HyDE → multi-strategy retrieval → reranking)
8. **Generator** → Synthesizes answer (MoA or single LLM, circuit breaker gated)
9. **Critic** → Self-evaluates (faithfulness/relevancy/hallucination)
10. **Retry** → If Critic fails, re-retrieve with expanded query (max 2 retries)
11. **Synthesizer** → Formats answer with citations, stores episode
12. **Feedback** → Records answer quality signal for future improvement
13. **Tracing** → OpenTelemetry spans for all operations

## Key Components

### HippoRAG Retriever
- Builds Knowledge Graph from document passages (triple extraction)
- Uses Personalized PageRank for multi-hop traversal
- Achieves 86% accuracy vs 79% for standard vector RAG
- Fallback to dense search when graph traversal fails

### Critic Agent (Self-Evaluation)
- LLM-as-Judge using DeepEval + Groq Llama 3.1 8B (fast)
- Evaluates faithfulness, relevancy, hallucination rate
- Triggers retry loop when answer quality is low
- Heuristic fallback when LLM is unavailable

### 4-Layer Memory
- **Working Memory** → LangGraph state (`PolyMindState` TypedDict, 25+ fields)
- **Episodic Memory** → Mem0-backed conversation history (`EpisodicStore`)
- **Semantic Memory** → Qdrant-backed extracted facts (`SemanticStore`)
- **Procedural Memory** → JSON-backed successful patterns (`ProceduralStore`)
- **Consolidation** → LLM-based fact extraction, deduplication, procedure extraction

### Adaptive Retrieval
- **SKIP**: Simple factual questions (no retrieval needed)
- **STANDARD**: Single-hop dense vector search
- **HIPPORAG**: Multi-hop reasoning (Knowledge Graph + Personalized PageRank)
- **SPECULATIVE**: Time-sensitive queries

### Circuit Breaker Pattern
Prevents cascading failures across external services:

| Service | Failure Threshold | Recovery Timeout | Behavior When Open |
|---------|-------------------|------------------|---------------------|
| Qdrant | 3 failures | 30 seconds | Skip retrieval, use LLM knowledge |
| LLM | 5 failures | 60 seconds | Heuristic classification + extractive generation |
| Embedder | 3 failures | 30 seconds | Keyword search fallback |
| Memory | 3 failures | 30 seconds | Skip memory recall, operate without context |

```python
# Usage in infrastructure/circuit_breaker.py
breaker = CircuitBreaker("qdrant", failure_threshold=3, recovery_timeout=30)

if breaker.allow_request():
    try:
        result = call_service()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        # Degrade gracefully
```

States: **CLOSED** (normal) → **OPEN** (failing) → **HALF_OPEN** (testing recovery)

### Graceful Degradation
The `DegradationManager` checks circuit breakers and provides fallback decisions:

```python
# From infrastructure/degradation.py
degradation = DegradationManager()

if degradation.should_skip_retrieval():
    # Both Qdrant AND embedder are down — skip retrieval entirely
    pass

if degradation.should_use_heuristic_classification():
    # LLM unavailable — use keyword-based intent classification
    pass

if degradation.should_skip_memory():
    # Memory service down — operate without context
    pass
```

### HyDE Query Expansion
Generates hypothetical answers to bridge the "vocabulary gap" between queries and documents:

```python
# From infrastructure/hyde.py
from polymind.infrastructure.hyde import expand_query_hyde, expand_query_multi

# Single expansion for complex queries
expanded = expand_query_hyde("What is RAG?")
# Returns: "RAG is a technique that combines retrieval and generation..."

# Multiple variants for diversified retrieval
variants = expand_query_multi("How does HippoRAG work?", num_variants=3)
# Returns: [original, variant_1, variant_2, variant_3]
```

Based on [HyDE (arxiv.org/abs/2212.10496)](https://arxiv.org/abs/2212.10496).

### Mixture-of-Agents (MoA)
Generates multiple answers with different temperatures, then merges the best parts:

```python
# From infrastructure/moa.py
from polymind.infrastructure.moa import generate_with_moa

answer = await generate_with_moa(query, context, num_agents=3)
```

| Agent | Temperature | Style |
|-------|-------------|-------|
| Precise | 0.0 | Factual, concise, cite sources |
| Comprehensive | 0.3 | Thorough, detailed, all aspects |
| Creative | 0.7 | Engaging, analogies, examples |

A merger agent (reasoning tier) combines the best parts into a final answer.

Based on [MoA (arxiv.org/abs/2401.04088)](https://arxiv.org/abs/2401.04088).

### Cross-Encoder Reranking
Two-stage retrieval for higher precision:

```python
# From infrastructure/rag/reranker.py
from polymind.infrastructure.rag.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(model_id="BAAI/bge-reranker-v2-m3", top_k=5)
reranked = reranker.rerank(query, candidates)  # candidates: list of DocumentChunk
# Returns top 5 most relevant chunks
```

Pipeline: bi-encoder retrieves 20 candidates → cross-encoder scores each (query, doc) pair → top 5 returned.

### CLIP Multimodal Retrieval
Unified embedding space for text and images via OpenAI CLIP:

```python
# From infrastructure/rag/clip_embedder.py
from polymind.infrastructure.rag.clip_embedder import CLIPEmbedder

clip = CLIPEmbedder()
text_vec = clip.embed_text("a photo of a cat")     # 512-dim
image_vec = clip.embed_image("photo.jpg")           # 512-dim
similarity = clip.compute_similarity(text_vec, image_vec)

# From infrastructure/rag/multimodal_retriever.py
from polymind.infrastructure.rag.multimodal_retriever import MultiModalRetriever

retriever = MultiModalRetriever()
results = await retriever.search("a cat on a table", modality="image")
```

Supports: text→image, image→text, text→text (multilingual) search.

### Feedback Loop System
Captures user satisfaction signals for continuous improvement:

```python
# From infrastructure/feedback.py
from polymind.infrastructure.feedback import FeedbackStore

store = FeedbackStore()
store.record(query_id="abc", query="What is RAG?", rating=5, feedback="thumbs_up")
stats = store.get_stats()
# {'total': 142, 'average_rating': 4.2, 'satisfaction_rate': 0.87, ...}

by_intent = store.get_satisfaction_by_intent()
by_strategy = store.get_satisfaction_by_strategy()
```

JSON-backed persistence with 10K entry cap. Analytics by intent and retrieval strategy.

### API Authentication & Rate Limiting

**API Key Auth** (`api/middleware/auth.py`):
```python
# Bearer token validation
# Authorization: Bearer <api_key>
# Public paths bypass: /health, /docs, /redoc, /openapi.json, /metrics, /feedback
# Dev mode: no POLYMIND_API_KEY set → all requests allowed
```

**Rate Limiting** (`api/middleware/rate_limit.py`):
```python
# Per-IP sliding window
DEFAULT_RATE_LIMIT = 30    # requests per minute (default)
QUERY_RATE_LIMIT = 10      # requests per minute (/query)
INGEST_RATE_LIMIT = 5      # requests per minute (/ingest)
```

### OpenTelemetry Tracing
Distributed tracing with OTLP and Console exporters:

```python
# From infrastructure/tracing.py
from polymind.infrastructure.tracing import trace_span, trace_graph_node, trace_llm_call

# Generic span
with trace_span("my_operation", {"key": "value"}):
    pass

# Graph node span
with trace_graph_node("planner"):
    pass  # auto-named: graph.node.planner

# LLM call span
with trace_llm_call(model="llama-3.3-70b", tier="reasoning"):
    pass  # auto-named: llm.call
```

Configure via env vars: `OTEL_EXPORTER_TYPE=otlp|console`, `OTEL_EXPORTER_ENDPOINT=localhost:4317`.

### Caching Layer
In-memory TTL cache with LRU eviction (swappable for Redis):

```python
# From infrastructure/cache.py
from polymind.infrastructure.cache import (
    cache, cached_embedding, store_embedding,
    cached_response, store_response,
    cached_classification, store_classification,
)

# Direct cache usage
cache.set("key", value, ttl=3600)
value = cache.get("key")

# Specialized helpers
store_embedding("What is RAG?", embedding_vector, ttl=86400)
cached_embedding("What is RAG?")  # Returns list[float] or None

store_response("query", response_dict, ttl=1800)
cached_response("query")  # Returns dict or None
```

Global instance: `TTLCache(max_size=10000, default_ttl=3600)`.

## Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | Qdrant | Hybrid search, payload filtering, scalability |
| Agent Graph | LangGraph | Stateful, conditional routing, checkpointing |
| LLM | Groq (Llama 3.3 70B) | 280 t/s, free tier, OpenAI-compatible |
| Embeddings | BAAI/bge-m3 | Multilingual, 1024-dim, high quality |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder, high precision |
| Multimodal | OpenAI CLIP (ViT-B/32) | Text+image shared embedding space |
| Evaluation | DeepEval + RAGAS | 6+ metrics, LLM-as-Judge, CI integration |
| Backend | FastAPI | Async, auto-docs, Pydantic v2 |
| Tracing | OpenTelemetry | OTLP export, Jaeger/Tempo integration |
| Metrics | Prometheus + Grafana | Counter, Histogram, Gauge |

## Security Considerations

- API keys stored in environment variables (never hardcoded)
- Qdrant collection per environment (dev/staging/prod)
- Input validation via Pydantic schemas (Layer 4)
- Rate limiting via per-IP sliding window (Layer 4)
- Auth middleware with public path bypass (Layer 4)
- No PII stored in logs (structlog redaction)
- Prompt injection prevention (input sanitization — planned)
- PII redaction in responses (output filtering — planned)

## Scalability

- **Horizontal:** FastAPI workers behind load balancer
- **Vertical:** Modal GPU for heavy model inference
- **Vector DB:** Qdrant sharding + replication
- **LLM:** Groq handles scaling (managed service)
- **Memory:** Mem0 + Qdrant for persistent storage
- **Caching:** In-memory now, Redis-backed in production (planned)
- **Rate Limiting:** In-memory now, Redis-backed for multi-worker (planned)
