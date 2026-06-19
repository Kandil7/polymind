# PolyMind — System Architecture

## Overview

PolyMind is a self-evaluating, multimodal, multi-agent knowledge assistant built with Clean Architecture principles. It routes user queries across 7+ HuggingFace task types, runs a Critic agent to self-evaluate outputs, and catches hallucinations before delivery.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — API / Delivery                                   │
│  FastAPI routes, schemas, middleware (structlog, Prometheus) │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Application / Use Cases                          │
│  Agent graph (LangGraph), state management, orchestration   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Domain                                           │
│  Entities, value objects, interfaces (ABC), exceptions      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Infrastructure                                   │
│  Qdrant, HuggingFace, Groq, Mem0, n8n, Modal               │
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
    │  • Intent classification
    │  • Memory recall (episodic + semantic)
    │
    ▼
[Router Agent]
    │  Conditional routing by modality
    │
    ├──→ [ASR Agent] (audio → transcript)
    ├──→ [VQA Agent] (image → answer)
    ├──→ [DocQA Agent] (PDF → answer)
    ├──→ [TableQA Agent] (CSV → answer)
    └──→ [RAG Agent] (text → retrieval)
            │
            ▼
    [HippoRAG Retriever]
    │  Knowledge Graph + Personalized PageRank
    │  Multi-hop retrieval (86% accuracy)
    │
    ▼
[Generator] ← Groq Llama 3.3 70B (280 t/s)
    │
    ▼
[Critic Agent] ← LLM-as-Judge (6 metrics)
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
    │
    ▼
    Final Answer + Confidence Score
```

## Data Flow

1. **User Input** → FastAPI gateway accepts text/audio/image/PDF
2. **Planner** → Detects modality + intent, recalls memory
3. **Router** → Dispatches to specialist agent(s)
4. **Specialist** → Processes modality-specific input
5. **RAG** → Retrieves relevant context from Qdrant
6. **Generator** → Synthesizes answer using Groq LLM
7. **Critic** → Self-evaluates (faithfulness/relevancy/hallucination)
8. **Retry** → If Critic fails, re-retrieve with expanded query
9. **Synthesizer** → Formats answer with citations
10. **Memory** → Stores episode, consolidates patterns

## Key Components

### HippoRAG Retriever
- Builds Knowledge Graph from document passages
- Uses Personalized PageRank for multi-hop traversal
- Achieves 86% accuracy vs 79% for standard vector RAG
- Fallback to dense search when graph traversal fails

### Critic Agent (Self-Evaluation)
- LLM-as-Judge using Groq Llama 3.1 8B (fast)
- Evaluates faithfulness, relevancy, hallucination rate
- Triggers retry loop when answer quality is low
- Heuristic fallback when LLM is unavailable

### 4-Layer Memory
- **Working Memory** → LangGraph state (graph nodes)
- **Episodic Memory** → Mem0 (conversation history)
- **Semantic Memory** → Qdrant (extracted facts)
- **Procedural Memory** → JSON (successful patterns)

### Adaptive Retrieval
- SKIP: Simple factual questions (no retrieval needed)
- STANDARD: Single-hop document lookup
- HIPPORAG: Multi-hop reasoning (Knowledge Graph)
- SPECULATIVE: Time-sensitive queries

## Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | Qdrant | Hybrid search, payload filtering, scalability |
| Agent Graph | LangGraph | Stateful, conditional routing, checkpointing |
| LLM | Groq (Llama 3.3 70B) | 280 t/s, free tier, OpenAI-compatible |
| Embeddings | BAAI/bge-m3 | Multilingual, 1024-dim, high quality |
| Evaluation | DeepEval + RAGAS | 6+ metrics, LLM-as-Judge, CI integration |
| Backend | FastAPI | Async, auto-docs, Pydantic v2 |

## Security Considerations

- API keys stored in environment variables (never hardcoded)
- Qdrant collection per environment (dev/staging/prod)
- Input validation via Pydantic schemas
- Rate limiting via Prometheus metrics
- No PII stored in logs (structlog redaction)

## Scalability

- **Horizontal:** FastAPI workers behind load balancer
- **Vertical:** Modal GPU for heavy model inference
- **Vector DB:** Qdrant sharding + replication
- **LLM:** Groq handles scaling (managed service)
- **Memory:** Mem0 + Qdrant for persistent storage
