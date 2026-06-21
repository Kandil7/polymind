# Clean Architecture in PolyMind

## What is Clean Architecture?

Clean Architecture, coined by Robert C. Martin ("Uncle Bob"), is a software design philosophy that centers the codebase around **business rules** (the domain) and pushes all external concerns (databases, UIs, frameworks) to the edges. The core promise: **your business logic never depends on anything that could change**.

The key insight is the **Dependency Rule**: source code dependencies must point only inward, toward higher-level policies. The domain knows nothing about the framework, the database, or the API.

## The Dependency Rule

```
  ┌─────────────────────────────────────────────────┐
  │  Outer Layer (Framework / Driver / Interface)    │
  │    depends on →                                  │
  │  ┌───────────────────────────────────────────┐   │
  │  │  Middle Layer (Use Cases / Application)   │   │
  │  │    depends on →                           │   │
  │  │  ┌─────────────────────────────────────┐  │   │
  │  │  │  Inner Layer (Domain / Entities)    │  │   │
  │  │  └─────────────────────────────────────┘  │   │
  │  └───────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
```

**The rule in practice:**
- Domain entities **never import** from infrastructure, API, or application layers
- Application use cases **never import** from infrastructure or API layers
- Infrastructure and API layers **import from** domain and application layers
- Dependencies flow inward only — the domain is the stable core

## Stable Dependencies Principle

The most stable layer (the one least likely to change) should be the innermost layer. In PolyMind:

| Layer | Stability | Changes When |
|-------|-----------|--------------|
| Domain (Layer 2) | Most stable | Business rules change (rarely) |
| Application (Layer 3) | Stable | Workflow/orchestration changes |
| Infrastructure (Layer 1) | Less stable | External services change |
| API (Layer 4) | Least stable | Client contracts change |

This means the domain is the **safest place to put code** — it's protected from framework churn, library upgrades, and API contract changes.

## PolyMind's 4-Layer Implementation

PolyMind implements Clean Architecture with 4 strict layers:

```
src/polymind/
├── domain/          ← Layer 2: Pure business logic (MOST STABLE)
│   ├── entities/      Pydantic models (Query, Answer, DocumentChunk)
│   ├── interfaces/    ABC contracts (IRetriever, ISpecialist, IMemoryStore, IGenerator, IEvaluator)
│   ├── value_objects/  Enums and immutable types (Modality, RetrievalStrategy, Score)
│   └── exceptions/    Domain-specific errors (DomainError hierarchy)
│
├── application/     ← Layer 3: Use cases and orchestration
│   ├── agents/        LangGraph nodes (planner, router, critic, generator, synthesizer)
│   ├── graph.py       Graph construction and compilation
│   ├── state.py       PolyMindState TypedDict (25+ fields)
│   └── use_cases/     QueryUseCase, IngestUseCase
│
├── infrastructure/  ← Layer 1: External systems
│   ├── qdrant/        Vector DB (QdrantChunkRepository, HippoRAGRetriever, AdaptiveRetriever)
│   ├── specialists/   HF model wrappers (ASR, VQA, DocQA, TableQA)
│   ├── llm/           LLM factory (Groq, OpenAI)
│   ├── memory/        4-layer memory (Episodic, Semantic, Procedural)
│   ├── rag/           Embedding, chunking, reranking, CLIP, multimodal
│   ├── eval/          DeepEval, RAGAS
│   └── *.py           Cross-cutting (circuit_breaker, cache, hyde, moa, feedback, tracing, degradation)
│
└── api/             ← Layer 4: Delivery mechanism
    ├── routes/        FastAPI endpoints (query, health, ingest, eval, feedback)
    ├── schemas/       Pydantic request/response models
    └── middleware/     Auth, rate limiting, logging, metrics
```

## Layer Boundaries: Domain Has Zero External Dependencies

The domain layer is the purest layer — it imports **nothing** from the outside world. Let's verify:

```python
# domain/entities/query.py — imports only Pydantic (a validation library, not an external service)
from pydantic import BaseModel, Field
from polymind.domain.entities.answer import Answer          # ← same layer
from polymind.domain.entities.chunk import DocumentChunk    # ← same layer

class Query(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    user_id: str
    # ... no Qdrant, no Groq, no LangChain, no FastAPI
```

```python
# domain/interfaces/retriever.py — imports only domain entities
from polymind.domain.entities.chunk import DocumentChunk  # ← same layer

class IRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        ...
```

**What the domain does NOT import:**
- No `qdrant_client`
- No `langchain`
- No `groq`
- No `fastapi`
- No `sentence_transformers`
- No `torch`

This is the dependency rule in action. The domain defines **contracts** (interfaces), and infrastructure implements them.

## The ABC Interfaces Pattern

PolyMind defines 5 ABC interfaces in `domain/interfaces/`:

```python
# domain/interfaces/retriever.py
class IRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]: ...
    @abstractmethod
    async def index(self, chunks: list[DocumentChunk]) -> None: ...

# domain/interfaces/specialist.py
class ISpecialist(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> dict[str, Any]: ...

# domain/interfaces/memory_store.py
class IMemoryStore(ABC):
    @abstractmethod
    def store(self, key: str, value: Any, **kwargs) -> None: ...
    @abstractmethod
    def recall(self, query: str, top_k: int = 5) -> list[Any]: ...
    @abstractmethod
    def consolidate(self) -> None: ...

# domain/interfaces/generator.py
class IGenerator(ABC):
    @abstractmethod
    async def generate(self, query: str, context: list[DocumentChunk], **kwargs) -> Answer: ...

# domain/interfaces/evaluator.py
class IEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, query: str, answer: str, contexts: list[str], **kwargs) -> dict[str, ScoreResult]: ...
```

**Why this matters:**
1. **Testability** — Tests can mock `IRetriever` without needing a running Qdrant instance
2. **Swappability** — Swap `QdrantChunkRepository` for `PineconeRepository` without changing application code
3. **Dependency inversion** — Application depends on abstractions, not implementations

## How Each Layer Maps to the Codebase

### Layer 2: Domain (Pure Business Logic)

| File | Purpose |
|------|---------|
| `domain/entities/query.py` | `Query`, `QueryResult`, `ScoreResult` — immutable Pydantic models |
| `domain/entities/answer.py` | `Answer` — text + confidence + sources |
| `domain/entities/chunk.py` | `DocumentChunk`, `ChunkMetadata` — vector DB records |
| `domain/entities/episode.py` | `ConversationEpisode` — memory input |
| `domain/interfaces/*.py` | 5 ABC contracts for all external systems |
| `domain/value_objects/modality.py` | `Modality` enum (TEXT, AUDIO, IMAGE, DOCUMENT, TABLE, MULTI) |
| `domain/value_objects/retrieval_strategy.py` | `RetrievalStrategy` enum (SKIP, STANDARD, HIPPORAG, SPECULATIVE) |
| `domain/value_objects/score.py` | `Score` with threshold-based `passed` property |
| `domain/exceptions/*.py` | `DomainError` hierarchy (RetrievalError, CriticFailedError, etc.) |

### Layer 3: Application (Orchestration)

| File | Purpose |
|------|---------|
| `application/state.py` | `PolyMindState` TypedDict — shared state across all nodes |
| `application/graph.py` | `build_graph()` — compiles the LangGraph StateGraph |
| `application/agents/planner.py` | Modality detection, intent classification, memory recall |
| `application/agents/router.py` | Strategy classification, conditional routing |
| `application/agents/rag_node.py` | Multi-strategy retrieval (HyDE, reranking, circuit breakers) |
| `application/agents/generator.py` | LLM generation with MoA support |
| `application/agents/critic.py` | LLM-as-Judge evaluation, retry decisions |
| `application/agents/synthesizer.py` | Answer formatting, episode storage |
| `application/agents/specialist_nodes.py` | ASR, VQA, DocQA, TableQA dispatch |
| `application/use_cases/query_use_case.py` | `QueryUseCase.execute()` — orchestrates the full pipeline |

### Layer 1: Infrastructure (External Systems)

| File | Purpose |
|------|---------|
| `infrastructure/qdrant/chunk_repository.py` | `QdrantChunkRepository(IRetriever)` — dense vector search |
| `infrastructure/qdrant/hipporag_retriever.py` | `HippoRAGRetriever(IRetriever)` — Knowledge Graph + PPR |
| `infrastructure/qdrant/adaptive_retriever.py` | Strategy routing + fallback logic |
| `infrastructure/specialists/*.py` | HF model wrappers (Whisper, BLIP, LayoutLM, TAPAS) |
| `infrastructure/llm/llm_factory.py` | `LLMFactory` — Groq (4 tiers) + OpenAI |
| `infrastructure/memory/*.py` | Episodic (Mem0), Semantic (Qdrant), Procedural (JSON) stores |
| `infrastructure/rag/embedder.py` | `Embedder` — BAAI/bge-m3 via sentence-transformers |
| `infrastructure/rag/reranker.py` | `CrossEncoderReranker` — BAAI/bge-reranker-v2-m3 |
| `infrastructure/rag/clip_embedder.py` | `CLIPEmbedder` — OpenAI CLIP for cross-modal search |
| `infrastructure/rag/multimodal_retriever.py` | `MultiModalRetriever(IRetriever)` — CLIP-based |
| `infrastructure/circuit_breaker.py` | `CircuitBreaker` — CLOSED/OPEN/HALF_OPEN state machine |
| `infrastructure/cache.py` | `TTLCache` — LRU in-memory cache with TTL |
| `infrastructure/degradation.py` | `DegradationManager` — fallback decisions per service |
| `infrastructure/hyde.py` | `expand_query_hyde()` — hypothetical answer generation |
| `infrastructure/moa.py` | `generate_with_moa()` — multi-agent generation + merger |
| `infrastructure/feedback.py` | `FeedbackStore` — user satisfaction tracking |
| `infrastructure/tracing.py` | OpenTelemetry setup + span helpers |

### Layer 4: API (Delivery)

| File | Purpose |
|------|---------|
| `api/main.py` | `create_app()` — FastAPI factory with lifespan |
| `api/routes/query.py` | `POST /query/`, `POST /query/stream` (SSE) |
| `api/routes/health.py` | `GET /health` — Qdrant, LLM, circuit breaker status |
| `api/routes/feedback.py` | `POST /feedback`, `GET /feedback/stats` |
| `api/schemas/*.py` | Pydantic request/response models |
| `api/middleware/auth.py` | `APIKeyAuthMiddleware` — Bearer token validation |
| `api/middleware/rate_limit.py` | `RateLimitMiddleware` — per-IP sliding window |
| `api/middleware/logging.py` | `RequestLoggingMiddleware` — structlog |
| `api/middleware/metrics.py` | `PrometheusMiddleware` — Counter, Histogram, Gauge |

## Common Mistakes and How PolyMind Avoids Them

### Mistake 1: Domain imports infrastructure

**Wrong:**
```python
# In domain layer — BAD: imports Qdrant
from qdrant_client import QdrantClient

class ChunkRepository:
    def __init__(self):
        self.client = QdrantClient()  # ← domain depends on infrastructure!
```

**PolyMind's approach:**
```python
# domain/interfaces/retriever.py — GOOD: defines contract only
class IRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]: ...

# infrastructure/qdrant/chunk_repository.py — GOOD: implements contract
class QdrantChunkRepository(IRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        # Qdrant-specific implementation here
        ...
```

### Mistake 2: Business logic in API routes

**Wrong:**
```python
# In API route — BAD: business logic in delivery layer
@app.post("/query/")
async def query(request: QueryRequest):
    chunks = qdrant.search(request.text)      # ← DB call in route
    answer = llm.generate(request.text, chunks) # ← business logic in route
    return {"answer": answer}
```

**PolyMind's approach:**
```python
# api/routes/query.py — GOOD: delegates to use case
@app.post("/query/")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    use_case = QueryUseCase(...)  # ← orchestrates the pipeline
    result = await use_case.execute(query_obj)
    return result
```

### Mistake 3: God objects that do everything

**Wrong:**
```python
class Agent:
    def route(self): ...
    def retrieve(self): ...
    def generate(self): ...
    def evaluate(self): ...
    def format(self): ...
    # ← one class does everything
```

**PolyMind's approach:** Each node is a focused module with a single responsibility:
- `planner.py` — detection + recall only
- `router.py` — strategy classification + routing only
- `rag_node.py` — retrieval only
- `generator.py` — generation only
- `critic.py` — evaluation only
- `synthesizer.py` — formatting + storage only

### Mistake 4: No fallback paths

**Wrong:**
```python
async def retrieve(query: str):
    return await qdrant.search(query)  # ← what if Qdrant is down?
```

**PolyMind's approach:** Circuit breakers + degradation manager:
```python
# infrastructure/degradation.py
degradation = DegradationManager()

async def retrieve(query: str):
    if degradation.should_skip_retrieval():
        return []  # ← graceful: skip retrieval, use LLM knowledge
    try:
        return await qdrant.search(query)
    except Exception:
        degradation.record_service_failure("qdrant")
        return []  # ← graceful fallback
```

### Mistake 5: Testing with real external services

**Wrong:**
```python
def test_retrieval():
    qdrant = QdrantClient()  # ← requires running Qdrant
    repo = QdrantChunkRepository(qdrant)
    results = repo.retrieve("test query")  # ← integration test, not unit
```

**PolyMind's approach:** Mock the interface:
```python
def test_retrieval():
    mock_retriever = AsyncMock(spec=IRetriever)
    mock_retriever.retrieve.return_value = [DocumentChunk(...)]

    # Test against the contract, not the implementation
    results = await mock_retriever.retrieve("test query")
    assert len(results) == 1
```

## Summary

Clean Architecture gives PolyMind:

1. **Testability** — 390+ tests, most running against mocked interfaces
2. **Swappability** — Swap Qdrant for Pinecone, Groq for OpenAI, without touching business logic
3. **Maintainability** — Each layer changes independently (domain rarely changes, API often does)
4. **Clarity** — New developers know exactly where to find code: domain logic? → `domain/`. Infrastructure? → `infrastructure/`.
5. **Resilience** — Circuit breakers + degradation at the infrastructure boundary protect the domain

The domain is the castle. Everything else is the moat.
