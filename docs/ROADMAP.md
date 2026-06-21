# PolyMind Roadmap

## Phase 1 — Foundation & Infrastructure ✅
- [x] pyproject.toml + Poetry
- [x] Domain entities + interfaces + exceptions
- [x] PolyMindState TypedDict
- [x] Docker Compose (Qdrant, Prometheus, Grafana)
- [x] FastAPI health endpoint
- [x] ADRs documented
- [x] Unit tests passing

## Phase 2 — Specialist Model Wrappers ✅
- [x] ASR Wrapper (Whisper — local + Groq fallback)
- [x] VQA Wrapper (BLIP)
- [x] DocQA Wrapper (LayoutLM)
- [x] TableQA Wrapper (TAPAS)
- [x] Summarizer Wrapper
- [x] Each wrapper with 3+ passing tests

## Phase 3 — HippoRAG Retriever ✅
- [x] Qdrant client factory (singleton via `@lru_cache`)
- [x] HippoRAG retriever (Knowledge Graph + Personalized PageRank)
- [x] Adaptive retriever (skip/standard/multi-hop/speculative)
- [x] Document ingestion pipeline
- [x] Multi-hop test passing (86% accuracy)

## Phase 4 — LangGraph Agent Graph ✅
- [x] Planner node with memory recall
- [x] Router node with conditional edges
- [x] Specialist nodes (ASR, VQA, DocQA, TableQA)
- [x] RAG node (multi-strategy retrieval)
- [x] Generator node (Mixture-of-Agents)
- [x] Critic node (DeepEval LLM-as-Judge)
- [x] Synthesizer node (answer formatting + episode storage)
- [x] Retry loop (max 2 retries, critic-driven)
- [x] End-to-end test passing (10 LangGraph nodes)

## Phase 5 — 4-Layer Memory ✅
- [x] Episodic store (Mem0-backed conversation history)
- [x] Semantic store (Qdrant-backed extracted facts)
- [x] Procedural store (JSON-backed successful patterns)
- [x] FourLayerMemory class (orchestrates all stores)
- [x] Memory integration in Planner + Synthesizer
- [x] Consolidation pipeline (LLM-based fact extraction + dedup)

## Phase 6 — Eval Harness & CI ✅
- [x] DeepEval critic implementation (LLM-as-Judge)
- [x] RAGAS runner (retrieval quality metrics)
- [x] 40-case benchmark dataset (20 text + 5 audio + 5 image + 5 document + 5 table)
- [x] Pytest eval suite (390+ passing tests)
- [x] GitHub Actions eval gate (`eval_gate.yml`)
- [x] PR fails when faithfulness < 0.72

## Phase 7 — API & Observability ✅
- [x] `/query` endpoint (multimodal, non-streaming + SSE streaming)
- [x] `/ingest` endpoint
- [x] `/eval` endpoint
- [x] `/feedback` endpoint (thumbs up/down, rating, stats)
- [x] `/health` endpoint (Qdrant, LLM API key, circuit breaker status)
- [x] structlog middleware (request/response logging with timing)
- [x] Prometheus metrics (Counter, Histogram, Gauge)
- [x] OpenTelemetry tracing (OTLP + Console exporters)
- [x] n8n webhook triggers (planned — not yet implemented)

## Phase 8 — Demo & Deploy ✅
- [x] Streamlit demo app (`app.py`)
- [x] Modal GPU deployment (`infra/modal_deploy.py`)
- [x] Dockerfile + docker-compose (production-ready)
- [x] Final README with metrics (433 lines, architecture diagram, eval results)

---

## Phase 9 — Production Hardening 🔧

> **Status: PLANNED** — Builds on all 8 completed phases.

### 9.1 Circuit Breaker Improvements
- [ ] Configurable thresholds per environment (dev/staging/prod)
- [ ] Circuit breaker state exposed in `/health` endpoint response
- [ ] Prometheus metrics for circuit breaker state transitions
- [ ] Auto-reset timer visibility in Grafana dashboard

### 9.2 HyDE Query Expansion Enhancements
- [ ] Selective HyDE activation (only for complex/ambiguous queries)
- [ ] HyDE confidence scoring (reject low-quality expansions)
- [ ] Multi-perspective HyDE variants for diversified retrieval
- [ ] HyDE results cached to avoid redundant LLM calls

### 9.3 Mixture-of-Agents (MoA) Enhancements
- [ ] Configurable agent count (1–5) via API parameter
- [ ] Per-agent latency tracking and fallback
- [ ] MoA quality gating (skip merger if single agent scores high)
- [ ] Cost-aware agent selection (cheaper tiers for simple queries)

### 9.4 Cross-Encoder Reranking Improvements
- [ ] Batch reranking for large candidate sets
- [ ] Reranker latency monitoring (Prometheus histogram)
- [ ] Model hot-swap support (switch reranker models without restart)
- [ ] Reranker cache for repeated (query, document) pairs

### 9.5 CLIP Multimodal Retrieval
- [ ] CLIP embeddings indexed in Qdrant (persistent, not in-memory)
- [ ] Image-to-image similarity search
- [ ] Multi-image query support (batch CLIP encoding)
- [ ] CLIP model versioning (support multiple model sizes)

### 9.6 Feedback Loop System
- [ ] Feedback-driven retrieval strategy adjustment
- [ ] A/B testing framework for retrieval strategies
- [ ] Feedback analytics dashboard (Grafana)
- [ ] Automatic retraining trigger on negative feedback patterns

### 9.7 Rate Limiting & Authentication
- [ ] Redis-backed rate limiting (replace in-memory for multi-worker)
- [ ] OAuth2 / JWT token authentication (replace simple API key)
- [ ] Role-based access control (admin, user, readonly)
- [ ] Rate limit headers in responses (`X-RateLimit-*`)

### 9.8 Caching Layer
- [ ] Redis-backed cache (replace in-memory TTLCache)
- [ ] Cache warming on startup (embeddings, common queries)
- [ ] Cache invalidation hooks (on document ingestion)
- [ ] Cache hit rate alerting (Grafana)

### 9.9 OpenTelemetry Tracing
- [ ] Auto-instrumentation for all LangGraph nodes
- [ ] Trace propagation across async boundaries
- [ ] Custom span attributes for query metadata
- [ ] Trace-to-log correlation (trace_id in structlog)

### 9.10 Performance & Scalability
- [ ] Async document ingestion pipeline
- [ ] Batch embedding generation
- [ ] Qdrant collection sharding strategy
- [ ] Load testing with Locust (target: 100 QPS)
- [ ] Connection pooling for LLM clients

### 9.11 Security Hardening
- [ ] Input sanitization (prompt injection prevention)
- [ ] Output filtering (PII redaction in responses)
- [ ] Audit logging for all write operations
- [ ] Dependency vulnerability scanning in CI

---

## Completion Summary

| Phase | Status | Key Deliverable |
|-------|--------|-----------------|
| 1 — Foundation | ✅ Complete | 4-layer architecture, ADRs, Docker Compose |
| 2 — Specialists | ✅ Complete | 5 HF model wrappers with tests |
| 3 — HippoRAG | ✅ Complete | Knowledge Graph + PPR (86% accuracy) |
| 4 — Agent Graph | ✅ Complete | 10 LangGraph nodes, retry loop |
| 5 — Memory | ✅ Complete | 4-layer memory (episodic/semantic/procedural/working) |
| 6 — Eval & CI | ✅ Complete | 390+ tests, 40-case benchmark, CI gate |
| 7 — API & Observability | ✅ Complete | FastAPI + SSE streaming, structlog, Prometheus, OTel |
| 8 — Demo & Deploy | ✅ Complete | Streamlit demo, Modal GPU, Docker |
| 9 — Production Hardening | 🔧 Planned | Circuit breakers, HyDE, MoA, CLIP, feedback, auth, caching, tracing |
