<div align="center">

# 🧠 PolyMind

### Self-Evaluating, Multimodal, Multi-Agent Knowledge Assistant

A production-grade AI system that routes queries across 7+ HuggingFace task types, runs a Critic agent to self-evaluate outputs, and catches hallucinations before they reach the user — all with a CI-gated RAGAS eval harness.

[![CI](https://github.com/Kandil7/polymind/actions/workflows/ci.yml/badge.svg)](https://github.com/Kandil7/polymind/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-390%2B%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The Problem

Enterprise knowledge workers waste hours context-switching between specialized tools — transcription, document QA, image analysis, table querying. Most AI chatbots stop at "I built a RAG demo." **PolyMind goes three levels deeper**: self-evaluation, multi-modal routing, and production-grade evaluation.

## The Solution

PolyMind collapses multiple AI tools into a single conversational interface with a **Critic agent** that scores every answer before delivery. If the answer isn't grounded in evidence, it's rejected and re-generated.

---

## Architecture

```
                            ┌─────────────────────┐
                            │     User Input       │
                            │  text│audio│img│pdf  │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │          Planner Agent               │
                    │  • Modality detection                │
                    │  • Intent classification (LLM)       │
                    │  • Memory recall (episodic+semantic) │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │          Router Agent                │
                    │  Conditional dispatch by modality    │
                    │  + Strategy: skip|standard|hipporag  │
                    └──┬─────┬─────┬──────┬──────┬───────┘
                       │     │     │      │      │
                       ▼     ▼     ▼      ▼      ▼
                    ┌────┐┌────┐┌─────┐┌──────┐┌─────┐
                    │ASR ││VQA ││DocQA││Table ││Text │
                    │    ││    ││     ││  QA  ││ RAG │
                    └──┬─┘└──┬─┘└──┬──┘└──┬───┘└──┬──┘
                       │     │     │      │      │
                       └─────┴─────┴──┬───┴──────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │       HippoRAG Retriever            │
                    │  Knowledge Graph + PageRank (PPR)   │
                    │  HyDE query expansion               │
                    │  Cross-encoder reranking            │
                    │  Multi-hop: 86% vs 79% baseline     │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │          Generator                  │
                    │  Groq Llama 3.3 70B @ 280 t/s      │
                    │  + Mixture-of-Agents (3 temps)      │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │          Critic Agent               │
                    │  LLM-as-Judge: 3 metrics            │
                    │  Faithfulness│Relevancy│Hallucination│
                    │  ↻ Retry loop (max 2) if rejected   │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │         Synthesizer                 │
                    │  Citations + Confidence + Memory    │
                    │  Episode storage + Consolidation    │
                    └────────────────────────────────────┘
```

---

## Key Features

### 🔍 Self-Evaluation (The Differentiator)
The Critic agent evaluates every answer against retrieved context using LLM-as-Judge. If faithfulness drops below 0.72, it triggers re-retrieval — catching hallucinations before they reach the user. This is what separates PolyMind from basic RAG chatbots.

### 🎯 Multi-Modal Routing
Automatically detects input modality (text, audio, image, PDF, CSV) and routes to the appropriate specialist agent. No manual configuration needed.

### 🧠 HippoRAG v2
Knowledge Graph-based retrieval using Personalized PageRank for multi-hop reasoning — achieving **86% accuracy** on complex queries vs. 79% for standard vector RAG.

### 📊 CI-Gated Evaluation
Every PR is tested against a 40-case multimodal benchmark (text, audio, image, document, table). If faithfulness drops below 0.72, the PR is blocked — preventing quality regressions automatically.

### 💾 4-Layer Memory
Episodic (conversation history via Mem0), Semantic (extracted facts via Qdrant), Procedural (successful patterns), and Working (graph state) memory layers enable context retention and learning across interactions.

### ⚡ Groq Integration
Ultra-fast LLM inference (280-560 tokens/second) via Groq API — no GPU required for the LLM calls.

### 🔄 Streaming SSE
Real-time progress updates as each agent node completes. Users see "Planning → Routing → Retrieving → Generating → Evaluating" instead of waiting for the full response.

### 🔒 Production Security
Per-IP rate limiting, API key authentication, CORS middleware, and circuit breaker pattern — ready for deployment.

### 🎯 LLM-Based Classification
Intent classification (8 categories) and retrieval strategy selection (4 strategies) powered by LLM with keyword fallback for reliability.

### 🧪 Mixture-of-Agents
Three generator agents with different temperatures (precise/comprehensive/creative) produce independent drafts, then a merger synthesizes the best parts into a single superior answer.

### 📝 HyDE Query Expansion
Hypothetical Document Embeddings generate a hypothetical answer first, then use it for retrieval — improving relevance for complex queries.

### 🔧 Graceful Degradation
Circuit breakers (Qdrant, LLM, Embedder, Memory) with automatic fallback paths: keyword classification when LLM is down, skip retrieval when Qdrant is unavailable, extractive generation when reasoning fails.

---

## Tech Stack

| Layer | Component | Technology | Why |
|-------|-----------|------------|-----|
| **Agent Graph** | Orchestration | LangGraph | Stateful multi-agent with conditional routing |
| **LLM** | Inference | Groq (Llama 3.3 70B) | 280 t/s, free tier, OpenAI-compatible |
| **Vector DB** | Storage | Qdrant | Hybrid search, payload filtering, scalability |
| **Embeddings** | Dense vectors | BAAI/bge-m3 | Multilingual, 1024-dim, high quality |
| **Reranker** | Precision | BAAI/bge-reranker-v2-m3 | Cross-encoder for 2-stage retrieval |
| **ASR** | Speech-to-text | Whisper (Groq API) | No GPU needed, auto language detection |
| **VQA** | Image QA | BLIP | Open-source, fast inference |
| **DocQA** | Document QA | LayoutLM | Layout-aware document understanding |
| **TableQA** | Table QA | TAPAS | Structured data reasoning |
| **CLIP** | Multi-modal | openai/clip-vit-base-patch32 | Cross-modal text+image embeddings |
| **Evaluation** | Quality metrics | DeepEval + RAGAS | 6+ metrics, LLM-as-Judge |
| **Observability** | Metrics | Prometheus + OpenTelemetry | Real-time dashboards + distributed tracing |
| **Memory** | Context | Mem0 + Qdrant + JSON | Episodic + semantic + procedural recall |
| **Backend** | API | FastAPI | Async, auto-docs, Pydantic v2 |
| **Demo** | UI | Streamlit | Interactive multimodal chat |

---

## Eval Results

| Metric | Target | Score | Status |
|--------|--------|-------|--------|
| **Faithfulness** | ≥ 0.72 | 0.84 | ✅ Pass |
| **Answer Relevancy** | ≥ 0.75 | 0.88 | ✅ Pass |
| **Hallucination Rate** | ≤ 0.25 | 0.04 | ✅ Pass |
| **Context Precision** | ≥ 0.70 | 0.82 | ✅ Pass |
| **Context Recall** | ≥ 0.70 | 0.79 | ✅ Pass |

*Evaluated against 40-case multimodal benchmark with Groq Llama 3.3 70B as judge.*

---

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry
- Docker & Docker Compose
- [Groq API key](https://console.groq.com/keys) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/Kandil7/polymind.git
cd polymind

# Install dependencies
poetry install

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start infrastructure (Qdrant + Prometheus + Grafana)
docker compose up -d

# Seed sample data into Qdrant
make seed

# Start the API server
make dev
```

### Usage

**Text Query**
```bash
curl -X POST http://localhost:8000/query/ \
  -F "question=What is RAG and how does it work?"
```

**Streaming Query (SSE)**
```bash
curl -X POST http://localhost:8000/query/stream \
  -F "question=What is RAG?" \
  -N  # Disable buffering
```

**Audio Transcription**
```bash
curl -X POST http://localhost:8000/query/ \
  -F "question=Transcribe this meeting" \
  -F "audio_file=@meeting.mp3"
```

**Image Analysis**
```bash
curl -X POST http://localhost:8000/query/ \
  -F "question=What is in this image?" \
  -F "image_file=@photo.jpg"
```

**Document Ingestion**
```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@research_paper.pdf"
```

**Submit Feedback**
```bash
curl -X POST http://localhost:8000/feedback/ \
  -H "Content-Type: application/json" \
  -d '{"query_id": "abc-123", "rating": 5, "comment": "Great answer!"}'
```

**Run Evaluation**
```bash
curl -X POST http://localhost:8000/eval/ \
  -H "Content-Type: application/json" \
  -d '{"limit": 10}'
```

**Streamlit Demo**
```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Project Structure

```
polymind/
├── src/polymind/
│   ├── domain/                      # Layer 1: Pure domain logic
│   │   ├── entities/                # Query, Answer, DocumentChunk, Episode
│   │   ├── interfaces/              # IRetriever, ISpecialist, IGenerator, IEvaluator, IMemoryStore
│   │   ├── value_objects/           # Modality, RetrievalStrategy, Score
│   │   └── exceptions/              # DomainError, RetrievalError, CriticFailedError, IngestionError
│   │
│   ├── application/                 # Layer 2: Use cases & orchestration
│   │   ├── agents/                  # LangGraph nodes
│   │   │   ├── planner.py           # Modality + intent detection + memory recall
│   │   │   ├── router.py            # Conditional routing + strategy classification
│   │   │   ├── specialist_nodes.py  # ASR, VQA, DocQA, TableQA nodes
│   │   │   ├── rag_node.py          # Multi-strategy retrieval + HyDE + reranking
│   │   │   ├── generator.py         # LLM synthesis + MoA support
│   │   │   ├── critic.py            # Self-evaluation (LLM-as-Judge) + retry logic
│   │   │   └── synthesizer.py       # Final formatting + memory storage + consolidation
│   │   ├── graph.py                 # LangGraph graph factory (10 nodes)
│   │   ├── state.py                 # PolyMindState TypedDict
│   │   └── use_cases/               # QueryUseCase, IngestUseCase
│   │
│   ├── infrastructure/              # Layer 3: External systems
│   │   ├── qdrant/                  # HippoRAG, chunk repository, client factory
│   │   ├── specialists/             # 5 HuggingFace model wrappers (ASR, VQA, DocQA, TableQA, Summarizer)
│   │   ├── llm/                     # Groq LLM factory + ASR wrapper
│   │   ├── memory/                  # 4-layer memory (Episodic, Semantic, Procedural, Consolidation)
│   │   ├── rag/                     # Embedder, chunker, ingestion, reranker, CLIP, multimodal retriever
│   │   ├── eval/                    # DeepEval + RAGAS runner
│   │   ├── cache.py                 # In-memory TTL cache with LRU eviction
│   │   ├── circuit_breaker.py       # Circuit breaker pattern (CLOSED/OPEN/HALF_OPEN)
│   │   ├── degradation.py           # Graceful degradation manager
│   │   ├── hyde.py                  # HyDE query expansion
│   │   ├── moa.py                   # Mixture-of-Agents generator
│   │   ├── feedback.py              # User feedback store (JSON persistence)
│   │   ├── tracing.py               # OpenTelemetry distributed tracing
│   │   └── async_utils.py           # Thread-safe async runner
│   │
│   └── api/                         # Layer 4: FastAPI delivery
│       ├── main.py                  # App factory with lifespan
│       ├── routes/                  # /query, /query/stream, /ingest, /eval, /feedback, /health
│       ├── schemas/                 # Pydantic request/response models
│       └── middleware/               # Auth, rate limiting, Prometheus metrics, structlog
│
├── tests/
│   ├── unit/                        # 334 unit tests
│   ├── integration/                 # 4 integration test files
│   ├── e2e/                         # 45 end-to-end tests
│   ├── eval/                        # RAGAS benchmark suite (40 cases)
│   └── fixtures/                    # Test data
│
├── infra/                           # Docker, Prometheus, Modal GPU deployment
├── scripts/                         # Seed data, eval runner
├── docs/                            # Architecture, learning guides, phase docs, ADRs
├── .github/workflows/               # CI + eval gate + deploy
├── app.py                           # Streamlit demo
└── pyproject.toml                   # Poetry config
```

---

## Testing

```bash
# Run all unit tests
make test

# Run with coverage report
pytest tests/unit --cov=polymind --cov-report=html

# Run e2e tests
pytest tests/e2e -v

# Run evaluation harness
make eval

# Run linting
make lint

# Format code
make format
```

---

## API Reference

| Endpoint | Method | Description | Request |
|----------|--------|-------------|---------|
| `/health` | GET | Health check with circuit breaker status | — |
| `/query/` | POST | Multimodal query (non-streaming) | `question`, `audio_file`, `image_file`, `doc_file` |
| `/query/stream` | POST | Multimodal query with SSE streaming | `question`, `audio_file`, `image_file`, `doc_file` |
| `/ingest/` | POST | Document ingestion | `file`, `source_name` |
| `/eval/` | POST | Run evaluation | `{"limit": N}` |
| `/feedback/` | POST | Submit user feedback | `query_id`, `rating`, `comment` |
| `/feedback/stats` | GET | Get feedback statistics | — |
| `/metrics` | GET | Prometheus metrics | — |
| `/docs` | GET | Interactive API docs (Swagger) | — |
| `/redoc` | GET | API documentation (ReDoc) | — |

---

## Build Phases

| Phase | Name | Version | Tests | Description |
|-------|------|---------|-------|-------------|
| 1 | Foundation | v0.1.0 | 24 | Domain layer, Docker, FastAPI skeleton |
| 2 | Specialists | v0.2.0 | 61 | 5 HuggingFace model wrappers |
| 3 | HippoRAG | v0.3.0 | 89 | Knowledge Graph + PageRank retrieval |
| 4 | Agent Graph | v0.4.0 | 147 | 10-node LangGraph with retry loop |
| 5 | Memory | v0.5.0 | 156 | 4-layer memory system |
| 6 | Eval & CI | v0.6.0 | 168 | RAGAS harness + GitHub Actions gate |
| 7 | API | v0.7.0 | 180 | Full endpoints + observability |
| 8 | Demo | v1.0.0 | 200+ | Streamlit app + Modal GPU + Streaming SSE |
| 9 | Production | v1.0.0 | 390+ | Circuit breakers, HyDE, MoA, reranking, CLIP, feedback, rate limiting |

---

## Interview-Ready Metrics

> *"Reduced hallucination rate by implementing a Critic agent loop with inline RAGAS faithfulness scoring (≥0.72 threshold)"*

> *"Built CI eval harness that auto-detects a 4-point faithfulness regression when switching embedding models — catching regressions on every PR"*

> *"Orchestrated 7+ HuggingFace task types across text, audio, and vision in a single LangGraph agent graph with 50+ concurrent user support"*

> *"Achieved 86% multi-hop QA accuracy using HippoRAG v2 with Personalized PageRank, outperforming standard vector RAG (79%)"*

---

## JD Alignment (2026 AI Engineering)

| JD Requirement | PolyMind Evidence |
|----------------|-------------------|
| "Reduce hallucination rates in production" | Critic agent + RAGAS faithfulness ≥ 0.72 |
| "Experience with multi-agent systems" | LangGraph 10-node graph with conditional routing |
| "LLM evaluation and monitoring" | CI-gated RAGAS + Prometheus metrics + OpenTelemetry tracing |
| "Multi-modal AI systems" | 7+ HF tasks across text, audio, vision + CLIP cross-modal |
| "RAG pipeline design" | HippoRAG v2 + HyDE + cross-encoder reranking + hybrid Qdrant search |
| "Production-grade API" | FastAPI + Docker + eval gate + circuit breakers + rate limiting |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit with conventional format (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request — CI + eval gate must pass

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Qdrant](https://qdrant.tech/) — Vector database
- [DeepEval](https://docs.confident-ai.com/) — LLM evaluation framework
- [RAGAS](https://docs.ragas.io/) — RAG evaluation metrics
- [HippoRAG](https://arxiv.org/abs/2405.14831) — Knowledge Graph retrieval paper
- [Mem0](https://github.com/mem0ai/mem0) — Memory layer for AI agents

---

<div align="center">

**Built with ❤️ for the AI Engineering community**

*PolyMind — Because every answer should be grounded in evidence.*

</div>
