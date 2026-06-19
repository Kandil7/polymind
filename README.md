# PolyMind

> A self-evaluating, multimodal, multi-agent knowledge assistant with a CI-gated RAGAS eval harness.

[![CI](https://github.com/your-username/polymind/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/polymind/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                               │
│              (text / audio / image / PDF / CSV)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Planner Agent]  ←  4-Layer Memory (Episodic + Semantic)      │
│  Detects modality + intent, recalls past context                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Router Agent]  —  Conditional routing by modality             │
│  ┌─────┬─────┬──────┬──────────┬──────┐                        │
│  ASR   VQA  DocQA  TableQA    Text   │                        │
│  └──┬──┘──┬─┘──┬───┘────┬─────┘      │                        │
└─────┼─────┼────┼────────┼────────────┘
      │     │    │        │
      ▼     ▼    ▼        ▼
┌─────────────────────────────────────────────────────────────────┐
│  [HippoRAG Retriever]  —  Knowledge Graph + Personalized PPR   │
│  Multi-hop retrieval with NetworkX + BAAI/bge-m3 embeddings     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Generator]  —  LLM-powered answer synthesis                   │
│  Groq Llama 3.3 70B (280 t/s)                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Critic Agent]  —  LLM-as-Judge self-evaluation                │
│  Faithfulness ≥ 0.72 │ Relevancy ≥ 0.75 │ Hallucination ≤ 0.25 │
│  ↻ Retry loop (max 2) if rejected                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Synthesizer]  —  Format answer with citations + confidence    │
│  Store episode → Memory consolidation                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Tech Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| **Agent Graph** | Orchestration | LangGraph |
| **LLM** | Inference | Groq (Llama 3.3 70B, Llama 3.1 8B) |
| **Vector DB** | Storage | Qdrant |
| **Embeddings** | Dense vectors | BAAI/bge-m3 |
| **ASR** | Speech-to-text | openai/whisper-large-v3 (Groq API) |
| **VQA** | Image QA | Salesforce/blip-vqa-base |
| **DocQA** | Document QA | impira/layoutlm-document-qa |
| **TableQA** | Table QA | google/tapas-base-finetuned-wtq |
| **Evaluation** | Quality metrics | DeepEval + RAGAS |
| **Observability** | Metrics/Traces | Prometheus + Grafana |
| **Memory** | Context retention | Mem0 + Qdrant |
| **Backend** | API | FastAPI + Pydantic v2 |
| **Deployment** | Container | Docker Compose |
| **GPU** | On-demand | Modal (T4) |

## 📊 Eval Results

| Metric | Target | Score | Status |
|--------|--------|-------|--------|
| Faithfulness | ≥ 0.72 | 0.84 | ✅ |
| Answer Relevancy | ≥ 0.75 | 0.88 | ✅ |
| Hallucination Rate | ≤ 0.25 | 0.04 | ✅ |
| Context Precision | ≥ 0.70 | 0.82 | ✅ |
| Context Recall | ≥ 0.70 | 0.79 | ✅ |

## 🏁 Build Phases

| Phase | Name | Status | Tests |
|-------|------|--------|-------|
| 1 | Foundation & Infrastructure | ✅ v0.1.0 | 24 |
| 2 | Specialist Model Wrappers | ✅ v0.2.0 | 61 |
| 3 | HippoRAG Retriever | ✅ v0.3.0 | 89 |
| 4 | LangGraph Agent Graph | ✅ v0.4.0 | 147 |
| 5 | 4-Layer Memory | ✅ v0.5.0 | 156 |
| 6 | Eval Harness & CI | ✅ v0.6.0 | 168 |
| 7 | API & Observability | ✅ v0.7.0 | — |
| 8 | Demo & Deploy | ✅ v1.0.0 | — |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Poetry
- Docker & Docker Compose
- Groq API key (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/polymind.git
cd polymind

# Install dependencies
poetry install

# Copy environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Start infrastructure
docker compose up -d

# Start the API server
make dev
```

### Usage

```bash
# Text query
curl -X POST http://localhost:8000/query/ \
  -F "question=What is RAG?"

# With audio file
curl -X POST http://localhost:8000/query/ \
  -F "question=Transcribe this" \
  -F "audio_file=@meeting.mp3"

# Ingest a document
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@document.pdf"

# Run evaluation
curl -X POST http://localhost:8000/eval/ \
  -H "Content-Type: application/json" \
  -d '{"limit": 10}'
```

### Streamlit Demo

```bash
# Start the demo app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🧪 Testing

```bash
# Run all unit tests
make test

# Run with coverage
pytest tests/unit --cov=polymind --cov-report=html

# Run evaluation harness
make eval

# Run linting
make lint

# Format code
make format
```

## 📁 Project Structure

```
polymind/
├── src/polymind/
│   ├── domain/              # Entities, interfaces, exceptions
│   ├── application/         # Agent graph, state, use cases
│   │   ├── agents/          # Planner, Router, Critic, etc.
│   │   └── graph.py         # LangGraph graph factory
│   ├── infrastructure/      # External integrations
│   │   ├── qdrant/          # HippoRAG, chunk repository
│   │   ├── specialists/     # ASR, VQA, DocQA, TableQA
│   │   ├── llm/             # Groq LLM factory
│   │   ├── memory/          # 4-layer memory system
│   │   ├── rag/             # Embedder, chunker, ingestion
│   │   └── eval/            # DeepEval, RAGAS runner
│   └── api/                 # FastAPI endpoints + middleware
├── tests/
│   ├── unit/                # 168 unit tests
│   └── eval/                # RAGAS benchmark suite
├── infra/                   # Docker, Prometheus, Modal
├── scripts/                 # Seed data, eval runner
├── .github/workflows/       # CI + eval gate
├── app.py                   # Streamlit demo
└── pyproject.toml           # Poetry config
```

## 🎯 Key Features

### Self-Evaluation (Critic Agent)
The Critic agent evaluates every answer against retrieved context using LLM-as-Judge. If faithfulness drops below threshold, it triggers re-retrieval — catching hallucinations before they reach the user.

### Multi-Modal Routing
Automatically detects input modality (text, audio, image, PDF, CSV) and routes to the appropriate specialist agent. No manual configuration needed.

### HippoRAG v2
Knowledge Graph-based retrieval using Personalized PageRank for multi-hop reasoning — achieving 86% accuracy on complex queries vs. 79% for standard vector RAG.

### CI-Gated Evaluation
Every PR is tested against a 20-case benchmark. If faithfulness drops below 0.72, the PR is blocked — preventing quality regressions.

### 4-Layer Memory
Episodic (conversation history), Semantic (extracted facts), Procedural (successful patterns), and Working (graph state) memory layers enable context retention and learning.

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/query/` | POST | Multimodal query processing |
| `/ingest/` | POST | Document ingestion |
| `/eval/` | POST | Run evaluation harness |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | OpenAPI documentation |

## 📈 Observability

- **Prometheus**: Query count, latency, faithfulness scores
- **Grafana**: Real-time dashboards (port 3000)
- **structlog**: Structured request/response logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Qdrant](https://qdrant.tech/) — Vector database
- [DeepEval](https://docs.confident-ai.com/) — LLM evaluation
- [RAGAS](https://docs.ragas.io/) — RAG evaluation
