# PolyMind

> Self-evaluating, multimodal, multi-agent knowledge assistant with a CI-gated RAGAS eval harness.

## Architecture

```
User Input (text / audio / image / PDF / CSV)
    ↓
[Planner Agent] ← 4-Layer Memory (Mem0 + Qdrant)
    ↓
[Router Agent] — detects modality + intent
    ↓ branches:
    ASR | VQA | DocQA | TableQA | Text
    ↓
[HippoRAG Retriever] — Knowledge Graph + PPR
    ↓
[Adaptive RAG Classifier] — skip / standard / multi-hop / speculative
    ↓
[Mixture-of-Agents Generator] — 3 proposers + 1 aggregator
    ↓
[Corrective RAG] — re-retrieve if confidence < 0.7
    ↓
[Critic Node] — DeepEval LLM-as-Judge (6 metrics)
    ↓
[Memory Consolidation] — store episode / extract semantic facts
    ↓
Final Answer + Citations + Confidence Scores
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Graph | LangGraph |
| Orchestration | LangChain |
| Vector DB | Qdrant |
| Embedding | BAAI/bge-m3 |
| ASR | openai/whisper-large-v3 |
| VQA | Salesforce/blip-vqa-base |
| DocQA | impira/layoutlm-document-qa |
| TableQA | google/tapas-base-finetuned-wtq |
| Evaluation | DeepEval + RAGAS |
| Observability | Prometheus + Grafana + LangSmith |
| Memory | Mem0 + Qdrant |
| Backend | FastAPI + Pydantic v2 |
| Deployment | Docker Compose + Modal |

## Quick Start

```bash
# Install
poetry install

# Start infrastructure
docker compose up -d

# Run app
make dev

# Test
make test

# Lint
make lint
```

## Eval Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.84 |
| Answer Relevancy | 0.88 |
| Hallucination Rate | 4.2% |

## Build Phases

1. Foundation & Infrastructure (Week 1)
2. Specialist Model Wrappers (Week 2)
3. HippoRAG Retriever (Week 3)
4. LangGraph Agent Graph (Weeks 4-5)
5. 4-Layer Memory (Week 6)
6. Eval Harness & CI Gate (Week 7)
7. API & Observability (Week 8)
8. Demo & Deploy (Week 9)

## License

MIT
