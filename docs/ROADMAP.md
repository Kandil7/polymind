# PolyMind Roadmap

## Phase 1 — Foundation & Infrastructure ✅
- [x] pyproject.toml + Poetry
- [x] Domain entities + interfaces + exceptions
- [x] PolyMindState TypedDict
- [x] Docker Compose (Qdrant, Prometheus, Grafana)
- [x] FastAPI health endpoint
- [x] ADRs documented
- [x] Unit tests passing

## Phase 2 — Specialist Model Wrappers
- [ ] ASR Wrapper (Whisper)
- [ ] VQA Wrapper (BLIP)
- [ ] DocQA Wrapper (LayoutLM)
- [ ] TableQA Wrapper (TAPAS)
- [ ] Summarizer Wrapper
- [ ] Each wrapper with 3+ passing tests

## Phase 3 — HippoRAG Retriever
- [ ] Qdrant client factory
- [ ] HippoRAG retriever (Knowledge Graph + PPR)
- [ ] Adaptive retriever (skip/standard/multi-hop)
- [ ] Document ingestion pipeline
- [ ] Multi-hop test passing

## Phase 4 — LangGraph Agent Graph
- [ ] Planner node with memory recall
- [ ] Router node with conditional edges
- [ ] Specialist nodes (ASR, VQA, DocQA, TableQA)
- [ ] RAG node
- [ ] Generator node (Mixture-of-Agents)
- [ ] Critic node (DeepEval LLM-as-Judge)
- [ ] Synthesizer node
- [ ] Retry loop (max 2 retries)
- [ ] End-to-end test passing

## Phase 5 — 4-Layer Memory
- [ ] Episodic store (Mem0)
- [ ] Semantic store (Qdrant)
- [ ] Procedural store
- [ ] FourLayerMemory class
- [ ] Memory integration in Planner + Synthesizer

## Phase 6 — Eval Harness & CI
- [ ] DeepEval critic implementation
- [ ] RAGAS runner
- [ ] 100-case benchmark dataset
- [ ] Pytest eval suite
- [ ] GitHub Actions eval gate
- [ ] PR fails when faithfulness < 0.72

## Phase 7 — API & Observability
- [ ] /query endpoint (multimodal)
- [ ] /ingest endpoint
- [ ] /eval endpoint
- [ ] structlog middleware
- [ ] Prometheus metrics
- [ ] OpenTelemetry tracing
- [ ] n8n webhook triggers

## Phase 8 — Demo & Deploy
- [ ] Streamlit demo app
- [ ] Modal GPU deployment
- [ ] HuggingFace Space
- [ ] Final README with metrics
