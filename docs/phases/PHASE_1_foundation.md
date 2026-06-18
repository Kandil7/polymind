# Phase 1 — Foundation & Infrastructure

**Goal:** Project skeleton with Clean Architecture, Docker infrastructure, domain layer, and a working FastAPI health endpoint.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `pyproject.toml` | Poetry config, all dependencies, tool configs | Root |
| `Makefile` | Dev commands: dev, test, lint, format, clean | Root |
| `.env.example` | Required env vars with placeholders | Root |
| `.gitignore` | Python/IDE/OS ignores | Root |
| `src/polymind/__init__.py` | Package root | Root |
| `src/polymind/domain/entities/query.py` | Query, QueryResult, ScoreResult | Domain |
| `src/polymind/domain/entities/answer.py` | Answer with confidence | Domain |
| `src/polymind/domain/entities/chunk.py` | DocumentChunk + ChunkMetadata | Domain |
| `src/polymind/domain/entities/episode.py` | ConversationEpisode | Domain |
| `src/polymind/domain/value_objects/modality.py` | Modality enum | Domain |
| `src/polymind/domain/value_objects/retrieval_strategy.py` | RetrievalStrategy enum | Domain |
| `src/polymind/domain/value_objects/score.py` | Score value object | Domain |
| `src/polymind/domain/interfaces/retriever.py` | IRetriever ABC | Domain |
| `src/polymind/domain/interfaces/specialist.py` | ISpecialist ABC | Domain |
| `src/polymind/domain/interfaces/memory_store.py` | IMemoryStore ABC | Domain |
| `src/polymind/domain/interfaces/generator.py` | IGenerator ABC | Domain |
| `src/polymind/domain/interfaces/evaluator.py` | IEvaluator ABC | Domain |
| `src/polymind/domain/exceptions/base.py` | DomainError + subclasses | Domain |
| `src/polymind/application/state.py` | PolyMindState TypedDict | Application |
| `src/polymind/application/graph.py` | build_graph() stub | Application |
| `src/polymind/application/use_cases/query_use_case.py` | QueryUseCase | Application |
| `src/polymind/application/use_cases/ingest_use_case.py` | IngestUseCase stub | Application |
| `src/polymind/api/main.py` | FastAPI app factory | API |
| `src/polymind/api/routes/health.py` | GET /health | API |
| `src/polymind/api/schemas/health.py` | HealthResponse model | API |
| `infra/docker-compose.yml` | Qdrant + Prometheus + Grafana | Infra |
| `infra/Dockerfile` | App container | Infra |
| `infra/prometheus.yml` | Prometheus config | Infra |
| `tests/conftest.py` | Shared fixtures | Test |
| `tests/unit/test_domain_entities.py` | Entity tests | Test |
| `tests/unit/test_domain_value_objects.py` | Value object tests | Test |
| `tests/unit/test_domain_interfaces.py` | Interface contract tests | Test |
| `tests/unit/test_state.py` | State schema tests | Test |
| `docs/architecture/ADR/ADR-001-qdrant-over-pgvector.md` | Architecture decision | Docs |
| `docs/architecture/ADR/ADR-002-langgraph-over-crewai.md` | Architecture decision | Docs |
| `docs/architecture/ADR/ADR-003-clean-architecture.md` | Architecture decision | Docs |
| `docs/architecture/diagrams/system-overview.mmd` | Mermaid architecture diagram | Docs |

## Key Design Decisions

1. **Poetry over uv** — plan.md mandates Poetry
2. **structlog** — structured logging, not loguru
3. **Pydantic v2** — for all schemas and domain validation
4. **ABC interfaces** — domain depends on abstractions only
5. **TypedDict for LangGraph state** — LangGraph's native pattern
6. **TYPE_CHECKING guards** — break circular imports between entities

## Dependencies Introduced

- fastapi, uvicorn — API serving
- langgraph, langchain — Agent orchestration (minimal)
- pydantic v2 — Data validation
- qdrant-client — Vector DB client (connection only)
- prometheus-client — Metrics
- structlog — Structured logging
- ruff, mypy — Code quality
- pytest, pytest-asyncio — Testing

## How to Run

```bash
# Install dependencies
poetry install

# Start infra
docker compose up -d

# Start app
poetry run uvicorn polymind.api.main:app --reload

# Test
curl http://localhost:8000/health
# → {"status":"ok","version":"0.1.0","service":"polymind"}

# Lint
make lint

# Test
make test
```

## What Phase 2 Builds On Top

Phase 2 adds specialist wrappers (ASR, VQA, DocQA, TableQA, Summarizer) as infrastructure implementations of the `ISpecialist` interface.
