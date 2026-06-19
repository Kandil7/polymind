# Phase 7 — API & Observability

**Goal:** Complete FastAPI endpoints with structlog logging and Prometheus metrics.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `api/routes/query.py` | Multimodal query endpoint | API |
| `api/routes/ingest.py` | Document ingestion endpoint | API |
| `api/routes/eval.py` | Evaluation endpoint | API |
| `api/schemas/query.py` | Query request/response models | API |
| `api/schemas/ingest.py` | Ingest response model | API |
| `api/schemas/eval.py` | Eval response model | API |
| `api/middleware/logging.py` | structlog request logging | API |
| `api/middleware/metrics.py` | Prometheus metrics | API |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query/` | POST | Multimodal query |
| `/ingest/` | POST | Document ingestion |
| `/eval/` | POST | Run evaluation |
| `/metrics` | GET | Prometheus metrics |

## Prometheus Metrics

- `polymind_queries_total` — Query count by modality
- `polymind_query_latency_seconds` — Latency histogram
- `polymind_faithfulness_score` — Faithfulness distribution
- `polymind_active_requests` — Active request gauge
