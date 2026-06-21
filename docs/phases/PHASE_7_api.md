# Phase 7 — API & Observability

**Goal:** Complete FastAPI endpoints with structlog logging, Prometheus metrics, authentication, and rate limiting.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `api/routes/query.py` | Multimodal query endpoint (streaming + non-streaming) | API |
| `api/routes/ingest.py` | Document ingestion endpoint | API |
| `api/routes/eval.py` | Evaluation endpoint | API |
| `api/routes/feedback.py` | User feedback capture (POST + stats) | API |
| `api/schemas/query.py` | Query request/response models | API |
| `api/schemas/ingest.py` | Ingest request/response models | API |
| `api/schemas/eval.py` | Eval request/response models | API |
| `api/schemas/feedback.py` | Feedback request/response models | API |
| `api/middleware/logging.py` | structlog request logging | API |
| `api/middleware/metrics.py` | Prometheus metrics | API |
| `api/middleware/auth.py` | Bearer token auth with public path skip | API |
| `api/middleware/rate_limit.py` | Per-IP sliding window rate limiter | API |
| `api/main.py` | FastAPI app factory with lifespan | API |

## API Endpoints

| Endpoint | Method | Auth | Rate Limit | Description |
|----------|--------|------|------------|-------------|
| `/health` | GET | Public | — | Health check with dependency status |
| `/query/` | POST | Bearer token | 10/min | Multimodal query (multipart form) |
| `/query/stream` | POST | Bearer token | 10/min | Multimodal query with SSE streaming |
| `/ingest/` | POST | Bearer token | 5/min | Document ingestion into knowledge base |
| `/eval/` | POST | Bearer token | 30/min | Run evaluation harness |
| `/feedback/` | POST | Public | 30/min | Submit user feedback |
| `/feedback/stats` | GET | Public | 30/min | Get feedback statistics |
| `/metrics` | GET | Public | — | Prometheus metrics |
| `/docs` | GET | Public | — | OpenAPI Swagger UI |
| `/redoc` | GET | Public | — | ReDoc API documentation |

## Authentication

Bearer token authentication via `POLYMIND_API_KEY` environment variable.

- **With API key set**: All protected endpoints require `Authorization: Bearer <api_key>` header
- **Without API key**: Development mode — all requests allowed (no auth required)
- **Public paths** (skip auth): `/health`, `/docs`, `/redoc`, `/openapi.json`, `/metrics`, `/feedback`

## Rate Limiting

Per-IP sliding window counter (in-memory, no external dependencies):

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/query/*` | 10 requests | 60 seconds |
| `/ingest/*` | 5 requests | 60 seconds |
| All others | 30 requests | 60 seconds |

Rate limit exceeded returns `429` with `Retry-After` header.

## Prometheus Metrics

- `polymind_queries_total` — Query count by modality
- `polymind_query_latency_seconds` — Latency histogram
- `polymind_faithfulness_score` — Faithfulness distribution
- `polymind_active_requests` — Active request gauge

## Middleware Stack

Applied in order (outermost first):

1. **CORS** — Allow all origins (configurable for production)
2. **APIKeyAuth** — Bearer token validation
3. **RateLimit** — Per-IP sliding window
4. **Prometheus** — Request metrics collection
5. **RequestLogging** — structlog structured request logging
