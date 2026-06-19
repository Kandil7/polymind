# Observability in PolyMind

## Three Pillars

### 1. Structured Logging (structlog)
Every request is logged with structured data:
```python
logger.info("query.processed",
    modality="text",
    latency_ms=245.3,
    faithfulness=0.84,
    passed_critic=True,
)
```

### 2. Metrics (Prometheus)
Key metrics tracked:
- `polymind_queries_total` — Total queries by modality
- `polymind_query_latency_seconds` — Latency histogram
- `polymind_faithfulness_score` — Faithfulness distribution
- `polymind_active_requests` — Current active requests

### 3. Traces (OpenTelemetry)
Each request is traced end-to-end:
- Planner → Router → Specialist → RAG → Generator → Critic
- Each span includes timing and metadata

## Dashboard (Grafana)

Access at http://localhost:3000 (admin/admin):
- Query rate by modality
- Latency percentiles (P50, P95, P99)
- Faithfulness score distribution
- Error rate by endpoint

## Alerting

Set up alerts for:
- Faithfulness drops below 0.72
- Latency P95 exceeds 2 seconds
- Error rate exceeds 5%
- Memory usage exceeds 80%

## Log Levels

- **DEBUG** — Detailed diagnostic info
- **INFO** — Normal operation events
- **WARNING** — Unexpected but recoverable
- **ERROR** — Failures requiring attention

Use `POLYMIND_LOG_LEVEL` env var to control verbosity.
