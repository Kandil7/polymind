# n8n Integration Patterns for PolyMind

> **Status: PLANNED** — This integration is not yet implemented. These patterns describe how n8n could be integrated with PolyMind in a future phase.

## What is n8n?

[n8n](https://n8n.io) is a workflow automation platform that connects apps and services via a visual node-based editor. It supports 400+ integrations and can be self-hosted, making it ideal for building automated pipelines that trigger on events and orchestrate multi-step workflows.

## Why n8n + PolyMind?

PolyMind processes multimodal queries (text, audio, image, document, table). n8n can:

1. **Trigger PolyMind** via webhooks (e.g., new email → extract text → query PolyMind)
2. **Route PolyMind outputs** to downstream systems (Slack, Notion, Google Sheets)
3. **Orchestrate batch processing** (upload 50 PDFs → ingest all → get summary)
4. **Build monitoring dashboards** (aggregate feedback stats → display in n8n)
5. **Automate feedback loops** (negative feedback → alert in Slack → create ticket)

## Integration Pattern 1: Webhook Trigger

The simplest integration — n8n calls PolyMind's `/query` endpoint.

```
[Trigger] → [HTTP Request to PolyMind] → [Process Response]
```

### n8n Workflow Configuration

```
1. Webhook node (trigger)
   - Method: POST
   - Path: /webhook/polyquery
   - Body: { "text": "{{ $json.message }}", "user_id": "n8n-bot" }

2. HTTP Request node
   - URL: http://localhost:8000/query/
   - Method: POST
   - Headers: { "Authorization": "Bearer {{ $env.POLYMIND_API_KEY }}" }
   - Body: JSON
     {
       "text": "{{ $json.text }}",
       "user_id": "{{ $json.user_id }}"
     }

3. IF node (check confidence)
   - Condition: {{ $json.answer.confidence }} >= 0.7

4. Response node (if high confidence)
   - Send answer to caller

5. HTTP Request node (if low confidence)
   - Log to monitoring system
```

### PolyMind API Contract

```json
// POST /query/
{
  "text": "What is the capital of France?",
  "user_id": "n8n-automation"
}

// Response
{
  "answer": {
    "text": "The capital of France is Paris.",
    "confidence": 0.95,
    "sources": ["knowledge_base_france.txt"]
  },
  "critic_scores": {
    "faithfulness": {"value": 0.92, "passed": true},
    "answer_relevancy": {"value": 0.88, "passed": true},
    "hallucination": {"value": 0.05, "passed": true}
  },
  "modality": "text",
  "retry_count": 0
}
```

## Integration Pattern 2: Document Ingestion Pipeline

Automate document ingestion when files are uploaded to cloud storage.

```
[Google Drive / S3 Upload] → [Download File] → [PolyMind /ingest] → [Notify]
```

### n8n Workflow

```
1. Google Drive Trigger node
   - Watch for new files in folder

2. HTTP Request node (ingest)
   - URL: http://localhost:8000/ingest/
   - Method: POST
   - Body: multipart/form-data
     - file: {{ $binary.fileData }}
     - metadata: { "source": "google-drive", "folder": "research" }

3. Slack node
   - Channel: #document-ingestion
   - Message: "Ingested: {{ $json.filename }} ({{ $json.chunk_count }} chunks)"
```

## Integration Pattern 3: Batch Query Processing

Process multiple queries in parallel using n8n's Split In Batches node.

```
[Google Sheets] → [Split In Batches] → [PolyMind /query] → [Aggregate Results] → [Write Back]
```

### n8n Workflow

```
1. Google Sheets node
   - Read all rows (column A: query text, column B: empty — to be filled)

2. Split In Batches node
   - Batch size: 5 (respect PolyMind's rate limit: 10/min for /query)

3. HTTP Request node
   - URL: http://localhost:8000/query/
   - Method: POST
   - Body: { "text": "{{ $json.query }}", "user_id": "batch-job" }

4. Wait node
   - Resume: 6 seconds (stay under rate limit)

5. Google Sheets node (write back)
   - Update row with answer and confidence
```

## Integration Pattern 4: Feedback-Driven Alerting

Monitor feedback and alert when satisfaction drops.

```
[Schedule Trigger] → [PolyMind /feedback/stats] → [IF satisfaction < 0.8] → [Slack Alert]
```

### n8n Workflow

```
1. Schedule Trigger node
   - Every 6 hours

2. HTTP Request node
   - URL: http://localhost:8000/feedback/stats
   - Method: GET

3. IF node
   - Condition: {{ $json.satisfaction_rate }} < 0.8

4. Slack node
   - Channel: #alerts
   - Message: "⚠️ PolyMind satisfaction dropped to {{ $json.satisfaction_rate }} ({{ $json.total }} reviews)"
```

## Integration Pattern 5: SSE Streaming with n8n

n8n can consume PolyMind's SSE streaming endpoint for real-time progress updates.

```
[Webhook] → [HTTP Request with SSE] → [Process Events] → [Respond]
```

### PolyMind SSE Events

```
POST /query/stream

event: node_start
data: {"node": "planner", "status": "running"}

event: node_complete
data: {"node": "planner", "modality": "text", "intent": "question_answering"}

event: node_complete
data: {"node": "router", "strategy": "standard"}

event: node_complete
data: {"node": "rag", "chunks_retrieved": 5}

event: node_complete
data: {"node": "generator", "length": 256}

event: node_complete
data: {"node": "critic", "passed": true, "faithfulness": 0.89}

event: node_complete
data: {"node": "synthesizer", "confidence": 0.92}

event: done
data: {"answer": "The capital of France is Paris.", "confidence": 0.92}
```

## Integration Pattern 6: Health Monitoring

n8n can periodically check PolyMind's health and alert on degradation.

```
[Schedule] → [PolyMind /health] → [IF degraded] → [PagerDuty / Slack]
```

### Health Response

```json
{
  "status": "healthy",
  "qdrant": "connected",
  "llm_api_key": "configured",
  "circuit_breakers": {
    "qdrant": {"state": "closed", "failures": 0},
    "llm": {"state": "closed", "failures": 0},
    "embedder": {"state": "closed", "failures": 0},
    "memory": {"state": "closed", "failures": 0}
  },
  "degradation": {
    "services": {"qdrant": "healthy", "llm": "healthy", "embedder": "healthy", "memory": "healthy"},
    "overall": "healthy"
  }
}
```

## n8n Self-Hosting

For production use, deploy n8n alongside PolyMind:

```yaml
# docker-compose.yml addition
services:
  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - GENERIC_TIMEZONE=UTC
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - polymind-api  # Ensure PolyMind is running
```

## Security Considerations

1. **API Key Management** — Store `POLYMIND_API_KEY` in n8n credentials, not in workflow JSON
2. **Webhook Authentication** — Use n8n's built-in header authentication for incoming webhooks
3. **Network Isolation** — Deploy n8n and PolyMind in the same Docker network
4. **Rate Limiting** — Respect PolyMind's rate limits (10/min for /query, 5/min for /ingest)
5. **Input Validation** — Sanitize user input before passing to PolyMind (prompt injection prevention)

## When to Use n8n vs Direct API Calls

| Scenario | Use n8n | Use Direct API |
|----------|---------|----------------|
| Simple query from a script | ❌ | ✅ (faster, no overhead) |
| Multi-step workflow with external services | ✅ | ❌ |
| Scheduled batch processing | ✅ | ❌ (need to build scheduler) |
| Event-driven automation | ✅ | ❌ (need webhook server) |
| Real-time dashboard | ❌ | ✅ (direct SSE) |
| Non-technical team automation | ✅ | ❌ |

## Summary

n8n integration with PolyMind enables:

- **Event-driven workflows** — Trigger on emails, file uploads, database changes
- **Multi-service orchestration** — PolyMind + Slack + Notion + Google Sheets
- **Batch processing** — Process hundreds of documents automatically
- **Monitoring & alerting** — Watch health and feedback metrics
- **No-code automation** — Let non-engineers build query pipelines

This integration is planned for Phase 9+ and will be documented with working examples once implemented.
