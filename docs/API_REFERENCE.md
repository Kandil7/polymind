# PolyMind API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Bearer token authentication via `POLYMIND_API_KEY` environment variable.

| Mode | Behavior |
|------|----------|
| **With API key** | Protected endpoints require `Authorization: Bearer <api_key>` header |
| **Without API key** | Development mode — all requests allowed |

**Public paths** (skip authentication):
- `GET /health`
- `GET /docs`, `GET /redoc`, `GET /openapi.json`
- `GET /metrics`
- `POST /feedback/`, `GET /feedback/stats`

---

## Rate Limiting

Per-IP sliding window (in-memory, 60-second window):

| Endpoint | Limit |
|----------|-------|
| `/query/*` | 10 requests/min |
| `/ingest/*` | 5 requests/min |
| All others | 30 requests/min |

Rate limit exceeded returns `429`:
```json
{
  "error": "Rate limit exceeded",
  "limit": 10,
  "window_seconds": 60,
  "retry_after": 60
}
```

---

## Error Response Format

All errors follow a consistent format:
```json
{
  "error": "Error description",
  "hint": "Optional suggestion"
}
```

HTTP status codes:
- `401` — Missing or invalid Authorization header
- `403` — Invalid API key
- `429` — Rate limit exceeded
- `500` — Internal server error

---

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "service": "polymind",
  "checks": {
    "qdrant": { "status": "ok", "collections": 1 },
    "llm": { "status": "ok", "provider": "groq" },
    "embedder": { "status": "configured", "model": "BAAI/bge-m3" }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Query (Non-Streaming)

```
POST /query/
```

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | User's question |
| `user_id` | string | No | User identifier (default: "anonymous") |
| `audio_file` | file | No | Audio file for ASR |
| `image_file` | file | No | Image file for VQA |
| `doc_file` | file | No | Document file (PDF/CSV) |

**Response:**
```json
{
  "answer": "The answer to your question...",
  "modality": "text",
  "confidence": 0.85,
  "citations": [
    { "source": "document.pdf", "score": 0.92 }
  ],
  "critic_scores": {
    "faithfulness": { "score": 0.88, "passed": true },
    "answer_relevancy": { "score": 0.91, "passed": true }
  },
  "retry_count": 0,
  "processing_time_ms": 1234.56
}
```

**Example:**
```bash
# Text query
curl -X POST http://localhost:8000/query/ \
  -F "question=What is HippoRAG?"

# With audio file
curl -X POST http://localhost:8000/query/ \
  -F "question=Transcribe this audio" \
  -F "audio_file=@recording.wav"

# With image file
curl -X POST http://localhost:8000/query/ \
  -F "question=What is in this image?" \
  -F "image_file=@photo.jpg"

# With auth
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer your-api-key" \
  -F "question=What is HippoRAG?"
```

---

### Query (Streaming)

```
POST /query/stream
```

**Request:** Same as `/query/` (multipart/form-data)

**Response:** `text/event-stream` (SSE)

**SSE Event Types:**

| Event | Description |
|-------|-------------|
| `node_start` | Agent node started processing |
| `node_done` | Agent node completed with state |
| `complete` | Final answer with all metadata |
| `error` | An error occurred |

**Example SSE Stream:**
```
event: node_start
data: {"node": "planner", "label": "Planning", "elapsed_ms": 0}

event: node_done
data: {"node": "planner", "label": "Planning", "modality": "text", "intent": "factual_qa", "elapsed_ms": 120}

event: node_start
data: {"node": "router", "label": "Routing", "elapsed_ms": 120}

event: node_done
data: {"node": "router", "label": "Routing", "strategy": "hipporag", "elapsed_ms": 180}

event: node_start
data: {"node": "rag", "label": "Retrieving context", "elapsed_ms": 180}

event: node_done
data: {"node": "rag", "label": "Retrieving context", "chunks_found": 5, "elapsed_ms": 890}

event: node_start
data: {"node": "generator", "label": "Generating answer", "elapsed_ms": 890}

event: node_done
data: {"node": "generator", "label": "Generating answer", "answer_preview": "HippoRAG is a retrieval-augmented generation...", "elapsed_ms": 2100}

event: node_start
data: {"node": "critic", "label": "Evaluating quality", "elapsed_ms": 2100}

event: node_done
data: {"node": "critic", "label": "Evaluating quality", "passed": true, "score_count": 6, "elapsed_ms": 2800}

event: complete
data: {"answer": "HippoRAG is a retrieval-augmented generation framework...", "modality": "text", "confidence": 0.85, "retry_count": 0, "elapsed_ms": 3200}
```

**Example:**
```bash
curl -X POST http://localhost:8000/query/stream \
  -F "question=What is HippoRAG?" \
  -N
```

---

### Ingest

```
POST /ingest/
```

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file (PDF, CSV, TXT) |
| `source_name` | string | No | Source name override |
| `collection` | string | No | Target collection (default: "polymind") |

**Response:**
```json
{
  "status": "completed",
  "chunks_created": 42,
  "source": "document.pdf",
  "processing_time_ms": 2345.67
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@document.pdf" \
  -F "source_name=my_document"
```

---

### Eval

```
POST /eval/
```

**Request:** `application/json`

```json
{
  "limit": 10,
  "run_full": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `limit` | int | No | Max cases to evaluate |
| `run_full` | bool | No | Run full benchmark |

**Response:**
```json
{
  "status": "completed",
  "total": 10,
  "passed": 8,
  "failed": 2,
  "pass_rate": 0.8,
  "averages": {
    "faithfulness": 0.82,
    "answer_relevancy": 0.79
  },
  "thresholds": {
    "faithfulness": 0.72,
    "answer_relevancy": 0.75
  },
  "processing_time_ms": 12345.67
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/eval/ \
  -H "Content-Type: application/json" \
  -d '{"limit": 5}'
```

---

### Submit Feedback

```
POST /feedback/
```

**Request:** `application/json`

```json
{
  "query_id": "abc-123",
  "query": "What is HippoRAG?",
  "rating": 5,
  "feedback": "thumbs_up",
  "intent": "factual_qa",
  "strategy": "hipporag",
  "modality": "text"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | Yes | Unique query identifier |
| `query` | string | Yes | Original user query |
| `rating` | int | Yes | Rating 1-5 (5=best) |
| `feedback` | string | No | Text feedback or thumbs_up/thumbs_down |
| `intent` | string | No | Query intent category |
| `strategy` | string | No | Retrieval strategy used |
| `modality` | string | No | Input modality |

**Response:**
```json
{
  "status": "recorded",
  "stats": {
    "total": 42,
    "average_rating": 4.2
  },
  "processing_time_ms": 12.34
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "abc-123",
    "query": "What is HippoRAG?",
    "rating": 5,
    "feedback": "thumbs_up"
  }'
```

---

### Get Feedback Stats

```
GET /feedback/stats
```

**Response:**
```json
{
  "total": 42,
  "average_rating": 4.2,
  "by_intent": {
    "factual_qa": { "count": 20, "avg_rating": 4.5 },
    "summarization": { "count": 15, "avg_rating": 4.0 }
  },
  "by_strategy": {
    "hipporag": { "count": 25, "avg_rating": 4.3 },
    "standard": { "count": 17, "avg_rating": 4.1 }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/feedback/stats
```

---

### Prometheus Metrics

```
GET /metrics
```

Returns Prometheus-format metrics:
- `polymind_queries_total` — Query count by modality
- `polymind_query_latency_seconds` — Latency histogram
- `polymind_faithfulness_score` — Faithfulness distribution
- `polymind_active_requests` — Active request gauge

**Example:**
```bash
curl http://localhost:8000/metrics
```

---

### Swagger UI

```
GET /docs
```

Interactive API documentation at `http://localhost:8000/docs`

### ReDoc

```
GET /redoc
```

Alternative API documentation at `http://localhost:8000/redoc`
