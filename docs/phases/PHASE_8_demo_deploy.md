# Phase 8 — Demo & Deploy

**Goal:** Create interactive demo, deployment config, CI/CD pipelines, and comprehensive documentation.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `app.py` | Streamlit demo app | Demo |
| `infra/modal_deploy.py` | GPU inference endpoints (Modal) | Deploy |
| `infra/docker-compose.yml` | Multi-service orchestration | Deploy |
| `infra/Dockerfile` | App container | Deploy |
| `infra/prometheus.yml` | Prometheus scrape config | Deploy |
| `.github/workflows/ci.yml` | Lint + test + eval gate | CI/CD |
| `.github/workflows/eval_gate.yml` | Eval threshold enforcement | CI/CD |
| `.github/workflows/deploy.yml` | Build + deploy to staging/prod | CI/CD |
| `README.md` | Project documentation | Docs |
| `CONTRIBUTING.md` | Contribution guidelines | Docs |
| `LICENSE` | MIT License | Docs |

## Streamlit Demo Features

The Streamlit app (`app.py`) provides an interactive interface:

- **Multimodal Upload** — Audio files (ASR), images (VQA), documents (PDF/CSV)
- **Real-time Chat** — Conversational query interface with streaming support
- **Critic Scores** — Visual display of faithfulness, relevancy, hallucination metrics
- **Source Citations** — Linked sources from HippoRAG retrieval
- **Confidence Metrics** — Color-coded confidence indicators
- **System Health** — Live health status from `/health` endpoint
- **Feedback Collection** — Thumbs up/down with optional comments

## Modal GPU Deployment

Serverless GPU inference via Modal (`infra/modal_deploy.py`):

| Function | Model | GPU | Timeout |
|----------|-------|-----|---------|
| `run_asr` | `openai/whisper-large-v3` | T4 | 300s |
| `run_vqa` | `Salesforce/blip-vqa-base` | T4 | 300s |
| `run_embedding` | `BAAI/bge-m3` | T4 | 300s |
| `run_clip_image` | `openai/clip-vit-base-patch32` | T4 | 300s |

**Deploy:**
```bash
modal deploy infra/modal_deploy.py
```

**Test locally:**
```bash
modal run infra/modal_deploy.py::run_asr --audio-bytes-file test.wav
```

## Docker Compose Services

```bash
docker compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| `api` | 8000 | PolyMind FastAPI app |
| `qdrant` | 6333, 6334 | Vector database (REST + gRPC) |
| `prometheus` | 9090 | Metrics collection (30d retention) |
| `grafana` | 3000 | Dashboards (admin/admin) |

**Volumes:** `qdrant_data`, `prometheus_data`, `grafana_data`

**Health checks:** API (curl `/health`), Qdrant (curl `/healthz`)

## CI/CD Workflows

### ci.yml — Lint & Test
- **Trigger:** Push to `main`/`develop`, PRs to `main`/`develop`
- **Steps:** Ruff lint → mypy type check → unit tests
- **Eval gate:** On PRs, runs eval harness subset and uploads report

### eval_gate.yml — Eval Threshold Enforcement
- **Trigger:** PRs to `main`
- **Steps:** Seed Qdrant → Run full eval harness → Check thresholds
- **Behavior:** Fails PR if any eval test fails

### deploy.yml — Build & Deploy
- **Trigger:** Push to `main`, tags `v*`, manual dispatch
- **Jobs:**
  1. **test** — Lint + unit tests (skip heavy tests)
  2. **build** — Docker build + push to GHCR
  3. **deploy-staging** — Modal deploy (on `main`)
  4. **deploy-production** — Modal deploy + GitHub Release (on `v*` tags)

## Deployment Options

1. **Local Development**
   ```bash
   docker compose up -d    # Start infra
   make dev                # Start app with hot reload
   ```

2. **GPU Inference (Modal)**
   ```bash
   modal deploy infra/modal_deploy.py
   ```

3. **Full Stack (Docker)**
   ```bash
   docker compose up -d --build
   ```

4. **Cloud (GitHub Actions)**
   - Push to `main` → staging auto-deploy
   - Push `v*` tag → production deploy + release
