# Phase 8 — Demo & Deploy

**Goal:** Create interactive demo, deployment config, and comprehensive documentation.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `app.py` | Streamlit demo app | Demo |
| `infra/modal_deploy.py` | GPU inference endpoints | Deploy |
| `README.md` | Project documentation | Docs |
| `CONTRIBUTING.md` | Contribution guidelines | Docs |
| `LICENSE` | MIT License | Docs |

## Demo Features

- Multimodal file upload (audio, image, document)
- Real-time chat interface
- Critic scores display
- Source citations
- Confidence metrics
- System health indicator

## Deployment Options

1. **Local:** Docker Compose + `make dev`
2. **GPU:** Modal T4 for ASR/VQA inference
3. **Cloud:** Any Docker-compatible platform
