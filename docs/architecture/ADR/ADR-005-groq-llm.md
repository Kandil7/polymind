# ADR-005: Groq for LLM Inference

**Date:** 2026-06-19
**Status:** Accepted

## Context

PolyMind needs LLM inference for Generator, Critic, and planning nodes. Local GPU inference is expensive and slow to set up.

## Decision

Use **Groq** as the primary LLM provider with OpenAI-compatible API.

## Rationale

- **280-560 tokens/second** — fastest inference available
- **Free tier** — generous rate limits (250K TPM, 1K RPM)
- **OpenAI-compatible** — works with langchain-openai
- **No GPU required** — managed service
- **Multiple models** — Llama 3.3 70B (reasoning), Llama 3.1 8B (fast)

## Alternatives Considered

| Option | Pros | Cons | Reason Rejected |
|--------|------|------|-----------------|
| OpenAI API | High quality | Expensive, rate limits | Cost |
| Local GPU (Ollama) | Free, private | Slow setup, limited models | Complexity |
| Anthropic Claude | High quality | Expensive, not OpenAI-compat | Cost + API |
| Groq | Fast, free, compatible | Newer service | Best balance |

## Consequences

- Requires GROQ_API_KEY environment variable
- Rate limits may affect high-throughput scenarios
- Model availability depends on Groq's offerings
