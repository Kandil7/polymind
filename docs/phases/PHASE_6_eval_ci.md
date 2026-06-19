# Phase 6 — Eval Harness & CI

**Goal:** Implement RAGAS evaluation with CI gate that blocks PRs on quality regressions.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `infrastructure/eval/deepeval_critic.py` | DeepEval LLM-as-Judge | Infrastructure |
| `infrastructure/eval/ragas_runner.py` | Benchmark evaluation runner | Infrastructure |
| `tests/eval/benchmark_v1.json` | 20-case benchmark dataset | Test |
| `tests/eval/test_harness.py` | Pytest eval suite | Test |
| `.github/workflows/ci.yml` | Lint + test CI | CI |
| `.github/workflows/eval_gate.yml` | Eval gate on PRs to main | CI |
| `scripts/run_eval.py` | Local eval runner | Script |

## Thresholds (enforced in CI)

| Metric | Threshold |
|--------|-----------|
| Faithfulness | ≥ 0.72 |
| Answer Relevancy | ≥ 0.75 |
| Hallucination Rate | ≤ 0.25 |

## Test Results

```
12 new eval tests (168 total), all passing
```
