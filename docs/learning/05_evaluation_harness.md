# Evaluation Harness

## Why Evaluate?

In production AI, **"it works" is not enough**. You need to prove:
- Answers are grounded in evidence (faithfulness)
- Answers address the question (relevancy)
- Answers don't fabricate information (hallucination)

## PolyMind's Approach

### 1. DeepEval (LLM-as-Judge)
Uses a fast LLM (Groq Llama 3.1 8B) to evaluate answers:
- Faithfulness: Is the answer supported by context?
- Answer Relevancy: Does it address the question?
- Hallucination Rate: How much is fabricated?

### 2. RAGAS (Benchmark)
Runs evaluation against a 20-case benchmark:
- Each case has query, ground_truth, and expected modality
- Measures faithfulness, relevancy, context precision/recall
- Tracks scores over time

### 3. CI Gate
Every PR is tested:
```yaml
# .github/workflows/eval_gate.yml
- name: Run eval harness
  run: pytest tests/eval/ --ragas-threshold=0.72
```
If faithfulness drops below 0.72, the PR is **blocked**.

## Metrics Explained

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| Faithfulness | Answer grounded in context? | ≥ 0.72 |
| Answer Relevancy | Answer addresses question? | ≥ 0.75 |
| Hallucination Rate | How much is fabricated? | ≤ 0.25 |
| Context Precision | Retrieved context relevant? | ≥ 0.70 |
| Context Recall | All info retrieved? | ≥ 0.70 |

## Running Evaluations

```bash
# Quick eval (10 cases)
make eval

# Full eval (20 cases)
poetry run python scripts/run_eval.py

# CI eval (runs on every PR)
# Automatically triggered by GitHub Actions
```

## Interpreting Results

- **Faithfulness drop** → Check if context is relevant
- **Relevancy drop** → Check if generator is off-topic
- **High hallucination** → Check if context is insufficient
- **Low precision** → Check retrieval quality
