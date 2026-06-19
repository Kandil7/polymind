# Phase 2 — Specialist Model Wrappers

**Goal:** Implement 5 HuggingFace model wrappers as infrastructure implementations of the `ISpecialist` interface, each with 3+ passing tests.

## Files Created

| File | Purpose | Layer |
|------|---------|-------|
| `src/polymind/infrastructure/specialists/__init__.py` | Package exports | Infrastructure |
| `src/polymind/infrastructure/specialists/asr_wrapper.py` | Whisper ASR wrapper | Infrastructure |
| `src/polymind/infrastructure/specialists/vqa_wrapper.py` | BLIP VQA wrapper | Infrastructure |
| `src/polymind/infrastructure/specialists/docqa_wrapper.py` | LayoutLM DocQA wrapper | Infrastructure |
| `src/polymind/infrastructure/specialists/tableqa_wrapper.py` | TAPAS TableQA wrapper | Infrastructure |
| `src/polymind/infrastructure/specialists/summarizer_wrapper.py` | BART Summarizer wrapper | Infrastructure |
| `tests/unit/test_asr_wrapper.py` | ASR wrapper tests (6 tests) | Test |
| `tests/unit/test_vqa_wrapper.py` | VQA wrapper tests (7 tests) | Test |
| `tests/unit/test_docqa_wrapper.py` | DocQA wrapper tests (7 tests) | Test |
| `tests/unit/test_tableqa_wrapper.py` | TableQA wrapper tests (7 tests) | Test |
| `tests/unit/test_summarizer_wrapper.py` | Summarizer wrapper tests (6 tests) | Test |
| `tests/unit/test_specialist_registry.py` | Registry structure tests (4 tests) | Test |

## Design Decisions

1. **Lazy loading**: Models are loaded on first `process()` call, not at `__init__`. This prevents GPU OOM at import time.
2. **Interface compliance**: All 5 wrappers implement `ISpecialist` ABC with `name` property and `process()` method.
3. **Error handling**: Each wrapper raises specific exceptions (`RuntimeError`, `FileNotFoundError`, `ValueError`) with descriptive messages.
4. **Test strategy**: Tests use `__new__` to create wrappers without calling `__init__`, avoiding model downloads. Mock pipelines test logic without GPU.
5. **structlog**: All wrappers use structured logging for observability.

## Models Used

| Specialist | HF Model | Task |
|------------|----------|------|
| ASR | openai/whisper-large-v3 | automatic-speech-recognition |
| VQA | Salesforce/blip-vqa-base | visual-question-answering |
| DocQA | impira/layoutlm-document-qa | document-question-answering |
| TableQA | google/tapas-base-finetuned-wtq | table-question-answering |
| Summarizer | facebook/bart-large-cnn | summarization |

## Test Results

```
61 passed in 4.05s
```

## What Phase 3 Builds On Top

Phase 3 adds the HippoRAG retriever with Knowledge Graph + Personalized PageRank for multi-hop retrieval.
