"""Infrastructure specialist wrappers — HuggingFace model integrations."""

from polymind.infrastructure.specialists.asr_wrapper import ASRWrapper
from polymind.infrastructure.specialists.docqa_wrapper import DocQAWrapper
from polymind.infrastructure.specialists.summarizer_wrapper import SummarizerWrapper
from polymind.infrastructure.specialists.tableqa_wrapper import TableQAWrapper
from polymind.infrastructure.specialists.vqa_wrapper import VQAWrapper

__all__ = [
    "ASRWrapper",
    "DocQAWrapper",
    "SummarizerWrapper",
    "TableQAWrapper",
    "VQAWrapper",
]
