"""LLM Factory — creates configured LLM clients for PolyMind.

Supports Groq (primary, ultra-fast), OpenAI, and local models.
Uses langchain-openai for LangGraph/LangChain compatibility.
"""

from __future__ import annotations

import os

import structlog
from langchain_core.language_models import BaseChatModel  # noqa: TCH002

logger = structlog.get_logger()

# ── Groq Models (Ultra-Fast Inference) ──────────────────
GROQ_MODELS = {
    "reasoning": "llama-3.3-70b-versatile",  # 280 t/s — Generator, Critic
    "fast": "llama-3.1-8b-instant",  # 560 t/s — Planner, Router
    "qwen": "qwen/qwen3-32b",  # 400 t/s — Alternative reasoning
    "code": "qwen/qwen3.6-27b",  # 500 t/s — Code-aware tasks
}

# ── Default Config ──────────────────────────────────────
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = GROQ_MODELS["reasoning"]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMFactory:
    """Factory for creating LLM clients with provider routing.

    Routes different tasks to different models based on complexity:
    - Planning/Classification → fast model (Llama 3.1 8B)
    - Generation/Critique → reasoning model (Llama 3.3 70B)
    - Fallback → configurable default
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize LLM factory.

        Args:
            provider: LLM provider ("groq", "openai").
            api_key: API key (falls back to env var).
        """
        self._provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        self._api_key = api_key or self._get_api_key()
        self._clients: dict[str, BaseChatModel] = {}

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if self._provider == "groq":
            key = os.getenv("GROQ_API_KEY", "")
        else:
            key = os.getenv("OPENAI_API_KEY", "")
        return key

    def get_llm(self, tier: str = "reasoning") -> BaseChatModel:
        """Get an LLM client for the specified tier.

        Args:
            tier: Model tier ("fast", "reasoning", "qwen", "code").

        Returns:
            Configured BaseChatModel instance.
        """
        cache_key = f"{self._provider}:{tier}"
        if cache_key in self._clients:
            return self._clients[cache_key]

        if self._provider == "groq":
            llm = self._create_groq_llm(tier)
        else:
            llm = self._create_openai_llm(tier)

        self._clients[cache_key] = llm
        return llm

    def _create_groq_llm(self, tier: str) -> BaseChatModel:
        """Create a Groq-backed LLM via langchain-openai."""
        from langchain_openai import ChatOpenAI

        model_id = GROQ_MODELS.get(tier, DEFAULT_MODEL)

        llm = ChatOpenAI(
            model=model_id,
            api_key=self._api_key,
            base_url=GROQ_BASE_URL,
            temperature=0.0 if tier == "fast" else 0.1,
            max_tokens=4096,
            timeout=30,
        )

        logger.info("llm.groq.created", model=model_id, tier=tier)
        return llm

    def _create_openai_llm(self, tier: str) -> BaseChatModel:
        """Create an OpenAI-backed LLM."""
        from langchain_openai import ChatOpenAI

        model_map = {
            "fast": "gpt-4o-mini",
            "reasoning": "gpt-4o",
            "qwen": "gpt-4o",
            "code": "gpt-4o",
        }
        model_id = model_map.get(tier, "gpt-4o")

        llm = ChatOpenAI(
            model=model_id,
            api_key=self._api_key,
            temperature=0.0 if tier == "fast" else 0.1,
            max_tokens=4096,
        )

        logger.info("llm.openai.created", model=model_id, tier=tier)
        return llm

    @property
    def available_tiers(self) -> list[str]:
        """List available model tiers."""
        return list(GROQ_MODELS.keys())

    @property
    def provider(self) -> str:
        """Current provider name."""
        return self._provider
