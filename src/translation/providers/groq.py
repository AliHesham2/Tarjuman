"""Groq — Llama 3.1 via the OpenAI-compatible API (free: 14,400 req/day)."""

from __future__ import annotations

from src.config.providers import TranslationProviderConfig
from src.translation.providers.openai_compat import OpenAICompatProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAICompatProvider):
    is_local = False

    def __init__(self, config: TranslationProviderConfig):
        config.base_url = GROQ_BASE_URL
        super().__init__(config)
        # "groq", or "groq_2" / "groq_3" for extra accounts: each one is its own
        # provider with its own quota counter and circuit breaker.
        self.name = config.name or "groq"
