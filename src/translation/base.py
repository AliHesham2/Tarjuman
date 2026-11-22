"""TranslationProvider ABC + request/result dataclasses.

The pluggable heart of the app: any translation backend implements
this interface and slots into the priority chain without touching
the rest of the application.

Input contract: ``TranslationRequest.text`` must already be sanitized
by :class:`src.translation.engine.TranslationEngine`. Providers are
internal components and do not re-run the security sanitizer.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranslationRequest:
    text: str                        # النص المراد ترجمته
    source_lang: str = "en"
    target_lang: str = "ar"
    context: str = ""                # آخر N جمل كـ context
    is_partial: bool = False         # True = hypothesis، False = confirmed
    fast_mode: bool = False          # True = latency priority، False = quality
    # "__T0__" -> "API": terms that must come back untouched (see protected_terms).
    protected_terms: dict = field(default_factory=dict)


@dataclass
class TranslationResult:
    translated: str                  # النص المترجم
    provider: str                    # من ترجم (groq / groq_2 / deepl / ...)
    latency_ms: float                # وقت الترجمة بالـ ms
    confidence: Optional[float] = None
    # Real tokens the provider billed for this call (prompt + completion), when
    # its response says so. The quota tracker prefers it to its own estimate.
    tokens: Optional[int] = None


def module_available(module: str) -> bool:
    """True if an optional dependency is importable, without importing it.

    ``find_spec`` can raise for malformed dotted names or a missing parent
    package; a provider must be skipped rather than crash the registry.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


class TranslationProvider(ABC):
    """Base class for every translation backend (cloud or local)."""

    name: str = "provider"
    supports_context: bool = False   # can use the last-N-sentences context
    is_local: bool = False           # offline = True (no quota, private)

    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate one request. Must not raise on provider-side errors —
        return/raise is handled by the engine; timeout is enforced by it."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Key present / settings valid? (cheap, no network)"""

    @abstractmethod
    def is_available(self) -> bool:
        """Service reachable? (may do a network check for local servers)"""

    def warmup(self) -> None:
        """Optional: pre-load models / open connections."""
