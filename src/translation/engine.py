"""TranslationEngine — orchestrates providers, quotas, protection, context.

Flow per request:
  sanitize → protect terms → pick provider (quota-aware) → translate
  → restore terms → record quota → return result.

Falls through the priority chain on failure, ending at the local
providers, which are always available.

Two deliberate policies from the Phase 3 audit:
  * Sanitization happens once here, before any provider/breaker is touched.
    A too-long or injection-shaped utterance must not trip circuit breakers.
  * Partial (hypothesis) translations never consume tracked quota.  They may
    use request-based providers, but providers with a monthly character
    budget (DeepL) are reserved for final translations.  A 429 still marks
    the provider exhausted and falls back immediately.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field

from src.config.settings import EngineConfig
from src.resilience.circuit_breaker import CircuitBreaker
from src.security.input_sanitizer import MAX_CHARS, SanitizationError, sanitize
from src.translation.base import TranslationProvider, TranslationRequest, TranslationResult
from src.translation.protected_terms import protect, restore
from src.translation.quota_router import QuotaAwareRouter

logger = logging.getLogger(__name__)

# Tokens a request costs beyond the sentence itself: system prompt + framing.
PROMPT_OVERHEAD_TOKENS = 120
# Start opening the connection to the next provider once the current one has
# used this much of its limits, so the hand-over costs no handshake.
PREWARM_AT = 0.8


@dataclass
class EngineMetrics:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latency_by_provider: dict = field(default_factory=dict)
    failures_by_provider: dict = field(default_factory=dict)

    def record(self, result: TranslationResult) -> None:
        stats = self.latency_by_provider.setdefault(
            result.provider, {"calls": 0, "total_ms": 0.0}
        )
        stats["calls"] += 1
        stats["total_ms"] += result.latency_ms

    def record_failure(self, provider: str) -> None:
        self.failures_by_provider[provider] = (
            self.failures_by_provider.get(provider, 0) + 1
        )

    def success_rate(self, provider: str | None = None) -> float:
        """Success ratio overall, or for one provider (0.0 when untried)."""
        if provider is not None:
            calls = self.latency_by_provider.get(provider, {}).get("calls", 0)
            failures = self.failures_by_provider.get(provider, 0)
        else:
            calls = self.successful
            failures = self.failed
        total = calls + failures
        return calls / total if total else 0.0


class TranslationEngine:
    def __init__(self, config: EngineConfig, providers: list[TranslationProvider]):
        self.config = config
        self.providers = providers
        # groq_2 / groq_3 are extra accounts of groq: same limits, own counters.
        limits = dict(config.quotas.limits or {})
        for provider in providers:
            if provider.name in limits:
                continue
            base = re.sub(r"_\d+$", "", provider.name)
            if base != provider.name and base in limits:
                limits[provider.name] = dict(limits[base] or {})
        self.router = QuotaAwareRouter(
            limits,
            state_path=config.quota_state_path,
            headroom=config.translation.quota_headroom,
        )
        self.breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(p.name) for p in providers
        }
        self.metrics = EngineMetrics()
        self._metrics_lock = threading.Lock()
        self._context: deque[str] = deque(
            maxlen=config.translation.context_window_sentences
        )
        self._context_lock = threading.Lock()
        # One hung provider must not starve the rest of the fallback chain.
        # Workers are per-provider (plus one spare); requests are serialized
        # by the pipeline's single translation thread.
        self._pool = ThreadPoolExecutor(
            max_workers=max(2, len(providers) + 1),
            thread_name_prefix="translator",
        )
        self._warmed: set[str] = set()
        self._warm_lock = threading.Lock()
        if providers:
            self._warm(providers[0])

    # -- public API ----------------------------------------------------------

    def translate_confirmed(self, text: str) -> TranslationResult | None:
        """Final translation of a confirmed utterance (quality mode)."""
        return self._translate(text, is_partial=False)

    def translate_partial(self, text: str) -> TranslationResult | None:
        """Quick translation of a hypothesis (latency mode)."""
        if not self.config.translation.partial_translation:
            return None
        return self._translate(text, is_partial=True)

    def remember_context(self, source_text: str, translated_text: str) -> None:
        """Add a finished pair to the rolling context window (bounded)."""
        if not source_text.strip() or not translated_text.strip():
            return
        try:
            safe_source = sanitize(source_text, max_chars=self._max_request_chars())
        except SanitizationError:
            logger.debug("Skipping context for rejected input")
            return
        limit = max(1, int(self.config.translation.context_max_chars))
        pair = f"{safe_source} => {translated_text}"[:limit]
        with self._context_lock:
            self._context.append(pair)

    def clear_context(self) -> None:
        with self._context_lock:
            self._context.clear()

    def current_provider_name(self) -> str | None:
        """The provider that will handle the next request (tray badge)."""
        for p in self.providers:
            if self.breakers[p.name].allow_request() and self.router.has_quota(p.name):
                return p.name
        return None

    # -- connection warm-up ---------------------------------------------------

    def _warm(self, provider: TranslationProvider) -> None:
        """Open the provider's connection in the background (once)."""
        with self._warm_lock:
            if provider.name in self._warmed:
                return
            self._warmed.add(provider.name)

        def run() -> None:
            try:
                provider.warmup()
            except Exception:
                logger.debug("Warm-up failed for %s", provider.name, exc_info=True)

        try:
            self._pool.submit(run)
        except RuntimeError:  # pool already shut down
            pass

    def _prewarm_next(self, name: str) -> None:
        """Warm the provider after ``name`` when ``name`` is nearly used up."""
        if self.router.usage_fraction(name) < PREWARM_AT:
            return
        names = [p.name for p in self.providers]
        if name not in names:
            return
        for candidate in self.providers[names.index(name) + 1:]:
            if candidate.is_local:
                continue
            self._warm(candidate)
            return

    def shutdown(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python < 3.9
            self._pool.shutdown(wait=False)

    # -- internals --------------------------------------------------------------

    def _max_request_chars(self) -> int:
        """Input cap, never above the sanitizer's own hard limit."""
        return max(1, min(int(self.config.translation.max_request_chars), MAX_CHARS))

    def _build_request(
        self, text: str, is_partial: bool, protected_terms: dict | None = None
    ) -> TranslationRequest:
        with self._context_lock:
            context = " ".join(self._context)
        limit = max(1, int(self.config.translation.context_max_chars))
        if len(context) > limit:
            context = context[-limit:]
        return TranslationRequest(
            text=text,
            source_lang=self.config.translation.source_lang,
            target_lang=self.config.translation.target_lang,
            context=context,
            is_partial=is_partial,
            fast_mode=is_partial,
            protected_terms=dict(protected_terms or {}),
        )

    def _translate(self, text: str, is_partial: bool) -> TranslationResult | None:
        with self._metrics_lock:
            self.metrics.total_requests += 1

        # Security audit H2: validate once, before any provider.  Sanitization
        # failure is a bad input, not a provider outage, so no breaker opens.
        try:
            clean = sanitize(text, max_chars=self._max_request_chars())
        except SanitizationError as exc:
            logger.warning("Rejected translation input: %s", exc)
            with self._metrics_lock:
                self.metrics.failed += 1
            return None

        protected = protect(clean)
        request = self._build_request(
            protected.safe_text, is_partial, protected.replacements
        )

        for provider in self.providers:
            name = provider.name
            breaker = self.breakers[name]

            if not breaker.allow_request():
                continue

            # Partial hypotheses are free-ish: never charge a monthly character
            # budget for text that may be replaced a second later.
            unit = self.router.unit(name)
            if is_partial and unit == "chars":
                continue

            cost = self.router.cost_for(name, protected.safe_text)
            # Real requests carry the system prompt and the context too, not
            # just the sentence: counting only the text under-counts 3-5x.
            tokens_in = (
                self.router.estimate_tokens(protected.safe_text)
                + (self.router.estimate_tokens(request.context) if request.context else 0)
                + PROMPT_OVERHEAD_TOKENS
            )

            if is_partial:
                # Drafts may only use a provider that is still comfortably
                # inside its limits; finals get the rest.
                if self.router.usage_fraction(name) >= self.config.translation.partial_max_usage:
                    continue
                if not self.router.has_quota(name, cost=cost, tokens=tokens_in):
                    continue
                reserved = False
            else:
                if not self.router.try_reserve(name, cost=cost, tokens=tokens_in):
                    continue
                reserved = True

            try:
                result = self._call_with_timeout(provider, request)
            except FuturesTimeoutError:
                # The request may already have reached (and been billed by) the
                # provider; keep the reservation rather than undercounting.
                logger.warning(
                    "Provider %s timed out; falling back", name
                )
                breaker.record_failure()
                self._record_provider_failure(name)
                continue
            except Exception as exc:
                quota_error = self._is_quota_error(exc)
                if reserved:
                    self.router.release(name, cost=cost, tokens=tokens_in)

                if quota_error:
                    # 429 is not a service outage: mark exhausted and switch
                    # immediately.  Untracked providers still need the breaker
                    # so a permanently rate-limited paid provider is not
                    # hammered on every sentence.
                    self.router.cool_down(name, self._retry_after_s(exc))
                    if name not in self.router.usage:
                        breaker.record_failure()
                else:
                    breaker.record_failure()

                self._record_provider_failure(name)
                logger.warning("Provider %s failed: %s", name, exc)
                continue

            translated = (getattr(result, "translated", "") or "").strip()
            if not translated:
                if reserved:
                    self.router.release(name, cost=cost, tokens=tokens_in)
                breaker.record_failure()
                self._record_provider_failure(name)
                logger.warning("Provider %s returned an empty translation", name)
                continue

            breaker.record_success()
            actual = getattr(result, "tokens", None)
            if reserved:
                # Replace the reservation's estimate by what the call really cost.
                if actual:
                    self.router.adjust_tokens(name, int(actual) - tokens_in)
                else:
                    self.router.adjust_tokens(
                        name, self.router.estimate_tokens(translated)
                    )
            else:
                # A draft is real traffic too: count it against the limits.
                self.router.record(
                    name, cost=cost,
                    tokens=int(actual) if actual else tokens_in
                    + self.router.estimate_tokens(translated),
                )
            self._prewarm_next(name)

            with self._metrics_lock:
                self.metrics.successful += 1
                self.metrics.record(result)

            result.translated = restore(translated, protected.replacements)
            return result

        with self._metrics_lock:
            self.metrics.failed += 1
        logger.error("All providers failed for request (%d chars)", len(text))
        return None

    def _record_provider_failure(self, name: str) -> None:
        with self._metrics_lock:
            self.metrics.record_failure(name)

    def _call_with_timeout(self, provider, request) -> TranslationResult:
        """Run one translation with the provider's own timeout enforced."""
        timeout_s = None
        cfg = getattr(provider, "cfg", None)
        if cfg is not None:
            try:
                timeout_s = float(cfg.timeout_s)
            except (TypeError, ValueError, AttributeError):
                timeout_s = None
        if timeout_s is None:
            timeout_s = 10.0 if provider.is_local else 3.0
        if provider.is_local:
            # An offline provider is CPU-bound and may need more than the
            # cloud default; never cut it off artificially.
            timeout_s = max(timeout_s, 10.0)
        timeout_s = max(0.05, timeout_s)

        future = self._pool.submit(provider.translate, request)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            raise

    @staticmethod
    def _retry_after_s(exc: Exception) -> float:
        """Seconds to cool a provider down after a 429 (5..300, default 60)."""
        try:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers is None:
                return 60.0
            raw = headers.get("retry-after")
            if raw is None:
                return 60.0
            seconds = float(str(raw).strip())
        except Exception:
            return 60.0
        if seconds != seconds:  # NaN
            return 60.0
        return min(300.0, max(5.0, seconds))

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Recognise 429/quota errors across all provider SDK exception types."""
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        if status == 429:
            return True

        name = type(exc).__name__.lower()
        if any(token in name for token in
               ("ratelimit", "quota", "resourceexhausted", "toomanyrequests")):
            return True

        text = str(exc).lower()
        return (
            re.search(r"\b429\b", text)
            or "quota" in text
            or "rate limit" in text
            or "too many requests" in text
        )
