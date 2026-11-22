"""ProviderRegistry — creates providers from config and builds the priority chain.

Auto-Skip Logic: a provider joins the chain only if BOTH
  1. is_configured()  — key present / not explicitly disabled
  2. is_available()   — service reachable (for local: server is up)
hold. Skipped providers are logged clearly.

Optional dependencies are imported one provider at a time.  Before Phase 3
a single missing package (for example a top-level ``httpx`` import inside
a provider) prevented *every* provider from registering.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

from src.config.providers import (
    GROQ_EXTRA_ACCOUNTS, TranslationProviderConfig, load_provider_config,
)
from src.translation.base import TranslationProvider

logger = logging.getLogger(__name__)

# name -> lazy factory(config) -> provider instance
PROVIDER_FACTORIES: dict[str, Callable[[TranslationProviderConfig], TranslationProvider]] = {}

# name, module, class — imported lazily and independently.
_PROVIDER_IMPORTS = [
    ("groq", "src.translation.providers.groq", "GroqProvider"),
    ("deepl", "src.translation.providers.deepl", "DeepLProvider"),
]


_PROVIDER_IMPORTS += [
    (name, "src.translation.providers.groq", "GroqProvider")
    for name in GROQ_EXTRA_ACCOUNTS
]

# Extra accounts of one service, tried right after the first one. Each account
# has its own key and therefore its own free quota: when "groq" reaches its
# limit the engine moves on to "groq_2", "groq_3", ... then the next provider.
KEY_ALIASES = {"groq": list(GROQ_EXTRA_ACCOUNTS)}


def _register(name: str):
    def deco(factory):
        PROVIDER_FACTORIES[name] = factory
        return factory
    return deco


def _default_factories() -> None:
    for name, module_name, class_name in _PROVIDER_IMPORTS:
        if name in PROVIDER_FACTORIES:
            continue
        try:
            module = importlib.import_module(module_name)
            PROVIDER_FACTORIES[name] = getattr(module, class_name)
        except Exception as exc:
            # One broken optional dependency must not take down the registry.
            logger.warning("Provider %s unavailable (%s)", name, exc)
            logger.debug("Provider import failure", exc_info=True)


class ProviderRegistry:
    def __init__(self):
        _default_factories()

    def is_configured(self, name: str) -> bool:
        """Cheap static check — is the API key present in the environment?"""
        try:
            provider = self._create(name)
            return bool(provider.is_configured())
        except Exception as exc:
            logger.warning("is_configured() failed for %s (%s)", name, exc)
            return False

    def _create(self, name: str, entry: dict | None = None) -> TranslationProvider:
        factory = PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown provider: {name}")
        config = load_provider_config(name, entry)
        return factory(config)

    def create_from_config(
        self, name: str, entry: dict | None = None
    ) -> TranslationProvider:
        """Public factory used by the app/tests to build one provider."""
        return self._create(name, entry)

    @staticmethod
    def with_key_aliases(provider_entries: list[dict]) -> list[dict]:
        """Add groq_2 .. groq_10 right after groq whenever their keys exist.

        Putting GROQ_API_KEY_2 in .env (or saving it in Settings) is enough;
        the provider list in config.yaml / settings.yaml need not mention it.
        """
        result = list(provider_entries or [])
        for base, aliases in KEY_ALIASES.items():
            base_entry = next(
                (e for e in result
                 if isinstance(e, dict) and e.get("name") == base),
                None,
            )
            if base_entry is None:
                continue
            present = {e.get("name") for e in result if isinstance(e, dict)}
            insert_at = result.index(base_entry) + 1
            for alias in aliases:
                if alias in present:
                    insert_at = max(
                        insert_at,
                        1 + next(
                            i for i, e in enumerate(result)
                            if isinstance(e, dict) and e.get("name") == alias
                        ),
                    )
                    continue
                if not load_provider_config(alias).api_key:
                    continue
                result.insert(insert_at, dict(base_entry, name=alias))
                insert_at += 1
        return result

    def build_chain(
        self,
        provider_entries: list[dict],
        confidential_mode: bool = False,
    ) -> tuple[list[TranslationProvider], list[str]]:
        """Build the priority chain, auto-skipping unconfigured/unavailable.

        Returns (chain, skipped_log_lines).
        In confidential_mode only local providers may join.
        """
        chain: list[TranslationProvider] = []
        skipped: list[str] = []
        seen: set[str] = set()

        for entry in self.with_key_aliases(provider_entries):
            if not isinstance(entry, dict):
                skipped.append("[INVALID ENTRY]")
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                skipped.append("[INVALID PROVIDER NAME]")
                continue
            name = name.strip()
            if name in seen:
                skipped.append(f"{name} [DUPLICATE]")
                continue

            try:
                provider = self._create(name, entry)
            except Exception as exc:
                logger.warning("Failed to create provider %s (%s) — skipping", name, exc)
                skipped.append(f"{name} [CREATE FAILED]")
                continue

            if confidential_mode and not provider.is_local:
                skipped.append(f"{name} [CONFIDENTIAL MODE]")
                continue

            try:
                configured = provider.is_configured()
            except Exception as exc:
                logger.warning("is_configured() failed for %s (%s) — skipping", name, exc)
                skipped.append(f"{name} [CONFIG CHECK FAILED]")
                continue
            if not configured:
                skipped.append(f"{name} [NO KEY]")
                continue

            try:
                available = provider.is_available()
            except Exception as exc:
                # H10: a flaky probe must not abort startup.
                logger.warning("is_available() failed for %s (%s) — skipping", name, exc)
                skipped.append(f"{name} [NOT AVAILABLE]")
                continue
            if not available:
                skipped.append(f"{name} [NOT AVAILABLE]")
                continue

            seen.add(name)
            chain.append(provider)

        logger.info("Translation chain: %s", [p.name for p in chain])
        if skipped:
            logger.info("Skipped providers: %s", skipped)

        return chain, skipped
