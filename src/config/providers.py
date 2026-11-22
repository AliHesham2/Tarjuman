"""TranslationProviderConfig — per-provider settings resolved from config.yaml + .env.

API keys live ONLY in .env / Windows Credential Store, never in config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TranslationProviderConfig:
    """Resolved settings for one provider in the priority chain."""
    name: str
    model: str | None = None
    timeout_ms: int = 2000
    api_key: str = ""                    # resolved from .env, never logged
    base_url: str | None = None          # local LLM endpoints only

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000.0


# Where each provider finds its API key in the environment
KEY_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "deepl": "DEEPL_API_KEY",
}

# Default models, overridden by .env MODEL vars or config.yaml
MODEL_ENV_VARS = {
    "groq": "GROQ_MODEL",
}

# Extra Groq ACCOUNTS (each has its own free quota): groq_2 .. groq_10, keys in
# GROQ_API_KEY_2 .. GROQ_API_KEY_10. Raise the range() end to allow more.
GROQ_EXTRA_ACCOUNTS = [f"groq_{n}" for n in range(2, 11)]
for _name in GROQ_EXTRA_ACCOUNTS:
    KEY_ENV_VARS[_name] = f"GROQ_API_KEY_{_name.rsplit('_', 1)[1]}"
    MODEL_ENV_VARS[_name] = "GROQ_MODEL"

def _stored_api_key(name: str) -> str:
    """Read a key saved by the settings dialog, if keyring is available."""
    try:
        from src.security.credential_store import CredentialStore
        return (CredentialStore().get_key(name) or "").strip()
    except Exception:
        # A missing/broken keyring must not prevent providers with .env keys
        # from starting.
        return ""


def load_provider_config(name: str, entry: dict | None = None) -> TranslationProviderConfig:
    """Build one provider's config from its config.yaml entry + .env overrides.

    .env always wins over config.yaml for keys and models (single place to rotate).
    """
    entry = entry or {}
    key_env = KEY_ENV_VARS.get(name, "")
    model_env = MODEL_ENV_VARS.get(name, "")

    api_key = os.getenv(key_env, "").strip() if key_env else ""
    if not api_key and key_env:
        api_key = _stored_api_key(name)
    model = entry.get("model")
    if model_env and os.getenv(model_env, "").strip():
        model = os.getenv(model_env).strip()

    base_url = None

    try:
        timeout_ms = int(entry.get("timeout_ms", 2000))
    except (TypeError, ValueError):
        timeout_ms = 2000

    return TranslationProviderConfig(
        name=name,
        model=model,
        timeout_ms=timeout_ms,
        api_key=api_key,
        base_url=base_url,
    )
