"""Quick live check of every configured translation key (Groq accounts + DeepL).

    python test_apis.py

Sends one short sentence through each provider that has a key and prints the
latency and the translation. Keys are never printed. Each call uses one
request from that account's allowance.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.config.providers import GROQ_EXTRA_ACCOUNTS  # noqa: E402
from src.config.settings import load_config  # noqa: E402
from src.translation.base import TranslationRequest  # noqa: E402
from src.translation.registry import ProviderRegistry  # noqa: E402

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SENTENCE = "The deployment pipeline is ready for review."


def main() -> int:
    config = load_config()
    base_entries = {e["name"]: e for e in config.translation.providers}
    registry = ProviderRegistry()
    results = {}

    for name in ["groq", *GROQ_EXTRA_ACCOUNTS, "deepl"]:
        entry = dict(base_entries.get(name) or base_entries.get(name.split("_")[0], {}),
                     name=name)
        try:
            provider = registry.create_from_config(name, entry)
        except Exception as exc:
            print(f"{name:8} cannot be created: {exc}")
            continue
        if not provider.is_configured():
            print(f"{name:8} no key set - skipped")
            continue
        try:
            result = provider.translate(TranslationRequest(text=SENTENCE))
            tokens = f", {result.tokens} tokens" if result.tokens else ""
            print(f"{name:8} OK  {result.latency_ms:6.0f} ms{tokens}  ->  {result.translated}")
            results[name] = True
        except Exception as exc:
            print(f"{name:8} FAILED  {type(exc).__name__}: {str(exc)[:120]}")
            results[name] = False

    if not results:
        print("\nNo key is configured. Fill .env (see .env.example).")
        return 1
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
