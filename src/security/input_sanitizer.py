"""Input sanitizer — validates text before every API call.

Whisper hallucinations can produce prompt-injection-shaped output
("ignore previous instructions..."). Cap length, strip control
characters, neutralize injection patterns before they reach an LLM.
"""

from __future__ import annotations

import re

MAX_CHARS = 500

# control chars except \n and tab
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# obvious prompt-injection payloads
# Tags are neutralised to a space, not deleted, so the words around them are
# not glued together ("hello</system>world" -> "hello world").
_TAG_PATTERN = r"</?(system|assistant|user)>"

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        _TAG_PATTERN,
    ]
]


class SanitizationError(ValueError):
    """Text failed validation — do not send it to any provider."""


def sanitize(text: str, max_chars: int = MAX_CHARS) -> str:
    """Return cleaned text or raise SanitizationError."""
    if not text or not text.strip():
        raise SanitizationError("Empty text")

    cleaned = _CONTROL_CHARS.sub(" ", text).strip()

    if len(cleaned) > max_chars:
        raise SanitizationError(f"Text too long ({len(cleaned)} > {max_chars} chars)")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            # strip the injected fragment rather than failing the whole
            # utterance — Whisper noise sometimes trips partial matches.
            # Role tags become a space so neighbouring words survive.
            replacement = " " if pattern.pattern == _TAG_PATTERN else ""
            cleaned = pattern.sub(replacement, cleaned).strip()

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if not cleaned:
        raise SanitizationError("Text empty after sanitization")

    return cleaned
