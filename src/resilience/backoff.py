"""Exponential backoff with full jitter — prevents Thundering Herd on provider failures.

Strategy: random(0, min(cap, base * 2^attempt))
Ref: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

from __future__ import annotations
import random

BASE_DELAY_S = 5.0
MAX_DELAY_S = 120.0
FACTOR = 2.0


def delay_for_attempt(attempt: int) -> float:
    """Return a jittered delay for the given 0-based attempt index.

    attempt=0 → uniform(0, 5s)
    attempt=1 → uniform(0, 10s)
    attempt=2 → uniform(0, 20s)
    attempt=3 → uniform(0, 40s)
    attempt≥4 → uniform(0, 120s)

    Full-jitter prevents thundering herd when multiple workers hit
    the same provider rate limit simultaneously.
    """
    if attempt < 0:
        attempt = 0
    cap = min(BASE_DELAY_S * (FACTOR ** attempt), MAX_DELAY_S)
    return random.uniform(0, cap)


def delay_no_jitter(attempt: int) -> float:
    """Deterministic delay — use only in tests or single-worker scenarios."""
    if attempt < 0:
        attempt = 0
    return min(BASE_DELAY_S * (FACTOR ** attempt), MAX_DELAY_S)
