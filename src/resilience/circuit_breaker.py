"""Circuit Breaker per provider — CLOSED / OPEN / HALF-OPEN.

With exponential backoff on repeated failures: 5s, 10s, 20s, 40s, max 120s.
Prevents hammering a failing provider (API abuse) and failing requests
with the same long timeout over and over.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"      # normal — requests flow
    OPEN = "open"          # failing — requests rejected instantly
    HALF_OPEN = "half_open"  # probing — one trial request allowed


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        base_backoff_s: float = 5.0,
        max_backoff_s: float = 120.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s

        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._backoff_s = base_backoff_s

    @property
    def state(self) -> BreakerState:
        with self._lock:
            if self._state == BreakerState.OPEN and self._retry_due():
                self._state = BreakerState.HALF_OPEN
                logger.info("Circuit breaker %s → HALF_OPEN (probing)", self.name)
            return self._state

    def allow_request(self) -> bool:
        """Can a request go through right now? Never blocks."""
        state = self.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.OPEN:
            return False
        return True  # HALF_OPEN: one trial request

    def record_success(self) -> None:
        with self._lock:
            if self._state != BreakerState.CLOSED:
                logger.info("Circuit breaker %s → CLOSED (recovered)", self.name)
            self._state = BreakerState.CLOSED
            self._consecutive_failures = 0
            self._backoff_s = self.base_backoff_s

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == BreakerState.HALF_OPEN or \
               self._consecutive_failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = time.monotonic()
        self._backoff_s = min(self._backoff_s * 2, self.max_backoff_s)
        logger.warning(
            "Circuit breaker %s → OPEN (failures=%d, retry in %.0fs)",
            self.name, self._consecutive_failures, self._backoff_s,
        )

    def _retry_due(self) -> bool:
        return (time.monotonic() - self._opened_at) >= self._backoff_s
