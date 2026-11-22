"""Thread-safe ring buffer for audio chunks.

Fixed duration (30s by default). Writes never block and never raise: when
the buffer is full the oldest audio is dropped, so a stalled STT worker
can never stall the capture callback.

Reads support both styles:
  * ``read_all()`` / ``consume_all()``      -- non-blocking snapshots
  * ``read(n, timeout=...)``                -- blocking, waits for n samples
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class RingBuffer:
    """Fixed-duration FIFO of float32 audio samples.

    All public methods are thread-safe. ``read()`` blocks on a condition
    variable and is woken by ``write()`` or ``close()``.
    """

    def __init__(self, max_seconds: float, sample_rate: int = 16000):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)

        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0
        self._dropped_samples = 0
        self._closed = False

        # Condition (not a bare Lock) so readers can sleep without polling.
        self._cond = threading.Condition()

    # -- writer side ---------------------------------------------------------

    def write(self, chunk: np.ndarray) -> int:
        """Append audio. Returns the number of samples dropped to make room.

        Never blocks and never raises; the PortAudio callback is the caller.
        """
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        if chunk.size == 0:
            return 0

        with self._cond:
            if self._closed:
                return 0

            self._chunks.append(chunk)
            self._sample_count += chunk.size

            dropped = 0
            # Trim exactly enough samples, possibly part of the oldest chunk,
            # so a single oversized chunk does not wipe the whole buffer.
            while self._sample_count > self.max_samples and self._chunks:
                excess = self._sample_count - self.max_samples
                first = self._chunks[0]
                if excess >= first.size:
                    self._chunks.popleft()
                    self._sample_count -= first.size
                    dropped += first.size
                else:
                    self._chunks[0] = first[excess:]
                    self._sample_count -= excess
                    dropped += excess

            self._dropped_samples += dropped
            if dropped:
                logger.debug(
                    "Ring buffer full — dropped %.0f ms of audio",
                    1000.0 * dropped / self.sample_rate,
                )

            self._cond.notify_all()
            return dropped

    # -- reader side ---------------------------------------------------------

    def read(self, n_samples: int, timeout: float | None = None) -> np.ndarray | None:
        """Block until ``n_samples`` are available, then consume them.

        Returns ``None`` on timeout or after ``close()`` — callers must
        distinguish that from an empty array.
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float32)

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        with self._cond:
            while self._sample_count < n_samples:
                if self._closed:
                    return None
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
            return self._take_locked(n_samples)

    def read_all(self) -> np.ndarray:
        """Return a copy of everything currently buffered (does not consume)."""
        with self._cond:
            if not self._chunks:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(list(self._chunks))

    def consume_all(self) -> np.ndarray:
        """Drain and return everything currently buffered."""
        with self._cond:
            return self._take_locked(self._sample_count)

    def clear(self) -> None:
        with self._cond:
            self._chunks.clear()
            self._sample_count = 0

    def close(self) -> None:
        """Wake every blocked reader and reject further writes."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # -- introspection -------------------------------------------------------

    @property
    def available_samples(self) -> int:
        with self._cond:
            return self._sample_count

    @property
    def duration_seconds(self) -> float:
        with self._cond:
            return self._sample_count / self.sample_rate

    @property
    def dropped_seconds(self) -> float:
        with self._cond:
            return self._dropped_samples / self.sample_rate

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def __len__(self) -> int:
        return self.available_samples

    # -- internals -----------------------------------------------------------

    def _take_locked(self, n_samples: int) -> np.ndarray:
        """Consume exactly ``n_samples``; caller must hold ``_cond``."""
        if n_samples <= 0:
            return np.empty(0, dtype=np.float32)

        parts: list[np.ndarray] = []
        remaining = n_samples
        while remaining > 0 and self._chunks:
            chunk = self._chunks[0]
            if chunk.size <= remaining:
                parts.append(chunk)
                self._chunks.popleft()
                self._sample_count -= chunk.size
                remaining -= chunk.size
            else:
                parts.append(chunk[:remaining])
                self._chunks[0] = chunk[remaining:]
                self._sample_count -= remaining
                remaining = 0

        if not parts:
            return np.empty(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)
