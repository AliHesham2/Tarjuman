"""HypothesisBuffer — Local Agreement Policy.

Ported from references/whisper_streaming/whisper_online.py (MIT).

A word is CONFIRMED (safe to translate) only when it appears in the
output of two consecutive Whisper runs over overlapping audio. The
buffer keeps the *previous* run's words in ``buffer`` and the current
run's words in ``new``; ``flush()`` walks both from the start and commits
the common prefix.

Changes vs. the reference:
  * ``reset(offset)`` — fixes the cross-utterance leak where the old
    implementation kept old timestamps and silently dropped every word
    of the second utterance.
  * the overlap de-duplication of up to 5 words is kept, so Whisper
    re-transcribing the committed tail does not duplicate text.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Whisper re-transcribes up to this many words of the committed tail.
MAX_OVERLAP_WORDS = 5
# Timeline tolerance for "same position" (seconds).
TIMELINE_TOLERANCE = 0.1


class HypothesisBuffer:
    """Accumulates Whisper outputs across runs and emits confirmed words."""

    def __init__(self):
        self.commited_in_buffer: list[tuple[float, float, str]] = []
        self.buffer: list[tuple[float, float, str]] = []   # previous run's pending words
        self.new: list[tuple[float, float, str]] = []      # current run's words

        self.last_commited_time = 0.0
        self.last_commited_word: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def reset(self, offset: float = 0.0) -> None:
        """Clear all state and start a new utterance at ``offset`` seconds."""
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = float(offset)
        self.last_commited_word = None

    # -- input ---------------------------------------------------------------

    def insert(self, new_words: list[tuple[float, float, str]], offset: float) -> None:
        """Insert the current run's output (timestamps are buffer-relative)."""
        absolute = [
            (start + offset, end + offset, word)
            for (start, end, word) in new_words
        ]
        self.new = [
            w for w in absolute
            if w[0] > self.last_commited_time - TIMELINE_TOLERANCE
        ]

        # Drop the committed tail that Whisper re-transcribed at the start
        # of this run (1..5 matching words).
        if self.new and self.commited_in_buffer:
            first_start = self.new[0][0]
            if abs(first_start - self.last_commited_time) < 1.0:
                committed_count = len(self.commited_in_buffer)
                new_count = len(self.new)
                for i in range(1, min(min(committed_count, new_count), MAX_OVERLAP_WORDS) + 1):
                    committed_tail = " ".join(
                        self.commited_in_buffer[-j][2]
                        for j in range(i, 0, -1)
                    )
                    new_prefix = " ".join(self.new[j][2] for j in range(i))
                    if committed_tail == new_prefix:
                        del self.new[:i]
                        break

    # -- output --------------------------------------------------------------

    def flush(self) -> list[tuple[float, float, str]]:
        """Return the longest common prefix of the last two runs."""
        committed: list[tuple[float, float, str]] = []

        while self.new and self.buffer:
            new_word = self.new[0]
            old_word = self.buffer[0]
            if new_word[2] != old_word[2]:
                break

            committed.append(new_word)
            self.last_commited_word = new_word[2]
            self.last_commited_time = new_word[1]
            self.buffer.pop(0)
            self.new.pop(0)

        # Only the current run survives as the pending hypothesis.
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(committed)
        return committed

    def pop_commited(self, time: float) -> None:
        """Drop committed words that ended before ``time`` (buffer trimming)."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[tuple[float, float, str]]:
        """End of utterance: everything still pending is now final."""
        return list(self.buffer)

    # -- text helpers ---------------------------------------------------------

    def full_text(self) -> str:
        return " ".join(w[2] for w in self.commited_in_buffer)

    def unconfirmed_text(self) -> str:
        return " ".join(w[2] for w in self.buffer)
