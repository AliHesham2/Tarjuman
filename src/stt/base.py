"""STTBackend abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTBackend(ABC):
    """A speech-to-text engine operating on 16kHz mono float32 audio."""

    name: str = "stt"

    @abstractmethod
    def load(self) -> None:
        """Load the model (heavy work — call once at startup)."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> list[dict]:
        """Transcribe one buffer. Returns word-level segments:
        [{"start": s, "end": e, "text": str, "no_speech_prob": f, "avg_logprob": f}, ...]
        """

    @abstractmethod
    def unload(self) -> None:
        """Free model memory."""
