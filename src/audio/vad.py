"""Voice Activity Detection — Silero VAD (ONNX, CPU) with an energy fallback.

Silero v5 consumes exactly 512 new samples per step at 16 kHz, with 64
samples of carried context (576 total) and a ``state`` tensor of shape
(2, batch, 128).  Older v4 exports use separate ``h`` / ``c`` tensors;
both signatures are supported by introspecting the ONNX inputs.

If the model or onnxruntime is missing, a small RMS-energy VAD takes over
so the pipeline still gets utterance boundaries instead of treating all
audio as speech until the end of time.

Events returned by ``process()`` / ``end_of_stream()``:
    {"type": "start",  "timestamp": <sample index>}
    {"type": "end",    "timestamp": <sample index>}
    {"type": "cancel", "timestamp": <sample index>}   # shorter than min_speech

Adapted from references/WhisperLiveKit (silero_vad_iterator.py) and
references/whisper_streaming (VADIterator padding rules).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from src.config.settings import EngineConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512      # Silero v5 step at 16 kHz (32 ms)
CONTEXT_SAMPLES = 64     # audio context carried between frames

# Release threshold hysteresis, same as Silero's VADIterator (0.15).
PROB_HYSTERESIS = 0.15

# Search order: user state dir first, then the bundled asset.
# In a PyInstaller frozen exe, __file__ points inside the .exe archive;
# sys._MEIPASS is the actual extraction directory.
def _asset_path() -> Path:
    import sys
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "assets" / "silero_vad.onnx"
    return Path(__file__).resolve().parents[2] / "assets" / "silero_vad.onnx"


MODEL_PATHS = [
    Path(os.path.expanduser("~/.liveTranslate/models/silero_vad.onnx")),
    _asset_path(),
]


def find_model() -> Path | None:
    """Return the first Silero ONNX model found, or None."""
    for path in MODEL_PATHS:
        if path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _SileroOnnx:
    """ONNX Runtime wrapper for Silero VAD (v5 ``state`` or v4 ``h``/``c``)."""

    name = "silero"

    def __init__(self, model_path: Path):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        self.session = onnxruntime.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name: i for i in self.session.get_inputs()}
        if "input" not in self._inputs or "sr" not in self._inputs:
            raise RuntimeError(
                f"Unexpected Silero ONNX inputs: {sorted(self._inputs)}"
            )
        self._stateful = "state" in self._inputs
        if not self._stateful and not {"h", "c"} <= set(self._inputs):
            raise RuntimeError(
                f"Unsupported Silero ONNX signature: {sorted(self._inputs)}"
            )
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        if self._stateful:
            self._state = _zeros_like_input(self._inputs["state"])
        else:
            self._h = _zeros_like_input(self._inputs["h"])
            self._c = _zeros_like_input(self._inputs["c"])

    def speech_prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame]).astype(np.float32).reshape(1, -1)
        feed = {
            "input": x,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }

        if self._stateful:
            feed["state"] = self._state
            out, state = self.session.run(None, feed)
            self._state = state
        else:
            feed["h"], feed["c"] = self._h, self._c
            out, h, c = self.session.run(None, feed)
            self._h, self._c = h, c

        self._context = x[0, -CONTEXT_SAMPLES:].copy()
        return float(np.asarray(out).reshape(-1)[0])


class _EnergyVad:
    """RMS fallback: maps dBFS to a pseudo-probability with hysteresis."""

    name = "energy"

    def __init__(self, threshold_dbfs: float, hysteresis_db: float):
        self.threshold_dbfs = float(threshold_dbfs)
        self.scale_db = max(1.0, float(hysteresis_db))

    def reset(self) -> None:
        pass

    def speech_prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        dbfs = 20.0 * np.log10(max(rms, 1e-8))
        # sigmoid: 0.5 at the threshold, approaching 0/1 a few dB away
        return float(1.0 / (1.0 + np.exp(-(dbfs - self.threshold_dbfs) / self.scale_db)))


def _zeros_like_input(inp) -> np.ndarray:
    """Build a zero tensor matching an ONNX input's declared shape."""
    shape = []
    for dim in inp.shape:
        shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Streaming VAD state machine
# ---------------------------------------------------------------------------


class VoiceActivityDetector:
    """Streaming VAD over 16 kHz mono float32 audio, any chunk size."""

    def __init__(self, config: EngineConfig, backend: str = "auto"):
        self.threshold = float(config.vad.threshold)
        self.release_threshold = max(0.0, self.threshold - PROB_HYSTERESIS)
        self.min_silence_samples = int(
            config.audio.sample_rate * config.vad.min_silence_ms / 1000
        )
        self.padding_samples = int(
            config.audio.sample_rate * config.vad.padding_ms / 1000
        )
        self.min_speech_samples = int(
            config.audio.sample_rate * config.vad.min_speech_ms / 1000
        )

        self._backend = self._build_backend(config, backend)

        self._buffer = np.empty(0, dtype=np.float32)
        self._current_sample = 0
        self._triggered = False
        self._temp_end = 0
        self._speech_start = 0

    # -- setup ---------------------------------------------------------------

    def _build_backend(self, config: EngineConfig, requested: str):
        if requested == "energy":
            return self._energy_backend(config)

        model_path = find_model()
        if model_path is None:
            logger.warning(
                "Silero VAD model not found (looked in %s) — using the RMS "
                "energy fallback. Copy silero_vad.onnx into assets/ for "
                "better accuracy.",
                ", ".join(str(p) for p in MODEL_PATHS),
            )
            return self._energy_backend(config)

        try:
            backend = _SileroOnnx(model_path)
            logger.info("Silero VAD loaded from %s", model_path)
            return backend
        except ImportError:
            logger.warning(
                "onnxruntime is not installed — using the RMS energy VAD fallback"
            )
        except Exception:
            logger.exception(
                "Failed to load Silero VAD from %s — using the RMS energy fallback",
                model_path,
            )
        return self._energy_backend(config)

    def _energy_backend(self, config: EngineConfig) -> _EnergyVad:
        return _EnergyVad(
            threshold_dbfs=config.vad.fallback_dbfs_threshold,
            hysteresis_db=config.vad.fallback_hysteresis_db,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Forget all state (new stream / new meeting)."""
        self._backend.reset()
        self._buffer = np.empty(0, dtype=np.float32)
        self._current_sample = 0
        self._triggered = False
        self._temp_end = 0
        self._speech_start = 0

    def process(self, samples: np.ndarray) -> list[dict]:
        """Feed audio of any length; return all events produced, in order."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if samples.size == 0:
            return []

        self._buffer = np.concatenate([self._buffer, samples])
        frames = self._buffer.size // FRAME_SAMPLES

        events: list[dict] = []
        for i in range(frames):
            frame = self._buffer[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            # Silero timestamps count *new* samples only (context excluded).
            self._current_sample += FRAME_SAMPLES
            event = self._step(self._backend.speech_prob(frame))
            if event is not None:
                events.append(event)

        if frames:
            self._buffer = self._buffer[frames * FRAME_SAMPLES:]

        return events

    def end_of_stream(self) -> list[dict]:
        """Close any open utterance when the audio source stops."""
        events: list[dict] = []
        if self._triggered:
            end = max(0, self._current_sample)
            start = self._speech_start
            self._triggered = False
            self._temp_end = 0
            self._speech_start = 0
            if end - start < self.min_speech_samples:
                events.append({"type": "cancel", "timestamp": int(start)})
            else:
                events.append({"type": "end", "timestamp": int(end)})
        return events

    # -- state machine -------------------------------------------------------

    def _step(self, prob: float) -> dict | None:
        # Speech resumed before the minimum silence elapsed: not an end yet.
        if prob >= self.threshold and self._temp_end:
            self._temp_end = 0

        if prob >= self.threshold and not self._triggered:
            self._triggered = True
            self._speech_start = max(
                0, self._current_sample - self.padding_samples - FRAME_SAMPLES
            )
            return {"type": "start", "timestamp": int(self._speech_start)}

        if prob < self.release_threshold and self._triggered:
            if not self._temp_end:
                self._temp_end = self._current_sample
            if self._current_sample - self._temp_end < self.min_silence_samples:
                return None

            end = max(0, self._temp_end + self.padding_samples - FRAME_SAMPLES)
            start = self._speech_start
            self._temp_end = 0
            self._triggered = False
            self._speech_start = 0

            if end - start < self.min_speech_samples:
                # Too short to be useful speech — retract the start.
                return {"type": "cancel", "timestamp": int(start)}
            return {"type": "end", "timestamp": int(end)}

        return None
