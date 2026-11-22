"""FasterWhisperBackend — faster-whisper with safe GPU → CPU fallback.

RTX 3060 Laptop (CC 8.6) → float16 hits the Tensor Cores (~1.5GB VRAM
for medium). If CUDA/CTranslate2 is unavailable we must *also* switch the
device to ``cpu`` — the previous code returned ``int8`` but still asked
for ``device="cuda"``, which failed at load time instead of falling back.
"""

from __future__ import annotations

import gc
import logging

import numpy as np

from src.config.settings import EngineConfig
from src.stt.base import STTBackend

logger = logging.getLogger(__name__)

# Approximate download size, only used to warn before a first-run download.
_MODEL_MB = {
    "tiny": 75, "base": 145, "small": 480, "medium": 1500, "large": 3000,
}


def _model_size_text(name: str) -> str:
    base = name.split(".")[0].split("-")[0]
    megabytes = _MODEL_MB.get(base)
    if megabytes is None:
        return "a large file"
    return f"about {megabytes / 1000:.1f} GB" if megabytes >= 1000 else f"about {megabytes} MB"


# Models too big to keep up in real time without a GPU, and what to use instead.
_CPU_TOO_SLOW = {"medium", "large", "large-v1", "large-v2", "large-v3"}
CPU_FALLBACK_MODEL = "base.en"

# Compute types that only exist on GPU.
_GPU_ONLY_COMPUTE_TYPES = {"float16", "int8_float16", "int8_bfloat16"}


def _resolve_runtime(config: EngineConfig) -> tuple[str, str]:
    """Return ``(device, compute_type)`` that CTranslate2 can actually use."""
    device = (config.engine.whisper_device or "auto").lower()
    requested = (config.engine.whisper_compute_type or "default").lower()

    try:
        import ctranslate2
    except ImportError:
        logger.warning("ctranslate2 not installed — using CPU int8")
        return "cpu", "int8"

    if device == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if device == "cuda":
        try:
            if ctranslate2.get_cuda_device_count() == 0:
                logger.warning(
                    "CUDA requested but no device is visible — using CPU int8"
                )
                return "cpu", "int8"
            supported = set(ctranslate2.get_supported_compute_types("cuda"))
        except Exception as exc:
            logger.warning("CUDA probe failed (%s) — using CPU int8", exc)
            return "cpu", "int8"

        if requested in supported:
            return "cuda", requested
        for candidate in ("float16", "int8_float16", "int8"):
            if candidate in supported:
                logger.warning(
                    "Compute type %r unsupported on CUDA (have %s) — using %s",
                    requested, sorted(supported), candidate,
                )
                return "cuda", candidate
        return "cuda", "float32"

    # CPU
    if requested in _GPU_ONLY_COMPUTE_TYPES:
        logger.info(
            "Compute type %r is GPU-only — using CPU int8", requested
        )
        return "cpu", "int8"
    try:
        supported = set(ctranslate2.get_supported_compute_types("cpu"))
        if requested in supported:
            return "cpu", requested
        for candidate in ("int8", "float32"):
            if candidate in supported:
                return "cpu", candidate
    except Exception:
        logger.exception("CPU compute-type probe failed")
    return "cpu", "int8"


class FasterWhisperBackend(STTBackend):
    name = "faster_whisper"

    def __init__(self, config: EngineConfig):
        self.config = config
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        # Optional hook the pipeline sets to show plain notices in the tray.
        self.on_info = None

    @staticmethod
    def _is_cached(name: str) -> bool:
        """True when the model is already on disk (no download needed)."""
        try:
            from faster_whisper.utils import download_model

            download_model(name, local_files_only=True)
            return True
        except Exception:
            return False

    def _announce_download(self, name: str) -> bool:
        if self.on_info is None or self._is_cached(name):
            return False
        try:
            self.on_info(
                f"Downloading the speech model '{name}' "
                f"({_model_size_text(name)}) - first run only. "
                "Translation starts when it finishes."
            )
        except Exception:
            logger.debug("on_info callback failed", exc_info=True)
        return True

    def load(self) -> None:
        from faster_whisper import WhisperModel

        from src.stt.cuda_paths import cuda_runtime_ready

        device, compute_type = _resolve_runtime(self.config)
        if device == "cuda" and not cuda_runtime_ready():
            logger.warning(
                "cuDNN 8 / cuBLAS 12 are not installed - Whisper will use the CPU"
            )
            device, compute_type = "cpu", "int8"

        # Most specific first: the resolved runtime, then CUDA int8 (a
        # float16 failure is usually a compute-type problem, not a missing
        # GPU), then CPU int8 as the last resort.
        candidates = [(device, compute_type)]
        if device == "cuda":
            if compute_type != "int8":
                candidates.append(("cuda", "int8"))
            candidates.append(("cpu", "int8"))

        last_error: Exception | None = None
        announced = False
        for candidate_device, candidate_compute in candidates:
            model_name = self.config.engine.whisper_model
            if candidate_device == "cpu" and model_name in _CPU_TOO_SLOW:
                # Measured: medium on CPU needs ~40-60 s per 15 s window.
                logger.warning(
                    "Whisper %s is far too slow on the CPU - using %s instead",
                    model_name, CPU_FALLBACK_MODEL,
                )
                model_name = CPU_FALLBACK_MODEL
            if not announced:
                announced = self._announce_download(model_name)
            logger.info(
                "Loading Whisper %s on %s (%s)...",
                model_name,
                candidate_device,
                candidate_compute,
            )
            try:
                model = WhisperModel(
                    model_name,
                    device=candidate_device,
                    compute_type=candidate_compute,
                    num_workers=1,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Whisper failed to load on %s (%s): %s",
                    candidate_device, candidate_compute, exc,
                )
                continue

            self.model = model
            self.device = candidate_device
            self.compute_type = candidate_compute
            logger.info(
                "Whisper model loaded (%s / %s)",
                self.device, self.compute_type,
            )
            if announced and self.on_info is not None:
                try:
                    self.on_info("Speech model ready.")
                except Exception:
                    logger.debug("on_info callback failed", exc_info=True)
            return

        raise RuntimeError(
            "Whisper failed to load on every available device"
        ) from last_error

    def transcribe(
        self, audio: np.ndarray, initial_prompt: str | None = None
    ) -> list[dict]:
        if self.model is None:
            raise RuntimeError("Whisper model not loaded — call load() first")

        segments, _info = self.model.transcribe(
            audio,
            language=self.config.engine.whisper_language,
            beam_size=self.config.engine.beam_size,
            condition_on_previous_text=(
                self.config.engine.condition_on_previous_text
            ),
            initial_prompt=initial_prompt,
            vad_filter=False,  # VAD is handled upstream by our own layer
            word_timestamps=True,
        )

        out = []
        for seg in segments:
            out.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": seg.no_speech_prob,
                "avg_logprob": seg.avg_logprob,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (seg.words or [])
                ],
            })
        return out

    def unload(self) -> None:
        self.model = None
        # CTranslate2 frees the model (and its GPU buffers) on collection;
        # no torch dependency is needed for that.
        gc.collect()
        logger.info("Whisper model unloaded")
