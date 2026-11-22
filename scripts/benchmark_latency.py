"""Latency benchmark for each pipeline stage.

Run from the project root:

    python scripts/benchmark_latency.py --iterations 50

The benchmark deliberately uses deterministic fake STT/translation providers.
That measures our own queue/conversion/engine overhead without pretending to
measure Whisper or a network provider.  Run the same script with real
providers and a real Whisper model when those are available; the stage names
stay the same.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.capture import AudioCapture, resample_audio  # noqa: E402
from src.audio.ring_buffer import RingBuffer  # noqa: E402
from src.audio.vad import VoiceActivityDetector  # noqa: E402
from src.config.settings import EngineConfig  # noqa: E402
from src.pipeline.ipc import Message, MessageType, UIQueue  # noqa: E402
from src.stt.base import STTBackend  # noqa: E402
from src.stt.online_processor import VACOnlineASRProcessor  # noqa: E402
from src.translation.base import (  # noqa: E402
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
)
from src.translation.engine import TranslationEngine  # noqa: E402

SR = 16000


@dataclass
class StageStats:
    stage: str
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class FixedBackend(STTBackend):
    name = "benchmark-fixed"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, initial_prompt=None) -> list[dict]:
        duration = audio.size / float(SR)
        return [{
            "start": 0.0,
            "end": max(duration, 0.01),
            "text": "hello world",
            "no_speech_prob": 0.01,
            "avg_logprob": -0.2,
            "words": [
                {"start": 0.0, "end": max(duration / 2, 0.01), "word": "hello"},
                {"start": max(duration / 2, 0.01), "end": max(duration, 0.02), "word": "world"},
            ],
        }]


class ImmediateProvider(TranslationProvider):
    name = "benchmark-provider"
    is_local = True

    def translate(self, request: TranslationRequest) -> TranslationResult:
        return TranslationResult(
            translated="AR:" + request.text,
            provider=self.name,
            latency_ms=0.0,
        )

    def is_configured(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True


def _stats(stage: str, values_ms: list[float]) -> StageStats:
    ordered = sorted(values_ms)
    if not ordered:
        return StageStats(stage, 0, 0.0, 0.0, 0.0, 0.0)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return StageStats(
        stage=stage,
        samples=len(ordered),
        mean_ms=round(statistics.fmean(ordered), 3),
        p50_ms=round(statistics.median(ordered), 3),
        p95_ms=round(ordered[p95_index], 3),
        max_ms=round(ordered[-1], 3),
    )


def _time_ms(fn: Callable[[], object]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def speech(duration_s: float = 1.8) -> np.ndarray:
    t = np.arange(int(duration_s * SR), dtype=np.float64) / SR
    return (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def raw_pcm_48k_stereo(duration_s: float = 0.1) -> bytes:
    t = np.arange(int(duration_s * 48000), dtype=np.float64) / 48000.0
    mono = (0.5 * np.sin(2.0 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    return stereo.astype(np.int16).tobytes()


def bench_ring(iterations: int) -> StageStats:
    ring = RingBuffer(max_seconds=5)
    chunk = speech(0.1)
    values = []

    def one() -> None:
        ring.write(chunk)
        result = ring.read(chunk.size, timeout=0.5)
        if result is None:
            raise RuntimeError("ring read timed out")

    for _ in range(iterations):
        values.append(_time_ms(one))
    return _stats("ring buffer write+read (100 ms)", values)


def bench_conversion(iterations: int) -> StageStats:
    config = EngineConfig()
    capture = AudioCapture(config, RingBuffer(max_seconds=1))
    capture._device_channels = 2
    pcm = raw_pcm_48k_stereo(0.1)
    values = []

    def one() -> None:
        mono = capture._to_float_mono(pcm)
        resample_audio(mono, 48000, SR, filter_cache={})

    for _ in range(iterations):
        values.append(_time_ms(one))
    return _stats("int16 stereo 48k -> float mono 16k (100 ms)", values)


def bench_vad(iterations: int) -> StageStats:
    config = EngineConfig()
    audio = np.concatenate([speech(1.0), np.zeros(int(0.6 * SR), dtype=np.float32)])
    values = []

    for _ in range(iterations):
        vad = VoiceActivityDetector(config, backend="energy")
        for start in range(0, audio.size, 1600):
            values.append(
                _time_ms(lambda chunk=audio[start:start + 1600]: vad.process(chunk))
            )
    return _stats("VAD process (100 ms input)", values)


def bench_vac(iterations: int) -> StageStats:
    config = EngineConfig()
    config.engine.chunk_duration_ms = 250
    audio = np.concatenate([speech(1.0), np.zeros(int(0.6 * SR), dtype=np.float32)])
    values = []

    for _ in range(iterations):
        vac = VACOnlineASRProcessor(
            FixedBackend(),
            config,
            vad=VoiceActivityDetector(config, backend="energy"),
        )
        for start in range(0, audio.size, 1600):
            chunk = audio[start:start + 1600]

            def one(chunk=chunk) -> None:
                vac.insert_audio_chunk(chunk)
                vac.process_iter()

            values.append(_time_ms(one))
    return _stats("VAC + local-agreement process (100 ms input)", values)


def bench_engine(iterations: int) -> StageStats:
    config = EngineConfig()
    engine = TranslationEngine(config, [ImmediateProvider()])
    values = []
    try:
        for _ in range(iterations):
            values.append(
                _time_ms(lambda: engine.translate_confirmed("hello world, this is a benchmark"))
            )
    finally:
        engine.shutdown()
    return _stats("translation engine overhead (in-process provider)", values)


def bench_ui_queue(iterations: int) -> StageStats:
    q = UIQueue()
    message = Message(MessageType.TEXT_UPDATE, {"arabic": "مرحبا", "english": "hello"})
    values = []

    def one() -> None:
        q.put_message(message)
        q.drain(limit=1)

    for _ in range(iterations):
        values.append(_time_ms(one))
    return _stats("UI queue put+drain", values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    stages = [
        bench_ring(args.iterations),
        bench_conversion(args.iterations),
        bench_vad(args.iterations),
        bench_vac(args.iterations),
        bench_engine(args.iterations),
        bench_ui_queue(args.iterations),
    ]
    if args.json:
        print(json.dumps([asdict(stage) for stage in stages], ensure_ascii=False, indent=2))
    else:
        width = max(len(stage.stage) for stage in stages)
        print(f"{'stage'.ljust(width)}  n     mean     p50     p95     max  (ms)")
        for stage in stages:
            print(
                f"{stage.stage.ljust(width)}  {stage.samples:<5} "
                f"{stage.mean_ms:>7.3f} {stage.p50_ms:>7.3f} "
                f"{stage.p95_ms:>7.3f} {stage.max_ms:>7.3f}"
            )
        print()
        print("NOTE: STT and translation use deterministic in-process fakes;")
        print("these numbers measure this code's overhead, not Whisper/network latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
