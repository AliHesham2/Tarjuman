"""Accelerated pipeline soak/stress test.

Examples:

    python scripts/soak_test.py --minutes 0.2 --speed 20
    python scripts/soak_test.py --minutes 120 --speed 1

The run uses synthetic meeting-like audio through the real coordinator,
ring buffer, VAC online processor and energy VAD.  STT and translation are
deterministic fakes; a real 2-hour Zoom run still needs a real desktop,
Whisper model and provider keys.

The default is the plan's two-hour duration in real time (speed 1).  Use a
higher ``--speed`` for a faster queue/thread stress run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.vad import VoiceActivityDetector  # noqa: E402
from src.config.settings import EngineConfig  # noqa: E402
from src.pipeline.coordinator import MainPipeline  # noqa: E402
from src.pipeline.ipc import MessageType, PipelineStatus  # noqa: E402
from src.stt.base import STTBackend  # noqa: E402
from src.translation.base import TranslationResult  # noqa: E402

SR = 16000
CHUNK = 1600  # 100 ms


class FakeCapture:
    def __init__(self) -> None:
        self.running = False

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class FixedBackend(STTBackend):
    name = "soak-fixed"

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


class EchoEngine:
    def __init__(self) -> None:
        self.confirmed = 0
        self.shutdown_called = False

    def translate_confirmed(self, text: str) -> TranslationResult:
        self.confirmed += 1
        return TranslationResult("AR:" + text, "soak-fake", 0.0)

    def translate_partial(self, text: str) -> TranslationResult:
        return TranslationResult("AR?:" + text, "soak-fake", 0.0)

    def remember_context(self, source: str, translated: str) -> None:
        pass

    def current_provider_name(self) -> str:
        return "soak-fake"

    def shutdown(self) -> None:
        self.shutdown_called = True


@dataclass
class SoakResult:
    simulated_seconds: float
    fed_chunks: int
    translated_finals: int
    ui_text_messages: int
    max_ui_queue: int
    workers_alive_at_end: dict
    final_status: str
    elapsed_real_seconds: float


def make_audio_cycle() -> np.ndarray:
    t = np.arange(int(1.8 * SR), dtype=np.float64) / SR
    speech = (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    return np.concatenate([speech, np.zeros(int(1.2 * SR), dtype=np.float32)])


def run(minutes: float, speed: float) -> SoakResult:
    import src.stt.online_processor as op

    config = EngineConfig()
    config.engine.chunk_duration_ms = 250
    config.vad.min_speech_ms = 200
    config.vad.min_silence_ms = 500
    config.vad.padding_ms = 100
    config.vad.fallback_dbfs_threshold = -42.0
    config.translation.partial_translation = False

    engine = EchoEngine()
    capture = FakeCapture()
    pipeline = MainPipeline(config, engine, capture=capture, backend=FixedBackend())
    cycle = make_audio_cycle()
    cycle_index = 0
    fed_chunks = 0
    ui_text = 0
    max_ui = 0
    target_seconds = max(0.1, minutes * 60.0)
    target_chunks = int(target_seconds * SR / CHUNK)
    interval = 0.1 / max(0.1, speed)
    real_vad = op.VoiceActivityDetector
    started = time.monotonic()

    try:
        with mock.patch.object(
            op,
            "VoiceActivityDetector",
            lambda cfg: real_vad(cfg, backend="energy"),
        ):
            if not pipeline.start():
                raise RuntimeError("pipeline failed to start")
            for _ in range(target_chunks):
                chunk = cycle[cycle_index:cycle_index + CHUNK]
                if chunk.size < CHUNK:
                    cycle_index = 0
                    chunk = cycle[:CHUNK]
                else:
                    cycle_index += CHUNK
                pipeline.ring_buffer.write(chunk)
                fed_chunks += 1

                if pipeline.status == PipelineStatus.ERROR:
                    raise RuntimeError("pipeline entered ERROR during soak")
                max_ui = max(max_ui, pipeline.ui_queue.qsize())
                for msg in pipeline.ui_queue.drain():
                    if msg.type == MessageType.TEXT_UPDATE:
                        ui_text += 1
                time.sleep(interval)

            # Let the final utterance finish and the queue drain.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and pipeline.ui_queue.qsize():
                for msg in pipeline.ui_queue.drain():
                    if msg.type == MessageType.TEXT_UPDATE:
                        ui_text += 1
                time.sleep(0.02)
    finally:
        pipeline.stop()
        pipeline.engine.shutdown()

    workers = {name: thread.is_alive() for name, thread in pipeline._threads.items()}
    return SoakResult(
        simulated_seconds=round(fed_chunks * CHUNK / SR, 2),
        fed_chunks=fed_chunks,
        translated_finals=engine.confirmed,
        ui_text_messages=ui_text,
        max_ui_queue=max_ui,
        workers_alive_at_end=workers,
        final_status=pipeline.status.value,
        elapsed_real_seconds=round(time.monotonic() - started, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=120.0,
                        help="simulated meeting duration (default: 120)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="audio feed speed; 1 = real time (default: 1)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.minutes <= 0 or args.speed <= 0:
        parser.error("--minutes and --speed must be positive")

    result = run(args.minutes, args.speed)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    ok = (
        result.final_status == "idle"
        and not any(result.workers_alive_at_end.values())
        and result.fed_chunks > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
