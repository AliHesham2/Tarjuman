"""Phase 6 integration: WAV → ring buffer → VAD/STT → translation → UI.

The WAV is real and the pipeline is the real coordinator + ring buffer +
VAC online processor + energy VAD.  Whisper and the translation provider are
deterministic fakes, because no model/API credentials are available in this
environment.  This still exercises every queue/thread/lifecycle boundary that
connects them.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
import wave
from unittest import mock

import numpy as np

from src.config.settings import EngineConfig
from src.pipeline.coordinator import MainPipeline
from src.pipeline.ipc import MessageType
from src.stt.base import STTBackend
from src.translation.base import TranslationResult

SR = 16000


class WavCapture:
    """Minimal capture stand-in; the test feeds the real ring buffer itself."""

    def __init__(self) -> None:
        self.running = False

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class FixedWhisperBackend(STTBackend):
    name = "fixed-wav"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, initial_prompt=None) -> list[dict]:
        if audio.size < int(0.2 * SR):
            return []
        duration = audio.size / float(SR)
        return [{
            "start": 0.0,
            "end": duration,
            "text": "hello world",
            "no_speech_prob": 0.01,
            "avg_logprob": -0.2,
            "words": [
                {"start": 0.0, "end": duration / 2, "word": "hello"},
                {"start": duration / 2, "end": duration, "word": "world"},
            ],
        }]


class WavTranslationEngine:
    def __init__(self) -> None:
        self.confirmed: list[str] = []
        self.context: list[tuple[str, str]] = []

    def translate_confirmed(self, text: str) -> TranslationResult:
        self.confirmed.append(text)
        return TranslationResult(f"AR:{text}", "wav-fake", 2.0)

    def translate_partial(self, text: str) -> TranslationResult:
        return TranslationResult(f"AR?:{text}", "wav-fake", 1.0)

    def remember_context(self, source: str, translated: str) -> None:
        self.context.append((source, translated))

    def current_provider_name(self) -> str:
        return "wav-fake"

    def shutdown(self) -> None:
        pass


def write_speech_wav(path: str) -> None:
    """Write 1.8s of speech-like tone followed by 1.2s of silence."""
    t = np.arange(int(1.8 * SR), dtype=np.float64) / SR
    speech = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    audio = np.concatenate([speech, np.zeros(int(1.2 * SR), dtype=np.float32)])
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == SR
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


class WavPipelineIntegrationTests(unittest.TestCase):
    def test_wav_runs_through_full_pipeline(self):
        import src.stt.online_processor as op

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        write_speech_wav(path)
        audio = read_wav(path)

        config = EngineConfig()
        config.engine.chunk_duration_ms = 250
        config.vad.min_speech_ms = 200
        config.vad.min_silence_ms = 500
        config.vad.padding_ms = 100
        config.vad.fallback_dbfs_threshold = -42.0
        config.translation.partial_translation = False

        engine = WavTranslationEngine()
        capture = WavCapture()
        pipeline = MainPipeline(
            config, engine, capture=capture, backend=FixedWhisperBackend()
        )
        ring = pipeline.ring_buffer
        real_vad = op.VoiceActivityDetector

        try:
            with mock.patch.object(
                op,
                "VoiceActivityDetector",
                lambda cfg: real_vad(cfg, backend="energy"),
            ):
                self.assertTrue(pipeline.start())

                # Feed the WAV in the same 100ms slices used by the live worker.
                for start in range(0, audio.size, 1600):
                    ring.write(audio[start:start + 1600])
                    time.sleep(0.005)

                deadline = time.monotonic() + 8.0
                text_messages = []
                while time.monotonic() < deadline:
                    for msg in pipeline.ui_queue.drain():
                        if msg.type == MessageType.TEXT_UPDATE:
                            text_messages.append(msg)
                    if text_messages:
                        break
                    time.sleep(0.02)

                self.assertTrue(text_messages, "no translated text reached the UI")
                self.assertIn("AR:", text_messages[-1].payload["arabic"])
                self.assertTrue(engine.confirmed)
                self.assertTrue(engine.context)
        finally:
            pipeline.stop()

        self.assertEqual(pipeline.status.value, "idle")
        self.assertFalse(capture.running)


if __name__ == "__main__":
    unittest.main()
