"""Phase 2 tests: HypothesisBuffer, OnlineASRProcessor, VAC, backend fallback."""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from src.audio.ring_buffer import RingBuffer
from src.audio.vad import VoiceActivityDetector
from src.config.settings import EngineConfig
from src.pipeline.coordinator import MainPipeline
from src.stt.base import STTBackend
from src.stt.faster_whisper import FasterWhisperBackend, _resolve_runtime
from src.stt.hypothesis_buffer import HypothesisBuffer
from src.stt.online_processor import (
    OnlineASRProcessor,
    VACOnlineASRProcessor,
    _trim_repeated_ngrams,
)

SR = 16000


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TimelineBackend(STTBackend):
    """One word per whole second of the *absolute* stream timeline.

    The audio carries the absolute time in sample 0 (signal[n] = n / SR),
    so the backend still knows the timeline after the processor trims the
    front of its buffer.
    """

    name = "timeline"

    def __init__(self):
        self.calls = 0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, initial_prompt=None) -> list[dict]:
        self.calls += 1
        t0 = float(audio[0])
        duration = len(audio) / SR
        t1 = t0 + duration

        segments = []
        second = int(math.floor(t0))
        while second < t1:
            slot_start = max(float(second), t0)
            slot_end = min(float(second + 1), t1)
            # only emit a word once the slot is (almost) fully covered
            if slot_end - slot_start >= 0.9:
                word = f"s{second}"
                segments.append({
                    "start": slot_start - t0,
                    "end": slot_end - t0,
                    "text": " " + word,
                    "no_speech_prob": 0.01,
                    "avg_logprob": -0.2,
                    "words": [{
                        "start": slot_start - t0,
                        "end": slot_end - t0,
                        "word": word,
                    }],
                })
            second += 1
        return segments


class GrowingBackend(STTBackend):
    """Returns one word per 0.5s of the window (grows with the audio)."""

    name = "growing"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, initial_prompt=None) -> list[dict]:
        duration = len(audio) / SR
        count = max(1, int(duration / 0.5 + 1e-9))
        words = [
            {"start": i * 0.5, "end": (i + 1) * 0.5, "word": f"w{i}"}
            for i in range(count)
        ]
        return [{
            "start": 0.0,
            "end": duration,
            "text": " ".join(w["word"] for w in words),
            "no_speech_prob": 0.01,
            "avg_logprob": -0.2,
            "words": words,
        }]


def timeline_audio(start_s: float, end_s: float) -> np.ndarray:
    n0 = int(round(start_s * SR))
    n1 = int(round(end_s * SR))
    return (np.arange(n0, n1, dtype=np.float64) / SR).astype(np.float32)


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def make_config() -> EngineConfig:
    config = EngineConfig()
    config.engine.chunk_duration_ms = 250
    config.engine.buffer_trimming_sec = 15
    config.vad.min_silence_ms = 500
    config.vad.min_speech_ms = 200
    config.vad.padding_ms = 100
    config.vad.fallback_dbfs_threshold = -30.0
    config.vad.fallback_hysteresis_db = 6.0
    return config


# ---------------------------------------------------------------------------
# HypothesisBuffer
# ---------------------------------------------------------------------------


class TestHypothesisBuffer(unittest.TestCase):
    def test_first_run_confirms_nothing(self):
        hb = HypothesisBuffer()
        hb.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0.0)
        self.assertEqual(hb.flush(), [])
        self.assertEqual(hb.unconfirmed_text(), "hello world")

    def test_confirms_common_prefix_of_two_runs(self):
        hb = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        hb.insert(words, offset=0.0)
        hb.flush()
        hb.insert(words, offset=0.0)
        committed = hb.flush()
        self.assertEqual([w[2] for w in committed], ["hello", "world"])
        self.assertEqual(hb.last_commited_time, 1.0)

    def test_partial_extension_is_not_committed(self):
        hb = HypothesisBuffer()
        hb.insert([(0.0, 0.5, "hello")], offset=0.0)
        hb.flush()
        hb.insert(
            [(0.0, 0.5, "hello"), (0.5, 1.0, "there")], offset=0.0
        )
        committed = hb.flush()
        self.assertEqual([w[2] for w in committed], ["hello"])
        self.assertEqual(hb.unconfirmed_text(), "there")

    def test_reset_starts_a_new_timeline(self):
        hb = HypothesisBuffer()
        hb.insert([(0.0, 0.5, "one"), (0.5, 1.0, "two")], offset=0.0)
        hb.flush()
        hb.insert([(0.0, 0.5, "one"), (0.5, 1.0, "two")], offset=0.0)
        hb.flush()

        # Utterance 2 starts at t=5s. Local timestamps restart at 0.
        hb.reset(offset=5.0)
        hb.insert([(0.0, 0.5, "three"), (0.5, 1.0, "four")], offset=5.0)
        committed = hb.flush()
        self.assertEqual(committed, [])
        self.assertEqual(hb.unconfirmed_text(), "three four")

    def test_overlap_dedupe_of_committed_tail(self):
        hb = HypothesisBuffer()
        hb.insert([(0.0, 0.5, "a"), (0.5, 1.0, "b")], offset=0.0)
        hb.flush()
        hb.insert([(0.0, 0.5, "a"), (0.5, 1.0, "b")], offset=0.0)
        hb.flush()
        self.assertEqual(hb.last_commited_time, 1.0)

        # Whisper re-transcribes "b" at the start of the next window.
        hb.insert([(0.95, 1.0, "b"), (1.0, 1.5, "c")], offset=0.0)
        self.assertEqual([w[2] for w in hb.new], ["c"])
        self.assertEqual(hb.unconfirmed_text(), "")

    def test_pop_commited_drops_old_words(self):
        hb = HypothesisBuffer()
        hb.insert([(0.0, 0.5, "a"), (0.5, 1.0, "b")], offset=0.0)
        hb.flush()
        hb.insert([(0.0, 0.5, "a"), (0.5, 1.0, "b")], offset=0.0)
        hb.flush()
        hb.pop_commited(0.5)
        self.assertEqual([w[2] for w in hb.commited_in_buffer], ["b"])


# ---------------------------------------------------------------------------
# OnlineASRProcessor
# ---------------------------------------------------------------------------


class TestOnlineASRProcessor(unittest.TestCase):
    def test_local_agreement_over_time(self):
        p = OnlineASRProcessor(TimelineBackend(), make_config())
        confirmed = []
        for end in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            text, _ = p.process_iter()
            if text:
                confirmed.append(text)
        final = p.finish()

        words = " ".join(confirmed + ([final] if final else [])).split()
        self.assertEqual(words, ["s0", "s1", "s2"])

    def test_second_utterance_is_not_dropped(self):
        """Regression for B1: old state made every utterance after #1 empty."""
        p = OnlineASRProcessor(TimelineBackend(), make_config())

        for end in (0.5, 1.0, 1.5, 2.0):
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            p.process_iter()
        self.assertTrue(p.finish())

        # Utterance 2 starts at 2.0s and must produce text again.
        seen = []
        for end in (2.5, 3.0, 3.5, 4.0):
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            confirmed, partial = p.process_iter()
            seen.extend(x for x in (confirmed, partial) if x)
        final = p.finish()
        if final:
            seen.append(final)

        self.assertTrue(seen, "utterance 2 produced no text (B1 regressed)")
        self.assertIn("s2", " ".join(seen))

    def test_finish_does_not_duplicate_confirmed_words(self):
        """Regression for B2: flush/complete used to repeat the whole buffer."""
        p = OnlineASRProcessor(TimelineBackend(), make_config())
        confirmed = []
        for end in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            text, _ = p.process_iter()
            if text:
                confirmed.append(text)

        final = p.finish()
        if final:
            confirmed.append(final)

        words = " ".join(confirmed).split()
        self.assertEqual(words, ["s0", "s1", "s2"])
        self.assertEqual(len(words), len(set(words)), words)

    def test_buffer_is_trimmed_on_long_stream(self):
        config = make_config()
        config.engine.buffer_trimming_sec = 5
        p = OnlineASRProcessor(TimelineBackend(), config)

        confirmed = []
        for i in range(1, 41):  # 20 seconds, 0.5s per step
            end = i * 0.5
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            text, _ = p.process_iter()
            if text:
                confirmed.append(text)

        self.assertLessEqual(p.buffer_duration, 6.0)
        final = p.finish()
        if final:
            confirmed.append(final)

        words = " ".join(confirmed).split()
        self.assertTrue(words)
        self.assertEqual(len(words), len(set(words)), words)
        self.assertIn("s0", words)

    def test_hard_trim_when_nothing_commits(self):
        class NoWords(TimelineBackend):
            def transcribe(self, audio, initial_prompt=None):
                return []

        config = make_config()
        config.engine.buffer_trimming_sec = 1
        p = OnlineASRProcessor(NoWords(), config)

        for i in range(1, 9):  # 4 seconds
            end = i * 0.5
            p.insert_audio_chunk(timeline_audio(end - 0.5, end))
            p.process_iter()

        self.assertLessEqual(p.buffer_duration, 1.5)

    def test_prompt_uses_only_scrolled_out_words(self):
        p = OnlineASRProcessor(TimelineBackend(), make_config())
        p.commited = [
            (0.0, 0.5, "before"),
            (2.0, 2.5, "inside"),
        ]
        p.buffer_time_offset = 2.0
        self.assertEqual(p._prompt(), "before")

    def test_segments_below_logprob_are_dropped(self):
        class LowConfidence(TimelineBackend):
            def transcribe(self, audio, initial_prompt=None):
                segs = super().transcribe(audio, initial_prompt)
                for seg in segs:
                    seg["avg_logprob"] = -2.0
                return segs

        p = OnlineASRProcessor(LowConfidence(), make_config())
        p.insert_audio_chunk(timeline_audio(0.0, 1.0))
        self.assertEqual(p.process_iter(), ("", ""))


# ---------------------------------------------------------------------------
# VACOnlineASRProcessor
# ---------------------------------------------------------------------------


class TestVAC(unittest.TestCase):
    def _run(self, vac, chunks):
        confirmed, partials = [], []
        for chunk in chunks:
            vac.insert_audio_chunk(chunk)
            text, partial = vac.process_iter()
            if text:
                confirmed.append(text)
            if partial:
                partials.append(partial)
        final = vac.finish_text()
        if final:
            confirmed.append(final)
        return confirmed, partials

    def test_speech_utterance_produces_text(self):
        config = make_config()
        vac = VACOnlineASRProcessor(
            GrowingBackend(), config,
            vad=VoiceActivityDetector(config, backend="energy"),
        )
        chunks = [tone(0.1) for _ in range(20)] + [silence(0.1) for _ in range(10)]
        confirmed, partials = self._run(vac, chunks)

        self.assertEqual(vac.status, "nonvoice")
        text = " ".join(confirmed)
        self.assertTrue(text, f"no confirmed text; partials={partials}")
        self.assertIn("w0", text)
        self.assertGreater(len(confirmed), 0)

    def test_short_blip_is_cancelled(self):
        config = make_config()
        vac = VACOnlineASRProcessor(
            GrowingBackend(), config,
            vad=VoiceActivityDetector(config, backend="energy"),
        )
        # 50ms of sound — with 100ms padding it is still below min_speech_ms.
        chunks = [tone(0.05)] + [silence(0.1) for _ in range(10)]
        confirmed, _ = self._run(vac, chunks)

        self.assertEqual(confirmed, [])
        self.assertEqual(vac.status, "nonvoice")

    def test_finish_text_resets_buffer_offset(self):
        """finish_text() restarts the VAD counter, so the buffer base must too.

        A stale _buffer_offset made later frame = timestamp - offset negative
        and broke utterance boundaries after the first flush.
        """
        config = make_config()
        vac = VACOnlineASRProcessor(
            GrowingBackend(), config,
            vad=VoiceActivityDetector(config, backend="energy"),
        )
        # Speech without silence: an utterance is open and the buffer has
        # advanced, but nothing has flushed it yet.
        for _ in range(12):
            vac.insert_audio_chunk(tone(0.1))
            vac.process_iter()
        self.assertGreater(vac._buffer_offset, 0)

        vac.finish_text()

        self.assertEqual(vac._buffer_offset, 0)
        self.assertEqual(vac.buffered_seconds, 0)

    def test_two_utterances_are_separated(self):
        config = make_config()
        vac = VACOnlineASRProcessor(
            GrowingBackend(), config,
            vad=VoiceActivityDetector(config, backend="energy"),
        )
        first = [tone(0.1) for _ in range(12)] + [silence(0.1) for _ in range(8)]
        second = [tone(0.1) for _ in range(12)] + [silence(0.1) for _ in range(8)]

        confirmed, _ = self._run(vac, first)
        first_text = " ".join(confirmed)
        confirmed2, _ = self._run(vac, second)
        second_text = " ".join(confirmed2)

        self.assertTrue(first_text)
        self.assertTrue(second_text, "second utterance was lost")


class TestSttWorkerIntegration(unittest.TestCase):
    def test_worker_flushes_text_to_translation_queue(self):
        """The real coordinator loop (minus Qt/audio hardware) end-to-end."""
        import src.stt.online_processor as op

        config = make_config()
        backend = GrowingBackend()
        ring = RingBuffer(max_seconds=10, sample_rate=SR)
        translation_queue: queue.Queue = queue.Queue()

        class Dummy:
            pass

        dummy = Dummy()
        dummy.config = config
        dummy.backend = backend
        dummy.ring_buffer = ring
        dummy._translation_queue = translation_queue
        dummy._stop = threading.Event()
        dummy._paused = threading.Event()
        dummy._queue_put = lambda q, item: q.put(item)
        dummy._notify_info = lambda text: None
        dummy._queue_text = lambda text, is_partial: translation_queue.put(
            (text, is_partial)
        )

        ring.write(np.concatenate([tone(1.0), silence(1.0)]))

        real_vad = op.VoiceActivityDetector
        with mock.patch.object(
            op, "VoiceActivityDetector",
            lambda cfg: real_vad(cfg, backend="energy"),
        ):
            thread = threading.Thread(
                target=MainPipeline._stt_worker, args=(dummy,), daemon=True
            )
            thread.start()
            deadline = time.monotonic() + 3.0
            while translation_queue.empty() and time.monotonic() < deadline:
                time.sleep(0.05)
            dummy._stop.set()
            thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        items = []
        while not translation_queue.empty():
            items.append(translation_queue.get_nowait())
        self.assertTrue(items, "STT worker emitted nothing")
        confirmed = [text for text, partial in items if not partial]
        self.assertTrue(confirmed)
        self.assertIn("w0", " ".join(confirmed))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestAntiLoop(unittest.TestCase):
    def test_collapses_repeated_words(self):
        self.assertEqual(
            _trim_repeated_ngrams("the the the the cat", 3), "the cat"
        )

    def test_collapses_repeated_phrase(self):
        self.assertEqual(
            _trim_repeated_ngrams("a b a b a b c", 3), "a b c"
        )

    def test_leaves_normal_text_alone(self):
        text = "this is a normal sentence with no loops"
        self.assertEqual(_trim_repeated_ngrams(text, 3), text)


# ---------------------------------------------------------------------------
# faster-whisper runtime selection (H11)
# ---------------------------------------------------------------------------


def _fake_ctranslate2(devices: int, supported: set[str]):
    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: devices
    module.get_supported_compute_types = lambda device: set(supported)
    return module


class TestResolveRuntime(unittest.TestCase):
    def test_missing_ctranslate2_uses_cpu(self):
        config = make_config()
        config.engine.whisper_device = "cuda"
        with mock.patch.dict(sys.modules, {"ctranslate2": None}):
            self.assertEqual(_resolve_runtime(config), ("cpu", "int8"))

    def test_cuda_without_devices_falls_back_to_cpu(self):
        config = make_config()
        config.engine.whisper_device = "cuda"
        config.engine.whisper_compute_type = "float16"
        fake = _fake_ctranslate2(0, {"float16", "int8"})
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(_resolve_runtime(config), ("cpu", "int8"))

    def test_cuda_float16_is_kept(self):
        config = make_config()
        config.engine.whisper_device = "cuda"
        config.engine.whisper_compute_type = "float16"
        fake = _fake_ctranslate2(1, {"float16", "int8"})
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(_resolve_runtime(config), ("cuda", "float16"))

    def test_cuda_without_float16_uses_supported_type(self):
        config = make_config()
        config.engine.whisper_device = "cuda"
        config.engine.whisper_compute_type = "float16"
        fake = _fake_ctranslate2(1, {"int8"})
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(_resolve_runtime(config), ("cuda", "int8"))

    def test_cpu_never_uses_float16(self):
        config = make_config()
        config.engine.whisper_device = "cpu"
        config.engine.whisper_compute_type = "float16"
        fake = _fake_ctranslate2(0, {"float32", "int8"})
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(_resolve_runtime(config), ("cpu", "int8"))

    def test_cuda_probe_exception_falls_back_to_cpu(self):
        config = make_config()
        config.engine.whisper_device = "cuda"

        module = types.ModuleType("ctranslate2")

        def boom():
            raise RuntimeError("CUDA driver missing")

        module.get_cuda_device_count = boom
        module.get_supported_compute_types = lambda device: {"float16"}
        with mock.patch.dict(sys.modules, {"ctranslate2": module}):
            self.assertEqual(_resolve_runtime(config), ("cpu", "int8"))


class TestBackendLoad(unittest.TestCase):
    def _load(self, failing: set[tuple[str, str]]):
        created: list[tuple[str, str]] = []

        class FakeModel:
            def __init__(self, name, device, compute_type, num_workers=1):
                created.append((device, compute_type))
                if (device, compute_type) in failing:
                    raise RuntimeError("cannot load")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeModel

        backend = FasterWhisperBackend(make_config())
        with mock.patch.dict(sys.modules, {"faster_whisper": fake}), \
             mock.patch(
                 "src.stt.faster_whisper._resolve_runtime",
                 return_value=("cuda", "float16"),
             ):
            backend.load()
        return backend, created

    def test_float16_failure_retries_cuda_int8(self):
        backend, created = self._load({("cuda", "float16")})
        self.assertEqual(
            created, [("cuda", "float16"), ("cuda", "int8")]
        )
        self.assertEqual((backend.device, backend.compute_type), ("cuda", "int8"))

    def test_cuda_failure_falls_back_to_cpu(self):
        backend, created = self._load(
            {("cuda", "float16"), ("cuda", "int8")}
        )
        self.assertEqual(created[-1], ("cpu", "int8"))
        self.assertEqual((backend.device, backend.compute_type), ("cpu", "int8"))

    def test_all_devices_failing_raises(self):
        with self.assertRaises(RuntimeError):
            self._load({("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")})


if __name__ == "__main__":
    unittest.main()
