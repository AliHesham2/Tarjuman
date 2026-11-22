"""Phase 1 tests — ring buffer, resampling, VAD state machine, capture math.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from src.audio.capture import (
    AudioCapture,
    LoopbackDeviceNotFoundError,
    _lowpass_taps,
    resample_audio,
)
from src.audio.ring_buffer import RingBuffer
from src.audio.vad import FRAME_SAMPLES, VoiceActivityDetector
from src.config.settings import EngineConfig


def tone(samples: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class RingBufferTests(unittest.TestCase):
    def test_write_and_snapshot(self):
        rb = RingBuffer(max_seconds=1.0)
        rb.write(np.arange(100, dtype=np.float32))
        rb.write(np.arange(100, 300, dtype=np.float32))

        self.assertEqual(rb.available_samples, 300)
        np.testing.assert_array_equal(rb.read_all(), np.arange(300, dtype=np.float32))
        # read_all does not consume
        self.assertEqual(rb.available_samples, 300)

    def test_consume_all(self):
        rb = RingBuffer(max_seconds=1.0)
        rb.write(np.ones(50, dtype=np.float32))
        out = rb.consume_all()
        self.assertEqual(out.size, 50)
        self.assertEqual(rb.available_samples, 0)
        self.assertEqual(rb.consume_all().size, 0)

    def test_drops_oldest_when_full(self):
        rb = RingBuffer(max_seconds=1.0)  # 16000 samples
        rb.write(np.ones(16000, dtype=np.float32))
        dropped = rb.write(np.full(1000, 2.0, dtype=np.float32))

        self.assertEqual(dropped, 1000)
        self.assertEqual(rb.available_samples, 16000)
        data = rb.read_all()
        self.assertTrue(np.all(data[-1000:] == 2.0))
        self.assertAlmostEqual(rb.dropped_seconds, 1000 / 16000)

    def test_oversized_chunk_keeps_only_the_tail(self):
        rb = RingBuffer(max_seconds=1.0)
        chunk = np.arange(20000, dtype=np.float32)
        rb.write(chunk)

        self.assertEqual(rb.available_samples, 16000)
        np.testing.assert_array_equal(rb.read_all(), chunk[-16000:])
        self.assertEqual(rb.dropped_seconds, 4000 / 16000)

    def test_read_exact_samples_across_chunks(self):
        rb = RingBuffer(max_seconds=1.0)
        rb.write(np.arange(10, dtype=np.float32))
        rb.write(np.arange(10, 30, dtype=np.float32))

        out = rb.read(15)
        np.testing.assert_array_equal(out, np.arange(15, dtype=np.float32))
        self.assertEqual(rb.available_samples, 15)
        np.testing.assert_array_equal(rb.read_all(), np.arange(15, 30, dtype=np.float32))

    def test_blocking_read_wakes_on_write(self):
        rb = RingBuffer(max_seconds=1.0)
        received = {}

        def reader():
            received["data"] = rb.read(1000, timeout=2.0)

        thread = threading.Thread(target=reader)
        thread.start()
        time.sleep(0.15)
        self.assertTrue(thread.is_alive(), "read() should be blocking")

        rb.write(np.ones(1000, dtype=np.float32))
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(received["data"])
        self.assertEqual(received["data"].size, 1000)

    def test_blocking_read_timeout_returns_none(self):
        rb = RingBuffer(max_seconds=1.0)
        start = time.monotonic()
        self.assertIsNone(rb.read(1000, timeout=0.1))
        self.assertLess(time.monotonic() - start, 1.0)

    def test_close_wakes_reader(self):
        rb = RingBuffer(max_seconds=1.0)
        result = {}

        thread = threading.Thread(target=lambda: result.update(data=rb.read(1000)))
        thread.start()
        time.sleep(0.1)
        rb.close()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNone(result["data"])
        self.assertEqual(rb.write(np.ones(10, dtype=np.float32)), 0)


class ResampleTests(unittest.TestCase):
    def test_48k_to_16k_length(self):
        out = resample_audio(np.zeros(12000, dtype=np.float32), 48000, 16000)
        self.assertEqual(out.size, 4000)

    def test_44k1_to_16k_length(self):
        out = resample_audio(np.zeros(11025, dtype=np.float32), 44100, 16000)
        self.assertEqual(out.size, 4000)

    def test_noop_at_target_rate(self):
        data = np.arange(100, dtype=np.float32)
        np.testing.assert_array_equal(resample_audio(data, 16000, 16000), data)

    def test_tone_is_preserved(self):
        freq = 1000.0
        src = np.sin(2 * np.pi * freq * np.arange(48000) / 48000).astype(np.float32)
        out = resample_audio(src, 48000, 16000)

        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(out.size, 1 / 16000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak_hz, freq, delta=30.0)
        # amplitude close to the original (FIR + decimation should not kill it)
        self.assertGreater(np.max(np.abs(out)), 0.9)

    def test_lowpass_taps_have_unity_dc_gain(self):
        taps = _lowpass_taps(7200, 48000)
        self.assertAlmostEqual(float(taps.sum()), 1.0, places=5)


class CaptureMathTests(unittest.TestCase):
    def setUp(self):
        self.capture = AudioCapture(EngineConfig(), RingBuffer(1.0))
    def test_mono_conversion(self):
        pcm = (np.array([0, 32767, -32768, 16384], dtype=np.int16)).tobytes()
        self.capture._device_channels = 1
        out = self.capture._to_float_mono(pcm)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.size, 4)
        self.assertAlmostEqual(float(out[1]), 32767 / 32768, places=6)

    def test_stereo_downmix(self):
        # frames: (1000, 3000), (-1000, -3000)
        pcm = np.array([1000, 3000, -1000, -3000], dtype=np.int16).tobytes()
        self.capture._device_channels = 2
        out = self.capture._to_float_mono(pcm)
        self.assertEqual(out.size, 2)
        self.assertAlmostEqual(float(out[0]), 2000 / 32768, places=6)
        self.assertAlmostEqual(float(out[1]), -2000 / 32768, places=6)

    def test_callback_feeds_ring_buffer(self):
        self.capture._device_channels = 1
        self.capture._device_rate = 16000
        self.capture._pa_module = None
        data = np.zeros(4000, dtype=np.int16).tobytes()

        self.capture._on_audio(data, 4000, None, 0)
        self.assertEqual(self.capture.ring_buffer.available_samples, 4000)

    def test_silent_stream_fills_ring_buffer_without_error(self):
        """A quiet loopback stream is normal: zeros, not an ERROR."""
        errors = []
        capture = AudioCapture(EngineConfig(), RingBuffer(5.0), on_error=errors.append)
        capture._stream = type("FakeStream", (), {"is_active": lambda self: True})()
        now = time.monotonic()
        capture._last_callback = now - 1.0
        capture._silence_until = 0.0

        capture._fill_silence(now)

        self.assertEqual(errors, [])
        available = capture.ring_buffer.available_samples
        self.assertGreater(available, 0)
        self.assertEqual(available, 16000)
        block = capture.ring_buffer.read(available)
        self.assertTrue(np.all(block == 0.0))


class _FakePyAudio:
    """Minimal PyAudioWPatch stand-in for device-selection tests."""

    def __init__(self, devices, default_output_index):
        self._devices = devices
        self._default = default_output_index

    def get_host_api_info_by_type(self, _host_api):
        return {"defaultOutputDevice": self._default}

    def get_device_info_by_index(self, index):
        return self._devices[index]

    def get_device_count(self):
        return len(self._devices)

    def get_loopback_device_info_generator(self):
        for info in self._devices:
            if info.get("isLoopbackDevice"):
                yield info


class _FakePyAudioModule:
    paWASAPI = 1
    paInputOverflow = 2


class DeviceSelectionTests(unittest.TestCase):
    def _capture(self, devices, default_index, **audio_overrides):
        cfg = EngineConfig()
        for key, value in audio_overrides.items():
            setattr(cfg.audio, key, value)
        capture = AudioCapture(cfg, RingBuffer(1.0))
        capture._pa = _FakePyAudio(devices, default_index)
        return capture

    def test_prefers_loopback_twin_of_default_output(self):
        devices = [
            {"index": 0, "name": "HDMI Monitor (loopback)", "isLoopbackDevice": False,
             "maxInputChannels": 2, "defaultSampleRate": 48000},
            {"index": 1, "name": "Speakers", "isLoopbackDevice": False,
             "maxInputChannels": 0, "defaultSampleRate": 48000},
            {"index": 2, "name": "HDMI Monitor (loopback)", "isLoopbackDevice": True,
             "maxInputChannels": 2, "defaultSampleRate": 48000},
            {"index": 3, "name": "Speakers (loopback)", "isLoopbackDevice": True,
             "maxInputChannels": 2, "defaultSampleRate": 48000},
        ]
        capture = self._capture(devices, default_index=1)

        device = capture._pick_device(_FakePyAudioModule)
        self.assertEqual(device["index"], 3, "must capture the default output, not HDMI")

    def test_falls_back_to_any_loopback_when_names_do_not_match(self):
        devices = [
            {"index": 0, "name": "Speakers", "isLoopbackDevice": False,
             "maxInputChannels": 0, "defaultSampleRate": 48000},
            {"index": 1, "name": "Some Virtual Cable", "isLoopbackDevice": True,
             "maxInputChannels": 2, "defaultSampleRate": 48000},
        ]
        capture = self._capture(devices, default_index=0)
        self.assertEqual(capture._pick_device(_FakePyAudioModule)["index"], 1)

    def test_raises_when_no_loopback_exists(self):
        devices = [
            {"index": 0, "name": "Speakers", "isLoopbackDevice": False,
             "maxInputChannels": 0, "defaultSampleRate": 48000},
        ]
        capture = self._capture(devices, default_index=0)
        with self.assertRaises(LoopbackDeviceNotFoundError):
            capture._pick_device(_FakePyAudioModule)

    def test_explicit_index_requires_input_channels(self):
        devices = [
            {"index": 0, "name": "Speakers", "isLoopbackDevice": False,
             "maxInputChannels": 0, "defaultSampleRate": 48000},
        ]
        capture = self._capture(
            devices, default_index=0, auto_select_device=False, device_index=0
        )
        with self.assertRaises(LoopbackDeviceNotFoundError):
            capture._pick_device(_FakePyAudioModule)


class VadTests(unittest.TestCase):
    def _make_vad(self, **overrides) -> VoiceActivityDetector:
        cfg = EngineConfig()
        cfg.vad.min_silence_ms = 300
        cfg.vad.padding_ms = 100
        cfg.vad.min_speech_ms = 200
        cfg.vad.fallback_dbfs_threshold = -40.0
        for key, value in overrides.items():
            setattr(cfg.vad, key, value)
        return VoiceActivityDetector(cfg, backend="energy")

    @staticmethod
    def _feed(vad: VoiceActivityDetector, audio: np.ndarray, window: int = 1600):
        events = []
        for start in range(0, len(audio), window):
            events.extend(vad.process(audio[start:start + window]))
        return events

    def test_speech_then_silence_emits_start_and_end(self):
        vad = self._make_vad()
        audio = np.concatenate([tone(16000), np.zeros(16000, dtype=np.float32)])
        events = self._feed(vad, audio)

        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("end", types)
        self.assertLess(types.index("start"), types.index("end"))

        start = next(e for e in events if e["type"] == "start")
        end = next(e for e in events if e["type"] == "end")
        self.assertGreaterEqual(start["timestamp"], 0)
        self.assertGreater(end["timestamp"], start["timestamp"])

    def test_short_blip_is_cancelled(self):
        vad = self._make_vad()
        audio = np.concatenate([tone(1024), np.zeros(8000, dtype=np.float32)])
        events = self._feed(vad, audio)

        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("cancel", types)
        self.assertNotIn("end", types)

    def test_no_events_for_pure_silence(self):
        vad = self._make_vad()
        events = self._feed(vad, np.zeros(16000, dtype=np.float32))
        self.assertEqual(events, [])

    def test_end_of_stream_closes_open_utterance(self):
        vad = self._make_vad()
        self._feed(vad, tone(16000))
        events = vad.end_of_stream()

        self.assertEqual([e["type"] for e in events], ["end"])

    def test_variable_chunk_sizes_do_not_lose_frames(self):
        vad = self._make_vad()
        audio = np.concatenate([tone(8000), np.zeros(8000, dtype=np.float32)])
        events = []
        for size in (1, 7, 511, 512, 513, 999, 1600, 23):
            chunk = audio[:size]
            audio = audio[size:]
            events.extend(vad.process(chunk))
        events.extend(vad.process(audio))

        self.assertEqual(len([e for e in events if e["type"] == "start"]), 1)
        self.assertEqual(len([e for e in events if e["type"] == "end"]), 1)

    def test_reset_clears_state(self):
        vad = self._make_vad()
        self._feed(vad, tone(8000))
        self.assertTrue(vad._triggered)
        vad.reset()
        self.assertFalse(vad._triggered)
        self.assertEqual(vad._current_sample, 0)

    def test_frame_size_is_silero_compliant(self):
        self.assertEqual(FRAME_SAMPLES, 512)


if __name__ == "__main__":
    unittest.main()
