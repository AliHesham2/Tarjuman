"""WASAPI loopback capture via PyAudioWPatch.

Captures whatever Windows is playing on the default output device (the
meeting audio), with no microphone involved.  The device is opened in its
native format and converted in the callback to float32 mono at 16 kHz for
Whisper:

  * int16 PCM -> float32 in [-1, 1]
  * N channels -> mono (average)
  * device rate -> 16 kHz (anti-aliased FIR + linear interpolation)

The callback never blocks and never raises: a stalled consumer must not
stall the audio thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from src.audio.ring_buffer import RingBuffer
from src.config.settings import EngineConfig

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
PA_CONTINUE = 0  # portaudio paContinue


class LoopbackDeviceNotFoundError(RuntimeError):
    """No WASAPI loopback device available / audio backend not installed."""


def _is_loopback(info: dict) -> bool:
    return bool(info.get("isLoopbackDevice") or info.get("isLoopback"))


def _iter_loopback_devices(pa, pyaudio_mod):
    """Yield every WASAPI loopback device, newest PyAudioWPatch API first."""
    generator = getattr(pa, "get_loopback_device_info_generator", None)
    if callable(generator):
        try:
            yield from generator()
            return
        except Exception:
            logger.debug("get_loopback_device_info_generator failed", exc_info=True)

    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if _is_loopback(info):
            yield info


def list_loopback_devices(pa, pyaudio_mod) -> list[dict]:
    """All loopback devices (for the settings dialog / diagnostics)."""
    return list(_iter_loopback_devices(pa, pyaudio_mod))


def find_loopback_device(pa, pyaudio_mod=None) -> dict | None:
    """Return the loopback device matching the *default* output, if possible.

    PyAudioWPatch exposes a loopback twin for every playback device, so
    simply taking the first one can capture the wrong monitor (e.g. HDMI
    instead of the speakers/headphones the meeting is actually on).
    """
    if pyaudio_mod is None:
        import pyaudiowpatch as pyaudio_mod

    default_name = ""
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if _is_loopback(default_out):
            return default_out
        default_name = str(default_out.get("name", ""))
    except Exception:
        logger.debug("Could not resolve the default WASAPI output", exc_info=True)

    if default_name:
        for info in _iter_loopback_devices(pa, pyaudio_mod):
            name = str(info.get("name", ""))
            if default_name in name or name.startswith(default_name):
                return info

    for info in _iter_loopback_devices(pa, pyaudio_mod):
        return info
    return None


class AudioCapture:
    """Callback-based loopback capture feeding a RingBuffer."""

    def __init__(
        self,
        config: EngineConfig,
        ring_buffer: RingBuffer,
        on_error: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.ring_buffer = ring_buffer
        self.on_error = on_error

        self._pa = None
        self._pa_module = None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._device_info: dict | None = None
        self._device_rate = TARGET_SAMPLE_RATE
        self._device_channels = 1
        self._resample_taps: dict[int, np.ndarray] = {}

        # diagnostics
        self._callback_count = 0
        self._overflow_count = 0
        self._last_callback = 0.0
        self._last_stall_notice = 0.0
        # Silence synthesis: WASAPI loopback sends no callbacks while the
        # system plays nothing, so the monitor injects zeros itself.
        self._silence_until = 0.0

    # -- device selection ----------------------------------------------------

    def _pick_device(self, pyaudio_mod) -> dict:
        if self._pa is None:
            self._pa = pyaudio_mod.PyAudio()
        pa = self._pa

        index = int(self.config.audio.device_index)
        if not self.config.audio.auto_select_device and index >= 0:
            info = pa.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                raise LoopbackDeviceNotFoundError(
                    f"Audio device {index} ({info.get('name', '?')}) has no input "
                    "channels — pick a loopback device instead."
                )
            if not _is_loopback(info):
                logger.warning(
                    "Device %d (%s) is not a WASAPI loopback device; capturing "
                    "its input directly.",
                    index, info.get("name", "?"),
                )
            return info

        device = find_loopback_device(pa, pyaudio_mod)
        if device is None:
            raise LoopbackDeviceNotFoundError(
                "No WASAPI loopback device found. Make sure a playback device "
                "is enabled in Windows, or install VB-Cable "
                "(https://vb-audio.com/Cable/) as a virtual output."
            )
        return device

    def list_devices(self) -> list[dict]:
        """Loopback devices available on this machine."""
        import pyaudiowpatch as pyaudio

        pa = self._pa or pyaudio.PyAudio()
        try:
            return list_loopback_devices(pa, pyaudio)
        finally:
            if self._pa is None:
                pa.terminate()

    # -- capture lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Open the loopback stream and start the monitor thread."""
        if self.is_running:
            return

        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise LoopbackDeviceNotFoundError(
                "PyAudioWPatch is not installed — run "
                "`pip install PyAudioWPatch` to enable audio capture."
            ) from exc

        self._pa_module = pyaudio
        try:
            device = self._pick_device(pyaudio)
            self._device_info = device
            self._device_rate = int(device["defaultSampleRate"])
            self._device_channels = max(1, int(device.get("maxInputChannels", 1)))

            logger.info(
                "Loopback device: %s (%d Hz, %d ch)",
                device.get("name", "?"), self._device_rate, self._device_channels,
            )

            frames_per_buffer = max(
                1,
                round(self.config.chunk_samples * self._device_rate / TARGET_SAMPLE_RATE),
            )

            self._stop_event.clear()
            self._callback_count = 0
            self._overflow_count = 0
            self._last_callback = time.monotonic()
            self._silence_until = time.monotonic()

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._device_channels,
                rate=self._device_rate,
                input=True,
                input_device_index=int(device["index"]),
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._on_audio,
            )
        except Exception:
            self._terminate()
            raise

        # The PortAudio callback runs on its own thread; this thread only
        # watches for a dead/stalled stream and keeps the object alive.
        self._thread = threading.Thread(
            target=self._monitor, name="AudioCapture", daemon=True
        )
        self._thread.start()

    def _on_audio(self, in_data, frame_count, time_info, status_flags):
        """PortAudio callback — must be fast and must never raise."""
        self._last_callback = time.monotonic()
        self._callback_count += 1

        if status_flags:
            overflow_flag = getattr(self._pa_module, "paInputOverflow", 0)
            if overflow_flag and status_flags & overflow_flag:
                self._overflow_count += 1
                if self._overflow_count == 1 or self._overflow_count % 100 == 0:
                    logger.warning(
                        "Audio input overflow (#%d) — the capture callback is "
                        "falling behind, audio was dropped",
                        self._overflow_count,
                    )
            else:
                logger.debug("PortAudio status flags: %s", status_flags)

        try:
            samples = self._to_float_mono(in_data)
            if self._device_rate != TARGET_SAMPLE_RATE:
                samples = resample_audio(
                    samples, self._device_rate, TARGET_SAMPLE_RATE,
                    filter_cache=self._resample_taps,
                )
            self.ring_buffer.write(samples)
        except Exception:
            logger.exception("Error in audio callback")

        return (None, PA_CONTINUE)

    def _to_float_mono(self, in_data: bytes) -> np.ndarray:
        pcm = np.frombuffer(in_data, dtype=np.int16)
        if pcm.size == 0:
            return np.empty(0, dtype=np.float32)

        channels = self._device_channels
        if channels > 1:
            usable = (pcm.size // channels) * channels
            if usable != pcm.size:
                pcm = pcm[:usable]
            samples = pcm.astype(np.float32) / 32768.0
            samples = samples.reshape(-1, channels).mean(axis=1)
            return samples.astype(np.float32, copy=False)

        return (pcm.astype(np.float32) / 32768.0).astype(np.float32, copy=False)

    def _monitor(self) -> None:
        """Watch the stream; keep a silent (callback-less) one flowing.

        WASAPI loopback delivers no callbacks at all while the system is
        quiet.  That is normal, not a fault: without injected silence the VAD
        never sees audio and an open utterance never ends.
        """
        while not self._stop_event.wait(0.25):
            if self._stream is None:
                return
            try:
                active = bool(self._stream.is_active())
            except Exception:
                active = False

            if not active:
                self._notify_error("Audio stream stopped unexpectedly")
                return

            now = time.monotonic()
            if now - max(self._last_callback, self._silence_until) > 0.3:
                self._fill_silence(now)

    def _fill_silence(self, now: float) -> None:
        """Write zeros covering the quiet gap up to ``now`` (max 2 s)."""
        ref = max(self._last_callback, self._silence_until)
        n = int(min(now - ref, 2.0) * TARGET_SAMPLE_RATE)
        if n <= 0:
            return
        try:
            self.ring_buffer.write(np.zeros(n, dtype=np.float32))
        except Exception:
            logger.debug("Could not write silence to the ring buffer", exc_info=True)
        self._silence_until = now

    def _notify_error(self, message: str) -> None:
        logger.error("%s", message)
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                logger.exception("on_error callback failed")

    def stop(self) -> None:
        """Stop capture and release PortAudio resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._terminate()

    def _terminate(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                logger.warning("Error closing audio stream", exc_info=True)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                logger.debug("Error terminating PyAudio", exc_info=True)
            self._pa = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "device": (self._device_info or {}).get("name"),
            "rate": self._device_rate,
            "channels": self._device_channels,
            "callbacks": self._callback_count,
            "overflows": self._overflow_count,
            "dropped_seconds": self.ring_buffer.dropped_seconds,
        }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _lowpass_taps(cutoff_hz: float, sample_rate: int, taps: int = 31) -> np.ndarray:
    """Windowed-sinc low-pass FIR, normalised to unity DC gain."""
    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / sample_rate
    h = np.sinc(2.0 * fc * n) * np.hamming(taps)
    total = h.sum()
    if total:
        h = h / total
    return h.astype(np.float32)


def resample_audio(
    samples: np.ndarray,
    orig_rate: int,
    target_rate: int,
    filter_cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Downsample/upsample with anti-aliasing (numpy only).

    For downsampling, a low-pass FIR at 45% of the target rate runs first
    so 48 kHz content above 8 kHz does not fold back into the speech band.
    """
    samples = np.asarray(samples, dtype=np.float32).ravel()
    if samples.size == 0 or orig_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    if orig_rate > target_rate:
        if filter_cache is not None and orig_rate in filter_cache:
            taps = filter_cache[orig_rate]
        else:
            taps = _lowpass_taps(0.45 * target_rate, orig_rate)
            if filter_cache is not None:
                filter_cache[orig_rate] = taps
        samples = np.convolve(samples, taps, mode="same").astype(np.float32)

    duration = samples.size / float(orig_rate)
    target_len = max(1, int(round(duration * target_rate)))

    # Integer decimation can skip interpolation entirely.
    if orig_rate > target_rate and orig_rate % target_rate == 0:
        step = orig_rate // target_rate
        return np.ascontiguousarray(samples[::step][:target_len])

    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)
