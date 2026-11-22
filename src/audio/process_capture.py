"""Process-loopback audio capture (Windows 10 20348+).

Windows' session enumerator only exposes metadata. Capturing one application's
audio requires the process-loopback activation path:

    ActivateAudioInterfaceAsync(
        L"VAD\\\\Process_Loopback",
        IID_IAudioClient,
        AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        ...)

The backend used here is the small MIT-licensed ``process-audio-capture``
package, which wraps that API in a DLL and writes a WAV stream. We hand it a
named pipe, parse the WAV header, and forward float32 mono 16 kHz samples to
the same RingBuffer used by device loopback. The backend is optional: if the
package/DLL is unavailable, :class:`ProcessCaptureUnavailable` is raised and
the source manager falls back to device loopback.

The capture scope is the target process *tree*. It is intentionally not
presented as tab-level isolation: Chrome, for example, may render several
tabs from one process tree.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import struct
import sys
import threading
import time
from ctypes import wintypes
from typing import Callable

import numpy as np

from src.audio.capture import TARGET_SAMPLE_RATE, resample_audio
from src.audio.ring_buffer import RingBuffer
from src.config.settings import EngineConfig

logger = logging.getLogger(__name__)

PROCESS_LOOPBACK_MODE_INCLUDE = 0
PROCESS_LOOPBACK_MODE_EXCLUDE = 1
PIPE_BUFFER_SIZE = 64 * 1024
CONNECT_TIMEOUT_S = 5.0
STARTUP_GRACE_S = 5.0


class ProcessCaptureUnavailable(RuntimeError):
    """The native process-loopback backend is not installed/supported."""


class ProcessCaptureError(RuntimeError):
    """The process-loopback backend failed after being selected."""


# ---------------------------------------------------------------------------
# Named pipe
# ---------------------------------------------------------------------------


class _NamedPipeServer:
    """Inbound byte-mode named pipe used to receive the backend's WAV stream."""

    def __init__(self, name: str):
        self.name = name
        self._handle: int | None = None
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateNamedPipeW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
            wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        ]
        self._kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
        self._kernel32.ConnectNamedPipe.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p,
        ]
        self._kernel32.ConnectNamedPipe.restype = wintypes.BOOL
        self._kernel32.ReadFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
        ]
        self._kernel32.ReadFile.restype = wintypes.BOOL
        self._kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
        self._kernel32.DisconnectNamedPipe.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def open(self) -> None:
        # PIPE_ACCESS_INBOUND
        # PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT
        handle = self._kernel32.CreateNamedPipeW(
            self.name,
            0x00000001,
            0x00000000,
            1,
            PIPE_BUFFER_SIZE,
            PIPE_BUFFER_SIZE,
            0,
            None,
        )
        if handle == wintypes.HANDLE(-1).value or handle is None:
            raise OSError(
                ctypes.get_last_error(), f"CreateNamedPipeW failed for {self.name}"
            )
        self._handle = int(handle)

    def connect(self) -> None:
        if self._handle is None:
            raise OSError("named pipe is closed")
        ok = self._kernel32.ConnectNamedPipe(self._handle, None)
        if not ok:
            error = ctypes.get_last_error()
            # ERROR_PIPE_CONNECTED means the client won the race before we
            # called ConnectNamedPipe; that is success, not a failure.
            if error != 535:  # ERROR_PIPE_CONNECTED
                raise OSError(error, "ConnectNamedPipe failed")

    def read(self, size: int = PIPE_BUFFER_SIZE) -> bytes:
        if self._handle is None:
            return b""
        buffer = ctypes.create_string_buffer(size)
        read = wintypes.DWORD()
        ok = self._kernel32.ReadFile(
            self._handle, buffer, size, ctypes.byref(read), None
        )
        if not ok:
            error = ctypes.get_last_error()
            # ERROR_BROKEN_PIPE / ERROR_PIPE_NOT_CONNECTED at shutdown.
            if error in (109, 233):
                return b""
            raise OSError(error, "ReadFile failed")
        return buffer.raw[: read.value]

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._kernel32.DisconnectNamedPipe(handle)
        except Exception:
            logger.debug("DisconnectNamedPipe failed", exc_info=True)
        try:
            self._kernel32.CloseHandle(handle)
        except Exception:
            logger.debug("CloseHandle failed", exc_info=True)


# ---------------------------------------------------------------------------
# WAV stream parser
# ---------------------------------------------------------------------------


class WavStreamDecoder:
    """Incrementally parse the WAV header and decode float32 mono samples."""

    def __init__(
        self,
        target_rate: int = TARGET_SAMPLE_RATE,
        filter_cache: dict[int, np.ndarray] | None = None,
    ):
        self.target_rate = target_rate
        self.filter_cache = filter_cache if filter_cache is not None else {}
        self.channels = 0
        self.sample_rate = 0
        self.bits_per_sample = 0
        self.block_align = 0
        self.format_tag = 0
        self.header_ready = False
        self._header = bytearray()
        self._pcm_remainder = b""

    def feed(self, data: bytes) -> np.ndarray:
        if not data:
            return np.empty(0, dtype=np.float32)

        if not self.header_ready:
            self._header.extend(data)
            pcm = self._consume_header()
            if not self.header_ready:
                return np.empty(0, dtype=np.float32)
        else:
            pcm = data

        if not pcm:
            return np.empty(0, dtype=np.float32)

        pcm = self._pcm_remainder + pcm
        frame_bytes = max(1, self.block_align or (self.channels * self.bits_per_sample // 8))
        usable = (len(pcm) // frame_bytes) * frame_bytes
        self._pcm_remainder = pcm[usable:]
        pcm = pcm[:usable]
        if not pcm:
            return np.empty(0, dtype=np.float32)

        samples = self._decode(pcm)
        if samples.size == 0:
            return samples
        if self.sample_rate and self.sample_rate != self.target_rate:
            samples = resample_audio(
                samples, self.sample_rate, self.target_rate,
                filter_cache=self.filter_cache,
            )
        return np.ascontiguousarray(samples, dtype=np.float32)

    def _consume_header(self) -> bytes:
        data = bytes(self._header)
        if len(data) < 12:
            return b""
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ProcessCaptureError("process-audio backend did not send a WAV stream")

        offset = 12
        while True:
            if len(data) < offset + 8:
                return b""
            chunk_id = data[offset:offset + 4]
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            body_start = offset + 8
            body_end = body_start + chunk_size

            if chunk_id == b"data":
                # A streaming WAV has no final payload size yet. Do not wait
                # for chunk_size bytes: everything after this 8-byte header is
                # PCM and can be forwarded immediately.
                if not self.channels or not self.sample_rate or not self.bits_per_sample:
                    raise ProcessCaptureError(
                        "process-audio WAV stream has data before a valid fmt chunk"
                    )
                self.header_ready = True
                self._header.clear()
                return data[body_start:]

            if len(data) < body_end:
                return b""

            if chunk_id == b"fmt ":
                self._parse_fmt(data[body_start:body_end])
            offset = body_end + (chunk_size & 1)

    def _parse_fmt(self, body: bytes) -> None:
        if len(body) < 16:
            raise ProcessCaptureError("process-audio fmt chunk is too short")
        (
            self.format_tag,
            self.channels,
            self.sample_rate,
            _avg_bytes,
            self.block_align,
            self.bits_per_sample,
        ) = struct.unpack_from("<HHIIHH", body, 0)
        # WAVE_FORMAT_EXTENSIBLE (0xFFFE) still carries the real sample size
        # in wBitsPerSample. The package writes 32-bit IEEE float.
        if self.channels <= 0 or self.sample_rate <= 0 or self.bits_per_sample <= 0:
            raise ProcessCaptureError("process-audio WAV stream has an invalid fmt chunk")

    def _decode(self, pcm: bytes) -> np.ndarray:
        if self.bits_per_sample == 32:
            # The backend uses IEEE float32 for its WAV output.
            samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32, copy=False)
        elif self.bits_per_sample == 16:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        elif self.bits_per_sample == 24:
            raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
            values = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            values = np.where(values & 0x800000, values - 0x1000000, values)
            samples = values.astype(np.float32) / 8388608.0
        else:
            raise ProcessCaptureError(
                f"unsupported process-audio sample size: {self.bits_per_sample} bit"
            )

        if self.channels > 1:
            usable = (samples.size // self.channels) * self.channels
            samples = samples[:usable].reshape(-1, self.channels).mean(axis=1)
        return np.ascontiguousarray(samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Capture source
# ---------------------------------------------------------------------------


class ProcessLoopbackCapture:
    """Capture one process tree and feed the shared RingBuffer."""

    def __init__(
        self,
        config: EngineConfig,
        ring_buffer: RingBuffer,
        pid: int,
        *,
        executable: str = "",
        label: str = "",
        on_error: Callable[[str], None] | None = None,
        on_lost: Callable[[str], None] | None = None,
        backend_factory=None,
    ):
        self.config = config
        self.ring_buffer = ring_buffer
        self.pid = int(pid)
        self.executable = executable
        self.label = label or executable or f"PID {pid}"
        self.on_error = on_error
        self.on_lost = on_lost
        self._backend_factory = backend_factory or _make_backend

        self._backend = None
        self._pipe: _NamedPipeServer | None = None
        self._reader: threading.Thread | None = None
        self._monitor: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._reader_stop = threading.Event()
        self._connected = threading.Event()
        self._closing = False
        self._starting = False
        self._lost_reported = False
        self._reader_error: Exception | None = None
        self._decoder = WavStreamDecoder(filter_cache={})
        self._bytes_received = 0
        self._last_audio = 0.0

    # -- support -------------------------------------------------------------

    @staticmethod
    def is_supported() -> bool:
        if sys.platform != "win32":
            return False
        if importlib.util.find_spec("process_audio_capture") is None:
            return False
        try:
            from process_audio_capture import ProcessAudioCapture

            return bool(ProcessAudioCapture.is_supported())
        except Exception:
            logger.debug("process-audio backend support check failed", exc_info=True)
            return False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self.is_running:
            return
        if sys.platform != "win32":
            raise ProcessCaptureUnavailable("process-loopback capture requires Windows")
        if not self.is_supported():
            raise ProcessCaptureUnavailable(
                "process-audio-capture is not installed or is unsupported. "
                "Install process-audio-capture to capture one application."
            )
        if self.pid <= 0:
            raise ProcessCaptureError("no target process id for process-loopback capture")

        self._stop_event.clear()
        self._reader_stop.clear()
        self._connected.clear()
        self._closing = False
        self._lost_reported = False
        self._reader_error = None
        self._decoder = WavStreamDecoder(filter_cache={})
        self._bytes_received = 0

        self._pipe = _NamedPipeServer(
            rf"\\.\pipe\liveTranslate_audio_{os.getpid()}_{time.monotonic_ns()}"
        )
        self._starting = True
        try:
            self._pipe.open()
            self._reader = threading.Thread(
                target=self._read_loop, name="ProcessCaptureReader", daemon=True
            )
            self._reader.start()

            self._backend = self._backend_factory(
                pid=self.pid,
                output_path=self._pipe.name,
                mode=PROCESS_LOOPBACK_MODE_INCLUDE,
            )
            self._backend.start()
        except Exception as exc:
            self._stop_event.set()
            self._reader_stop.set()
            self._starting = False
            self._close_pipe()
            self._join_threads()
            self._backend = None
            if isinstance(exc, (ProcessCaptureUnavailable, ProcessCaptureError)):
                raise
            raise ProcessCaptureError(f"could not start process capture: {exc}") from exc

        if not self._connected.wait(CONNECT_TIMEOUT_S):
            self.stop()
            raise ProcessCaptureError(
                f"process-audio backend did not connect for {self.label}"
            )
        if self._reader_error is not None:
            self.stop()
            raise ProcessCaptureError(
                f"process-audio stream failed for {self.label}: {self._reader_error}"
            )

        self._starting = False
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="ProcessCaptureMonitor", daemon=True
        )
        self._monitor.start()
        logger.info("Process-loopback capture started: %s (PID %d)", self.label, self.pid)

    def _read_loop(self) -> None:
        try:
            assert self._pipe is not None
            self._pipe.connect()
            self._connected.set()
            while not self._reader_stop.is_set():
                data = self._pipe.read()
                if not data:
                    break
                self._bytes_received += len(data)
                self._last_audio = time.monotonic()
                samples = self._decoder.feed(data)
                if samples.size:
                    self.ring_buffer.write(samples)
        except Exception as exc:
            self._reader_error = exc
            self._connected.set()
            if not self._closing:
                self._notify_lost(f"Process audio stream failed: {exc}")
        finally:
            if not self._connected.is_set():
                self._connected.set()
            if not self._closing and self._reader_error is None:
                self._notify_lost("Process audio stream ended unexpectedly")

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            if self._backend is None:
                return
            try:
                if not bool(self._backend.is_capturing):
                    self._notify_lost("Process audio capture stopped unexpectedly")
                    return
            except Exception:
                logger.debug("process backend state check failed", exc_info=True)

            try:
                import psutil

                if not psutil.pid_exists(self.pid):
                    self._notify_lost(
                        f"{self.label} exited; switching to system audio"
                    )
                    return
            except ImportError:
                pass
            except Exception:
                logger.debug("process existence check failed", exc_info=True)

    def _notify_lost(self, message: str) -> None:
        if self._starting:
            # start() will surface the reader error and clean up; reporting
            # from the reader thread here can deadlock the source manager.
            return
        if self._lost_reported:
            return
        self._lost_reported = True
        logger.warning("%s", message)
        if self.on_lost is not None:
            try:
                self.on_lost(message)
                return
            except Exception:
                logger.exception("on_lost callback failed")
        self._notify_error(message)

    def _notify_error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                logger.exception("on_error callback failed")

    def _close_pipe(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.close()

    def _join_threads(self) -> None:
        for thread in (self._reader, self._monitor):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._reader = None
        self._monitor = None

    def stop(self) -> None:
        """Stop the backend, close the pipe, and release all threads."""
        self._closing = True
        self._stop_event.set()
        self._starting = False
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.stop()
            except Exception:
                logger.warning("process-audio backend stop failed", exc_info=True)

        # Let the reader drain until the writer closes, then remove its pipe.
        self._reader_stop.set()
        self._close_pipe()
        self._join_threads()

    @property
    def is_running(self) -> bool:
        if self._backend is None:
            return False
        try:
            if not bool(self._backend.is_capturing):
                return False
        except Exception:
            return False
        return self._reader is not None and self._reader.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "source": self.label,
            "mode": "session",
            "pid": self.pid,
            "bytes": self._bytes_received,
            "dropped_seconds": self.ring_buffer.dropped_seconds,
        }


def _make_backend(*, pid: int, output_path: str, mode: int):
    from process_audio_capture import ProcessAudioCapture

    return ProcessAudioCapture(pid=pid, output_path=output_path, mode=mode)
