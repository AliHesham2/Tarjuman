"""Select between device loopback and per-process loopback at runtime.

``MainPipeline`` only knows that it owns an object with ``start``, ``stop`` and
``is_running``. This manager hides the two capture implementations and owns
the fallback policy:

  * Process capture stops because the app exited or the stream failed
    -> start device loopback, keep the pipeline running.
  * The requested app is not running
    -> use device loopback and tell the UI why.
  * Device loopback fails
    -> report the error through the normal pipeline error path.

A fallback is triggered by process exit/capture failure, never by silence:
silence is valid meeting audio and must not cause a source switch.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.audio.capture import AudioCapture, LoopbackDeviceNotFoundError
from src.audio.process_capture import (
    ProcessCaptureError,
    ProcessCaptureUnavailable,
    ProcessLoopbackCapture,
)
from src.audio.ring_buffer import RingBuffer
from src.audio.session_manager import AudioSessionInfo, AudioSessionManager
from src.config.settings import EngineConfig

logger = logging.getLogger(__name__)


class SourceSelectionError(RuntimeError):
    """The requested app/session could not be resolved."""


@dataclass(frozen=True)
class AudioSelection:
    """A user-visible audio-source choice."""

    mode: str = "device"          # device | session | auto
    executable: str = ""
    pid: int = 0
    label: str = "All system audio"

    def normalized(self) -> "AudioSelection":
        mode = self.mode if self.mode in ("device", "session", "auto") else "device"
        # The dataclass default label is for the common device choice; do not
        # let it leak into a session/auto choice that used the default.
        label = self.label
        if label == "All system audio":
            label = ""
        if mode == "device":
            return AudioSelection("device", label=label or "All system audio")
        if mode == "auto":
            return AudioSelection("auto", label=label or "Auto-detect meeting app")
        return AudioSelection(
            "session",
            executable=self.executable,
            pid=int(self.pid or 0),
            label=label or self.executable or f"PID {self.pid}",
        )

    @classmethod
    def from_session(cls, session: AudioSessionInfo) -> "AudioSelection":
        return cls(
            mode="session",
            executable=session.executable or session.name,
            pid=session.pid,
            label=session.label,
        )


class AudioSourceManager:
    """Capture facade with runtime source switching and safe fallback."""

    def __init__(
        self,
        config: EngineConfig,
        ring_buffer: RingBuffer,
        *,
        on_error: Callable[[str], None] | None = None,
        on_source_changed: Callable[[AudioSelection, str], None] | None = None,
        session_manager: AudioSessionManager | None = None,
        capture_factory=None,
        process_factory=None,
    ):
        self.config = config
        self.ring_buffer = ring_buffer
        self.on_error = on_error
        self.on_source_changed = on_source_changed
        self.session_manager = session_manager or AudioSessionManager(
            getattr(config.audio, "auto_detect_apps", None),
            exclude_pid=os.getpid(),
        )

        self._capture_factory = capture_factory or self._default_capture
        self._process_factory = process_factory or self._default_process
        self._lock = threading.RLock()
        self._source = None
        self._stop_requested = False
        self._fallback_thread: threading.Thread | None = None
        self._generation = 0

        mode = getattr(config.audio, "capture_mode", "device")
        selected_app = getattr(config.audio, "selected_app", "") or ""
        self._selection = AudioSelection(
            mode=mode,
            executable=selected_app,
            label=self._initial_label(mode, selected_app),
        ).normalized()
        self._active = self._selection
        self._last_reason = ""

    # -- factories -----------------------------------------------------------

    def _default_capture(self, config, ring_buffer, *, on_error):
        return AudioCapture(config, ring_buffer, on_error=on_error)

    def _default_process(self, config, ring_buffer, pid, **kwargs):
        return ProcessLoopbackCapture(config, ring_buffer, pid, **kwargs)

    @staticmethod
    def _initial_label(mode: str, executable: str) -> str:
        if mode == "device":
            return "All system audio"
        if mode == "auto":
            return "Auto-detect meeting app"
        return executable or "Selected application"

    # -- discovery -----------------------------------------------------------

    @property
    def process_capture_available(self) -> bool:
        return (
            ProcessLoopbackCapture.is_supported()
            and AudioSessionManager.available()
        )

    def list_sessions(self, active_only: bool = True) -> list[AudioSessionInfo]:
        return self.session_manager.list_sessions(active_only=active_only)

    def auto_detect_meeting(self) -> AudioSessionInfo | None:
        return self.session_manager.auto_detect_meeting()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_requested = False
            self._generation += 1
            token = self._generation
            selection = self._selection

        # Do not hold the manager lock while a process backend starts: its
        # reader thread may report a startup failure through on_lost, which
        # would otherwise block on this lock while start() joins that thread.
        source, active, fallback_reason = self._activate(selection)

        with self._lock:
            if self._stop_requested or token != self._generation:
                keep = False
            else:
                self._source = source
                self._active = active
                self._last_reason = fallback_reason or ""
                keep = True

        if not keep:
            try:
                source.stop()
            except Exception:
                logger.warning("Could not stop superseded audio source", exc_info=True)
            return

        self._notify(active, fallback_reason or "")
        if fallback_reason:
            logger.warning("Audio source fallback: %s", fallback_reason)

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._generation += 1
            source, self._source = self._source, None
        if source is not None:
            try:
                source.stop()
            except Exception:
                logger.warning("Audio source stop failed", exc_info=True)

    def switch_to(self, selection: AudioSelection) -> bool:
        """Stop the current source and start ``selection``.

        The caller decides whether the pipeline is running. This method is
        intentionally safe to call from the Qt thread while STT is running:
        the RingBuffer is shared and the VAD sees the short hand-off gap as
        ordinary silence.
        """
        selection = selection.normalized()
        with self._lock:
            self._generation += 1
            token = self._generation
            if self._stop_requested:
                self._selection = selection
                self._active = selection
                return True
            old_source, self._source = self._source, None

        # source.stop() joins the capture/monitor threads. Never do that while
        # holding the manager lock: a capture thread may be trying to schedule
        # its own fallback through that same lock.
        if old_source is not None:
            try:
                old_source.stop()
            except Exception:
                logger.warning("Old audio source stop failed", exc_info=True)

        error: str | None = None
        active = selection
        reason: str | None = None
        with self._lock:
            if token != self._generation:
                # A newer switch/stop won while the old source was stopping.
                return True
            if old_source is not None:
                self._insert_source_gap()
            try:
                source, active, reason = self._activate(selection)
            except Exception as exc:
                logger.error("Could not switch audio source: %s", exc)
                self._selection = selection
                self._active = selection
                error = str(exc)
            else:
                self._source = source
                self._selection = selection
                self._active = active
                self._last_reason = reason or ""

        if error is not None:
            self._notify(selection, error)
            return False
        self._notify(active, reason or "")
        if reason:
            logger.warning("Audio source fallback: %s", reason)
        return True

    def set_selection(self, selection: AudioSelection) -> None:
        """Remember a choice while the pipeline is stopped; do not start audio."""
        selection = selection.normalized()
        with self._lock:
            self._generation += 1
            self._selection = selection
            self._active = selection
        self._notify(selection, "")

    def _resolve(self, selection: AudioSelection) -> AudioSelection:
        selection = selection.normalized()
        if selection.mode == "device":
            return AudioSelection("device")
        if selection.mode == "auto":
            session = self.session_manager.auto_detect_meeting()
            if session is None:
                return AudioSelection("device")
            return AudioSelection.from_session(session)

        # Session mode: prefer a live PID, but re-resolve by executable when
        # the app restarted since the tray menu was built.
        sessions = self.session_manager.list_sessions()
        if selection.pid > 0:
            for session in sessions:
                if session.pid == selection.pid:
                    return AudioSelection.from_session(session)
        if selection.executable:
            session = self.session_manager.find_by_executable(
                selection.executable, sessions
            )
            if session is not None:
                return AudioSelection.from_session(session)
        raise SourceSelectionError(
            f"Audio app is not running: {selection.label}"
        )

    def _activate(
        self, selection: AudioSelection
    ) -> tuple[object, AudioSelection, str | None]:
        """Resolve a choice and start it, falling back to device audio."""
        selection = selection.normalized()
        try:
            resolved = self._resolve(selection)
        except Exception as exc:
            if selection.mode == "device":
                if isinstance(exc, LoopbackDeviceNotFoundError):
                    raise
                raise LoopbackDeviceNotFoundError(str(exc)) from exc
            reason = f"{selection.label} unavailable ({exc}); using system audio"
            logger.warning("%s", reason)
            try:
                source = self._start_source(AudioSelection("device"))
            except Exception as fallback_exc:
                raise LoopbackDeviceNotFoundError(
                    f"{reason}; device loopback also failed: {fallback_exc}"
                ) from fallback_exc
            return source, AudioSelection("device", label="All system audio"), reason
        return self._start_with_fallback(resolved)

    def _start_with_fallback(
        self, selection: AudioSelection
    ) -> tuple[object, AudioSelection, str | None]:
        try:
            return self._start_source(selection), selection, None
        except Exception as exc:
            if selection.mode == "device":
                if isinstance(exc, LoopbackDeviceNotFoundError):
                    raise
                raise LoopbackDeviceNotFoundError(str(exc)) from exc

            reason = f"{selection.label} unavailable ({exc}); using system audio"
            logger.warning("%s", reason)
            try:
                source = self._start_source(AudioSelection("device"))
            except Exception as fallback_exc:
                raise LoopbackDeviceNotFoundError(
                    f"{reason}; device loopback also failed: {fallback_exc}"
                ) from fallback_exc
            return source, AudioSelection("device", label="All system audio"), reason

    def _start_source(self, selection: AudioSelection):
        if selection.mode == "device":
            source = self._capture_factory(
                self.config, self.ring_buffer, on_error=self._on_device_error
            )
            source.start()
            return source

        holder: dict[str, object] = {}

        def on_lost(message: str) -> None:
            source = holder.get("source")
            if source is not None:
                self._schedule_process_fallback(source, message)

        source = self._process_factory(
            self.config,
            self.ring_buffer,
            selection.pid,
            executable=selection.executable,
            label=selection.label,
            on_error=self._on_process_error,
            on_lost=on_lost,
        )
        holder["source"] = source
        source.start()
        if not bool(getattr(source, "is_running", True)):
            try:
                source.stop()
            except Exception:
                logger.debug("Failed to stop non-running process source", exc_info=True)
            raise ProcessCaptureError(
                f"process capture for {selection.label} stopped during startup"
            )
        return source

    # -- error / fallback ----------------------------------------------------

    def _insert_source_gap(self) -> None:
        """Put a short silence between the old and new source.

        The VAD sees old audio -> silence -> new audio, so it closes the old
        utterance instead of splicing two unrelated sources together. This is
        non-blocking: the STT worker consumes the gap in order with the rest of
        the ring buffer.
        """
        try:
            gap_ms = int(
                self.config.vad.min_silence_ms
                + 2 * self.config.vad.padding_ms
            )
            samples = int(self.config.audio.sample_rate * gap_ms / 1000)
            if samples > 0:
                self.ring_buffer.write(np.zeros(samples, dtype=np.float32))
        except Exception:
            logger.debug("Could not insert audio-source gap", exc_info=True)

    def _on_device_error(self, message: str) -> None:
        if self.on_error is not None:
            self.on_error(message)

    def _on_process_error(self, message: str) -> None:
        # Used only when a process capture has no on_lost path (e.g. an
        # unexpected reader failure during shutdown). The normal path is
        # fallback, not a red pipeline error.
        logger.warning("Process capture error: %s", message)

    def _schedule_process_fallback(self, source, message: str) -> None:
        with self._lock:
            if self._stop_requested or self._source is not source:
                return
            if self._fallback_thread is not None and self._fallback_thread.is_alive():
                return
            token = self._generation
            self._fallback_thread = threading.Thread(
                target=self._process_fallback,
                args=(source, message, token),
                name="AudioSourceFallback",
                daemon=True,
            )
            self._fallback_thread.start()

    def _process_fallback(self, source, message: str, token: int) -> None:
        # Do not call source.stop() from its own reader thread: that would
        # join the current thread. This runs on AudioSourceFallback.
        with self._lock:
            if (
                self._stop_requested
                or token != self._generation
                or self._source is not source
            ):
                return
            self._source = None
        try:
            source.stop()
        except Exception:
            logger.warning("Failed to stop lost process capture", exc_info=True)

        with self._lock:
            if self._stop_requested or token != self._generation:
                return
            self._insert_source_gap()
            try:
                device = self._start_source(AudioSelection("device"))
            except Exception as exc:
                logger.error("Device fallback failed: %s", exc)
                if self.on_error is not None:
                    self.on_error(
                        f"{message}; device loopback fallback failed: {exc}"
                    )
                return
            active = AudioSelection("device", label="All system audio")
            self._source = device
            self._active = active
            self._last_reason = message
        logger.warning("Audio source fallback: %s", message)
        self._notify(active, message)

    # -- introspection -------------------------------------------------------

    @property
    def is_running(self) -> bool:
        source = self._source
        if source is None:
            return False
        try:
            return bool(source.is_running)
        except Exception:
            return False

    @property
    def current(self) -> AudioSelection:
        return self._active

    @property
    def selection(self) -> AudioSelection:
        return self._selection

    @property
    def stats(self) -> dict:
        source = self._source
        data = {}
        if source is not None:
            try:
                data.update(source.stats)
            except Exception:
                logger.debug("Audio source stats failed", exc_info=True)
        data.update({
            "mode": self._active.mode,
            "source": self._active.label,
            "fallback": self._last_reason,
        })
        return data

    def _notify(self, selection: AudioSelection, reason: str) -> None:
        callback = self.on_source_changed
        if callback is None:
            return
        try:
            callback(selection, reason)
        except Exception:
            logger.exception("on_source_changed callback failed")
