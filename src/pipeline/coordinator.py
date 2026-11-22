"""MainPipeline — orchestrates all worker threads.

Thread architecture:
  Thread 1: Audio capture (PortAudio callback) → RingBuffer
  Thread 2: VAD + STT worker (VACOnlineASRProcessor → Whisper → Local
            Agreement) → confirmed text
  Thread 3: Translation worker → engine → Arabic text → UI queue
  Main thread: Qt event loop (polls UI queue)

VAD and STT share one thread on purpose: VACOnlineASRProcessor must see
every audio sample in order to keep its absolute timestamps and its
utterance state consistent. Passing pre-gated audio through a queue (the
old design) duplicated VAD state and lost the pre-roll on cancel.

Error propagation: a dead worker is detected and restarted; shutdown
flushes the open utterance before the translation worker drains.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time

from src.audio.capture import LoopbackDeviceNotFoundError
from src.audio.ring_buffer import RingBuffer
from src.audio.source_manager import AudioSelection, AudioSourceManager
from src.config.settings import EngineConfig
from src.pipeline.ipc import (
    Message, MessageType, PipelineStatus, make_ui_queue, put_message,
)
from src.stt.faster_whisper import FasterWhisperBackend
from src.stt.online_processor import VACOnlineASRProcessor
from src.translation.engine import TranslationEngine

logger = logging.getLogger(__name__)

# How much audio the STT worker pulls from the ring buffer at a time
# (samples @16kHz). 100ms keeps VAD latency low without busy-looping.
AUDIO_WINDOW_SAMPLES = 1600

# Sentence boundary: whitespace after terminal punctuation (incl. Arabic '؟').
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def _split_text(text: str, limit: int = 450) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` chars.

    The sanitizer rejects anything over its own cap, so long confirmations are
    chopped here instead of being silently dropped.  Sentence ends are tried
    first; a sentence that is still too long is cut at the last space before
    the limit (hard cut when it has no space at all).  Pieces are stripped and
    rejoining them with single spaces loses no words.
    """
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        while len(sentence) > limit:
            head = sentence[:limit]
            cut = head.rfind(" ")
            if cut <= 0:
                cut = limit  # no space to break on: hard cut
            chunk = sentence[:cut].strip()
            if chunk:
                pieces.append(chunk)
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


class MainPipeline:
    """Lifecycle owner for capture + STT + translation workers."""

    JOIN_TIMEOUT_S = 10.0
    RESTART_DELAY_S = 2.0
    MAX_RESTARTS = 3
    # A worker that ran this long before dying is not "crashing repeatedly";
    # its restart budget starts over, so unrelated failures across a long
    # meeting cannot add up to a permanent ERROR.
    HEALTHY_RUN_S = 60.0

    def __init__(
        self,
        config: EngineConfig,
        engine: TranslationEngine,
        *,
        capture=None,
        backend=None,
    ):
        self.config = config
        self.engine = engine

        self.ring_buffer = RingBuffer(
            max_seconds=config.audio.ring_buffer_seconds,
            sample_rate=config.audio.sample_rate,
        )
        self.capture = capture or AudioSourceManager(
            config, self.ring_buffer, on_error=self._on_capture_error
        )
        self.backend = backend or FasterWhisperBackend(config)

        self.ui_queue = make_ui_queue()

        self._translation_queue: "queue.Queue[tuple[str, bool]]" = queue.Queue(maxsize=50)
        # Latest-only hypothesis slot: stale partials are replaced, never
        # queued behind finals.
        self._pending_partial: str | None = None
        self._last_partial_sent: str = ""
        self._partial_lock = threading.Lock()

        self._threads: dict[str, threading.Thread] = {}
        self._lifecycle_lock = threading.RLock()
        self._status_lock = threading.RLock()
        self._restart_lock = threading.Lock()
        self._stop = threading.Event()
        self._translation_stop = threading.Event()
        self._paused = threading.Event()
        self._status = PipelineStatus.IDLE
        self._restart_counts: dict[str, int] = {}
        # Kept as an instance attribute so tests can shorten the backoff.
        self._restart_delay_s = self.RESTART_DELAY_S

    # -- lifecycle ---------------------------------------------------------

    @property
    def status(self) -> PipelineStatus:
        with self._status_lock:
            return self._status

    def start(self) -> bool:
        """Start capture + all workers. Returns False if audio setup fails."""
        with self._lifecycle_lock:
            if self.status in (PipelineStatus.RECORDING, PipelineStatus.PAUSED):
                if any(t.is_alive() for t in self._threads.values()):
                    return True
            if any(t.is_alive() for t in self._threads.values()):
                logger.error("Cannot start: previous workers are still alive")
                return False

            self._stop.clear()
            self._translation_stop.clear()
            self._paused.clear()
            self._restart_counts.clear()
            self._threads.clear()
            self.ring_buffer.clear()
            self._clear_translation_queue()
            try:
                self.capture.start()
            except LoopbackDeviceNotFoundError as exc:
                logger.error("%s", exc)
                self._set_status(PipelineStatus.ERROR, str(exc))
                return False
            except Exception as exc:
                logger.exception("Audio capture failed to start")
                self._set_status(PipelineStatus.ERROR, f"Audio capture failed: {exc}")
                return False

            workers = [
                ("stt", self._stt_worker),
                ("translation", self._translation_worker),
            ]
            # Set RECORDING before the threads start: a worker can fail
            # immediately and set ERROR, which must not be overwritten by a
            # late status update from start().
            self._set_status(PipelineStatus.RECORDING)
            try:
                for name, target in workers:
                    t = threading.Thread(
                        target=self._supervised,
                        args=(name, target),
                        name=name,
                        daemon=True,
                    )
                    t.start()
                    self._threads[name] = t
            except Exception as exc:
                logger.exception("Could not start pipeline workers")
                self._stop.set()
                try:
                    self.capture.stop()
                except Exception:
                    logger.debug("Capture cleanup after start failure failed", exc_info=True)
                # Keep any worker that did start. Clearing _threads here would
                # lose the only reference to a live consumer and let a later
                # start() create a duplicate.
                self._set_status(PipelineStatus.ERROR, f"Could not start workers: {exc}")
                return False

        logger.info("Pipeline started")
        return True

    def stop(self) -> None:
        """Graceful shutdown: flush STT, drain queues, stop capture.

        Idempotent and safe to call after a failed start. The lifecycle lock is
        not held while joining workers: worker status updates must not block.
        """
        with self._lifecycle_lock:
            threads = dict(self._threads)
            if not threads and not self.capture.is_running and self.status == PipelineStatus.IDLE:
                return
            logger.info("Pipeline stopping...")
            self._stop.set()
            self._paused.clear()

        try:
            self.capture.stop()
        except Exception:
            logger.exception("Audio capture failed to stop cleanly")

        # Join the STT worker first: it may still flush the open utterance
        # into the translation queue. Only then tell the translator to stop,
        # otherwise the final sentence has nobody left to translate it.
        for name in ("stt",):
            self._join_worker(threads.get(name), name)
        with self._lifecycle_lock:
            self._translation_stop.set()
        for name, thread in threads.items():
            if name != "stt":
                self._join_worker(thread, name)

        self.ring_buffer.clear()
        self._clear_translation_queue()
        with self._lifecycle_lock:
            alive = {name: t for name, t in threads.items() if t.is_alive()}
            if alive:
                # Never leave a live worker behind while reporting IDLE:
                # a later start() would create a duplicate consumer.
                self._threads = alive
                self._restart_counts.clear()
                self._set_status(
                    PipelineStatus.ERROR,
                    "worker did not stop within the shutdown timeout",
                )
            else:
                self._threads.clear()
                self._restart_counts.clear()
                try:
                    self.backend.unload()
                except Exception:
                    logger.debug("STT backend unload failed", exc_info=True)
                self._set_status(PipelineStatus.IDLE)
        logger.info("Pipeline stopped")

    def _join_worker(self, thread: threading.Thread | None, name: str) -> None:
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=self.JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.error("Worker '%s' did not stop within %.1fs", name, self.JOIN_TIMEOUT_S)

    def restart(self) -> bool:
        """Stop a failed pipeline, then start a fresh one."""
        self.stop()
        return self.start()

    def pause(self) -> bool:
        with self._lifecycle_lock:
            if self.status != PipelineStatus.RECORDING:
                return False
            self._paused.set()
            self._set_status(PipelineStatus.PAUSED)
            return True

    def resume(self) -> bool:
        with self._lifecycle_lock:
            if self.status != PipelineStatus.PAUSED:
                return False
            self._paused.clear()
            self._set_status(PipelineStatus.RECORDING)
            return True

    def toggle(self) -> bool:
        """Start when stopped/failed, otherwise pause or resume."""
        status = self.status
        if status == PipelineStatus.ERROR:
            return self.restart()
        if status == PipelineStatus.IDLE:
            return self.start()
        if status == PipelineStatus.PAUSED:
            return self.resume()
        if status == PipelineStatus.RECORDING:
            return self.pause()
        return False

    @property
    def audio_source(self) -> AudioSelection | None:
        """Current source label/selection, when the capture backend supports it."""
        current = getattr(self.capture, "current", None)
        return current if isinstance(current, AudioSelection) else None

    def switch_audio_source(self, selection: AudioSelection) -> bool:
        """Switch source without restarting STT or the translation worker.

        While stopped, only remember the choice. While recording/paused, the
        source manager performs the hand-off and falls back to device audio
        if the selected app has already exited.
        """
        selection = selection.normalized()
        with self._lifecycle_lock:
            if self.status in (PipelineStatus.RECORDING, PipelineStatus.PAUSED):
                switcher = getattr(self.capture, "switch_to", None)
                if switcher is None:
                    return False
                try:
                    return bool(switcher(selection))
                except Exception:
                    logger.exception("Could not switch audio source")
                    return False

            setter = getattr(self.capture, "set_selection", None)
            if setter is None:
                return False
            try:
                setter(selection)
                return True
            except Exception:
                logger.exception("Could not remember audio source")
                return False

    # -- supervision -----------------------------------------------------------

    def _supervised(self, name: str, target) -> None:
        """Run a worker; restart it if it dies (bounded by MAX_RESTARTS)."""
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                target()
                if self._stop.is_set():
                    return
                logger.error("Worker '%s' exited unexpectedly", name)
            except Exception:
                logger.exception("Worker '%s' crashed", name)

            # A long healthy run earns a fresh restart budget: three crashes
            # spread over a meeting are not "failed repeatedly".
            if time.monotonic() - started >= self.HEALTHY_RUN_S:
                with self._restart_lock:
                    self._restart_counts[name] = 0

            if self._stop.is_set():
                return
            with self._restart_lock:
                count = self._restart_counts.get(name, 0) + 1
                self._restart_counts[name] = count
            if count >= self.MAX_RESTARTS:
                logger.error("Worker '%s' exceeded restart limit — pipeline error", name)
                self._set_status(PipelineStatus.ERROR, f"worker '{name}' failed")
                put_message(self.ui_queue, Message(
                    MessageType.ERROR_EVENT,
                    {"error": f"Pipeline worker '{name}' failed repeatedly"},
                ))
                return
            logger.warning("Restarting worker '%s' (attempt %d)", name, count)
            if self._stop.wait(self._restart_delay_s):
                return

    # -- workers -----------------------------------------------------------------

    def _on_capture_error(self, message: str) -> None:
        """Report an audio-device failure to the UI (called from the monitor thread)."""
        self._set_status(PipelineStatus.ERROR, message)
        put_message(self.ui_queue, Message(MessageType.ERROR_EVENT, {"error": message}))

    def _notify_info(self, text: str) -> None:
        """Show a plain notice in the tray (called from worker threads)."""
        put_message(self.ui_queue, Message(MessageType.INFO_EVENT, {"text": text}))

    def _stt_worker(self) -> None:
        """VAD + Whisper + Local Agreement Policy → confirmed English text."""
        self.backend.on_info = self._notify_info
        self.backend.load()
        vac = VACOnlineASRProcessor(self.backend, self.config)
        logger.info("STT ready (VAD backend: %s)", vac.backend_name)

        paused = False
        while not self._stop.is_set():
            if self._paused.is_set():
                if not paused:
                    # Close the sentence cleanly instead of joining audio
                    # from before and after the pause into one utterance.
                    final = vac.finish_text()
                    if final:
                        self._queue_text(final, False)
                    paused = True
                self.ring_buffer.consume_all()
                time.sleep(0.1)
                continue
            paused = False

            chunk = self.ring_buffer.read(AUDIO_WINDOW_SAMPLES, timeout=0.1)
            if chunk is not None:
                vac.insert_audio_chunk(chunk)

            confirmed, partial = vac.process_iter()
            if partial and self.config.translation.partial_translation:
                self._queue_text(partial, True)
            if confirmed:
                self._queue_text(confirmed, False)

        # graceful shutdown: flush the open utterance before the
        # translation worker drains the queue
        remaining = self.ring_buffer.consume_all()
        if remaining.size:
            vac.insert_audio_chunk(remaining)
        final = vac.finish_text()
        if final:
            self._queue_text(final, False)
        logger.info("STT worker stopped")

    def _translation_worker(self) -> None:
        """Translate confirmed/partial text, push Arabic to the UI queue.

        After ``_translation_stop`` is set the worker keeps draining until
        the queue is empty, so the last sentence flushed by the STT worker
        is not lost.
        """
        while (
            not self._translation_stop.is_set()
            or not self._translation_queue.empty()
        ):
            try:
                text, is_partial = self._translation_queue.get(timeout=0.1)
            except queue.Empty:
                if self._translation_stop.is_set():
                    break
                # Latest-only partials live outside the FIFO: a newer
                # hypothesis replaces an untranslated one instead of queueing.
                pending = None
                with self._partial_lock:
                    if self._pending_partial is not None:
                        pending = self._pending_partial
                        self._pending_partial = None
                        self._last_partial_sent = pending
                if pending is None:
                    continue
                text, is_partial = pending, True
            if not text.strip():
                continue

            try:
                if is_partial:
                    result = self.engine.translate_partial(text)
                else:
                    result = self.engine.translate_confirmed(text)
            except Exception:
                logger.exception("Translation failed")
                continue

            if result is None:
                continue

            msg_type = MessageType.PARTIAL_UPDATE if is_partial else MessageType.TEXT_UPDATE
            put_message(self.ui_queue, Message(msg_type, {
                "arabic": result.translated,
                "english": text,
                "provider": result.provider,
                "latency_ms": result.latency_ms,
            }))

            if not is_partial:
                self.engine.remember_context(text, result.translated)
                put_message(self.ui_queue, Message(MessageType.PROVIDER_UPDATE, {
                    "provider": self.engine.current_provider_name(),
                }))

    # -- helpers -------------------------------------------------------------------

    def _queue_text(self, text: str, is_partial: bool) -> None:
        """Queue text for translation, splitting over-long finals by sentence.

        A confirmation longer than the sanitizer cap would otherwise be
        rejected and lost, so it is translated piece by piece.  A partial is
        only a preview of the newest sentence: it goes into a latest-only
        slot, never the FIFO that finals must pass through.
        """
        pieces = _split_text(text)
        if not pieces:
            return
        if is_partial:
            piece = pieces[-1]
            with self._partial_lock:
                if piece != self._last_partial_sent:
                    self._pending_partial = piece
            return
        with self._partial_lock:
            self._pending_partial = None
            self._last_partial_sent = ""
        for piece in pieces:
            self._queue_put(self._translation_queue, (piece, False))

    def _queue_put(self, q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _clear_translation_queue(self) -> None:
        try:
            while True:
                self._translation_queue.get_nowait()
        except queue.Empty:
            pass
        with self._partial_lock:
            self._pending_partial = None
            self._last_partial_sent = ""

    def _set_status(self, status: PipelineStatus, detail: str = "") -> None:
        with self._status_lock:
            self._status = status
        put_message(self.ui_queue, Message(
            MessageType.STATUS_UPDATE, {"status": status.value, "detail": detail}
        ))
