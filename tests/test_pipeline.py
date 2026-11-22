"""Phase 5 tests: UI IPC contract and MainPipeline lifecycle/supervision."""

from __future__ import annotations

# onnxruntime must be imported BEFORE PyQt6 (PyQt6's old MSVCP140.dll makes a later
# onnxruntime import crash the process; see main.py). Keep this line first.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import importlib.util
import threading
import unittest
from unittest import mock

from src.config.settings import EngineConfig
from src.pipeline.coordinator import MainPipeline
from src.pipeline.ipc import (
    Message,
    MessageType,
    PipelineStatus,
    UIQueue,
    make_ui_queue,
    put_message,
)
from src.translation.base import TranslationResult


class FakeCapture:
    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.fail_with: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class FakeBackend:
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class FakeEngine:
    def __init__(self) -> None:
        self.confirmed: list[str] = []
        self.partials: list[str] = []
        self.context: list[tuple[str, str]] = []
        self.provider = "fake"
        self.fail_confirmed = False

    def translate_confirmed(self, text: str) -> TranslationResult:
        if self.fail_confirmed:
            raise RuntimeError("provider down")
        self.confirmed.append(text)
        return TranslationResult(f"AR:{text}", self.provider, 1.0)

    def translate_partial(self, text: str) -> TranslationResult:
        self.partials.append(text)
        return TranslationResult(f"AR?:{text}", self.provider, 0.5)

    def remember_context(self, source: str, translated: str) -> None:
        self.context.append((source, translated))

    def current_provider_name(self) -> str:
        return self.provider

    def clear_context(self) -> None:
        self.context.clear()

    def shutdown(self) -> None:
        pass


def make_config() -> EngineConfig:
    config = EngineConfig()
    config.translation.partial_translation = False
    return config


def make_pipeline() -> tuple[MainPipeline, FakeCapture, FakeEngine]:
    capture = FakeCapture()
    engine = FakeEngine()
    pipeline = MainPipeline(
        make_config(), engine, capture=capture, backend=FakeBackend()
    )
    return pipeline, capture, engine


class IPCQueueTests(unittest.TestCase):
    def test_make_ui_queue_is_bounded(self):
        q = make_ui_queue()
        self.assertIsInstance(q, UIQueue)
        self.assertEqual(q.maxsize, 200)

    def test_put_message_drops_oldest_when_full(self):
        q = UIQueue(maxsize=2)
        for i in range(3):
            put_message(q, Message(MessageType.TEXT_UPDATE, {"n": i}))

        received = [q.get_nowait().payload["n"] for _ in range(2)]
        self.assertEqual(received, [1, 2])

    def test_drain_is_fifo_and_respects_limit(self):
        q = UIQueue()
        for i in range(4):
            q.put_message(Message(MessageType.TEXT_UPDATE, {"n": i}))

        first = q.drain(limit=3)
        self.assertEqual([m.payload["n"] for m in first], [0, 1, 2])
        self.assertEqual(q.drain()[0].payload["n"], 3)

    def test_all_message_types_are_stable_strings(self):
        self.assertEqual(MessageType.TEXT_UPDATE.value, "text_update")
        self.assertEqual(MessageType.QUOTA_UPDATE.value, "quota_update")
        self.assertEqual(PipelineStatus.ERROR.value, "error")

    def test_typed_messages_populate_the_payload(self):
        from src.pipeline.ipc import ErrorEvent, QuotaUpdate, StatusUpdate, TextUpdate

        text = TextUpdate(arabic="مرحبا", english="hello")
        self.assertEqual(text.payload["arabic"], "مرحبا")
        self.assertEqual(text.payload["english"], "hello")

        status = StatusUpdate(status=PipelineStatus.PAUSED, detail="user")
        self.assertEqual(status.payload["status"], "paused")
        self.assertEqual(status.payload["detail"], "user")

        error = ErrorEvent(error="boom")
        self.assertEqual(error.payload["error"], "boom")

        quota = QuotaUpdate(quotas={"gemini": 4})
        self.assertEqual(quota.payload["quotas"], {"gemini": 4})


class PipelineLifecycleTests(unittest.TestCase):
    def _wait_workers(self, pipeline: MainPipeline):
        """Replace real audio/model workers with deterministic waiters."""
        pipeline._stt_worker = lambda: pipeline._stop.wait(2.0)
        pipeline._translation_worker = lambda: pipeline._translation_stop.wait(2.0)

    def test_start_pause_resume_stop(self):
        pipeline, capture, _engine = make_pipeline()
        self._wait_workers(pipeline)

        self.assertTrue(pipeline.start())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)
        self.assertTrue(capture.running)
        self.assertEqual(set(pipeline._threads), {"stt", "translation"})

        self.assertTrue(pipeline.pause())
        self.assertEqual(pipeline.status, PipelineStatus.PAUSED)
        self.assertTrue(pipeline.resume())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)

        pipeline.stop()
        self.assertEqual(pipeline.status, PipelineStatus.IDLE)
        self.assertFalse(capture.running)
        self.assertEqual(pipeline._threads, {})
        self.assertFalse(any(t.is_alive() for t in pipeline._threads.values()))

        # Idempotent and reusable.
        pipeline.stop()
        self.assertEqual(capture.stop_calls, 1)
        self.assertTrue(pipeline.start())
        pipeline.stop()
        self.assertEqual(capture.start_calls, 2)

    def test_toggle_starts_from_idle(self):
        pipeline, _capture, _engine = make_pipeline()
        self._wait_workers(pipeline)

        self.assertTrue(pipeline.toggle())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)
        self.assertTrue(pipeline.toggle())
        self.assertEqual(pipeline.status, PipelineStatus.PAUSED)
        self.assertTrue(pipeline.toggle())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)
        pipeline.stop()

    def test_audio_start_failure_sets_error_and_can_retry(self):
        pipeline, capture, _engine = make_pipeline()
        self._wait_workers(pipeline)
        from src.audio.capture import LoopbackDeviceNotFoundError

        capture.fail_with = LoopbackDeviceNotFoundError("no loopback device")
        self.assertFalse(pipeline.start())
        self.assertEqual(pipeline.status, PipelineStatus.ERROR)
        self.assertEqual(pipeline._threads, {})

        capture.fail_with = None
        self.assertTrue(pipeline.start())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)
        pipeline.stop()

    def test_capture_error_is_reported_to_ui(self):
        pipeline, _capture, _engine = make_pipeline()
        pipeline._on_capture_error("device disappeared")

        self.assertEqual(pipeline.status, PipelineStatus.ERROR)
        messages = pipeline.ui_queue.drain()
        self.assertTrue(any(m.type == MessageType.ERROR_EVENT for m in messages))
        statuses = [
            m.payload["status"] for m in messages
            if m.type == MessageType.STATUS_UPDATE
        ]
        self.assertIn("error", statuses)

    def test_restart_clears_error_state(self):
        pipeline, _capture, _engine = make_pipeline()
        pipeline._stt_worker = lambda: pipeline._stop.wait(2.0)
        pipeline._translation_worker = lambda: pipeline._translation_stop.wait(2.0)
        pipeline._set_status(PipelineStatus.ERROR, "previous failure")

        self.assertTrue(pipeline.restart())
        self.assertEqual(pipeline.status, PipelineStatus.RECORDING)
        pipeline.stop()


class PipelineSupervisionTests(unittest.TestCase):
    def test_worker_is_restarted_then_marked_error_after_limit(self):
        pipeline, _capture, _engine = make_pipeline()
        pipeline._restart_delay_s = 0.0
        calls: list[int] = []

        def always_crashes():
            calls.append(1)
            raise RuntimeError("boom")

        pipeline._supervised("stt", always_crashes)

        self.assertEqual(len(calls), pipeline.MAX_RESTARTS)
        self.assertEqual(pipeline.status, PipelineStatus.ERROR)
        messages = pipeline.ui_queue.drain()
        self.assertTrue(any(m.type == MessageType.ERROR_EVENT for m in messages))

    def test_unexpected_clean_exit_is_supervised(self):
        pipeline, _capture, _engine = make_pipeline()
        pipeline._restart_delay_s = 0.0
        calls: list[int] = []

        def exits_immediately():
            calls.append(1)

        pipeline._supervised("translation", exits_immediately)
        self.assertEqual(len(calls), pipeline.MAX_RESTARTS)
        self.assertEqual(pipeline.status, PipelineStatus.ERROR)

    def test_shutdown_is_not_treated_as_a_crash(self):
        pipeline, _capture, _engine = make_pipeline()
        calls: list[int] = []

        def exits_on_stop():
            calls.append(1)
            pipeline._stop.set()

        pipeline._supervised("stt", exits_on_stop)
        self.assertEqual(calls, [1])
        self.assertEqual(pipeline.status, PipelineStatus.IDLE)


class PipelineWorkerTests(unittest.TestCase):
    def test_final_text_flushed_on_stop_is_translated(self):
        pipeline, _capture, engine = make_pipeline()
        delivered = threading.Event()

        def stt_worker():
            pipeline._stop.wait(2.0)
            pipeline._queue_put(
                pipeline._translation_queue, ("final sentence", False)
            )
            delivered.set()

        pipeline._stt_worker = stt_worker
        self.assertTrue(pipeline.start())
        self.assertFalse(delivered.is_set())  # stop has not flushed yet
        pipeline.stop()
        self.assertTrue(delivered.is_set())

        messages = pipeline.ui_queue.drain()
        text = [m for m in messages if m.type == MessageType.TEXT_UPDATE]
        self.assertTrue(text, "final flush was not translated")
        self.assertEqual(text[-1].payload["arabic"], "AR:final sentence")
        self.assertEqual(engine.context, [("final sentence", "AR:final sentence")])

    def test_partial_translation_does_not_enter_context(self):
        pipeline, _capture, engine = make_pipeline()
        pipeline._translation_stop.set()
        pipeline._translation_queue.put(("hypothesis", True))
        MainPipeline._translation_worker(pipeline)

        messages = pipeline.ui_queue.drain()
        partial = [m for m in messages if m.type == MessageType.PARTIAL_UPDATE]
        self.assertEqual(partial[-1].payload["arabic"], "AR?:hypothesis")
        self.assertEqual(engine.partials, ["hypothesis"])
        self.assertEqual(engine.context, [])

    def test_translation_failure_does_not_kill_drain(self):
        pipeline, _capture, engine = make_pipeline()
        engine.fail_confirmed = True
        pipeline._translation_stop.set()
        pipeline._translation_queue.put(("first", False))
        pipeline._translation_queue.put(("second", False))

        pipeline._translation_worker()

        # Both attempts were made; neither escaped the worker.
        self.assertEqual(pipeline._translation_queue.qsize(), 0)
        messages = pipeline.ui_queue.drain()
        self.assertFalse(any(m.type == MessageType.TEXT_UPDATE for m in messages))

    def test_partials_use_latest_only_slot(self):
        pipeline, _capture, _engine = make_pipeline()

        pipeline._queue_text("a", True)
        pipeline._queue_text("b", True)

        self.assertEqual(pipeline._pending_partial, "b")
        self.assertTrue(pipeline._translation_queue.empty())

    def test_final_supersedes_pending_partial(self):
        pipeline, _capture, _engine = make_pipeline()
        pipeline._queue_text("b", True)

        pipeline._queue_text("done.", False)

        self.assertIsNone(pipeline._pending_partial)
        self.assertEqual(
            pipeline._translation_queue.get_nowait(), ("done.", False)
        )

    def test_duplicate_partial_is_not_stored(self):
        pipeline, _capture, _engine = make_pipeline()
        with pipeline._partial_lock:
            pipeline._last_partial_sent = "same"

        pipeline._queue_text("same", True)

        self.assertIsNone(pipeline._pending_partial)
        self.assertTrue(pipeline._translation_queue.empty())


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 is not installed")
class MainSignalTests(unittest.TestCase):
    def test_sigint_handler_quits_the_app(self):
        import signal

        import main

        app = mock.Mock()
        with mock.patch.object(main.signal, "signal") as register:
            self.assertTrue(main.install_sigint_handler(app))
            handler = register.call_args.args[1]
            handler(signal.SIGINT, None)
        app.quit.assert_called_once_with()

    def test_sigint_failure_is_reported_without_crashing(self):
        import main

        app = mock.Mock()
        with mock.patch.object(main.signal, "signal", side_effect=ValueError("no")):
            self.assertFalse(main.install_sigint_handler(app))
        app.quit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
