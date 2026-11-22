"""Phase 8 tests: session discovery, WAV stream decoding, source fallback, tray."""

from __future__ import annotations

# onnxruntime must be imported BEFORE PyQt6 (PyQt6's old MSVCP140.dll makes a later
# onnxruntime import crash the process; see main.py). Keep this line first.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import importlib.util
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.audio.ring_buffer import RingBuffer
from src.audio.session_manager import AudioSessionInfo
from src.audio.source_manager import AudioSelection, AudioSourceManager
from src.config.settings import EngineConfig, load_config, save_user_settings
from src.pipeline.coordinator import MainPipeline
from src.pipeline.ipc import PipelineStatus

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def wav_stream(
    frames: np.ndarray,
    *,
    channels: int = 2,
    rate: int = 48000,
    bits: int = 32,
    fmt_tag: int = 3,
) -> bytes:
    """Build the WAV shape emitted by process-audio-capture."""
    if bits == 32:
        pcm = np.asarray(frames, dtype="<f4").tobytes()
    elif bits == 16:
        pcm = np.asarray(frames, dtype="<i2").tobytes()
    else:
        raise ValueError(bits)
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH", fmt_tag, channels, rate, rate * block_align, block_align, bits
    )
    return (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )


class FakeSource:
    def __init__(self, kind: str):
        self.kind = kind
        self.started = 0
        self.stopped = 0
        self.running = False
        self.kwargs: dict = {}

    def start(self):
        self.started += 1
        self.running = True

    def stop(self):
        self.stopped += 1
        self.running = False

    @property
    def is_running(self):
        return self.running

    @property
    def stats(self):
        return {"source": self.kind}


class FakeSessionManager:
    def __init__(self, sessions):
        self.sessions = list(sessions)

    def list_sessions(self, active_only=True):
        return list(self.sessions)

    def find_by_executable(self, executable, sessions=None):
        wanted = os.path.basename(str(executable or "")).lower()
        for session in sessions if sessions is not None else self.sessions:
            if session.executable_name == wanted:
                return session
        return None

    def auto_detect_meeting(self, sessions=None):
        return self.sessions[0] if self.sessions else None


class SessionModelTests(unittest.TestCase):
    def test_session_identity_uses_executable_not_pid(self):
        session = AudioSessionInfo(
            pid=123,
            name="zoom.exe",
            executable=r"C:\Apps\Zoom\zoom.exe",
        )
        self.assertEqual(session.identity, r"c:\apps\zoom\zoom.exe")
        self.assertEqual(session.executable_name, "zoom.exe")
        self.assertIn("zoom.exe", session.label)

    def test_auto_detect_uses_active_session_list(self):
        from src.audio.session_manager import AudioSessionManager

        sessions = [
            AudioSessionInfo(pid=1, name="chrome.exe", executable="chrome.exe"),
            AudioSessionInfo(pid=2, name="zoom.exe", executable="zoom.exe"),
            AudioSessionInfo(pid=3, name="slack.exe", executable="slack.exe"),
        ]
        manager = AudioSessionManager(["zoom.exe", "chrome.exe"], exclude_pid=-1)
        detected = manager.auto_detect_meeting(sessions)
        self.assertIsNotNone(detected)
        self.assertEqual(detected.pid, 2)

    def test_inactive_sessions_are_not_auto_detected(self):
        from src.audio.session_manager import AudioSessionManager

        sessions = [
            AudioSessionInfo(
                pid=2, name="zoom.exe", executable="zoom.exe", is_active=False
            )
        ]
        manager = AudioSessionManager(["zoom.exe"], exclude_pid=-1)
        self.assertIsNone(manager.auto_detect_meeting(sessions))


class WavDecoderTests(unittest.TestCase):
    def test_float32_stereo_stream_is_resampled_and_downmixed(self):
        from src.audio.process_capture import WavStreamDecoder

        frames = np.zeros((4800, 2), dtype=np.float32)
        frames[:, 0] = 0.5
        frames[:, 1] = 0.5
        data = wav_stream(frames)
        decoder = WavStreamDecoder()
        out = np.concatenate([decoder.feed(data[:20]), decoder.feed(data[20:])])
        self.assertTrue(decoder.header_ready)
        self.assertEqual(out.size, 1600)
        self.assertAlmostEqual(float(out.mean()), 0.5, places=2)
        self.assertEqual(out.dtype, np.float32)

    def test_int16_mono_stream(self):
        from src.audio.process_capture import WavStreamDecoder

        frames = np.full(1600, 16384, dtype=np.int16)
        decoder = WavStreamDecoder()
        out = decoder.feed(wav_stream(frames, channels=1, rate=16000, bits=16, fmt_tag=1))
        self.assertEqual(out.size, 1600)
        self.assertAlmostEqual(float(out.mean()), 0.5, places=3)

    def test_partial_frame_is_retained(self):
        from src.audio.process_capture import WavStreamDecoder

        frames = np.zeros((100, 2), dtype=np.float32)
        data = wav_stream(frames, channels=2, rate=16000)
        decoder = WavStreamDecoder()
        first = decoder.feed(data[:60])  # header is 44 bytes; 16 bytes = 2 frames
        second = decoder.feed(data[60:])
        self.assertEqual(first.size, 2)
        self.assertEqual(second.size, 98)


class SourceManagerTests(unittest.TestCase):
    def _manager(
        self,
        session_manager,
        process_factory=None,
        capture_factory=None,
        *,
        mode="session",
        selected="zoom.exe",
    ):
        config = EngineConfig()
        config.audio.capture_mode = mode
        config.audio.selected_app = selected
        created = {"device": [], "process": []}

        def make_capture(cfg, rb, *, on_error):
            source = FakeSource("device")
            created["device"].append(source)
            return source

        def make_process(cfg, rb, pid, **kwargs):
            source = FakeSource("process")
            source.kwargs = kwargs
            source.pid = pid
            created["process"].append(source)
            return source

        manager = AudioSourceManager(
            config,
            RingBuffer(5.0),
            session_manager=session_manager,
            capture_factory=capture_factory or make_capture,
            process_factory=process_factory or make_process,
        )
        created["ring"] = manager.ring_buffer
        return manager, created

    def test_session_source_is_started_for_selected_app(self):
        sessions = [AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe")]
        manager, created = self._manager(FakeSessionManager(sessions))
        manager.start()
        self.assertTrue(manager.is_running)
        self.assertEqual(manager.current.mode, "session")
        self.assertEqual(created["process"][0].pid, 42)
        self.assertEqual(created["process"][0].started, 1)
        manager.stop()
        self.assertEqual(created["process"][0].stopped, 1)

    def test_auto_mode_selects_detected_meeting_app(self):
        sessions = [AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe")]
        manager, created = self._manager(FakeSessionManager(sessions), mode="auto")
        manager.start()
        self.assertEqual(manager.current.mode, "session")
        self.assertEqual(created["process"][0].pid, 42)
        manager.stop()

    def test_missing_app_falls_back_to_device(self):
        manager, created = self._manager(FakeSessionManager([]))
        manager.start()
        self.assertFalse(created["process"])
        self.assertEqual(created["device"][0].started, 1)
        self.assertEqual(manager.current.mode, "device")
        manager.stop()

    def test_unavailable_backend_falls_back_to_device(self):
        from src.audio.process_capture import ProcessCaptureUnavailable

        def broken(*args, **kwargs):
            raise ProcessCaptureUnavailable("DLL not installed")

        sessions = [AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe")]
        manager, created = self._manager(
            FakeSessionManager(sessions), process_factory=broken
        )
        manager.start()
        self.assertEqual(created["device"][0].started, 1)
        self.assertEqual(manager.current.mode, "device")
        manager.stop()

    def test_process_loss_falls_back_without_stopping_pipeline(self):
        sessions = [AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe")]
        manager, created = self._manager(FakeSessionManager(sessions))
        manager.start()
        process = created["process"][0]
        process.kwargs["on_lost"]("Zoom exited")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and manager.current.mode != "device":
            time.sleep(0.02)
        self.assertEqual(manager.current.mode, "device")
        self.assertEqual(process.stopped, 1)
        self.assertEqual(len(created["device"]), 1)
        self.assertTrue(manager.is_running)
        manager.stop()

    def test_switch_to_session_stops_previous_source(self):
        sessions = [AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe")]
        manager, created = self._manager(FakeSessionManager(sessions))
        manager.set_selection(AudioSelection("device"))
        manager.start()
        device = created["device"][0]
        self.assertTrue(manager.switch_to(
            AudioSelection("session", executable="zoom.exe", pid=42)
        ))
        self.assertEqual(device.stopped, 1)
        self.assertEqual(manager.current.mode, "session")
        self.assertGreater(created["ring"].available_samples, 0, "source switch must insert a VAD gap")
        manager.stop()


class FakeSwitchCapture:
    def __init__(self):
        self.running = False
        self.switched = None
        self.selected = None

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    @property
    def is_running(self):
        return self.running

    def switch_to(self, selection):
        self.switched = selection
        return True

    def set_selection(self, selection):
        self.selected = selection


class CoordinatorSwitchTests(unittest.TestCase):
    def test_switch_while_stopped_only_remembers_selection(self):
        config = EngineConfig()
        capture = FakeSwitchCapture()
        pipeline = MainPipeline(config, engine=mock.Mock(), capture=capture, backend=mock.Mock())
        selection = AudioSelection("device")
        self.assertTrue(pipeline.switch_audio_source(selection))
        self.assertEqual(capture.selected, selection)
        self.assertIsNone(capture.switched)

    def test_switch_while_recording_delegates_to_manager(self):
        config = EngineConfig()
        capture = FakeSwitchCapture()
        pipeline = MainPipeline(config, engine=mock.Mock(), capture=capture, backend=mock.Mock())
        pipeline._stt_worker = lambda: pipeline._stop.wait(2.0)
        pipeline._translation_worker = lambda: pipeline._translation_stop.wait(2.0)
        self.assertTrue(pipeline.start())
        selection = AudioSelection("device")
        self.assertTrue(pipeline.switch_audio_source(selection))
        self.assertEqual(capture.switched, selection)
        pipeline.stop()


class ConfigPhase8Tests(unittest.TestCase):
    def test_audio_fields_round_trip(self):
        config = EngineConfig()
        config.audio.capture_mode = "session"
        config.audio.selected_app = "zoom.exe"
        config.audio.auto_detect_apps = ["zoom.exe", "teams.exe"]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "settings.yaml"
            self.assertTrue(save_user_settings(config, target))
            loaded = load_config(
                path=Path(tmp) / "missing-config.yaml",
                user_path=target,
            )
        self.assertEqual(loaded.audio.capture_mode, "session")
        self.assertEqual(loaded.audio.selected_app, "zoom.exe")
        self.assertEqual(loaded.audio.auto_detect_apps, ["zoom.exe", "teams.exe"])


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not installed")
class TrayAudioMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _create(self, provider, available=True):
        from src.ui.tray import TrayController

        selected = []
        tray = TrayController(
            on_toggle_listening=mock.Mock(),
            on_open_settings=mock.Mock(),
            on_exit=mock.Mock(),
            on_select_audio_source=lambda choice: selected.append(choice) or True,
            audio_sources_provider=provider,
            current_audio_source=AudioSelection("device"),
            process_capture_available=available,
        )
        with mock.patch(
            "PyQt6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
            return_value=True,
        ):
            created = tray.create()
        self.addCleanup(created.hide)
        self.addCleanup(tray.menu.deleteLater)
        return tray, selected

    def test_menu_lists_live_sessions_and_selects_one(self):
        sessions = [
            AudioSessionInfo(pid=42, name="zoom.exe", executable="zoom.exe"),
            AudioSessionInfo(pid=43, name="chrome.exe", executable="chrome.exe"),
        ]
        tray, selected = self._create(lambda: sessions)
        texts = [action.text() for action in tray.audio_menu.actions()]
        self.assertIn("All system audio", texts)
        self.assertIn("Auto-detect meeting app", texts)
        self.assertIn("zoom.exe", texts)
        action = next(a for a in tray.audio_menu.actions() if a.text() == "zoom.exe")
        action.trigger()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].mode, "session")
        self.assertEqual(selected[0].pid, 42)

    def _texts(self, tray):
        return [action.text() for action in tray.audio_menu.actions()]

    def test_silent_apps_are_listed_after_the_playing_ones_and_selectable(self):
        sessions = [
            AudioSessionInfo(pid=1, name="vlc.exe", executable="vlc.exe", is_active=True),
            AudioSessionInfo(pid=2, name="Microsoft.Media.Player.exe",
                             executable="Microsoft.Media.Player.exe", is_active=False),
        ]
        tray, selected = self._create(lambda: sessions)
        texts = self._texts(tray)
        self.assertIn("Listen to one application:", texts)
        self.assertIn("vlc.exe", texts)
        self.assertIn("Microsoft.Media.Player.exe  (not playing)", texts)
        action = next(a for a in tray.audio_menu.actions() if "Media.Player" in a.text())
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.assertEqual(selected[0].pid, 2)
        self.assertEqual(selected[0].executable, "Microsoft.Media.Player.exe")

    def test_saved_choice_is_shown_even_when_its_app_is_closed(self):
        from src.ui.tray import TrayController

        tray = TrayController(
            on_toggle_listening=mock.Mock(), on_open_settings=mock.Mock(),
            on_exit=mock.Mock(), on_select_audio_source=lambda choice: True,
            audio_sources_provider=lambda: [],
            current_audio_source=AudioSelection(
                "session", executable=r"C:\Apps\Microsoft.Media.Player.exe",
            ),
            process_capture_available=True,
        )
        with mock.patch(
            "PyQt6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable", return_value=True,
        ):
            created = tray.create()
        self.addCleanup(created.hide)
        self.addCleanup(tray.menu.deleteLater)
        missing = [a for a in tray.audio_menu.actions() if "not running" in a.text()]
        self.assertEqual(len(missing), 1)
        self.assertIn("Microsoft.Media.Player.exe", missing[0].text())
        self.assertTrue(missing[0].isChecked())
        self.assertFalse(missing[0].isEnabled())

    def test_empty_list_tells_the_user_what_to_do(self):
        tray, _selected = self._create(lambda: [])
        self.assertTrue(any("play something" in t for t in self._texts(tray)))

    def test_missing_backend_disables_per_app_choices(self):
        tray, _selected = self._create(lambda: [], available=False)
        texts = [action.text() for action in tray.audio_menu.actions()]
        self.assertIn("Per-app capture unavailable", texts)
        self.assertNotIn("Listen to one application:", texts)
        self.assertFalse(any("play something" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
