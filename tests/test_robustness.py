"""Tests for the hardening fixes: config validation, DeepL term protection,
the "already running" notice and the first-run model-download notice."""

from __future__ import annotations

# onnxruntime must be imported BEFORE PyQt6 (PyQt6's old MSVCP140.dll makes a later
# onnxruntime import crash the process; see main.py). Keep this line first.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.config.providers import TranslationProviderConfig
from src.config.settings import EngineConfig, load_config, validate_config
from src.pipeline.coordinator import MainPipeline
from src.pipeline.ipc import MessageType, make_ui_queue
from src.stt.faster_whisper import FasterWhisperBackend, _model_size_text
from src.translation.base import TranslationRequest
from src.translation.providers.deepl import DeepLProvider


class ConfigValidationTests(unittest.TestCase):
    def test_a_valid_config_is_left_alone(self):
        self.assertEqual(validate_config(EngineConfig()), [])

    def test_bad_values_are_replaced_by_defaults_and_reported(self):
        config = EngineConfig()
        config.audio.ring_buffer_seconds = 0          # would crash RingBuffer
        config.overlay.position = "nowhere"
        config.vad.threshold = "high"                 # not even a number
        config.overlay.max_lines = True               # bool is not a count
        problems = validate_config(config)
        self.assertEqual(len(problems), 4)
        defaults = EngineConfig()
        self.assertEqual(config.audio.ring_buffer_seconds,
                         defaults.audio.ring_buffer_seconds)
        self.assertEqual(config.overlay.position, defaults.overlay.position)
        self.assertEqual(config.vad.threshold, defaults.vad.threshold)
        self.assertEqual(config.overlay.max_lines, defaults.overlay.max_lines)

    def test_load_config_survives_a_broken_yaml_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "audio:\n  ring_buffer_seconds: 0\n"
                "translation:\n  quota_headroom: 5\n",
                encoding="utf-8",
            )
            config = load_config(path, user_path=path)
        self.assertEqual(config.audio.ring_buffer_seconds, 30.0)
        self.assertEqual(config.translation.quota_headroom, 0.05)

    def test_zero_is_valid_where_it_means_automatic(self):
        config = EngineConfig()
        config.overlay.width = 0
        config.overlay.height = 0
        self.assertEqual(validate_config(config), [])


class DeepLTermTests(unittest.TestCase):
    def _provider(self, reply: str):
        provider = DeepLProvider(TranslationProviderConfig(name="deepl", api_key="k"))
        calls = []

        class FakeTranslator:
            def translate_text(self, text, **kwargs):
                calls.append((text, kwargs))
                return SimpleNamespace(text=reply)

        provider._translator = FakeTranslator()
        return provider, calls

    def test_protected_terms_travel_inside_ignored_xml_tags(self):
        provider, calls = self._provider("انشر <keep>API</keep> الآن")
        request = TranslationRequest(
            text="Deploy the __T0__ now & here",
            protected_terms={"__T0__": "API"},
        )
        result = provider.translate(request)

        text, kwargs = calls[0]
        self.assertEqual(text, "Deploy the <keep>API</keep> now &amp; here")
        self.assertEqual(kwargs["tag_handling"], "xml")
        self.assertEqual(kwargs["ignore_tags"], ["keep"])
        self.assertEqual(result.translated, "انشر API الآن")

    def test_text_without_terms_is_sent_as_plain_text(self):
        provider, calls = self._provider("مرحبا")
        provider.translate(TranslationRequest(text="Hello <b>"))
        text, kwargs = calls[0]
        self.assertEqual(text, "Hello <b>")
        self.assertNotIn("tag_handling", kwargs)

    def test_special_characters_in_the_reply_are_unescaped(self):
        provider, _ = self._provider("a &amp; <keep>R&amp;D</keep>")
        result = provider.translate(TranslationRequest(
            text="a & __T0__", protected_terms={"__T0__": "R&D"},
        ))
        self.assertEqual(result.translated, "a & R&D")


class AlreadyRunningTests(unittest.TestCase):
    def test_second_launch_says_so_in_a_message_box(self):
        import ctypes

        import main

        fake_windll = mock.MagicMock()
        with mock.patch.object(ctypes, "windll", fake_windll, create=True), \
                mock.patch.object(main.sys, "platform", "win32"):
            main.notify_already_running()
        args = fake_windll.user32.MessageBoxW.call_args.args
        self.assertIn("already running", args[1])


class ModelDownloadNoticeTests(unittest.TestCase):
    def _load(self, cached: bool, model: str = "medium"):
        notices: list[str] = []

        class FakeModel:
            def __init__(self, name, device, compute_type, num_workers=1):
                pass

        def download_model(name, local_files_only=False, **kwargs):
            if not cached:
                raise RuntimeError("not on disk")
            return "/cache/" + name

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeModel
        utils = types.ModuleType("faster_whisper.utils")
        utils.download_model = download_model
        fake.utils = utils

        config = EngineConfig()
        config.engine.whisper_model = model
        backend = FasterWhisperBackend(config)
        backend.on_info = notices.append
        with mock.patch.dict(sys.modules, {
            "faster_whisper": fake, "faster_whisper.utils": utils,
        }), mock.patch(
            "src.stt.faster_whisper._resolve_runtime", return_value=("cuda", "float16"),
        ), mock.patch("src.stt.cuda_paths.cuda_runtime_ready", return_value=True):
            backend.load()
        return notices

    def test_first_run_announces_the_download_and_the_end(self):
        notices = self._load(cached=False)
        self.assertEqual(len(notices), 2)
        self.assertIn("Downloading", notices[0])
        self.assertIn("medium", notices[0])
        self.assertIn("1.5 GB", notices[0])
        self.assertEqual(notices[1], "Speech model ready.")

    def test_a_cached_model_loads_silently(self):
        self.assertEqual(self._load(cached=True), [])

    def test_size_text(self):
        self.assertEqual(_model_size_text("base.en"), "about 145 MB")
        self.assertEqual(_model_size_text("large-v3"), "about 3.0 GB")
        self.assertEqual(_model_size_text("my-own-model"), "a large file")

    def test_pipeline_forwards_notices_to_the_tray_queue(self):
        pipeline = SimpleNamespace(ui_queue=make_ui_queue())
        MainPipeline._notify_info(pipeline, "hello")
        message = pipeline.ui_queue.get_nowait()
        self.assertEqual(message.type, MessageType.INFO_EVENT)
        self.assertEqual(message.payload["text"], "hello")


class StartupOrderTests(unittest.TestCase):
    def test_onnxruntime_is_loaded_before_qt(self):
        """PyQt6's old MSVCP140.dll crashes a later onnxruntime import.

        Run in a fresh interpreter: importing main must load onnxruntime and
        must not have pulled PyQt6 in first.
        """
        import subprocess

        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import main\n"
            "assert 'onnxruntime' in sys.modules, 'onnxruntime not imported by main'\n"
            "assert 'PyQt6' not in sys.modules, 'PyQt6 was loaded before onnxruntime'\n"
            "print('order ok')\n" % str(root)
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertIn("order ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
