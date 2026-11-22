"""Phase 4 tests — hotkeys, settings persistence, stealth, Qt widgets.

Qt tests run offscreen when PyQt6 is importable. The non-Qt tests always run.
"""

from __future__ import annotations

# onnxruntime must be imported BEFORE PyQt6 (PyQt6's old MSVCP140.dll makes a later
# onnxruntime import crash the process; see main.py). Keep this line first.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import ctypes
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from ctypes import wintypes

from src.config.providers import load_provider_config
from src.config.settings import EngineConfig, load_config, save_user_settings
from src.ui import stealth, win_effects
from src.ui.overlay import OverlayWindow, _choose_font_family, _load_bundled_fonts
from src.ui.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    WM_HOTKEY,
    GlobalHotkeyManager,
    parse_hotkey,
    validate_hotkeys,
)
from src.ui.settings_dialog import (
    SettingsDialogController,
    build_provider_entries,
    normalize_provider_entries,
)

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class HotkeyParserTests(unittest.TestCase):
    def test_parse_ctrl_shift_t(self):
        hotkey = parse_hotkey("Ctrl+Shift+T")
        self.assertEqual(hotkey.modifiers, MOD_CONTROL | MOD_SHIFT)
        self.assertEqual(hotkey.vk, ord("T"))
        self.assertEqual(hotkey.canonical, "Ctrl+Shift+T")

    def test_parse_is_case_insensitive_and_canonicalizes_order(self):
        hotkey = parse_hotkey("shift+ctrl+alt+k")
        self.assertEqual(hotkey.modifiers, MOD_CONTROL | MOD_SHIFT | MOD_ALT)
        self.assertEqual(hotkey.canonical, "Ctrl+Alt+Shift+K")

    def test_parse_function_and_named_keys(self):
        self.assertEqual(parse_hotkey("Ctrl+F8").vk, 0x77)
        self.assertEqual(parse_hotkey("Alt+PageDown").vk, 0x22)
        self.assertEqual(parse_hotkey("Ctrl+Space").canonical, "Ctrl+SPACE")
        self.assertEqual(parse_hotkey("Ctrl++").canonical, "Ctrl++")

    def test_parse_requires_modifier_and_key(self):
        with self.assertRaises(ValueError):
            parse_hotkey("T")
        with self.assertRaises(ValueError):
            parse_hotkey("Ctrl+Shift")
        with self.assertRaises(ValueError):
            parse_hotkey("")
        with self.assertRaises(ValueError):
            parse_hotkey("Ctrl+NotAKey")

    def test_validate_duplicates_and_empty_bindings(self):
        self.assertEqual(validate_hotkeys({}), [])
        self.assertEqual(
            validate_hotkeys({"one": "", "two": "Ctrl+Shift+T"}), []
        )
        errors = validate_hotkeys({
            "one": "Ctrl+Shift+T",
            "two": "shift+ctrl+t",
        })
        self.assertEqual(len(errors), 1)
        self.assertIn("already used", errors[0])


class StealthTests(unittest.TestCase):
    def test_non_windows_is_noop(self):
        with mock.patch.object(stealth, "is_windows", return_value=False):
            self.assertFalse(stealth.apply_stealth(123))
            self.assertFalse(stealth.remove_stealth(123))

    def test_supported_uses_exclude_from_capture(self):
        calls = []

        def fake_set(hwnd, affinity):
            calls.append(affinity)
            return True

        with mock.patch.object(stealth, "is_windows", return_value=True), \
                mock.patch.object(stealth, "is_supported", return_value=True), \
                mock.patch.object(stealth, "_set_affinity", side_effect=fake_set):
            self.assertTrue(stealth.apply_stealth(1))
        self.assertEqual(calls, [stealth.WDA_EXCLUDEFROMCAPTURE])

    def test_falls_back_to_monitor(self):
        calls = []

        def fake_set(hwnd, affinity):
            calls.append(affinity)
            return affinity == stealth.WDA_MONITOR

        with mock.patch.object(stealth, "is_windows", return_value=True), \
                mock.patch.object(stealth, "is_supported", return_value=True), \
                mock.patch.object(stealth, "_set_affinity", side_effect=fake_set):
            self.assertTrue(stealth.apply_stealth(1))
        self.assertEqual(
            calls,
            [stealth.WDA_EXCLUDEFROMCAPTURE, stealth.WDA_MONITOR],
        )

    def test_remove_uses_wda_none(self):
        calls = []
        with mock.patch.object(stealth, "is_windows", return_value=True), \
                mock.patch.object(
                    stealth, "_set_affinity",
                    side_effect=lambda h, a: calls.append(a) or True,
                ):
            self.assertTrue(stealth.remove_stealth(1))
        self.assertEqual(calls, [stealth.WDA_NONE])

    def test_build_support(self):
        with mock.patch.object(stealth.sys, "platform", "win32"), \
                mock.patch.object(
                    stealth.sys, "getwindowsversion",
                    return_value=SimpleNamespace(build=19041),
                ):
            self.assertTrue(stealth.is_supported())
        with mock.patch.object(stealth.sys, "platform", "win32"), \
                mock.patch.object(
                    stealth.sys, "getwindowsversion",
                    return_value=SimpleNamespace(build=19040),
                ):
            self.assertFalse(stealth.is_supported())


class Win32EffectTests(unittest.TestCase):
    def test_click_through_uses_ptr_api_and_preserves_other_styles(self):
        class FakeUser32:
            def __init__(self):
                self.style = 0x00000008  # unrelated existing style
                self.set_calls = []
                self.pos_calls = []

            def GetWindowLongPtrW(self, hwnd, index):
                return self.style

            def SetWindowLongPtrW(self, hwnd, index, value):
                old = self.style
                self.style = value
                self.set_calls.append(value)
                return old

            def SetWindowPos(self, *args):
                self.pos_calls.append(args)
                return True

        fake = FakeUser32()
        with mock.patch.object(win_effects.sys, "platform", "win32"), \
                mock.patch.object(win_effects, "get_user32", return_value=fake):
            self.assertTrue(win_effects.set_click_through(123, True))
            self.assertTrue(fake.style & win_effects.WS_EX_TRANSPARENT)
            self.assertTrue(fake.style & win_effects.WS_EX_LAYERED)
            self.assertTrue(fake.style & 0x00000008)
            self.assertTrue(win_effects.set_click_through(123, False))
            self.assertFalse(fake.style & win_effects.WS_EX_TRANSPARENT)
            self.assertTrue(fake.style & win_effects.WS_EX_LAYERED)

        self.assertEqual(len(fake.set_calls), 2)
        self.assertEqual(len(fake.pos_calls), 2)

    def test_acrylic_falls_back_to_blurbehind(self):
        calls = []

        def fake_accent(hwnd, state, tint=0):
            calls.append(state)
            return state == win_effects.ACCENT_ENABLE_BLURBEHIND

        with mock.patch.object(win_effects.sys, "platform", "win32"), \
                mock.patch.object(
                    win_effects.sys, "getwindowsversion",
                    return_value=SimpleNamespace(build=19045),
                ), \
                mock.patch.object(win_effects, "_set_accent", side_effect=fake_accent):
            self.assertTrue(win_effects.apply_acrylic(1))
        self.assertEqual(
            calls,
            [
                win_effects.ACCENT_ENABLE_ACRYLICBLURBEHIND,
                win_effects.ACCENT_ENABLE_BLURBEHIND,
            ],
        )


class HotkeyDispatchTests(unittest.TestCase):
    def test_native_hotkey_message_invokes_callback(self):
        manager = GlobalHotkeyManager()
        called = []
        manager._ids[0x4C54] = "toggle"
        manager._callbacks["toggle"] = lambda: called.append(True)
        msg = wintypes.MSG()
        msg.message = WM_HOTKEY
        msg.wParam = 0x4C54
        handled, result = manager._native_event(
            "windows_generic_MSG", ctypes.addressof(msg)
        )
        self.assertTrue(handled)
        self.assertEqual(result, 0)
        self.assertEqual(called, [True])

    def test_unrelated_native_message_is_ignored(self):
        manager = GlobalHotkeyManager()
        msg = wintypes.MSG()
        msg.message = 0x0010
        handled, result = manager._native_event(
            "windows_generic_MSG", ctypes.addressof(msg)
        )
        self.assertFalse(handled)
        self.assertEqual(result, 0)


class ProviderSelectionTests(unittest.TestCase):
    def test_normalize_dedupes_and_keeps_order(self):
        entries = [
            {"name": "gemini", "model": "a"},
            {"name": "gemini", "model": "b"},
            {"name": ""},
            "not-a-dict",
            {"name": "nllb", "model": "c"},
        ]
        result = normalize_provider_entries(entries)
        self.assertEqual([e["name"] for e in result], ["gemini", "nllb"])
        self.assertEqual(result[0]["model"], "a")

    def test_rebuild_preserves_models_and_priority(self):
        current = [
            {"name": "gemini", "model": "custom-gemini", "timeout_ms": 999},
            {"name": "groq", "model": "custom-groq"},
        ]
        result = build_provider_entries(
            current,
            ["groq", "gemini", "deepl"],
            {"gemini", "deepl"},
        )
        self.assertEqual([e["name"] for e in result], ["gemini", "deepl"])
        self.assertEqual(result[0]["model"], "custom-gemini")
        self.assertEqual(result[0]["timeout_ms"], 999)
        self.assertEqual(result[1]["timeout_ms"], 2000)


class SettingsPersistenceTests(unittest.TestCase):
    def test_user_settings_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig()
            cfg.privacy.state_dir = tmp
            cfg.engine.whisper_model = "small"
            cfg.engine.whisper_device = "cpu"
            cfg.overlay.opacity = 0.33
            cfg.overlay.max_lines = 2
            cfg.overlay.click_through = False
            cfg.hotkeys.toggle_overlay = "Ctrl+Alt+O"
            cfg.translation.providers = [
                {"name": "gemini", "model": "custom", "timeout_ms": 123}
            ]
            cfg.privacy.confidential_mode = True

            self.assertTrue(save_user_settings(cfg))
            loaded = load_config(
                path=Path(tmp) / "does-not-exist.yaml",
                user_path=Path(tmp) / "settings.yaml",
            )

            self.assertEqual(loaded.engine.whisper_model, "small")
            self.assertEqual(loaded.engine.whisper_device, "cpu")
            self.assertAlmostEqual(loaded.overlay.opacity, 0.33)
            self.assertEqual(loaded.overlay.max_lines, 2)
            self.assertFalse(loaded.overlay.click_through)
            self.assertEqual(loaded.hotkeys.toggle_overlay, "Ctrl+Alt+O")
            self.assertEqual(loaded.translation.providers[0]["model"], "custom")
            self.assertTrue(loaded.privacy.confidential_mode)

    def test_save_does_not_write_api_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig()
            cfg.privacy.state_dir = tmp
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "super-secret"}):
                self.assertTrue(save_user_settings(cfg))
            content = (Path(tmp) / "settings.yaml").read_text(encoding="utf-8")
            self.assertNotIn("super-secret", content)
            self.assertNotIn("api_key", content)

    def test_credential_store_fallback_for_provider_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                mock.patch(
                    "src.config.providers._stored_api_key",
                    return_value="stored-key",
                ):
            provider_cfg = load_provider_config("groq", {"name": "groq"})
        self.assertEqual(provider_cfg.api_key, "stored-key")


class SettingsDialogLogicTests(unittest.TestCase):
    def test_save_key_uses_credential_store_and_session_env(self):
        controller = SettingsDialogController(EngineConfig())
        controller.store = mock.Mock()
        controller.store.set_key.return_value = True
        edit = mock.Mock()
        edit.text.return_value = "  secret  "
        with mock.patch.dict(os.environ, {}, clear=False):
            controller._save_key("groq", edit)
            self.assertEqual(os.environ.get("GROQ_API_KEY"), "secret")
        controller.store.set_key.assert_called_once_with("groq", "secret")
        edit.clear.assert_called_once()

    def test_quota_summary_handles_unlimited_and_exhausted(self):
        config = EngineConfig()
        router = mock.Mock()
        router.remaining.side_effect = lambda name: {"gemini": 5, "nllb": -1, "deepl": 0}[name]
        engine = SimpleNamespace(
            providers=[
                SimpleNamespace(name="gemini"),
                SimpleNamespace(name="nllb"),
                SimpleNamespace(name="deepl"),
            ],
            router=router,
        )
        controller = SettingsDialogController(config, engine=engine)
        summary = controller.quota_summary()
        self.assertIn("gemini: 5", summary)
        self.assertIn("nllb: unlimited", summary)
        self.assertIn("deepl: exhausted", summary)

    def test_bool_parent_from_qaction_is_ignored(self):
        controller = SettingsDialogController(EngineConfig())
        fake_dialog = mock.Mock()
        fake_dialog.exec.return_value = 0
        with mock.patch.object(
            controller, "_build_dialog", return_value=fake_dialog
        ) as build:
            self.assertIs(controller.show(True), fake_dialog)
            build.assert_called_once_with(None)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not installed")
class QtOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._clock = [1000.0]      # fake time for paragraph pauses (seconds)
        self._patches = [
            mock.patch("src.ui.overlay._now", side_effect=lambda: self._clock[0]),
            mock.patch("src.ui.overlay.win_effects.apply_acrylic", return_value=True),
            mock.patch("src.ui.overlay.win_effects.set_click_through", return_value=True),
            mock.patch("src.ui.overlay.win_effects.alt_is_down", return_value=False),
            mock.patch("src.ui.overlay.stealth.apply_stealth", return_value=True),
            mock.patch("src.ui.overlay.stealth.remove_stealth", return_value=True),
        ]
        for patch in self._patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.config = EngineConfig()
        self.overlay = OverlayWindow(self.config)
        self.widget = self.overlay.create()
        self.assertIsNotNone(self.widget)

    def tearDown(self):
        if self.widget is not None:
            self.widget.hide()
            self.widget.deleteLater()
        self.app.processEvents()

    def test_partial_replaces_instead_of_appending(self):
        self.overlay.update_text("مرحبا", is_partial=True)
        self.assertEqual(len(self.widget._labels), 1)
        first = self.widget._labels[0]
        self.overlay.update_text("مرحبا بكم", is_partial=True)
        self.assertEqual(len(self.widget._labels), 1)
        self.assertIs(self.widget._labels[0], first)
        self.assertEqual(first.text(), "مرحبا بكم")

        self.overlay.update_text("أهلاً وسهلاً", is_partial=False)
        self.assertEqual(len(self.widget._labels), 1)
        self.assertEqual(first.text(), "أهلاً وسهلاً")
        self.assertIsNone(self.widget._partial_label)

    def _advance(self, seconds):
        self._clock[0] += seconds

    def test_history_is_kept_and_scrolls(self):
        for index in range(5):
            self.overlay.update_text(f"جملة {index}")
            self._advance(10)          # a pause between them: one paragraph each
        self.assertEqual(len(self.widget._labels), 5)
        self.assertEqual(
            [label.text() for label in self.widget._labels],
            [f"جملة {i}" for i in range(5)],
        )

    def _event(self, kind, x, y, buttons=None, modifiers=None):
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent
        point = QPointF(float(x), float(y))
        global_point = QPointF(
            float(self.widget.x() + x), float(self.widget.y() + y)
        )
        left = Qt.MouseButton.LeftButton
        return QMouseEvent(
            kind, point, global_point, left,
            left if buttons is None else buttons,
            Qt.KeyboardModifier.NoModifier if modifiers is None else modifiers,
        )

    def test_never_smaller_than_two_lines(self):
        from PyQt6.QtGui import QFontMetrics
        self.widget.resize(10, 10)
        self.app.processEvents()
        two_lines = 2 * QFontMetrics(self.widget._font).lineSpacing()
        self.assertGreaterEqual(self.widget.minimumHeight(), two_lines)
        self.assertGreaterEqual(self.widget.height(), self.widget.minimumHeight())
        self.assertGreaterEqual(self.widget.width(), self.widget.minimumWidth())

    def _settle(self):
        # Qt lays the scroll content out over a couple of event-loop passes.
        for _ in range(4):
            self.app.processEvents()

    def test_latest_line_stays_in_view_until_user_scrolls_up(self):
        self.widget.resize(self.widget.width(), self.widget.minimumHeight())
        for index in range(30):
            self.overlay.update_text(f"جملة رقم {index} " * 3)
        self._settle()
        bar = self.widget._scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        self.assertEqual(bar.value(), bar.maximum(), "newest line not in view")

        bar.setValue(0)                      # the user reads older text
        self.assertFalse(self.widget._follow)
        self.overlay.update_text("جملة جديدة")
        self._settle()
        self.assertEqual(bar.value(), 0, "view jumped while reading history")

        self.widget.scroll_to_latest()
        self.overlay.update_text("وجملة أخرى")
        self._settle()
        self.assertEqual(bar.value(), bar.maximum())

    def test_handle_strip_drags_without_alt(self):
        from PyQt6.QtCore import QEvent
        x = self.widget.width() // 2
        self.widget.mousePressEvent(self._event(QEvent.Type.MouseButtonPress, x, 5))
        self.assertTrue(self.widget._dragging)
        self.widget.mouseReleaseEvent(
            self._event(QEvent.Type.MouseButtonRelease, x, 5)
        )
        self.assertEqual(self.config.overlay.position, "custom")

    def test_text_area_press_does_not_drag(self):
        from PyQt6.QtCore import QEvent
        self.widget.mousePressEvent(self._event(
            QEvent.Type.MouseButtonPress, self.widget.width() // 2,
            self.widget.height() // 2,
        ))
        self.assertFalse(self.widget._dragging)
        self.assertFalse(self.widget._resizing)

    def test_corner_grip_resizes_clamps_to_minimum_and_persists(self):
        from PyQt6.QtCore import QEvent
        moved = []
        self.overlay.on_moved = lambda: moved.append(True)
        w, h = self.widget.width(), self.widget.height()

        # Grow from the bottom-right corner.
        self.widget.mousePressEvent(
            self._event(QEvent.Type.MouseButtonPress, w - 5, h - 5)
        )
        self.assertTrue(self.widget._resizing)
        self.widget.mouseMoveEvent(
            self._event(QEvent.Type.MouseMove, w - 5 + 60, h - 5 + 40)
        )
        self.assertEqual(self.widget.width(), w + 60)
        self.assertEqual(self.widget.height(), h + 40)
        self.widget.mouseReleaseEvent(
            self._event(QEvent.Type.MouseButtonRelease, w - 5 + 60, h - 5 + 40)
        )
        self.assertEqual(self.config.overlay.width, w + 60)
        self.assertEqual(self.config.overlay.height, h + 40)
        self.assertEqual(moved, [True])

        # Shrink far past the limit from the bottom-left corner.
        w, h = self.widget.width(), self.widget.height()
        right_edge = self.widget.x() + w
        self.widget.mousePressEvent(
            self._event(QEvent.Type.MouseButtonPress, 5, h - 5)
        )
        self.widget.mouseMoveEvent(
            self._event(QEvent.Type.MouseMove, 5 + 5000, h - 5 - 5000)
        )
        self.assertEqual(self.widget.width(), self.widget.minimumWidth())
        self.assertEqual(self.widget.height(), self.widget.minimumHeight())
        self.assertEqual(
            self.widget.x() + self.widget.width(), right_edge,
            "left grip must keep the right edge fixed",
        )
        self.widget.mouseReleaseEvent(
            self._event(QEvent.Type.MouseButtonRelease, 5, h - 5)
        )

    def test_hide_folds_to_strip_and_show_restores_size(self):
        for index in range(4):
            self.overlay.update_text(f"line {index}")
            self._advance(10)
        self._settle()
        full = self.widget.height()
        self.assertTrue(self.widget._btn_hide.isEnabled())
        self.assertFalse(self.widget._btn_show.isEnabled())

        self.widget._btn_hide.click()
        self._settle()
        self.assertTrue(self.widget._collapsed)
        self.assertEqual(self.widget.height(), self.widget._handle_h)
        self.assertFalse(self.widget._scroll.isVisible())
        self.assertFalse(self.widget._btn_hide.isEnabled())
        self.assertTrue(self.widget._btn_show.isEnabled())

        # Text keeps arriving while folded and is there after Show.
        self._advance(10)
        self.overlay.update_text("arrived while folded")
        self.assertEqual(len(self.widget._labels), 5)

        self.widget._btn_show.click()
        self._settle()
        self.assertFalse(self.widget._collapsed)
        self.assertEqual(self.widget.height(), full)
        self.assertTrue(self.widget._scroll.isVisible())

    def test_folded_window_saves_the_full_height(self):
        self._settle()
        full = self.widget.height()
        self.widget._btn_hide.click()
        self.widget._persist_geometry()
        self.assertEqual(self.config.overlay.height, full)

    def test_folded_window_has_no_resize_grips(self):
        self.widget._btn_hide.click()
        w, h = self.widget.width(), self.widget.height()
        from PyQt6.QtCore import QPointF
        self.assertEqual(self.widget._hit(QPointF(w - 3.0, h - 3.0)), "move")
        self.assertEqual(len(self.widget._interactive_zones()), 1)

    def test_close_needs_two_clicks(self):
        closed = []
        self.overlay.on_close = lambda: closed.append(True)
        self.widget._btn_close.click()
        self.assertEqual(closed, [], "one stray click must not quit")
        self.assertTrue(self.widget._close_armed)
        self.assertEqual(self.widget._btn_close.text(), "Close?‎")

        self.widget._btn_close.click()
        self.assertEqual(closed, [True])
        self.assertFalse(self.widget._close_armed)

    def test_close_disarms_when_the_second_click_never_comes(self):
        closed = []
        self.overlay.on_close = lambda: closed.append(True)
        self.widget._btn_close.click()
        self.widget._disarm_close()          # what the 2.5 s timer does
        self.assertFalse(self.widget._close_armed)
        self.widget._btn_close.click()       # arms again, does not quit
        self.assertEqual(closed, [])

    def test_buttons_stay_inside_the_window(self):
        for width in (self.widget.minimumWidth(), 500, 900):
            self.widget.resize(width, self.widget.height())
            self._settle()
            for button in (self.widget._btn_hide, self.widget._btn_show,
                           self.widget._btn_close):
                geo = button.geometry()
                self.assertGreaterEqual(geo.left(), 0)
                self.assertLessEqual(geo.right(), self.widget.width())
                self.assertLessEqual(geo.bottom(), self.widget._handle_h)

    def _text_extent(self, label):
        """(leftmost, rightmost) x of the painted text inside a label."""
        import numpy as np

        image = label.grab().toImage().convertToFormat(
            __import__("PyQt6.QtGui", fromlist=["QImage"]).QImage.Format.Format_ARGB32
        )
        pointer = image.constBits()
        pointer.setsize(image.sizeInBytes())
        pixels = np.frombuffer(pointer, dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine() // 4, 4
        )[:, : image.width(), :]
        columns = np.where((pixels[:, :, 3] > 128).any(axis=0))[0]
        self.assertGreater(columns.size, 0, "the label painted no text")
        return int(columns.min()), int(columns.max()), image.width()

    def test_short_lines_hug_the_right_edge_in_rtl(self):
        # Qt mirrors a plain AlignRight into "left" in a right-to-left window,
        # which put every short line / last line of a paragraph on the left.
        self.config.overlay.fade_duration_ms = 0      # no fade-in while we measure
        self.overlay.update_text("مرحباً بكم في الاجتماع.")
        self._settle()
        label = self.widget._labels[0]
        left, right, width = self._text_extent(label)
        self.assertGreater(right, width - 48, "text is not on the right edge")
        self.assertGreater(left, width * 0.4, "a short line should not reach the far left")

    def test_short_lines_hug_the_left_edge_in_ltr(self):
        self.config.overlay.fade_duration_ms = 0
        self.config.overlay.text_direction = "ltr"
        self.overlay.apply_settings()
        self.overlay.update_text("Hello everyone, welcome.")
        self._settle()
        left, right, width = self._text_extent(self.widget._labels[0])
        self.assertLess(left, 48)
        self.assertLess(right, width - 48, "a short LTR line should not reach the right edge")

    def test_pieces_flow_into_one_paragraph(self):
        # The translator delivers a few words at a time; they must read as one
        # flowing text, not one short line per piece.
        for piece in ("شاهدت سفينة جيش فيكنج كبير في", "جنوب ويلز.", "إذا وصلوا مسارهم،"):
            self.overlay.update_text(piece)
            self._advance(1.5)
        self.assertEqual(len(self.widget._labels), 1)
        self.assertEqual(
            self.widget._labels[0].text(),
            "شاهدت سفينة جيش فيكنج كبير في جنوب ويلز. إذا وصلوا مسارهم،",
        )

    def test_draft_follows_the_confirmed_text_in_the_same_paragraph(self):
        self.overlay.update_text("أول", is_partial=False)
        self.overlay.update_text("تجريبي", is_partial=True)
        label = self.widget._labels[0]
        self.assertEqual(len(self.widget._labels), 1)
        self.assertEqual(label.text(), "أول تجريبي")
        self.assertIs(self.widget._partial_label, label)     # still has a draft

        self.overlay.update_text("ثاني", is_partial=False)   # confirmed replaces the draft
        self.assertEqual(label.text(), "أول ثاني")
        self.assertIsNone(self.widget._partial_label)

    def test_a_pause_starts_a_new_paragraph(self):
        self.overlay.update_text("قبل التوقف")
        self._advance(3.0)                       # shorter than the 4 s default
        self.overlay.update_text("بعد وقفة قصيرة")
        self.assertEqual(len(self.widget._labels), 1)
        self._advance(5.0)                       # a real pause
        self.overlay.update_text("بعد وقفة طويلة")
        self.assertEqual(len(self.widget._labels), 2)
        self.assertEqual(self.widget._labels[1].text(), "بعد وقفة طويلة")

    def test_the_pause_length_comes_from_the_config(self):
        self.config.overlay.paragraph_pause_s = 0.5
        self.overlay.update_text("أول")
        self._advance(1.0)
        self.overlay.update_text("ثاني")
        self.assertEqual(len(self.widget._labels), 2)

    def test_a_very_long_paragraph_is_closed(self):
        self.overlay.update_text("كلمة " * 160)          # ~800 characters
        self._advance(1.0)
        self.overlay.update_text("جملة جديدة")
        self.assertEqual(len(self.widget._labels), 2)

    def test_clear_text_starts_a_fresh_paragraph(self):
        self.overlay.update_text("قديم")
        self.overlay.clear_text()
        self.overlay.update_text("جديد")
        self.assertEqual([l.text() for l in self.widget._labels], ["جديد"])

    def test_a_flowing_paragraph_fills_the_window_width_before_wrapping(self):
        # Several short pieces must wrap at the window edge, not before it.
        self.config.overlay.fade_duration_ms = 0
        for piece in ("شاهدت سفينة جيش فيكنج كبير في", "جنوب ويلز.", "إذا وصلوا مسارهم،"):
            self.overlay.update_text(piece)
        self.widget.resize(900, self.widget.height())
        self._settle()
        label = self.widget._labels[0]
        left, right, width = self._text_extent(label)
        # 58 characters at this size need well over half of 900 px on one line:
        # if every piece had its own line the text would be < 40% wide.
        self.assertGreater(right - left, width * 0.55)

    def test_hover_zones_cover_handle_and_grips_only(self):
        zones = self.widget._interactive_zones()
        geo = self.widget.geometry()
        from PyQt6.QtCore import QPoint
        handle = QPoint(geo.center().x(), geo.top() + 3)
        corner = QPoint(geo.right() - 3, geo.bottom() - 3)
        middle = geo.center()
        self.assertTrue(any(z.contains(handle) for z in zones))
        self.assertTrue(any(z.contains(corner) for z in zones))
        self.assertFalse(any(z.contains(middle) for z in zones))

    def test_clear_text(self):
        self.overlay.update_text("نص")
        self.overlay.clear_text()
        self.assertEqual(self.widget._labels, [])
        self.assertIsNone(self.widget._partial_label)

    def test_alt_drag_is_persisted_as_custom_position(self):
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        moved = []
        self.overlay.on_moved = lambda: moved.append(True)

        self.widget.move(123, 77)
        self.widget._dragging = True  # as set by an Alt+left press
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.widget.mouseReleaseEvent(event)

        self.assertFalse(self.widget._dragging)
        self.assertEqual(self.config.overlay.position, "custom")
        self.assertEqual(self.config.overlay.custom_x, self.widget.x())
        self.assertEqual(self.config.overlay.custom_y, self.widget.y())
        self.assertEqual(moved, [True])

        # The next update must keep the dropped position, not snap back.
        self.overlay.update_text("نص")
        self.assertEqual(
            (self.widget.x(), self.widget.y()),
            (self.config.overlay.custom_x, self.config.overlay.custom_y),
        )

    def test_apply_settings_updates_font_and_max_lines(self):
        for index in range(4):
            self.overlay.update_text(f"line {index}")
            self._advance(10)
        self.config.overlay.opacity = 0.5
        self.config.overlay.font_size = 31
        self.config.overlay.font_family = "Definitely Not Installed Font"
        self.config.overlay.max_lines = 2
        self.config.overlay.position = "top-left"
        self.overlay.apply_settings()

        self.assertAlmostEqual(self.widget.windowOpacity(), 0.5, places=2)
        self.assertEqual(self.widget._labels[0].font().pointSize(), 31)
        self.assertIn(self.widget._font_family, self.widget._loaded_fonts)
        self.assertEqual(len(self.widget._labels), 4, "history must be kept")
        self.assertGreaterEqual(self.widget.height(), self.widget.minimumHeight())

    def test_toggle_visibility(self):
        self.assertTrue(self.widget.isVisible())
        self.assertFalse(self.overlay.toggle_visibility())
        self.assertFalse(self.widget.isVisible())
        self.assertTrue(self.overlay.toggle_visibility())
        self.assertTrue(self.widget.isVisible())

    def test_glass_panel_and_stable_readable_width(self):
        self.overlay.update_text(
            "مرحباً بكم في الاجتماع. سنبدأ بعد دقيقة. "
            "دعونا نراجع جدول الأعمال أولاً.",
            is_partial=False,
        )
        self.app.processEvents()
        self.assertGreater(
            self.widget.width(), 250,
            "word-wrapped overlay collapsed to a narrow column",
        )
        image = self.widget.grab().toImage()
        sample = image.pixelColor(
            10, max(1, self.widget.height() // 2)
        )
        self.assertGreater(
            sample.alpha(), 0, "glass panel background was not painted"
        )

    def test_bundled_arabic_font_loads(self):
        from PyQt6.QtGui import QFontDatabase
        families = _load_bundled_fonts(QFontDatabase)
        self.assertTrue(
            any("Naskh" in family for family in families),
            f"bundled font not registered: {families}",
        )
        chosen = _choose_font_family(
            "Definitely Not Installed Font", families, QFontDatabase
        )
        self.assertIn(chosen, families)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not installed")
class QtSettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_build_dialog_does_not_crash(self):
        from PyQt6.QtWidgets import QFontComboBox, QLabel, QTabWidget
        controller = SettingsDialogController(EngineConfig())
        controller.store = mock.Mock()
        controller.store.get_key.return_value = None
        dialog = controller._build_dialog(None)
        self.assertIsNotNone(dialog)
        self.assertIn("Settings", dialog.windowTitle())
        tabs = dialog.findChild(QTabWidget)
        self.assertEqual(tabs.currentIndex(), 1)  # first run starts on API Keys
        font_combo = dialog.findChild(QFontComboBox)
        self.assertEqual(font_combo.currentFont().family(), "Noto Naskh Arabic")
        notes = [
            label for label in dialog.findChildren(QLabel)
            if "Credential Manager" in label.text()
        ]
        self.assertTrue(notes)
        self.assertTrue(notes[0].wordWrap())
        dialog.close()

    def test_accept_collects_settings_and_calls_callback(self):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        config = EngineConfig()
        changed = []
        controller = SettingsDialogController(
            config, on_config_changed=changed.append
        )
        controller.store = mock.Mock()
        controller.store.get_key.return_value = None
        with mock.patch(
            "src.ui.settings_dialog.save_user_settings", return_value=True
        ) as save:
            dialog = controller._build_dialog(None)
            box = dialog.findChild(QDialogButtonBox)
            self.assertIsNotNone(box)
            box.accepted.emit()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        save.assert_called_once_with(config)
        self.assertEqual(changed, [config])

    def test_key_save_checks_matching_provider(self):
        from PyQt6.QtCore import Qt
        config = EngineConfig()
        controller = SettingsDialogController(config)
        controller.store = mock.Mock()
        controller.store.set_key.return_value = True
        dialog = controller._build_dialog(None)
        controller._key_edits["deepl"].setText("secret")
        with mock.patch.dict(os.environ, {}, clear=False):
            controller._save_key("deepl", controller._key_edits["deepl"])
        matching = None
        for row in range(controller._provider_list.count()):
            item = controller._provider_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == "deepl":
                matching = item
                break
        self.assertIsNotNone(matching)
        self.assertEqual(matching.checkState(), Qt.CheckState.Checked)
        dialog.close()


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not installed")
class FirstRunSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _run(self, result):
        import main
        from PyQt6.QtWidgets import QDialog

        seen = {}

        class FakeController:
            def __init__(self, config, engine=None):
                seen["config"] = config
                seen["engine"] = engine

            def show(self):
                return SimpleNamespace(
                    result=lambda: getattr(QDialog.DialogCode, result)
                )

        config = EngineConfig()
        with mock.patch.object(main, "SettingsDialogController", FakeController), \
                mock.patch("PyQt6.QtWidgets.QMessageBox.information") as info:
            accepted = main.open_first_run_settings(config)
        return accepted, info, seen, config

    def test_accept_shows_restart_prompt(self):
        accepted, info, seen, config = self._run("Accepted")
        self.assertTrue(accepted)
        info.assert_called_once()
        self.assertIs(seen["config"], config)
        self.assertIsNone(seen["engine"])

    def test_cancel_does_not_show_prompt(self):
        accepted, info, _seen, _config = self._run("Rejected")
        self.assertFalse(accepted)
        info.assert_not_called()


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not installed")
class QtTrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_menu_status_provider_and_bool_callbacks(self):
        from src.pipeline.ipc import PipelineStatus
        from src.ui.tray import TrayController

        open_settings = mock.Mock()
        toggle_listening = mock.Mock()
        exit_app = mock.Mock()
        click_through = mock.Mock()
        stealth = mock.Mock()
        tray = TrayController(
            on_toggle_listening=toggle_listening,
            on_open_settings=open_settings,
            on_exit=exit_app,
            on_toggle_click_through=click_through,
            on_toggle_stealth=stealth,
            click_through=True,
            stealth=True,
        )
        with mock.patch(
            "PyQt6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
            return_value=True,
        ):
            created = tray.create()
        self.assertIsNotNone(created)
        self.addCleanup(created.hide)
        self.addCleanup(tray.menu.deleteLater)

        actions = {action.text(): action for action in tray.menu.actions()}
        actions["Settings"].trigger()
        open_settings.assert_called_once_with()

        tray.update_status(PipelineStatus.RECORDING)
        tray.update_provider("gemini")
        self.assertEqual(tray.status_action.text(), "Status: Recording")
        self.assertEqual(tray.provider_action.text(), "Provider: gemini")
        self.assertEqual(tray.toggle_action.text(), "Pause listening")
        self.assertIn("Recording", tray.tray.toolTip())
        self.assertIn("gemini", tray.tray.toolTip())

        tray.click_through_action.setChecked(False)
        click_through.assert_called_once_with(False)
        tray.stealth_action.setChecked(False)
        stealth.assert_called_once_with(False)

        tray.sync_overlay_actions(True, True)
        self.assertTrue(tray.click_through_action.isChecked())
        self.assertTrue(tray.stealth_action.isChecked())
        self.assertEqual(click_through.call_count, 1)
        self.assertEqual(stealth.call_count, 1)


if __name__ == "__main__":
    unittest.main()
