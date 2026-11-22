"""Global hotkeys on Windows via RegisterHotKey + Qt's native event filter.

The parser is pure Python (and unit-testable without Qt). The manager is a
no-op on non-Windows systems and reports registration conflicts instead of
crashing the tray/UI.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
HOTKEY_ID_BASE = 0x4C54

HOTKEY_ACTIONS = (
    ("toggle_overlay", "Show / hide overlay"),
    ("toggle_listening", "Start / pause listening"),
    ("clear_history", "Clear translation history"),
    ("open_settings", "Open settings"),
)

_MODIFIERS = {
    "CTRL": MOD_CONTROL,
    "CONTROL": MOD_CONTROL,
    "ALT": MOD_ALT,
    "SHIFT": MOD_SHIFT,
    "WIN": MOD_WIN,
    "META": MOD_WIN,
    "SUPER": MOD_WIN,
    "CMD": MOD_WIN,
}
_MODIFIER_NAMES = (
    (MOD_CONTROL, "Ctrl"),
    (MOD_ALT, "Alt"),
    (MOD_SHIFT, "Shift"),
    (MOD_WIN, "Win"),
)

_NAMED_KEYS = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "RETURN": 0x0D,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "PAGEUP": 0x21,
    "PGUP": 0x21,
    "PAGEDOWN": 0x22,
    "PGDN": 0x22,
    "END": 0x23,
    "HOME": 0x24,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "INSERT": 0x2D,
    "INS": 0x2D,
    "DELETE": 0x2E,
    "DEL": 0x2E,
    "SEMICOLON": 0xBA,
    "EQUAL": 0xBB,
    "PLUS": 0xBB,
    "COMMA": 0xBC,
    "MINUS": 0xBD,
    "PERIOD": 0xBE,
    "SLASH": 0xBF,
    "BACKQUOTE": 0xC0,
    "BRACKETLEFT": 0xDB,
    "BACKSLASH": 0xDC,
    "BRACKETRIGHT": 0xDD,
    "QUOTE": 0xDE,
}
_PUNCTUATION_KEYS = {
    ";": 0xBA,
    "=": 0xBB,
    "+": 0xBB,
    ",": 0xBC,
    "-": 0xBD,
    ".": 0xBE,
    "/": 0xBF,
    "`": 0xC0,
    "[": 0xDB,
    "\\": 0xDC,
    "]": 0xDD,
    "'": 0xDE,
}


@dataclass(frozen=True)
class Hotkey:
    modifiers: int
    vk: int
    canonical: str


def _canonical_key(key: str) -> str:
    return key.strip().upper()


def _key_to_vk(key: str) -> int:
    key = _canonical_key(key)
    if key in _NAMED_KEYS:
        return _NAMED_KEYS[key]
    if len(key) == 1:
        char = key.upper()
        if "A" <= char <= "Z" or "0" <= char <= "9":
            return ord(char)
        if char in _PUNCTUATION_KEYS:
            return _PUNCTUATION_KEYS[char]
    if key.startswith("F") and key[1:].isdigit():
        number = int(key[1:])
        if 1 <= number <= 24:
            return 0x70 + number - 1
    raise ValueError(f"Unsupported key: {key or '<empty>'}")


def parse_hotkey(sequence: str) -> Hotkey:
    """Parse ``Ctrl+Shift+T`` into Win32 modifiers/VK and a canonical name."""
    if not sequence or not sequence.strip():
        raise ValueError("Hotkey is empty")

    text = sequence.strip()
    if text.endswith("++"):
        # QKeySequence's portable text for Ctrl+Plus is "Ctrl++".
        parts = [part.strip() for part in text[:-2].split("+") if part.strip()]
        parts.append("+")
    else:
        parts = [part.strip() for part in text.split("+")]
    if any(not part for part in parts):
        raise ValueError(f"Invalid hotkey: {sequence!r}")

    modifiers = 0
    key: str | None = None
    for part in parts:
        upper = part.upper()
        if upper in _MODIFIERS:
            modifiers |= _MODIFIERS[upper]
        elif key is None:
            key = part
        else:
            raise ValueError(f"Only one non-modifier key is allowed: {sequence!r}")

    if key is None:
        raise ValueError(f"Hotkey has no key: {sequence!r}")
    if not modifiers:
        raise ValueError(f"Hotkey needs Ctrl, Alt, Shift or Win: {sequence!r}")

    vk = _key_to_vk(key)
    canonical_key = key.upper() if len(key) == 1 else key
    if len(canonical_key) == 1 and "A" <= canonical_key <= "Z":
        canonical_key = canonical_key.upper()
    else:
        canonical_key = canonical_key.upper()

    names = [name for bit, name in _MODIFIER_NAMES if modifiers & bit]
    return Hotkey(modifiers=modifiers, vk=vk, canonical="+".join(names + [canonical_key]))


def validate_hotkeys(bindings: Mapping[str, str]) -> list[str]:
    """Return human-readable validation errors; empty means valid.

    Empty bindings are allowed and simply disable that action.
    """
    errors: list[str] = []
    seen: dict[str, str] = {}
    for action, sequence in bindings.items():
        if not sequence or not sequence.strip():
            continue
        try:
            parsed = parse_hotkey(sequence)
        except ValueError as exc:
            errors.append(f"{action}: {exc}")
            continue
        previous = seen.get(parsed.canonical)
        if previous is not None:
            errors.append(
                f"{action}: {parsed.canonical} is already used by {previous}"
            )
        else:
            seen[parsed.canonical] = action
    return errors


class GlobalHotkeyManager:
    """Register global hotkeys and dispatch them on the Qt thread."""

    def __init__(self):
        self._app = None
        self._window = None
        self._filter = None
        self._user32 = None
        self._hwnd = 0
        self._ids: dict[int, str] = {}
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._registered: dict[str, str] = {}

    @property
    def registered(self) -> dict[str, str]:
        return dict(self._registered)

    def start(
        self,
        bindings: Mapping[str, str],
        callbacks: Mapping[str, Callable[[], None]],
    ) -> list[str]:
        self.stop()
        errors: list[str] = []
        if sys.platform != "win32":
            return ["Global hotkeys are only supported on Windows"]

        try:
            from PyQt6.QtCore import QAbstractNativeEventFilter, Qt
            from PyQt6.QtWidgets import QApplication, QWidget
        except ImportError:
            return ["PyQt6 is not installed"]

        app = QApplication.instance()
        if app is None:
            return ["QApplication is not running"]

        try:
            from ctypes import wintypes
            self._user32 = ctypes.WinDLL("user32", use_last_error=True)
            self._user32.RegisterHotKey.argtypes = [
                wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT
            ]
            self._user32.RegisterHotKey.restype = wintypes.BOOL
            self._user32.UnregisterHotKey.argtypes = [
                wintypes.HWND, wintypes.INT
            ]
            self._user32.UnregisterHotKey.restype = wintypes.BOOL
        except Exception:
            logger.exception("Could not load user32 for hotkeys")
            return ["Could not load the Windows hotkey API"]

        self._app = app
        self._window = QWidget()
        self._window.setWindowFlags(Qt.WindowType.Tool)
        self._window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self._window.resize(1, 1)
        self._hwnd = int(self._window.winId())

        manager = self

        class _Filter(QAbstractNativeEventFilter):
            def nativeEventFilter(self, eventType, message):
                return manager._native_event(eventType, message)

        self._filter = _Filter()
        app.installNativeEventFilter(self._filter)

        for index, (action, sequence) in enumerate(bindings.items()):
            if not sequence or not sequence.strip():
                continue
            try:
                parsed = parse_hotkey(sequence)
            except ValueError as exc:
                errors.append(f"{action}: {exc}")
                continue
            callback = callbacks.get(action)
            if callback is None:
                continue
            hotkey_id = HOTKEY_ID_BASE + index
            if not self._user32.RegisterHotKey(
                self._hwnd, hotkey_id, parsed.modifiers | MOD_NOREPEAT, parsed.vk
            ):
                error = ctypes.get_last_error()
                errors.append(
                    f"{action}: {parsed.canonical} is unavailable (error {error})"
                )
                continue
            self._ids[hotkey_id] = action
            self._callbacks[action] = callback
            self._registered[action] = parsed.canonical

        if not self._registered and self._filter is not None:
            # Keep the filter installed only when at least one hotkey works.
            try:
                app.removeNativeEventFilter(self._filter)
            except Exception:
                pass
            self._filter = None
        return errors

    def reload(
        self,
        bindings: Mapping[str, str],
        callbacks: Mapping[str, Callable[[], None]],
    ) -> list[str]:
        return self.start(bindings, callbacks)

    def stop(self) -> None:
        if self._user32 is not None and self._hwnd:
            for hotkey_id in list(self._ids):
                try:
                    self._user32.UnregisterHotKey(self._hwnd, hotkey_id)
                except Exception:
                    logger.debug("UnregisterHotKey failed", exc_info=True)
        self._ids.clear()
        self._callbacks.clear()
        self._registered.clear()

        if self._app is not None and self._filter is not None:
            try:
                self._app.removeNativeEventFilter(self._filter)
            except Exception:
                pass
        self._filter = None
        if self._window is not None:
            try:
                self._window.deleteLater()
            except Exception:
                pass
        self._window = None
        self._hwnd = 0
        self._app = None
        self._user32 = None

    def _native_event(self, event_type, message):
        if hasattr(event_type, "data"):
            try:
                event_type = event_type.data()
            except Exception:
                pass
        if isinstance(event_type, bytes):
            try:
                event_type = event_type.decode("ascii", "ignore")
            except Exception:
                return False, 0
        if str(event_type) not in ("windows_generic_MSG", "windows_dispatcher_MSG"):
            return False, 0
        try:
            from ctypes import wintypes
            msg = ctypes.cast(
                int(message), ctypes.POINTER(wintypes.MSG)
            ).contents
            if msg.message != WM_HOTKEY:
                return False, 0
            action = self._ids.get(int(msg.wParam))
            if action is None:
                return False, 0
            callback = self._callbacks.get(action)
            if callback is not None:
                try:
                    callback()
                except Exception:
                    logger.exception("Hotkey callback failed: %s", action)
            return True, 0
        except Exception:
            logger.debug("Could not decode native hotkey message", exc_info=True)
            return False, 0
