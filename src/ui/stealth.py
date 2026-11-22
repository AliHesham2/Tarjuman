"""Stealth Mode — SetWindowDisplayAffinity(WDA_*).

Makes the overlay invisible to screen capture: it shows on the physical
monitor but never in Zoom/Meet share, screenshots, or recordings.

Windows 10 2004+ supports WDA_EXCLUDEFROMCAPTURE (the window disappears
from capture completely). Older builds fall back to WDA_MONITOR, which
renders the window black in captured frames.
"""

from __future__ import annotations

import ctypes
import logging
import sys

from src.ui.win_effects import get_user32

logger = logging.getLogger(__name__)

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Win10 2004+


def is_windows() -> bool:
    return sys.platform == "win32"


def is_supported() -> bool:
    """WDA_EXCLUDEFROMCAPTURE needs Windows 10 build 19041+."""
    if not is_windows():
        return False
    try:
        return int(sys.getwindowsversion().build) >= 19041
    except (AttributeError, TypeError, ValueError):
        return False


def _set_affinity(hwnd: int, affinity: int) -> bool:
    user32 = get_user32()
    if user32 is None:
        return False
    try:
        ctypes.set_last_error(0)
        ok = bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
    except Exception:
        logger.exception("SetWindowDisplayAffinity raised")
        return False
    if not ok:
        try:
            error = ctypes.get_last_error()
        except Exception:
            error = 0
        logger.error(
            "SetWindowDisplayAffinity(hwnd=%s, affinity=0x%x) failed (error=%s)",
            hwnd, affinity, error,
        )
    return ok


def apply_stealth(hwnd: int) -> bool:
    """Enable the best available display affinity for this Windows build."""
    if not is_windows():
        logger.warning("Stealth mode only supported on Windows")
        return False
    if not hwnd:
        return False

    preferred = WDA_EXCLUDEFROMCAPTURE if is_supported() else WDA_MONITOR
    if _set_affinity(hwnd, preferred):
        logger.info("Stealth mode ON (affinity=0x%x)", preferred)
        return True

    # Some capture drivers reject EXCLUDEFROMCAPTURE even on a new build.
    if preferred != WDA_MONITOR and _set_affinity(hwnd, WDA_MONITOR):
        logger.info("Stealth mode ON (WDA_MONITOR fallback)")
        return True
    return False


def remove_stealth(hwnd: int) -> bool:
    if not is_windows() or not hwnd:
        return False
    return _set_affinity(hwnd, WDA_NONE)


def set_stealth(hwnd: int, enable: bool) -> bool:
    """Toggle helper used by the tray/settings UI."""
    return apply_stealth(hwnd) if enable else remove_stealth(hwnd)
