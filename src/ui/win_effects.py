"""Small Win32 helpers used by the overlay.

Kept separate from the Qt widgets so the native calls can be mocked in
tests.  Every function is a safe no-op when not running on Windows or when
the requested feature is unavailable.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger(__name__)

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_TOOLWINDOW = 0x00000080

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

VK_MENU = 0x12  # Alt

# SetWindowCompositionAttribute accent states
ACCENT_DISABLED = 0
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4

# DWM_BLURBEHIND flags
DWM_BB_ENABLE = 0x00000001

_user32 = None


class _AccentPolicy(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_int),
        ("AccentFlags", ctypes.c_int),
        ("GradientColor", ctypes.c_uint32),
        ("AnimationId", ctypes.c_int),
    ]


class _WindowCompositionAttributeData(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_uint32),
        ("Data", ctypes.c_void_p),
        ("SizeOfData", ctypes.c_size_t),
    ]


class _DwmBlurBehind(ctypes.Structure):
    _fields_ = [
        ("dwFlags", wintypes.DWORD),
        ("fEnable", wintypes.BOOL),
        ("hRgnBlur", wintypes.HANDLE),
        ("fTransitionOnMaximized", wintypes.BOOL),
    ]


def _configure_user32(user32):
    try:
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.argtypes = [
            wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t
        ]
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    except AttributeError:
        # 32-bit Python only exposes the non-Ptr variants.
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = wintypes.LONG
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        user32.SetWindowLongW.restype = wintypes.LONG

    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL


def get_user32():
    """Return a cached, correctly-typed user32 DLL handle (or None)."""
    global _user32
    if sys.platform != "win32":
        return None
    if _user32 is None:
        try:
            _user32 = ctypes.WinDLL("user32", use_last_error=True)
            _configure_user32(_user32)
        except Exception:
            logger.exception("Could not load user32")
            _user32 = None
    return _user32


def set_click_through(hwnd: int, enable: bool) -> bool:
    """Set/clear WS_EX_TRANSPARENT on the overlay.

    Uses SetWindowLongPtrW rather than SetWindowLongW so the HWND is not
    truncated on 64-bit Python.
    """
    if not hwnd or sys.platform != "win32":
        return False
    user32 = get_user32()
    if user32 is None:
        return False

    get_long = getattr(user32, "GetWindowLongPtrW", None)
    set_long = getattr(user32, "SetWindowLongPtrW", None)
    if get_long is None or set_long is None:
        get_long = user32.GetWindowLongW
        set_long = user32.SetWindowLongW

    try:
        ex_style = int(get_long(hwnd, GWL_EXSTYLE))
        if enable:
            new_style = ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
        else:
            new_style = ex_style & ~WS_EX_TRANSPARENT
        if new_style != ex_style:
            ctypes.set_last_error(0)
            set_long(hwnd, GWL_EXSTYLE, new_style)
            user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
                | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
        return True
    except Exception:
        logger.exception("Could not change click-through style")
        return False


def alt_is_down() -> bool:
    """True while the physical Alt key is held (works when app is unfocused)."""
    user32 = get_user32()
    if user32 is None:
        return False
    try:
        return bool(user32.GetAsyncKeyState(VK_MENU) & 0x8000)
    except Exception:
        return False


def _set_accent(hwnd: int, state: int, tint: int = 0) -> bool:
    user32 = get_user32()
    if user32 is None:
        return False
    func = getattr(user32, "SetWindowCompositionAttribute", None)
    if func is None:
        return False
    try:
        policy = _AccentPolicy(state, 2, tint, 0)
        data = _WindowCompositionAttributeData(
            19,  # WCA_ACCENT_POLICY
            ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p),
            ctypes.sizeof(policy),
        )
        func.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(_WindowCompositionAttributeData),
        ]
        func.restype = wintypes.BOOL
        return bool(func(hwnd, ctypes.byref(data)))
    except Exception:
        logger.debug("SetWindowCompositionAttribute failed", exc_info=True)
        return False


def _apply_dwm_blur(hwnd: int, enable: bool) -> bool:
    if sys.platform != "win32":
        return False
    try:
        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        func = dwmapi.DwmEnableBlurBehindWindow
        func.argtypes = [wintypes.HWND, ctypes.POINTER(_DwmBlurBehind)]
        func.restype = ctypes.c_long
        blur = _DwmBlurBehind(
            dwFlags=DWM_BB_ENABLE if enable else 0,
            fEnable=enable,
            hRgnBlur=None,
            fTransitionOnMaximized=False,
        )
        return func(hwnd, ctypes.byref(blur)) == 0
    except Exception:
        logger.debug("DwmEnableBlurBehindWindow failed", exc_info=True)
        return False


def apply_acrylic(hwnd: int, tint: int = 0xCC14141C) -> bool:
    """Enable Windows acrylic blur, falling back to classic DWM blur."""
    if not hwnd or sys.platform != "win32":
        return False
    try:
        build = int(sys.getwindowsversion().build)
    except Exception:
        build = 0
    if build >= 17763 and _set_accent(hwnd, ACCENT_ENABLE_ACRYLICBLURBEHIND, tint):
        return True
    if _set_accent(hwnd, ACCENT_ENABLE_BLURBEHIND, tint):
        return True
    return _apply_dwm_blur(hwnd, True)


def remove_effects(hwnd: int) -> None:
    """Remove acrylic/blur when the overlay is hidden or reconfigured."""
    if not hwnd or sys.platform != "win32":
        return
    _set_accent(hwnd, ACCENT_DISABLED)
    _apply_dwm_blur(hwnd, False)
