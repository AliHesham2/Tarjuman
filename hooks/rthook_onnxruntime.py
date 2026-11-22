# rthook_onnxruntime.py — Runtime hook for onnxruntime DLL path fix
# Injected before main.py runs in the frozen exe.
# Adds the onnxruntime directory to os.add_dll_directory so Windows
# can find onnxruntime.dll / onnxruntime_providers_shared.dll at import time.
import os, sys
from pathlib import Path

_base = Path(sys._MEIPASS if hasattr(sys, '_MEIPASS') else '.')
_ort_dir = _base / "onnxruntime"
if _ort_dir.exists():
    try:
        os.add_dll_directory(str(_ort_dir))
    except Exception:
        pass
_ort_capi = _base / "onnxruntime" / "capi"
if _ort_capi.exists():
    try:
        os.add_dll_directory(str(_ort_capi))
    except Exception:
        pass

# Windows 11 ships its own (older, v1.17) onnxruntime.dll in System32. Resolving
# the DLL by name can pick that one, and the bundled onnxruntime_pybind11_state
# then crashes with an access violation on import. Load ours by full path first:
# later by-name lookups reuse the module that is already in the process.
_ort_handles = []
if sys.platform == "win32":
    import ctypes
    for _name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        _dll = _ort_capi / _name
        if _dll.is_file():
            try:
                _ort_handles.append(ctypes.WinDLL(str(_dll)))
            except OSError:
                pass

