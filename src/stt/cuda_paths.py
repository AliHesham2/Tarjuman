"""Make the pip-installed NVIDIA runtime DLLs visible to CTranslate2.

CTranslate2 4.4 needs cuDNN 8 and cuBLAS 12 at model-load time. The
``nvidia-cudnn-cu12`` / ``nvidia-cublas-cu12`` wheels put them under
``site-packages/nvidia/<lib>/bin``, which Windows does not search by default.
Without this, loading on CUDA fails with "Could not locate
cudnn_ops_infer64_8.dll". A missing wheel is not an error: the caller then
falls back to the CPU as before.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_NVIDIA_PACKAGES = ("nvidia.cudnn", "nvidia.cublas")
_registered: list[str] = []
_handles: list = []  # keep add_dll_directory handles alive for the process


def _candidate_bin_dirs() -> list[Path]:
    """Folders that may hold cuDNN/cuBLAS: the PyInstaller bundle, then pip."""
    dirs: list[Path] = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        for sub in ("cudnn", "cublas"):
            dirs.append(Path(bundle) / "nvidia" / sub / "bin")
    for package in _NVIDIA_PACKAGES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        dirs.append(Path(list(spec.submodule_search_locations)[0]) / "bin")
    return [d for d in dirs if d.is_dir()]


def cuda_runtime_ready() -> bool:
    """True when the DLLs CTranslate2 loads lazily can really be resolved.

    A missing cuDNN/cuBLAS does not raise at model load: it aborts the whole
    process at the first inference. Probe the libraries up front instead.
    """
    if sys.platform != "win32":
        return True
    import ctypes

    register_cuda_dlls()
    for name in ("cublas64_12.dll", "cudnn_ops_infer64_8.dll"):
        try:
            ctypes.WinDLL(name)
        except OSError:
            logger.warning("CUDA runtime library not found: %s", name)
            return False
    return True


def register_cuda_dlls() -> list[str]:
    """Add every nvidia/*/bin folder to the DLL search path (idempotent)."""
    if sys.platform != "win32":
        return []

    for bin_dir in _candidate_bin_dirs():
        folder = str(bin_dir)
        if not bin_dir.is_dir() or folder in _registered:
            continue
        try:
            _handles.append(os.add_dll_directory(folder))
        except (OSError, AttributeError):
            logger.debug("add_dll_directory failed for %s", folder, exc_info=True)
            continue
        # Dependent DLLs (cublasLt, cudnn_*_infer) resolve through PATH too.
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
        _registered.append(folder)
        logger.info("CUDA runtime folder registered: %s", folder)

    return list(_registered)
