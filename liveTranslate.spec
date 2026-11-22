# -*- mode: python ; coding: utf-8 -*-
# ============================================================
#  liveTranslate.spec — PyInstaller packaging spec
#  Build: python -m PyInstaller liveTranslate.spec --clean
#  Output: dist/liveTranslate/liveTranslate.exe  (folder mode)
# ============================================================

import sys, os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

project_root = Path(SPECPATH)

# ── Data files ──────────────────────────────────────────────
datas = [
    (str(project_root / "assets" / "fonts"), "assets/fonts"),
    (str(project_root / "assets" / "silero_vad.onnx"), "assets"),
    (str(project_root / "config.yaml"), "."),
]

for pkg in ("ctranslate2", "process_audio_capture"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ── Hidden imports ──────────────────────────────────────────
hiddenimports = [
    "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets", "PyQt6.QtNetwork",
    "src.translation.providers.groq",
    "src.translation.providers.deepl",
    "src.translation.providers.openai_compat",
    "keyring.backends.Windows",
    "keyring.backends.fail",
    "keyring.util",
    "pyaudiowpatch", "process_audio_capture",
    "comtypes", "comtypes.client",
    "pycaw", "pycaw.pycaw",
    "ctranslate2",
    "faster_whisper",
    "src.resilience.circuit_breaker",
    "src.resilience.backoff",
    "dotenv", "yaml", "psutil", "deepl",
]

# ── Exclusions ──────────────────────────────────────────────
# onnxruntime is excluded from PyInstaller binary analysis because importing
# onnxruntime in an isolated subprocess causes an access violation (0xC0000005).
# We inject its files manually into a.datas after Analysis (see below).
excludes = [
    "onnxruntime",            # <-- key fix: skip binary dep scan
    "onnxruntime.capi",
    # The app is cloud-only (Groq + DeepL): none of these SDKs is imported.
    "google", "googleapiclient", "anthropic", "azure", "grpc",
    "torch", "torchgen", "silero_vad",
    "matplotlib", "tkinter", "pydoc", "doctest", "difflib", "test",
    "IPython", "jupyter", "notebook", "tensorboard", "onnx",
]

# ── Pre-Analysis binaries (NO onnxruntime here!) ─────────────
binaries = []

# ctranslate2 CUDA DLLs — safe to add, no crash
try:
    import ctranslate2 as _ct2
    _ct2_dir = Path(_ct2.__file__).parent
    for dll in _ct2_dir.glob("*.dll"):
        binaries.append((str(dll), "ctranslate2"))
except Exception:
    pass

# NVIDIA cuDNN 8 / cuBLAS 12 runtime (pip wheels) so the exe can run Whisper on
# the GPU. Inference-only DLLs: the *_train* ones are ~hundreds of MB and unused.
# src/stt/cuda_paths.py finds them under <bundle>/nvidia/<lib>/bin.
import importlib.util as _ilu
for _pkg, _sub in (("nvidia.cudnn", "cudnn"), ("nvidia.cublas", "cublas")):
    try:
        _spec = _ilu.find_spec(_pkg)
        _bin = Path(list(_spec.submodule_search_locations)[0]) / "bin"
        for _dll in _bin.glob("*.dll"):
            if "_train" in _dll.name or _dll.name.startswith("nvblas"):
                continue
            binaries.append((str(_dll), f"nvidia/{_sub}/bin"))
    except Exception as _e:
        print(f"WARNING: NVIDIA {_sub} DLLs not bundled: {_e}")

# ProcessAudioCapture.dll lives inside the process_audio_capture package; without
# it the exe cannot capture a single application (the tray menu then says
# "Per-app capture unavailable").
try:
    binaries += collect_dynamic_libs("process_audio_capture")
except Exception as _e:
    print(f"WARNING: process_audio_capture DLL not bundled: {_e}")

# ── Analysis ────────────────────────────────────────────────
a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=["hooks/rthook_onnxruntime.py"],
    excludes=excludes,
    noarchive=False,
)

# ── Inject onnxruntime files AFTER Analysis (bypasses binary dep scan crash) ──
# Files added to a.datas are copied as-is with no subprocess import scanning.
try:
    import onnxruntime as _ort
    _ort_dir = Path(_ort.__file__).parent
    _ort_parent = _ort_dir.parent
    _injected = 0
    for _f in _ort_dir.rglob("*"):
        if _f.is_file() and _f.suffix in (".pyd", ".dll", ".json", ".py"):
            _dest = str(_f.relative_to(_ort_parent)).replace("\\", "/")
            a.datas.append((_dest, str(_f), "DATA"))
            _injected += 1
    print(f"INFO: Injected {_injected} onnxruntime files into a.datas")
except Exception as _e:
    print(f"WARNING: Could not inject onnxruntime: {_e}")

pyz = PYZ(a.pure)

# ── Folder output ────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="liveTranslate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime140.dll", "vcruntime140_1.dll", "python3*.dll"],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime140.dll", "vcruntime140_1.dll"],
    name="liveTranslate",
)
