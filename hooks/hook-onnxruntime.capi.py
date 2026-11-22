# hook-onnxruntime.capi.py
# Override default hook — manually collect DLLs to avoid isolated subprocess crash (exit 0xC0000005)
from PyInstaller.utils.hooks import collect_data_files
import pathlib, site

binaries = []
datas = []
hiddenimports = []

try:
    import onnxruntime as _ort
    _ort_dir = pathlib.Path(_ort.__file__).parent
    # Collect capi DLLs and .pyd directly (no subprocess)
    for _f in (_ort_dir / 'capi').glob('*.dll'):
        binaries.append((str(_f), 'onnxruntime/capi'))
    for _f in (_ort_dir / 'capi').glob('*.pyd'):
        binaries.append((str(_f), 'onnxruntime/capi'))
    for _f in _ort_dir.glob('*.dll'):
        binaries.append((str(_f), 'onnxruntime'))
    datas += collect_data_files('onnxruntime', includes=['**/*.json', '**/*.onnx'])
except Exception as e:
    print(f'hook-onnxruntime.capi: skipping ({e})')
