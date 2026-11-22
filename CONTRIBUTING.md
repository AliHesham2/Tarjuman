# Contributing to Tarjuman

Thank you for helping people talk across languages.

## Before you start

1. Search existing issues. For anything bigger than a small fix, open an issue first and say what you plan to do.
2. Check the [Roadmap](ROADMAP.md). Items there are the most useful contributions.

## Set up

Requirements: Windows 10/11, Python 3.12.

```powershell
git clone https://github.com/AliHesham2/Tarjuman.git
cd Tarjuman
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m unittest discover -s tests
```

All tests must pass before you open a pull request. Tests use fakes for speech and network, so they need no keys or GPU.

## Pull requests

- One topic per pull request, with a clear description of what and why.
- Add or update tests for behaviour you change.
- Do not commit secrets, `.env`, model files, build output or personal paths.
- Keep new user-facing claims honest: if you state a number (latency, accuracy), say how it was measured and on what hardware.
- By contributing you agree your work is released under the project's [GPL-3.0 license](LICENSE).

## Good first contributions

- Measure latency on your hardware with `scripts/benchmark_latency.py` and share the results in an issue.
- Report audio sources or apps that fail to capture.
- Improve translation provider adapters in `src/translation/providers/`.
- Documentation fixes and translations.

## Code of conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
