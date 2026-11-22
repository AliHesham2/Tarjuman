<div align="center">

# Tarjuman

**Live translated captions for any meeting, any app, on your own PC.**

Tarjuman (ترجمان, "interpreter") listens to the audio your computer is playing, transcribes it locally with Whisper, translates it in near real time, and shows the result in a small always-on-top overlay. It is not tied to Zoom, Meet or Teams: if you can hear it, Tarjuman can caption it.

[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-3776AB.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4.svg)
![Tests](https://img.shields.io/badge/tests-241%20passing-brightgreen.svg)

</div>

> **Status: early but working (v0.1).** Today Tarjuman translates **English → Arabic** on **Windows**. The goal is any language to any language, for everyone. See the [Roadmap](ROADMAP.md) for what is done and what is missing.

## Language plan

1. **Now:** English → Arabic (implemented).
2. **Next:** Arabic → English, so conversations work in both directions (planned, not implemented yet).
3. **Then:** more languages, one at a time, using the same pipeline (planned).

Right now the focus is **speed and quality** of the translation, not the number of languages. New languages come after the core is fast and accurate.

## Why Tarjuman

Most live-translation features are built into one product (Zoom, Google Meet, Teams) and work only inside it. Tarjuman sits outside the meeting app, so it works with **any** source: a conference platform, a webinar, a YouTube lecture, a livestream, a recorded call.

The mission: help people around the world talk to each other without a language barrier.

## How it works

```
system audio ─► VAD ─► Whisper (local, GPU) ─► translation (Groq → DeepL) ─► overlay
```

| Stage | What it does | Code |
|---|---|---|
| Capture | Records all system audio, or a single application (per-app loopback) | `src/audio/` |
| VAD | Silero VAD cuts silence so Whisper only sees speech | `src/audio/vad.py` |
| Speech to text | `faster-whisper` on your GPU, with a hypothesis buffer that only commits words the model agrees on twice (less flicker) | `src/stt/` |
| Translation | A quota-aware router tries Groq accounts in order, then DeepL, with circuit breakers and backoff | `src/translation/` |
| Overlay | Frameless, click-through, always-on-top window with proper right-to-left Arabic and a bundled Arabic font | `src/ui/` |

## Features

- **Works with any audio source**, not one meeting app.
- **Speech recognition runs on your machine.** Audio is never uploaded; only the transcribed text is sent to the translation provider.
- **Quota-aware translation routing.** Add several free Groq accounts; Tarjuman moves to the next key at 95% of a limit, before errors happen, then falls back to DeepL.
- **Per-application capture**, so you can caption one app and ignore the rest.
- **Private by default.** The overlay is excluded from screen capture on Windows (`SetWindowDisplayAffinity`), so your captions do not appear in your own screen share. You can turn this off.
- **Click-through overlay** that never steals focus. Hold Alt to grab it.
- **Safe secrets handling.** Keys come from `.env` or Windows Credential Manager, and logs are sanitised.
- **Tested.** 241 unit and integration tests (run locally, Python 3.12, Windows).

## Quick start

Requirements: Windows 10/11, Python 3.12, and an NVIDIA GPU is strongly recommended (without one Tarjuman falls back to the CPU and a smaller model).

```powershell
git clone https://github.com/AliHesham2/Tarjuman.git
cd Tarjuman
pip install -r requirements.txt
copy .env.example .env      # then add your Groq / DeepL keys
.\start.bat
```

Then right-click the tray icon (near the clock) and choose **Start listening**, or press `Ctrl+Shift+L`.

You can also type keys in **Settings → API Keys** instead of editing `.env`. Test them with `python test_apis.py`.

### Hotkeys

| Hotkey | Action |
|---|---|
| `Ctrl+Shift+L` | Start / pause listening |
| `Ctrl+Shift+T` | Show / hide the window |
| `Ctrl+Shift+C` | Clear the text |
| `Ctrl+Shift+S` | Settings |

### Free API keys

| Provider | Link | Free allowance |
|---|---|---|
| Groq | <https://console.groq.com/keys> | Per account and per model; see `config.yaml` → `quotas.groq` |
| DeepL | <https://www.deepl.com/pro-api> | 500,000 characters / month |

Limits change. The values in `config.yaml` were checked against Groq's response headers when written; re-check them before relying on them.

## Performance

These numbers were measured by the author on one machine and have **not** yet been reproduced elsewhere or measured in a real live meeting.

| Setup | Result |
|---|---|
| Whisper `medium`, `int8_float16`, RTX 3060 laptop GPU | about 0.7 s to transcribe 15 s of audio |
| Same model on CPU | 40 to 60 s for 15 s of audio (use the `base.en` fallback) |

End-to-end latency (speech → Arabic on screen) with real models and a real network is **not yet measured**. It is the first item on the [Roadmap](ROADMAP.md), and we will publish the method and the numbers when we have them.

## Configuration

- `config.yaml`: models, hotkeys, quotas, overlay defaults. Invalid values are replaced by defaults and listed in the log.
- `~/.liveTranslate/settings.yaml`: what Settings and window moves save. Delete it to go back to defaults.
- Logs: `~/.liveTranslate/logs/live_translate.log` (and `crash.log` for hard crashes).

> The `~/.liveTranslate` folder and some internal names still use the old project name. Renaming them would orphan existing settings and stored keys, so it will be done with a migration (see Roadmap).

## Build the .exe

```powershell
python -m PyInstaller liveTranslate.spec --clean --noconfirm
```

Run it from the project folder, not from inside `dist`. Prebuilt binaries are not published yet.

## Tests

```powershell
python -m unittest discover -s tests
```

## Comparison with similar projects

Tarjuman is not the first live Whisper-based tool. Related open-source projects include [whisper_streaming](https://github.com/ufal/whisper_streaming), [WhisperLive](https://github.com/collabora/WhisperLive), [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit), [Speech Translate](https://github.com/Dadangdut33/Speech-Translate) and [live-translation](https://github.com/AbdullahHendy/live-translation). We have not benchmarked Tarjuman against them yet, so we make no claim of being faster. What Tarjuman focuses on: a ready-to-use desktop overlay that works with any audio source, a quota-aware cloud translation chain, and first-class right-to-left rendering.

## Contributing

Contributions are very welcome, especially new languages, macOS/Linux capture back-ends, and latency measurements on other hardware. Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Security issues: see [SECURITY.md](SECURITY.md).

## Privacy

- Speech recognition is local. Audio does not leave your machine.
- The transcribed text is sent to Groq and/or DeepL for translation. Read their privacy policies before using Tarjuman for confidential meetings.
- Do not record or caption people without the consent that your local law and your meeting's rules require.

## License

Tarjuman is released under the [GNU General Public License v3.0](LICENSE). It currently depends on PyQt6, which is GPLv3 or commercially licensed. Moving to a more permissive license is possible after migrating to PySide6 (see Roadmap).

## Acknowledgements

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Silero VAD](https://github.com/snakers4/silero-vad), the Whisper-Streaming "local agreement" idea ([paper](https://arxiv.org/abs/2307.14743)), [Groq](https://groq.com), [DeepL](https://www.deepl.com) and the Noto Naskh Arabic font (SIL OFL).
