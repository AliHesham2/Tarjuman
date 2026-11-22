# Roadmap

Honest status of what exists and what is still missing on the way to fast, reliable, any-language live translation. Items are ordered by priority. "Verified" means checked by a test or a measurement in this repository.

## Done (verified by the 241-test suite on Windows, Python 3.12)

- [x] System-audio and per-application capture, ring buffer, Silero VAD
- [x] Local Whisper (faster-whisper) with GPU and CPU fallback
- [x] Streaming hypothesis buffer (local agreement)
- [x] Translation chain with quota-aware router, circuit breaker and backoff (Groq → DeepL)
- [x] Always-on-top, click-through overlay with right-to-left Arabic and a bundled font
- [x] Tray, hotkeys, settings dialog, credential storage, log sanitising
- [x] PyInstaller build

## Not done yet, in priority order

### 1. Prove the speed (the core promise)
- [ ] Measure real end-to-end latency (speech → text on screen) with real models and network, and publish the method and numbers
- [ ] Test in a real Zoom / Meet / Teams call and a real 2-hour meeting (not yet done)
- [ ] Reproduce the Whisper timings on other GPUs and on CPU
- [ ] Accuracy evaluation (WER for English, translation quality for Arabic) on a public test set
- [ ] Benchmark against whisper_streaming, WhisperLiveKit and others on the same audio

### 2. Languages (the mission)
- [ ] Choose any source and target language in Settings (today: English → Arabic only)
- [ ] Automatic source-language detection
- [ ] Add right-to-left and non-Latin font handling beyond Arabic
- [ ] Arabic dialect handling (decide scope and test data first)

### 3. Independence from cloud quotas
- [ ] Fully offline translation option (a local model) so the tool works with no internet and no free-tier limits
- [ ] More providers behind the existing provider interface
- [ ] Optional local-only mode for confidential meetings

### 4. Platforms and distribution
- [ ] Prebuilt `.exe` on GitHub Releases, with checksums
- [ ] Hardware-independent path: better CPU performance and AMD / Intel GPU support
- [ ] macOS and Linux audio capture back-ends
- [ ] Optional on-device speech-to-text for lower-end machines

### 5. Project quality
- [ ] Continuous integration (run the test suite on every pull request)
- [ ] Rename internal identifiers (`liveTranslate`, `~/.liveTranslate`) to Tarjuman, **with a settings and key migration** so existing users lose nothing
- [ ] Migrate PyQt6 → PySide6, then consider relicensing to a permissive license
- [ ] Demo GIF and screenshots in the README
- [ ] Localised README files and UI strings
- [ ] Type checking and linting in CI

### 6. Ideas, not committed
- Speaker labels, transcript export, custom vocabulary per meeting, a "second screen" view for phones

Want to help with any item? Open an issue first so we do not duplicate work.
