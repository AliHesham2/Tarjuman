# Performance: current numbers

**Measured on 2026-11-22 with `scripts/measure_real_latency.py`.** These are real measurements with the real Whisper model and a real translation provider, but on **one machine, one short clip, one language pair**. Treat them as a first data point, not a benchmark. Raw results: [`benchmarks/2026-11-22-rtx3060-laptop-jfk-clip.json`](benchmarks/2026-11-22-rtx3060-laptop-jfk-clip.json).

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop (6 GB), driver 591.74 |
| CPU threads | 16 |
| OS / Python | Windows 11, Python 3.12.10 |
| Whisper | `medium`, `int8_float16` on CUDA, `beam_size=1` (from `config.yaml`) |
| Audio | 14 s: 1 s silence + an 11 s English clip (public-domain JFK speech, 2 sentences) + 2 s silence |
| Feeding | Real time, 250 ms chunks, through the same classes the app uses (`VACOnlineASRProcessor`, `TranslationEngine`) |
| Runs | 3 streaming runs, 5 batch runs |
| Translation provider actually used | **DeepL only.** All configured Groq keys returned `400 Organization has been restricted` during this test (see "Known problems") |

## Results

| Measurement | Result |
|---|---|
| Model load (first start, cached model) | 7.1 s |
| Transcribe the whole 14 s clip in one go | **0.49 s** (real-time factor 0.035, i.e. about 28x faster than real time) |
| One streaming Whisper pass | mean 0.28 s, p95 0.35 s, max 0.50 s |
| **STT lag**: word spoken → word confirmed on screen | median **0.65 s**, p95 3.0 s |
| Translation round trip (DeepL, steady state) | median **0.17 s**, p95 0.35 s |
| **End to end**: end of spoken words → translation available (runs 2 and 3, n=14) | median **0.87 s**, p95 3.35 s, max 3.8 s |
| End to end, all 3 runs (n=21) | median 1.17 s, p95 4.30 s, max 4.43 s |

The all-runs figures are worse because in run 1 the app first tried the restricted Groq keys, which failed, before falling back to DeepL (translations of 1.4 to 3.7 s). The steady-state rows are what you get once the failed provider is skipped.

### What "p95 3.0 s" means

The slowest confirmations are the **last words of a sentence**. The app only confirms text when the same words appear in two consecutive passes or when the speaker pauses, so the end of a sentence waits for the pause. A typical mid-sentence word is confirmed in about 0.5 to 1.0 s.

## Known problems found by this measurement

1. **Streaming transcription lost part of the sentence.** The clip says *"And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."* In streaming mode the app confirmed *"And so my fellow Americans. ASS Not! Ask what you can do for your country."*, which dropped "what your country can do for you" and misheard "ask" as "ASS". Transcribing the same audio in one go is correct. The result was identical in all 3 runs, and the streaming text differed from the batch text by **36% of words**. The cause is not yet found. This is the top item on the [Roadmap](../ROADMAP.md).
2. **Groq keys were restricted.** Every configured Groq account returned `Error code: 400 - Organization has been restricted`. Groq community staff say this message usually means a possible terms violation or a flagged key, and that support can unlock it. A third-party summary of Groq's Acceptable Use Policy says registering multiple accounts to get around rate limits is not allowed; we could not confirm the exact wording in Groq's own document, so check it yourself. Until we know, the README no longer recommends using several free accounts.
3. **Failed providers cost time.** Before the circuit breaker opens, each failed provider adds latency to the first sentences.

## How these numbers were computed

- *STT lag*: time a confirmed text appeared minus the time its last word finished being spoken. Word end times come from Whisper's own word timestamps on a one-shot transcription of the same audio, matched to confirmed text in order. This is an approximation, not human annotation.
- *End to end*: time the translation came back minus the same word end time.
- The streaming/batch text difference is a word-level edit distance against the one-shot transcription, not against a human transcript. It is not a WER.

## Limits of this data

- One 11-second clip, one speaker, clean audio, one language pair (English → Arabic path, but the Arabic output text was not scored).
- One laptop. No measurement of CPU-only, other GPUs, other models, or a real meeting with overlapping speech and noise.
- Network and provider load vary; DeepL numbers here are from a single session.
- Translation quality was not evaluated.

## Reproduce it

```powershell
python scripts/measure_real_latency.py --audio path\to\clip.flac --runs 3 --json results.json
```

Needs your `.env` keys for the translation part (`--no-translate` measures speech recognition only). Only the transcribed text is sent to the provider. If you run it on other hardware, please open an issue with the JSON so we can add your numbers here.
