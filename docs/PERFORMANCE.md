# Performance: current numbers

**Measured on 2026-11-22 with `scripts/measure_real_latency.py`.** These are real measurements with the real Whisper model and a real translation provider, but on **one machine, one short clip, one language pair**. Treat them as a first data point, not a benchmark.

Raw results: [before the fix](benchmarks/2026-11-22-rtx3060-laptop-jfk-clip.json) and [after the fix](benchmarks/2026-11-22-rtx3060-laptop-jfk-clip-after-fix.json).

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
| Translation provider actually used | **DeepL only.** All configured Groq keys returned `400 Organization has been restricted` (see "Known problems") |

## Results (after the streaming fix)

| Measurement | Result |
|---|---|
| Model load (cached model) | 8.7 s |
| Transcribe the whole 14 s clip in one go | **0.59 s** (real-time factor 0.042, about 24x faster than real time) |
| One streaming Whisper pass | mean 0.32 s, p95 0.40 s, max 0.49 s |
| **STT lag**: word spoken → word confirmed on screen | median **0.68 s**, p95 1.1 s |
| Translation round trip (DeepL, steady state) | median **0.18 s**, p95 0.54 s |
| **End to end**: end of spoken words → translation available (runs 2 and 3, n=20) | median **0.93 s**, p95 1.4 s, max 1.4 s |
| End to end, all 3 runs (n=30) | median 0.97 s, p95 3.6 s, max 3.6 s |

The all-runs tail is worse only because in run 1 the app first tried the restricted Groq keys, which failed, before falling back to DeepL (two translations took 2.7 and 3.1 s). Once the failed provider is skipped, the numbers are the steady-state row.

Batch timings vary by about 20% between sessions on this laptop (0.49 s earlier the same day, 0.59 s here), so read them as a range.

## What changed in this measurement

The first measurement (same day) found that streaming mode lost words. The cause was found and fixed:

| | Before | After |
|---|---|---|
| Streaming text | *And so my fellow Americans. ASS Not! Ask what you can do for your country.* | *And so my fellow Americans. ASS Not! what your country can do for you. Ask what you can do for your country.* |
| Words different from the one-shot transcription | 36% | **4.5%** (one word: "ASS" instead of "ask") |
| STT lag p95 | 3.0 s | 1.1 s |
| End to end, runs 2 and 3, p95 | 3.35 s | 1.4 s |

**Cause (fixed):** the voice detector can report the end of one utterance and the start of the next inside the same audio chunk. The code then threw away the unconfirmed words of the first utterance and the first audio of the second. The fix finishes the first utterance before starting the second and keeps the audio after the "end" marker. A regression test covers it (`tests/test_stt.py::TestVACBoundaries`).

## Known problems

1. **Isolated short words are misheard.** In this clip the speaker pauses between "ask" and "not". The voice detector treats each as its own utterance, and Whisper transcribes the 0.5 s fragment with no context: "ask" came out as "ASS". Things we tried on this one clip, none shipped:
   - Passing earlier confirmed text to Whisper as a prompt did **not** help ("Not" became "Knock", "ask" stayed "ASS"), so it was not kept.
   - Raising `vad.min_silence_ms` to 1000 gave "ASK NOT!" correctly but delays the end of every utterance, and it also showed other differences ("Watch your country" for "what your country"). 800 did not fix it. A change to the default needs more audio than one clip.
2. **Groq keys were restricted.** Every configured Groq account returned `Error code: 400 - Organization has been restricted`. Groq community staff say this message usually means a possible terms violation or a flagged key, and that support can unlock it. A third-party summary of Groq's Acceptable Use Policy says registering multiple accounts to get around rate limits is not allowed; we could not confirm the exact wording in Groq's own document, so check it yourself. Until we know, the README no longer recommends using several free accounts.
3. **Failed providers cost time.** Before the circuit breaker opens, each failed provider adds latency to the first sentences.

## How these numbers were computed

- *STT lag*: time a confirmed text appeared minus the time its last word finished being spoken. Word end times come from Whisper's own word timestamps on a one-shot transcription of the same audio, matched to confirmed text in order. This is an approximation, not human annotation.
- *End to end*: time the translation came back minus the same word end time.
- The streaming/batch text difference is a word-level edit distance against the one-shot transcription, not against a human transcript. It is not a WER.

## Limits of this data

- One 11-second clip, one speaker, clean audio, one language pair (the Arabic output text was not scored).
- One laptop. No measurement of CPU-only, other GPUs, other models, or a real meeting with overlapping speech and noise.
- Network and provider load vary; the DeepL numbers are from a single session.
- Translation quality was not evaluated.
- The fix was verified on this clip and by unit tests, not on a larger audio set.

## Reproduce it

```powershell
python scripts/measure_real_latency.py --audio path\to\clip.flac --runs 3 --json results.json
```

Needs your `.env` keys for the translation part (`--no-translate` measures speech recognition only). Only the transcribed text is sent to the provider. If you run it on other hardware, please open an issue with the JSON so we can add your numbers here.
