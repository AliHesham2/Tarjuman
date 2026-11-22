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

## Pause handling: synthetic stress test

The first clip suggested a weakness: a word after a pause was misheard. To test it properly we generated a **synthetic** corpus with the Windows built-in voices (`scripts/make_tts_corpus.py`): 8 sentences x 5 pause patterns = 40 clips with known text, about 470 words per setting. Scored with `scripts/eval_streaming_accuracy.py` against the **true** text. Raw data: [`benchmarks/2026-11-22-synthetic-pause-eval.json`](benchmarks/2026-11-22-synthetic-pause-eval.json).

Word error rate (lower is better), streaming mode, by `vad.min_silence_ms`:

| min_silence_ms | continuous | first word alone (500 ms pause) | mid pauses (800 ms) | word by word (500 ms) | last word alone (1200 ms) | **all** |
|---|---|---|---|---|---|---|
| 500 (current default) | 2.1% | 4.3% | 3.2% | **13.8%** | 3.2% | **5.3%** |
| 800 | 2.1% | 4.3% | 3.2% | 3.2% | 1.1% | **2.8%** |
| 1000 | 2.1% | 4.3% | 3.2% | 3.2% | 3.2% | **3.2%** |
| one-shot transcription (reference) | | | | | | 3.0% |

What it shows:

- With the default 500 ms, a pause of 500 ms or more splits the speech into separate utterances and Whisper transcribes tiny fragments with no context. Worst case ("word by word") had 13.8% errors, including a hallucinated sentence ("...please visit the link in the description").
- At 800 or 1000 ms those pauses no longer split and the error rate matches the one-shot transcription. 800 and 1000 are not distinguishable here (a difference of about 2 words out of 470).
- About 2% of the error floor in every column is number formatting ("ten percent" vs "10%"), not misrecognition.

**Latency cost of 800 ms**, measured in real time on 2 of these clips (3 runs each, speech recognition only): the median lag was unchanged (0.76 s vs 0.73 s on a continuous clip; 0.83 s vs 0.72 s on a clip with pauses). The p95 on the clip with pauses rose from 1.1 s to 1.6 s. Confirmation of the last words does not wait for the silence timer, because a word is confirmed as soon as two passes agree.

Caveats: the voices are synthetic, clean and regular; real speakers pause less evenly and there is noise and overlap. Two voices, one language, 40 clips. No real meeting. **The default has not been changed**: the synthetic corpus favours 800 ms, a real-speech clip disfavours it (next section).

### Counter-evidence: the real-speech clip got worse at 800 ms

The same 800 ms setting on the original real-speech clip (JFK, 3 runs, DeepL; [raw data](benchmarks/2026-11-22-rtx3060-laptop-jfk-clip-min-silence-800.json)) was **worse** than 500 ms:

| Real clip, runs 2 and 3 | 500 ms | 800 ms |
|---|---|---|
| STT lag, median / p95 | 0.68 s / 1.1 s | **1.12 s / 1.33 s** |
| End to end, median / p95 | 0.93 s / 1.4 s | **1.30 s / 1.5 s** |
| Words different from the one-shot transcription | 4.5% | **9.1%** |

At 800 ms the sentence became *"And so my fellow Americans. Not! Watch your country can do for you. ask what you can do for your country."* ("ask" was lost). The clip has only 22 words and every run was identical, so this is one data point, but it is real human speech, and it disagrees with the synthetic corpus. **Because the two disagree, the default was kept at 500 ms.** Deciding needs several real recordings.

## Known problems

1. **Isolated short words after a pause can be misheard** (see the stress test above; in the first clip "ask" came out as "ASS"). Passing earlier confirmed text to Whisper as a prompt was tried and did **not** help ("Not" became "Knock"), so it was not kept. A longer `vad.min_silence_ms` (800) helps on synthetic speech but hurt on the one real clip, so it is undecided.
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
