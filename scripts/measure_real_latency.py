"""Measure REAL latency with the real Whisper model and real translation providers.

Unlike ``benchmark_latency.py`` (fake providers, framework overhead only), this
runs the same classes the app uses (``FasterWhisperBackend``,
``VACOnlineASRProcessor``, ``TranslationEngine``) on a real audio file that is
fed in real time, in 250 ms chunks, and reports:

* model load time
* batch transcription time (real-time factor)
* STT lag: how long after a word was spoken it was confirmed
* translation round trip per confirmed text
* end-to-end lag: end of the spoken words -> translation available

Run from the project root (needs your .env keys and, for speed, an NVIDIA GPU):

    python scripts/measure_real_latency.py --audio path/to/clip.flac --runs 3
    python scripts/measure_real_latency.py --audio clip.wav --no-translate

Only the transcribed text of the clip is sent to the translation provider.
Results depend on hardware, network and provider load; always publish them
together with the hardware description printed in the output.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000
CHUNK_S = 0.25


def load_audio_16k(path: str) -> np.ndarray:
    """Decode any audio file to mono float32 at 16 kHz using PyAV."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    chunks: list[np.ndarray] = []
    with av.open(path, mode="r") as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().ravel())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().ravel())
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance / reference length (case and punctuation ignored)."""
    norm = lambda t: re.sub(r"[^\w\s']", " ", t.lower()).split()
    r, h = norm(reference), norm(hypothesis)
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[k]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(pct(values, 50), 3),
        "p95": round(pct(values, 95), 3),
        "max": round(max(values), 3),
    }


def hardware() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version()}
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        info["gpu"] = out or "none"
    except Exception:
        info["gpu"] = "none / nvidia-smi unavailable"
    try:
        import os

        info["cpu_threads"] = os.cpu_count()
    except Exception:
        pass
    return info


def stream_once(config, backend, audio, engine, pool, ref_ends):
    """Feed ``audio`` in real time; return per-confirmation measurements.

    ``ref_ends`` are the end times (seconds) of every word in a batch
    transcription of the same audio. Each confirmed text is matched to the
    next reference words in order, so lag = confirmation time - the time the
    last of those words finished being spoken (approximate; the reference is
    Whisper's own word timing, not human annotation).
    """
    from src.stt.online_processor import VACOnlineASRProcessor

    vac = VACOnlineASRProcessor(backend, config)
    chunk = int(CHUNK_S * SR)
    n_chunks = (len(audio) + chunk - 1) // chunk

    events: list[dict] = []
    futures: list[tuple[dict, object]] = []
    ptr = 0
    fed = 0
    t0 = time.perf_counter()
    iter_times: list[float] = []

    while fed < n_chunks:
        now = time.perf_counter() - t0
        # Chunk k becomes available once its last sample has been "spoken".
        available = min(n_chunks, int(now / CHUNK_S))
        if available <= fed:
            time.sleep(0.01)
            continue
        vac.insert_audio_chunk(audio[fed * chunk: available * chunk])
        fed = available

        t_start = time.perf_counter()
        confirmed, _partial = vac.process_iter()
        t_end = time.perf_counter()
        if t_end - t_start > 0.005:
            iter_times.append(t_end - t_start)

        if confirmed:
            n = len(confirmed.split())
            last_end = ref_ends[min(ptr + n, len(ref_ends)) - 1]
            ptr += n
            event = {
                "text": confirmed,
                "last_word_end_s": round(last_end, 3),
                "confirmed_at_s": round(t_end - t0, 3),
                "stt_lag_s": round((t_end - t0) - last_end, 3),
            }
            events.append(event)
            if engine is not None:
                futures.append((event, pool.submit(_translate, engine, confirmed, t0)))

    tail = vac.finish_text()
    t_final = time.perf_counter()
    if tail:
        n = len(tail.split())
        last_end = ref_ends[min(ptr + n, len(ref_ends)) - 1]
        event = {
            "text": tail,
            "last_word_end_s": round(last_end, 3),
            "confirmed_at_s": round(t_final - t0, 3),
            "stt_lag_s": round((t_final - t0) - last_end, 3),
            "flushed_at_end": True,
        }
        events.append(event)
        if engine is not None:
            futures.append((event, pool.submit(_translate, engine, tail, t0)))

    for event, fut in futures:
        result, done_at, took = fut.result()
        event["translation_s"] = round(took, 3)
        event["provider"] = result.provider if result else None
        event["translated_ok"] = bool(result and result.translated)
        if result and result.translated:
            event["e2e_lag_s"] = round(done_at - event["last_word_end_s"], 3)
    return events, iter_times


def _translate(engine, text, t0):
    start = time.perf_counter()
    result = engine.translate_confirmed(text)
    done = time.perf_counter()
    return result, done - t0, done - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio", required=True, help="audio file (wav/flac/mp3...)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--min-silence", type=int, help="override vad.min_silence_ms")
    parser.add_argument("--json", help="write full results to this file")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from src.config.settings import load_config
    from src.stt.faster_whisper import FasterWhisperBackend

    load_dotenv(ROOT / ".env")
    # Ignore per-user overrides so results are reproducible from config.yaml.
    config = load_config(ROOT / "config.yaml", user_path=ROOT / "config.yaml")
    if args.min_silence:
        config.vad.min_silence_ms = args.min_silence

    audio = load_audio_16k(args.audio)
    # 1 s of silence before and 2 s after, like a real stream.
    audio = np.concatenate([np.zeros(SR, np.float32), audio, np.zeros(2 * SR, np.float32)])
    duration = len(audio) / SR

    backend = FasterWhisperBackend(config)
    t = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - t

    backend.transcribe(audio[: 5 * SR])  # warm-up (kernel init), not measured
    batch = []
    for _ in range(5):
        t = time.perf_counter()
        backend.transcribe(audio)
        batch.append(time.perf_counter() - t)

    ref_segments = backend.transcribe(audio)
    ref_words = [w for seg in ref_segments for w in seg["words"] if w["word"].strip()]
    ref_ends = [w["end"] for w in ref_words]
    ref_text = " ".join(w["word"].strip() for w in ref_words)

    engine = None
    skipped = []
    if not args.no_translate:
        from src.translation.engine import TranslationEngine
        from src.translation.registry import ProviderRegistry

        providers, skipped = ProviderRegistry().build_chain(config.translation.providers)
        if providers:
            engine = TranslationEngine(config, providers)
        else:
            print("No translation provider configured; measuring STT only.")

    pool = ThreadPoolExecutor(max_workers=2)
    all_events: list[dict] = []
    all_iters: list[float] = []
    stream_texts: list[str] = []
    for run in range(args.runs):
        events, iters = stream_once(config, backend, audio, engine, pool, ref_ends)
        for e in events:
            e["run"] = run + 1
        stream_texts.append(" ".join(e["text"] for e in events))
        all_events.extend(events)
        all_iters.extend(iters)

    report = {
        "hardware": hardware(),
        "model": config.engine.whisper_model,
        "vad_min_silence_ms": config.vad.min_silence_ms,
        "device": f"{backend.device}/{backend.compute_type}",
        "audio_seconds": round(duration, 2),
        "runs": args.runs,
        "model_load_s": round(load_s, 2),
        "batch_transcribe_s": summarize(batch),
        "batch_real_time_factor": round(statistics.fmean(batch) / duration, 3),
        "streaming_whisper_pass_s": summarize(all_iters),
        "stt_lag_s": summarize([e["stt_lag_s"] for e in all_events]),
        "translation_s": summarize([e["translation_s"] for e in all_events if "translation_s" in e]),
        "e2e_lag_s": summarize([e["e2e_lag_s"] for e in all_events if "e2e_lag_s" in e]),
        "reference_transcript": ref_text,
        "streaming_transcripts": stream_texts,
        "streaming_word_error_vs_batch": [round(word_error_rate(ref_text, t), 3) for t in stream_texts],
        "confirmations": len(all_events),
        "translation_failures": sum(1 for e in all_events if "translation_s" in e and not e.get("translated_ok")),
        "providers_used": sorted({e["provider"] for e in all_events if e.get("provider")}),
        "skipped_providers": skipped,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": report, "events": all_events}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
