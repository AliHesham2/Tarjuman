"""Score streaming speech recognition against known transcripts.

Runs each clip of a corpus (see make_tts_corpus.py) through the same
``VACOnlineASRProcessor`` the app uses, for one or more ``vad.min_silence_ms``
values, and reports word error rate against the TRUE text. It also scores a
one-shot (non-streaming) transcription as a reference.

    python scripts/eval_streaming_accuracy.py --min-silence 500 800 1000

Audio is fed in 250 ms chunks as fast as the model allows (not paced in real
time), one chunk per ``process_iter`` call. This measures accuracy, not
latency; use measure_real_latency.py for latency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_real_latency import SR, load_audio_16k  # noqa: E402

CHUNK = int(0.25 * SR)


def norm(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def edit_counts(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def stream_text(config, backend, audio: np.ndarray) -> str:
    from src.stt.online_processor import VACOnlineASRProcessor

    vac = VACOnlineASRProcessor(backend, config)
    out: list[str] = []
    for i in range(0, len(audio), CHUNK):
        vac.insert_audio_chunk(audio[i:i + CHUNK])
        confirmed, _partial = vac.process_iter()
        if confirmed:
            out.append(confirmed)
    tail = vac.finish_text()
    if tail:
        out.append(tail)
    return " ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", default=str(ROOT / "benchmarks" / "corpus"))
    parser.add_argument("--min-silence", type=int, nargs="+", default=[500])
    parser.add_argument("--limit", type=int, default=0, help="only the first N clips")
    parser.add_argument("--json", help="write per-clip results here")
    args = parser.parse_args()

    from src.config.settings import load_config
    from src.stt.faster_whisper import FasterWhisperBackend

    corpus = Path(args.corpus)
    clips = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        clips = clips[: args.limit]

    config = load_config(ROOT / "config.yaml", user_path=ROOT / "config.yaml")
    backend = FasterWhisperBackend(config)
    backend.load()

    audios = {}
    for clip in clips:
        a = load_audio_16k(str(corpus / f"{clip['id']}.wav"))
        audios[clip["id"]] = np.concatenate(
            [np.zeros(SR, np.float32), a, np.zeros(2 * SR, np.float32)]
        )
    backend.transcribe(audios[clips[0]["id"]][: 5 * SR])  # warm-up

    results = {"clips": []}
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # setting -> variant -> [err, words]

    for clip in clips:
        ref = norm(clip["text"])
        batch_text = " ".join(
            w["word"] for seg in backend.transcribe(audios[clip["id"]]) for w in seg["words"]
        )
        e = edit_counts(ref, norm(batch_text))
        totals["one-shot"][clip["variant"]][0] += e
        totals["one-shot"][clip["variant"]][1] += len(ref)
        row = {"id": clip["id"], "variant": clip["variant"], "truth": clip["text"],
               "one_shot": batch_text.strip(), "streaming": {}}
        for ms in args.min_silence:
            config.vad.min_silence_ms = ms
            text = stream_text(config, backend, audios[clip["id"]])
            err = edit_counts(ref, norm(text))
            key = f"stream min_silence={ms}"
            totals[key][clip["variant"]][0] += err
            totals[key][clip["variant"]][1] += len(ref)
            row["streaming"][str(ms)] = {"text": text, "errors": err}
        results["clips"].append(row)
        print(f"done {clip['id']}", flush=True)

    variants = list(dict.fromkeys(c["variant"] for c in clips))
    print("\nWord error rate vs the TRUE text (lower is better)")
    header = f"{'setting':28s}" + "".join(f"{v:>22s}" for v in variants) + f"{'ALL':>8s}"
    print(header)
    summary = {}
    for setting, per in totals.items():
        cells, te, tw = [], 0, 0
        for v in variants:
            err, words = per[v]
            te += err
            tw += words
            cells.append(f"{100 * err / words:>21.1f}%")
        summary[setting] = round(100 * te / tw, 2)
        print(f"{setting:28s}" + "".join(cells) + f"{100 * te / tw:>7.1f}%")
    results["summary_wer_percent"] = summary
    results["words_per_variant"] = {v: totals["one-shot"][v][1] for v in variants}
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
