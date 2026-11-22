"""Generate a small SYNTHETIC English test corpus with controlled pauses.

Uses the Windows built-in voices (no downloads). Every clip has a known
transcript, so streaming accuracy can be scored against the truth instead of
against another Whisper run. The pause patterns target one known weakness:
short isolated words after a pause being misheard.

    python scripts/make_tts_corpus.py                 # writes benchmarks/corpus/
    python scripts/eval_streaming_accuracy.py ...     # scores it

Synthetic speech is cleaner and more regular than real speech. Treat the
results as a stress test for pause handling, not as a real-world accuracy
figure. The generated audio is not committed (see .gitignore).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "corpus"

SENTENCES = [
    "Good morning everyone, let's begin the meeting with the quarterly results.",
    "We will release the new version next Tuesday after the final review.",
    "Please send me the report before the end of the day.",
    "The budget for this project has increased by ten percent.",
    "Can you hear me clearly or should I move closer to the microphone?",
    "Our customers in Egypt and Saudi Arabia asked for this feature.",
    "I think we should postpone the launch until the testing is complete.",
    "Thank you all for joining, and see you at the same time next week.",
]

# name -> (split after these 0-based word indices, pause in ms)
VARIANTS = {
    "continuous": ([], 0),
    "first_word_alone": ([0], 500),
    "mid_pauses_800": ([1, 3], 800),
    "word_by_word_500": ([0, 1, 2], 500),
    "last_word_alone_1200": ([-2], 1200),   # split before the last word
}

VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]


def build_ssml(words: list[str], splits: list[int], pause_ms: int) -> str:
    n = len(words)
    cut = {(s if s >= 0 else n + s) for s in splits}
    parts = []
    for i, w in enumerate(words):
        parts.append(escape(w))
        if i in cut and i < n - 1:
            parts.append(f'<break time="{pause_ms}ms"/>')
    body = " ".join(parts)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="en-US">{body}</speak>'
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    k = 0
    for si, sentence in enumerate(SENTENCES):
        words = sentence.split()
        for name, (splits, pause) in VARIANTS.items():
            clips.append({
                "id": f"s{si:02d}_{name}",
                "voice": VOICES[k % len(VOICES)],
                "variant": name,
                "pause_ms": pause,
                "text": sentence,
                "ssml": build_ssml(words, splits, pause),
            })
            k += 1

    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps(clips, indent=2), encoding="utf-8")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(ROOT / "scripts" / "tts_render.ps1"), str(manifest), str(OUT)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
