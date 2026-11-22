"""OnlineASRProcessor + VACOnlineASRProcessor.

Ported from references/whisper_streaming/whisper_online.py (MIT).

``OnlineASRProcessor`` owns the sliding window over one utterance:

  * every ``process_iter()`` transcribes the whole window and feeds the
    words to the Local Agreement Policy (``HypothesisBuffer``),
  * only words seen in two consecutive runs are confirmed,
  * once the window exceeds ``buffer_trimming_sec`` it is cut at the end
    of the last completed Whisper segment, so the window stays bounded
    instead of growing quadratically for the whole meeting.

``VACOnlineASRProcessor`` wraps it with our Silero VAD. It accepts raw
audio of any size, detects utterance boundaries itself, and only feeds
speech to the ASR. The pipeline therefore needs a single worker for
VAD + STT instead of two threads passing gated audio around.
"""

from __future__ import annotations

import logging

import numpy as np

from src.audio.vad import VoiceActivityDetector
from src.config.settings import EngineConfig
from src.stt.base import STTBackend
from src.stt.hypothesis_buffer import HypothesisBuffer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Phrases Whisper hallucinates on silence/music. Deliberately excludes
# words that are legitimate speech ("okay", "thank you", "hello", ...):
# dropping those would silently lose real meeting content.
COMMON_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "amara.org",
    "www.amara.org",
}

# Keep at most this many committed words for ``initial_prompt``. Only the
# last ~200 characters are used, so this is plenty and bounds memory.
MAX_PROMPT_WORDS = 400


class OnlineASRProcessor:
    """Sliding-window online transcription over one utterance."""

    def __init__(self, backend: STTBackend, config: EngineConfig):
        self.backend = backend
        self.buffer_trimming_sec = float(config.engine.buffer_trimming_sec)
        self.no_speech_threshold = float(config.filters.no_speech_threshold)
        self.min_avg_logprob = float(
            getattr(config.filters, "min_avg_logprob", -1.0)
        )
        self.same_output_threshold = int(config.filters.same_output_threshold)
        self.anti_loop_ngram = int(config.filters.anti_loop_ngram)

        self._same_output_count = 0
        self._last_output = ""
        self.init()

    # -- lifecycle -----------------------------------------------------------

    def init(self, offset: float = 0.0) -> None:
        """Start a new utterance whose first sample is at ``offset`` seconds."""
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.buffer_time_offset = float(offset)
        self.transcript_buffer = HypothesisBuffer()
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited: list[tuple[float, float, str]] = []
        self._same_output_count = 0
        self._last_output = ""

    def discard(self, offset: float | None = None) -> None:
        """Throw the current utterance away without emitting text (VAD cancel)."""
        if offset is None:
            offset = self.buffer_time_offset + self.buffer_duration
        self.init(offset=offset)

    # -- input ---------------------------------------------------------------

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        if audio is None:
            return
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio])

    @property
    def buffer_duration(self) -> float:
        """Seconds of audio in the current window."""
        return self.audio_buffer.size / float(SAMPLE_RATE)

    # -- inference -----------------------------------------------------------

    def process_iter(self) -> tuple[str, str]:
        """Transcribe the window and return ``(confirmed, partial)``.

        ``confirmed`` is the newly committed text (safe to translate);
        ``partial`` is the still-unconfirmed hypothesis shown in the UI.
        Both are empty strings when there is nothing new.
        """
        if self.audio_buffer.size == 0:
            return "", ""

        segments = self._transcribe()
        words = self._segments_to_words(segments)
        self.transcript_buffer.insert(words, offset=self.buffer_time_offset)
        committed_words = self.transcript_buffer.flush()

        confirmed = self._apply_filters(
            " ".join(w[2] for w in committed_words).strip()
        )
        partial = _trim_repeated_ngrams(
            self.transcript_buffer.unconfirmed_text().strip(),
            self.anti_loop_ngram,
        )

        self.commited.extend(committed_words)
        self._prune_prompt_words()

        if self.buffer_duration > self.buffer_trimming_sec:
            self._trim_completed_segments(segments)
            # Pathological fallback: if Whisper never commits (very noisy
            # audio), segment trimming cannot cut the buffer. Cap it hard
            # instead of letting memory and inference time grow forever.
            if self.buffer_duration > 2.0 * self.buffer_trimming_sec:
                self._hard_trim()

        return confirmed, partial

    def finish(self) -> str:
        """End of utterance: transcribe any remaining audio and flush it all."""
        flushed: list[tuple[float, float, str]] = []
        if self.audio_buffer.size:
            segments = self._transcribe()
            self.transcript_buffer.insert(
                self._segments_to_words(segments), offset=self.buffer_time_offset
            )
            flushed = self.transcript_buffer.flush()

        pending = self.transcript_buffer.complete()
        text = self._apply_filters(
            " ".join(w[2] for w in flushed + pending).strip()
        )

        end = self.buffer_time_offset + self.buffer_duration
        self.init(offset=end)
        return text

    # -- prompt --------------------------------------------------------------

    def _prompt(self) -> str | None:
        """Last 200 chars of committed text that already scrolled out."""
        if not self.commited:
            return None

        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        words = [w[2] for w in self.commited[:k]]
        prompt: list[str] = []
        length = 0
        while words and length < 200:
            word = words.pop()
            length += len(word) + 1
            prompt.append(word)
        if not prompt:
            return None
        return " ".join(reversed(prompt))

    def _prune_prompt_words(self) -> None:
        if len(self.commited) > MAX_PROMPT_WORDS:
            del self.commited[: len(self.commited) - MAX_PROMPT_WORDS]

    # -- transcription helpers ----------------------------------------------

    def _transcribe(self) -> list[dict]:
        return self.backend.transcribe(
            self.audio_buffer, initial_prompt=self._prompt()
        )

    def _segments_to_words(
        self, segments: list[dict]
    ) -> list[tuple[float, float, str]]:
        words: list[tuple[float, float, str]] = []
        for seg in segments:
            if seg.get("no_speech_prob", 0.0) > self.no_speech_threshold:
                logger.debug(
                    "Dropped segment (no_speech_prob=%.2f)",
                    seg.get("no_speech_prob", 0.0),
                )
                continue
            if seg.get("avg_logprob", 0.0) < self.min_avg_logprob:
                logger.debug(
                    "Dropped segment (avg_logprob=%.2f)",
                    seg.get("avg_logprob", 0.0),
                )
                continue

            seg_words = seg.get("words") or []
            if not seg_words:
                # Word timestamps unavailable — fall back to the segment
                # itself so the transcript is not silently emptied.
                text = str(seg.get("text", "")).strip()
                if text:
                    words.append(
                        (float(seg.get("start", 0.0)),
                         float(seg.get("end", 0.0)), text)
                    )
                continue

            for w in seg_words:
                text = str(w.get("word", "")).strip()
                if text:
                    words.append(
                        (float(w["start"]), float(w["end"]), text)
                    )
        return words

    def _apply_filters(self, text: str) -> str:
        if not text:
            return text

        normalized = text.lower().strip().rstrip(".!?")
        if normalized in COMMON_HALLUCINATIONS:
            logger.debug("Dropped common hallucination (%d chars)", len(text))
            return ""

        # Same output N times in a row = decoding loop.
        if text == self._last_output:
            self._same_output_count += 1
            if self._same_output_count >= self.same_output_threshold:
                logger.warning(
                    "Output repeated %d times — dropping as hallucination",
                    self._same_output_count,
                )
                return ""
        else:
            self._same_output_count = 0
            self._last_output = text

        return _trim_repeated_ngrams(text, self.anti_loop_ngram)

    # -- buffer trimming -----------------------------------------------------

    def _trim_completed_segments(self, segments: list[dict]) -> None:
        """Cut the window at a completed segment boundary (reference logic).

        Whisper timestamps are relative to ``audio_buffer``, so the segment
        end is shifted by ``buffer_time_offset`` before comparing with the
        committed words' absolute timeline.
        """
        if not self.commited:
            return

        ends = [
            float(seg.get("end", 0.0)) + self.buffer_time_offset
            for seg in segments
            if seg.get("end") is not None
        ]
        if len(ends) <= 1:
            return

        last_committed_end = self.commited[-1][1]
        cut = ends[-2]
        while len(ends) > 2 and cut > last_committed_end:
            ends.pop(-1)
            cut = ends[-2]

        if cut <= last_committed_end and cut > self.buffer_time_offset:
            self._chunk_at(cut)

    def _chunk_at(self, time: float) -> None:
        """Drop audio and committed words before ``time`` (absolute seconds)."""
        end = self.buffer_time_offset + self.buffer_duration
        time = min(max(time, self.buffer_time_offset), end)
        self.transcript_buffer.pop_commited(time)
        cut = max(0.0, time - self.buffer_time_offset)
        cut_samples = min(int(cut * SAMPLE_RATE), self.audio_buffer.size)
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = time
        logger.debug(
            "Buffer trimmed at %.2fs — %.2fs remain",
            time,
            self.buffer_duration,
        )

    def _hard_trim(self) -> None:
        """Keep only the last ``buffer_trimming_sec`` when nothing commits."""
        keep = int(self.buffer_trimming_sec * SAMPLE_RATE)
        drop = max(0, self.audio_buffer.size - keep)
        if not drop:
            return
        self.audio_buffer = self.audio_buffer[drop:]
        self.buffer_time_offset += drop / float(SAMPLE_RATE)
        # The hypothesis refers to audio that no longer exists; start a new
        # one at the new offset. Prompt words are kept for context.
        self.transcript_buffer.reset(offset=self.buffer_time_offset)
        logger.warning(
            "ASR buffer hard-trimmed to %.1fs (no committed words)",
            self.buffer_duration,
        )


class VACOnlineASRProcessor:
    """VAD-driven utterance segmentation in front of ``OnlineASRProcessor``.

    Usage::

        vac = VACOnlineASRProcessor(backend, config)
        while audio := source.read():
            vac.insert_audio_chunk(audio)
            confirmed, partial = vac.process_iter()
        final = vac.finish_text()      # optional end-of-stream flush

    The class is deliberately independent of the Qt/UI layer, so it can be
    tested with a fake backend and the energy VAD.
    """

    def __init__(
        self,
        backend: STTBackend,
        config: EngineConfig,
        vad: VoiceActivityDetector | None = None,
    ):
        self.online = OnlineASRProcessor(backend, config)
        self.vad = vad or VoiceActivityDetector(config)

        # Transcribe at most this often while speech is flowing. The plan's
        # 250ms chunks → 500ms windows, which keeps partials fresh without
        # re-running Whisper on every 100ms slice.
        chunk_s = max(0.1, 2.0 * config.engine.chunk_duration_ms / 1000.0)
        self.online_chunk_samples = int(chunk_s * SAMPLE_RATE)

        self._audio = np.empty(0, dtype=np.float32)
        # Absolute sample index of self._audio[0] (VAD timestamps are
        # relative to the start of the stream).
        self._buffer_offset = 0
        self._pending_online_samples = 0
        self._status = "nonvoice"
        self._is_final = False

    # -- info ----------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def buffered_seconds(self) -> float:
        return self._audio.size / float(SAMPLE_RATE)

    @property
    def backend_name(self) -> str:
        return getattr(self.vad, "backend_name", "unknown")

    # -- input ---------------------------------------------------------------

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        """Feed raw 16 kHz audio of any length."""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if not audio.size:
            return

        self._audio = np.concatenate([self._audio, audio])

        events = self.vad.process(audio)
        for event in events:
            self._handle_event(event)

        if events:
            return

        if self._status == "voice":
            # Inside an utterance: everything belongs to the current window.
            self._feed_online(self._audio)
            self._clear_audio()
        else:
            # No speech yet: keep 1s so a VAD start can still see the
            # first syllable (VAD adds its own 100ms pre-roll).
            self._keep_last_seconds(1.0)

    # -- output --------------------------------------------------------------

    def process_iter(self) -> tuple[str, str]:
        """Return ``(confirmed, partial)`` if it is time to run the ASR."""
        if self._is_final:
            text = self.finish_text()
            return (text, "")

        if self._pending_online_samples >= self.online_chunk_samples:
            self._pending_online_samples = 0
            return self.online.process_iter()

        return "", ""

    def finish_text(self) -> str:
        """Flush the open utterance (end of stream / pipeline stop)."""
        text = self.online.finish()
        self._pending_online_samples = 0
        self._is_final = False
        self._status = "nonvoice"
        self._clear_audio()
        self.vad.reset()
        # vad.reset() restarts the VAD's absolute sample counter at 0, so the
        # buffer's base index must go back to 0 as well or every later
        # frame = timestamp - _buffer_offset turns negative.
        self._buffer_offset = 0
        return text

    def flush(self) -> str:
        """Alias used by the coordinator when the stream stops."""
        return self.finish_text()

    def finish(self) -> str:
        """Alias matching the whisper_streaming VAC API."""
        return self.finish_text()

    # -- VAD event handling --------------------------------------------------

    def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        timestamp = int(event.get("timestamp", 0))

        if kind == "start":
            frame = max(0, timestamp - self._buffer_offset)
            self.online.init(offset=timestamp / float(SAMPLE_RATE))
            self._pending_online_samples = 0
            self._feed_online(self._audio[frame:])
            self._clear_audio()
            self._status = "voice"
            self._is_final = False

        elif kind == "end":
            frame = timestamp - self._buffer_offset
            frame = min(max(0, frame), self._audio.size)
            self._feed_online(self._audio[:frame])
            self._clear_audio()
            self._status = "nonvoice"
            self._is_final = True

        elif kind == "cancel":
            # Utterance shorter than min_speech_ms — never send it to ASR.
            self._clear_audio()
            self.online.discard(offset=timestamp / float(SAMPLE_RATE))
            self._status = "nonvoice"
            self._is_final = False
            self._pending_online_samples = 0

    # -- audio bookkeeping ---------------------------------------------------

    def _feed_online(self, audio: np.ndarray) -> None:
        if audio.size:
            self.online.insert_audio_chunk(audio)
            self._pending_online_samples += int(audio.size)

    def _clear_audio(self) -> None:
        self._buffer_offset += self._audio.size
        self._audio = np.empty(0, dtype=np.float32)

    def _keep_last_seconds(self, seconds: float) -> None:
        keep = int(seconds * SAMPLE_RATE)
        drop = max(0, self._audio.size - keep)
        if drop:
            self._audio = self._audio[drop:].copy()
            self._buffer_offset += drop


def _trim_repeated_ngrams(text: str, n: int) -> str:
    """Collapse consecutive repeated words/phrases (Whisper decoding loops).

    ``"the the the the"`` → ``"the"``; ``"a b a b a b"`` → ``"a b"``.
    """
    n = max(1, int(n))
    words = text.split()
    if len(words) < 2:
        return text

    out: list[str] = []
    for word in words:
        out.append(word)
        for size in range(1, n + 1):
            if len(out) >= 2 * size and out[-size:] == out[-2 * size:-size]:
                del out[-size:]
                break
    return " ".join(out)
