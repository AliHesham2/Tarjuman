"""QuotaAwareRouter — proactive quota tracking + smart provider switching.

Switches providers BEFORE the 429 arrives, not after. State persists in
~/.liveTranslate/quota_state.json (atomic write + SHA256 checksum) and
recovers across app restarts.

Quota units differ by provider: DeepL counts characters per month while
Groq counts requests and tokens (per day and per minute), per account.
``cost_for()``/``unit()`` expose that distinction so the engine does not
decrement a 490K-character budget by one per sentence.

Thread-safety (security audit CRITICAL 2): check-and-increment happens
atomically in ``try_reserve()`` — two threads cannot both take the last
request.  ``has_quota()`` is a read-only check used for partials and the
tray badge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Persistent state only.  Limits always come from config.yaml, so changing
# a limit takes effect on the next start without being shadowed by an old
# value written to disk.
_STATE_KEYS = {"date", "month", "count", "chars", "tokens", "rpm_timestamps"}


class QuotaExhaustedError(RuntimeError):
    """Provider has no quota left right now."""


def _today() -> str:
    return date.today().isoformat()


def _this_month() -> str:
    return datetime.now().strftime("%Y-%m")


def _nonneg_int(value) -> int | None:
    """Parse a persisted counter, rejecting bools/strings/negatives."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    return None


class QuotaAwareRouter:
    """Tracks per-provider usage; tells you who can take the next request."""

    def __init__(
        self,
        quota_limits: dict,
        state_path: Path | None,
        headroom: float = 0.0,
    ):
        self._lock = threading.Lock()
        # Stop using a provider this fraction before its real limit (0.05 =
        # at 95 %), so the next key takes over before a 429 can happen.
        self._headroom = min(max(float(headroom or 0.0), 0.0), 0.5)
        self._state_path = Path(state_path) if state_path else None
        self.usage: dict[str, dict] = {}
        # Transient 429 cooldowns (monotonic deadlines).  Deliberately NOT
        # persisted and NOT part of _STATE_KEYS: one per-minute rate-limit
        # must not disable a provider for the rest of the day.
        self._cooldowns: dict[str, float] = {}

        for name, limits in (quota_limits or {}).items():
            limits = limits or {}
            if "monthly_char_limit" in limits:
                self.usage[name] = {
                    "kind": "monthly", "month": _this_month(),
                    "chars": 0, "limit": int(limits["monthly_char_limit"]),
                }
            else:
                entry = {
                    "kind": "daily", "date": _today(), "count": 0,
                    "limit": int(limits.get("daily_limit", 0)),
                }
                if "daily_token_limit" in limits:
                    entry["token_limit"] = int(limits["daily_token_limit"])
                    entry["tokens"] = 0
                if "rpm_limit" in limits:
                    entry["rpm_limit"] = int(limits["rpm_limit"])
                    entry["rpm_timestamps"] = []
                if "tpm_limit" in limits:
                    # Tokens used in the last 60 s; in memory only.
                    entry["tpm_limit"] = int(limits["tpm_limit"])
                    entry["token_events"] = []
                self.usage[name] = entry

        self._load_state()

    # -- units -----------------------------------------------------------------

    def unit(self, provider: str) -> str:
        """What one unit of ``cost`` means for this provider."""
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return "unlimited"
            return "chars" if entry["kind"] == "monthly" else "requests"

    def cost_for(self, provider: str, text: str) -> int:
        """Cost of translating ``text`` (chars for DeepL, 1 request otherwise)."""
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return 0
            return len(text) if entry["kind"] == "monthly" else 1

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (English/Arabic mixed); used for Groq's cap."""
        return max(1, (len(text) + 3) // 4)

    # -- availability ----------------------------------------------------------

    def has_quota(self, provider: str, cost: int = 1, tokens: int = 0) -> bool:
        """Read-only check: can this provider take a call of this size?"""
        with self._lock:
            if self._cooling_down_locked(provider):
                return False
            entry = self.usage.get(provider)
            if entry is None:
                return True  # untracked (local/paid-unlimited) providers
            self._maybe_reset(entry)
            return self._has_capacity_locked(entry, max(0, cost), max(0, tokens))

    def try_reserve(self, provider: str, cost: int = 1, tokens: int = 0) -> bool:
        """Atomically check capacity and consume it (for final requests)."""
        cost = max(0, cost)
        tokens = max(0, tokens)
        with self._lock:
            if self._cooling_down_locked(provider):
                return False
            entry = self.usage.get(provider)
            if entry is None:
                return True
            self._maybe_reset(entry)
            if not self._has_capacity_locked(entry, cost, tokens):
                return False

            if entry["kind"] == "monthly":
                entry["chars"] += cost
            else:
                entry["count"] += cost
                if "rpm_timestamps" in entry:
                    entry["rpm_timestamps"].append(datetime.now().timestamp())
                if "tokens" in entry:
                    entry["tokens"] += tokens
                self._note_tokens_locked(entry, tokens)
            self._save_state()
            return True

    def record(self, provider: str, cost: int = 1, tokens: int = 0) -> None:
        """Unconditional increment (kept for callers that do their own check)."""
        cost = max(0, cost)
        tokens = max(0, tokens)
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return
            self._maybe_reset(entry)
            if entry["kind"] == "monthly":
                entry["chars"] += cost
            else:
                entry["count"] += cost
                if "rpm_timestamps" in entry:
                    entry["rpm_timestamps"].append(datetime.now().timestamp())
                if "tokens" in entry:
                    entry["tokens"] += tokens
                self._note_tokens_locked(entry, tokens)
            self._save_state()

    def release(self, provider: str, cost: int = 1, tokens: int = 0) -> None:
        """Give a reservation back after a failed call."""
        cost = max(0, cost)
        tokens = max(0, tokens)
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return
            if entry["kind"] == "monthly":
                entry["chars"] = max(0, entry["chars"] - cost)
            else:
                entry["count"] = max(0, entry["count"] - cost)
                if "tokens" in entry:
                    entry["tokens"] = max(0, entry["tokens"] - tokens)
                stamps = entry.get("rpm_timestamps")
                if stamps:
                    stamps.pop()  # this reservation was appended last
                events = entry.get("token_events")
                if events:
                    events.pop()
            self._save_state()

    def add_tokens(self, provider: str, tokens: int) -> None:
        """Account for output tokens after a successful daily request."""
        if tokens <= 0:
            return
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None or "tokens" not in entry:
                return
            entry["tokens"] += tokens
            self._save_state()

    def adjust_tokens(self, provider: str, delta: int) -> None:
        """Correct the token count once the real usage is known (delta may be < 0)."""
        if not delta:
            return
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return
            if "tokens" in entry:
                entry["tokens"] = max(0, entry["tokens"] + delta)
            self._note_tokens_locked(entry, delta)
            self._save_state()

    def usage_fraction(self, provider: str) -> float:
        """How much of its tightest limit the provider has used (0.0 - 1.0+)."""
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return 0.0
            self._maybe_reset(entry)
            ratios = []
            if entry["kind"] == "monthly":
                ratios.append(entry["chars"] / max(1, entry["limit"]))
            else:
                ratios.append(entry["count"] / max(1, entry["limit"]))
                if "token_limit" in entry:
                    ratios.append(entry.get("tokens", 0) / max(1, entry["token_limit"]))
                if "rpm_limit" in entry:
                    stamps = entry.get("rpm_timestamps") or []
                    cutoff = datetime.now().timestamp() - 60.0
                    live = sum(1 for t in stamps if t > cutoff)
                    ratios.append(live / max(1, entry["rpm_limit"]))
                if "tpm_limit" in entry:
                    ratios.append(
                        self._tpm_used(entry) / max(1, entry["tpm_limit"])
                    )
            return max(ratios)

    def remaining(self, provider: str) -> int:
        """How many requests/chars remain (for the settings dialog)."""
        with self._lock:
            entry = self.usage.get(provider)
            if entry is None:
                return -1  # unlimited
            self._maybe_reset(entry)
            used = entry["chars"] if entry["kind"] == "monthly" else entry["count"]
            return max(0, entry["limit"] - used)

    def mark_exhausted_for_session(self, provider: str) -> None:
        """429 arrived despite tracking — treat as exhausted until reset."""
        with self._lock:
            entry = self.usage.get(provider)
            if entry is not None:
                if entry["kind"] == "monthly":
                    entry["chars"] = entry["limit"]
                else:
                    entry["count"] = entry["limit"]
                    if "token_limit" in entry:
                        entry["tokens"] = entry["token_limit"]
                self._save_state()
            logger.warning("Provider %s marked exhausted for this period", provider)

    def cool_down(self, provider: str, seconds: float = 60.0) -> None:
        """Skip ``provider`` for a short while after a 429 (in-memory only).

        Unlike mark_exhausted_for_session() this never touches the persisted
        counters, so a per-minute rate-limit costs one minute, not a day.
        """
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 60.0
        if seconds <= 0:
            return
        with self._lock:
            self._cooldowns[provider] = time.monotonic() + seconds
        logger.warning("Provider %s cooling down for %.0fs after 429", provider, seconds)

    # -- internals -------------------------------------------------------------

    def _cooling_down_locked(self, provider: str) -> bool:
        """True while a 429 cooldown is active (caller holds the lock)."""
        deadline = self._cooldowns.get(provider)
        if deadline is None:
            return False
        if time.monotonic() < deadline:
            return True
        self._cooldowns.pop(provider, None)
        return False

    def _cap(self, limit: int) -> int:
        """A limit reduced by the headroom (never below 1 for a real limit)."""
        if limit <= 0:
            return limit
        return max(1, int(limit * (1.0 - self._headroom)))

    def _has_capacity_locked(self, entry: dict, cost: int, tokens: int) -> bool:
        if entry["kind"] == "monthly":
            limit = self._cap(entry["limit"])
            if entry["chars"] >= limit:
                return False
            return entry["chars"] + cost <= limit

        if not self._rpm_ok(entry):
            return False
        if not self._tpm_ok(entry, tokens):
            return False
        limit = self._cap(entry["limit"])
        if entry["count"] >= limit:
            return False
        if entry["count"] + cost > limit:
            return False
        token_limit = entry.get("token_limit")
        if token_limit is not None:
            token_limit = self._cap(token_limit)
        if token_limit is not None:
            if entry.get("tokens", 0) >= token_limit:
                return False
            if entry.get("tokens", 0) + tokens > token_limit:
                return False
        return True

    @staticmethod
    def _note_tokens_locked(entry: dict, tokens: int) -> None:
        events = entry.get("token_events")
        if events is not None and tokens:
            events.append((datetime.now().timestamp(), int(tokens)))

    @staticmethod
    def _tpm_used(entry: dict) -> int:
        events = entry.get("token_events")
        if not events:
            return 0
        cutoff = datetime.now().timestamp() - 60.0
        events[:] = [e for e in events if e[0] > cutoff]
        return max(0, sum(tokens for _, tokens in events))

    def _tpm_ok(self, entry: dict, tokens: int) -> bool:
        limit = entry.get("tpm_limit")
        if limit is None:
            return True
        used = self._tpm_used(entry)
        cap = self._cap(limit)
        return used < cap and used + max(0, tokens) <= cap

    def _rpm_ok(self, entry: dict) -> bool:
        stamps = entry.get("rpm_timestamps")
        if stamps is None:
            return True
        now = datetime.now().timestamp()
        cutoff = now - 60.0
        stamps[:] = [t for t in stamps if isinstance(t, (int, float)) and t > cutoff]
        return len(stamps) < self._cap(entry.get("rpm_limit", 0))

    # -- reset ------------------------------------------------------------------

    def _maybe_reset(self, entry: dict) -> None:
        if entry["kind"] == "daily" and entry.get("date") != _today():
            logger.info("Daily quota reset for period starting %s", _today())
            entry["date"] = _today()
            entry["count"] = 0
            if "tokens" in entry:
                entry["tokens"] = 0
            if "rpm_timestamps" in entry:
                entry["rpm_timestamps"] = []
        elif entry["kind"] == "monthly" and entry.get("month") != _this_month():
            logger.info("Monthly quota reset for %s", _this_month())
            entry["month"] = _this_month()
            entry["chars"] = 0

    # -- persistence (atomic write + SHA256, security audit HIGH 2) --------------

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.is_file():
            return
        try:
            raw = self._state_path.read_bytes()
            payload, _, checksum = raw.rpartition(b"\n#checksum:")
            if not payload or not checksum:
                raise ValueError("missing checksum")
            if hashlib.sha256(payload).hexdigest().encode() != checksum.strip():
                raise ValueError("checksum mismatch — file corrupted")
            saved = json.loads(payload.decode("utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("state root is not an object")
        except (ValueError, json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error("quota_state.json unreadable (%s) — starting fresh", exc)
            return

        # Restore counters only.  Limits always come from the current config,
        # otherwise raising a limit in config.yaml would appear to do nothing.
        with self._lock:
            for name, saved_entry in saved.items():
                entry = self.usage.get(name)
                if entry is None or not isinstance(saved_entry, dict):
                    continue
                if entry["kind"] == "monthly":
                    chars = _nonneg_int(saved_entry.get("chars"))
                    if chars is not None:
                        entry["chars"] = chars
                    month = saved_entry.get("month")
                    if isinstance(month, str):
                        entry["month"] = month
                else:
                    count = _nonneg_int(saved_entry.get("count"))
                    if count is not None:
                        entry["count"] = count
                    day = saved_entry.get("date")
                    if isinstance(day, str):
                        entry["date"] = day
                    if "tokens" in entry:
                        tokens = _nonneg_int(saved_entry.get("tokens"))
                        if tokens is not None:
                            entry["tokens"] = tokens
                    stamps = saved_entry.get("rpm_timestamps")
                    if isinstance(stamps, list):
                        entry["rpm_timestamps"] = [
                            float(t) for t in stamps
                            if isinstance(t, (int, float)) and not isinstance(t, bool)
                        ]
                self._maybe_reset(entry)
        logger.info("Quota state restored from %s", self._state_path)

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        tmp = None
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    k: {kk: vv for kk, vv in v.items() if kk in _STATE_KEYS}
                    for k, v in self.usage.items()
                },
                ensure_ascii=False,
            ).encode("utf-8")
            checksum = hashlib.sha256(payload).hexdigest().encode()
            fd, tmp = tempfile.mkstemp(dir=str(self._state_path.parent), suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.write(b"\n#checksum:")
                f.write(checksum)
            os.replace(tmp, self._state_path)  # atomic on Windows (same dir)
            tmp = None
        except OSError:
            logger.exception("Failed to persist quota state")
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
