"""EngineConfig dataclass — every tunable parameter lives here.

Single source of truth for runtime settings, loaded from config.yaml.
No magic numbers in the code: if a value is tunable, it belongs here.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

import sys as _sys
if hasattr(_sys, "_MEIPASS"):
    # Frozen PyInstaller exe: data files are extracted to sys._MEIPASS
    PROJECT_ROOT = Path(_sys._MEIPASS)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


@dataclass
class EngineSettings:
    whisper_model: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "en"
    chunk_duration_ms: int = 250
    buffer_trimming_sec: int = 15
    beam_size: int = 1
    condition_on_previous_text: bool = True


@dataclass
class VadSettings:
    threshold: float = 0.5
    min_silence_ms: int = 500
    padding_ms: int = 100
    min_speech_ms: int = 200
    # Used only when the Silero ONNX model is unavailable (RMS fallback).
    fallback_dbfs_threshold: float = -42.0
    fallback_hysteresis_db: float = 6.0


@dataclass
class FilterSettings:
    no_speech_threshold: float = 0.45
    # Whisper avg_logprob is usually -0.2..-0.6; below -1.0 is unreliable.
    min_avg_logprob: float = -1.0
    same_output_threshold: int = 7
    anti_loop_ngram: int = 3


@dataclass
class TranslationSettings:
    source_lang: str = "en"
    target_lang: str = "ar"
    partial_translation: bool = True
    context_window_sentences: int = 5
    context_max_chars: int = 2000          # hard limit (memory-leak guard)
    max_request_chars: int = 500           # input validation cap
    # Switch to the next provider/key this fraction BEFORE a limit is reached
    # (0.05 = at 95 % of the daily/minute/token limit).
    quota_headroom: float = 0.05
    # Draft (partial) translations are only sent to a provider while it has
    # used less than this share of its limits; the rest is kept for the
    # final translations, which must never be starved.
    partial_max_usage: float = 0.6
    providers: list[dict] = field(default_factory=list)


@dataclass
class QuotaSettings:
    limits: dict[str, dict] = field(default_factory=lambda: {
        # Groq, per account (qwen/qwen3.8-27b); sources are listed in config.yaml.
        "groq": {
            "daily_limit": 1000, "daily_token_limit": 200000,
            "rpm_limit": 30, "tpm_limit": 8000,
        },
        "deepl": {"monthly_char_limit": 490000},
    })


@dataclass
class OverlaySettings:
    opacity: float = 0.85
    font_family: str = "Noto Naskh Arabic"
    font_size: int = 20
    position: str = "bottom-center"
    custom_x: int = 100
    custom_y: int = 800
    # Default window height, in text lines (the user can resize it freely).
    max_lines: int = 3
    # Consecutive translations flow into ONE paragraph that wraps at the window
    # edge; a new paragraph starts after the speaker pauses this many seconds.
    paragraph_pause_s: float = 4.0
    # Window size saved after the user resizes it; 0 = automatic.
    width: int = 0
    height: int = 0
    fade_duration_ms: int = 200
    stealth_mode: bool = True
    # Mouse clicks pass through to the meeting app. Hold Alt to drag.
    click_through: bool = True
    text_direction: str = "rtl"


@dataclass
class HotkeySettings:
    toggle_overlay: str = "Ctrl+Shift+T"
    toggle_listening: str = "Ctrl+Shift+L"
    clear_history: str = "Ctrl+Shift+C"
    open_settings: str = "Ctrl+Shift+S"


@dataclass
class PrivacySettings:
    confidential_mode: bool = False
    log_translation_content: bool = False
    state_dir: str = "~/.liveTranslate"


@dataclass
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    exclusive_mode: bool = False
    auto_select_device: bool = True
    device_index: int = -1
    ring_buffer_seconds: float = 30.0      # max audio kept in the ring buffer
    # Phase 8: device = all system audio, session = one app/process tree,
    # auto = propose a meeting app at startup and fall back to device.
    capture_mode: str = "device"
    auto_detect_apps: list[str] = field(default_factory=lambda: [
        "zoom.exe",
        "ms-teams.exe",
        "teams.exe",
        "webex.exe",
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "discord.exe",
        # media players
        "Microsoft.Media.Player.exe",
        "vlc.exe",
        "wmplayer.exe",
        "mpc-hc64.exe",
        "PotPlayerMini64.exe",
    ])
    # Persist the executable name/path, never a PID: PIDs change on restart.
    selected_app: str = ""


@dataclass
class LoggingSettings:
    level: str = "info"
    file: str | None = "~/.liveTranslate/logs/live_translate.log"
    max_size_mb: int = 10
    backup_count: int = 3


@dataclass
class EngineConfig:
    """Top-level config object passed to every component."""
    engine: EngineSettings = field(default_factory=EngineSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    quotas: QuotaSettings = field(default_factory=QuotaSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @property
    def state_dir(self) -> Path:
        return _expand(self.privacy.state_dir)

    @property
    def quota_state_path(self) -> Path:
        return self.state_dir / "quota_state.json"

    @property
    def user_settings_path(self) -> Path:
        """User-level overrides written by the settings dialog.

        Keeping these out of config.yaml preserves the documented comments
        and lets a packaged app update settings without touching bundled files.
        """
        return self.state_dir / "settings.yaml"

    @property
    def chunk_samples(self) -> int:
        """Audio chunk size in samples at 16kHz."""
        return int(self.audio.sample_rate * self.engine.chunk_duration_ms / 1000)

    @property
    def provider_priority(self) -> list[str]:
        return [p["name"] for p in self.translation.providers]


def _update_dataclass(instance, data: dict) -> None:
    """Apply a dict of settings onto a dataclass, ignoring unknown keys."""
    if not data:
        return
    known = set(instance.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown config keys %s in section %s",
                       sorted(unknown), type(instance).__name__)
    for key, value in data.items():
        if key in known:
            setattr(instance, key, value)


def _apply_raw_config(config: EngineConfig, raw: dict) -> None:
    """Apply a parsed YAML mapping onto an EngineConfig."""
    if not isinstance(raw, dict):
        return

    section_map = {
        "engine": config.engine,
        "vad": config.vad,
        "filters": config.filters,
        "translation": config.translation,
        "overlay": config.overlay,
        "hotkeys": config.hotkeys,
        "privacy": config.privacy,
        "audio": config.audio,
        "logging": config.logging,
    }
    for section_name, target in section_map.items():
        if isinstance(raw.get(section_name), dict):
            _update_dataclass(target, raw[section_name])

    # translation.providers and quotas are dicts/lists, not scalars
    t = raw.get("translation", {})
    if isinstance(t, dict) and isinstance(t.get("providers"), list):
        config.translation.providers = [
            entry for entry in t["providers"] if isinstance(entry, dict)
        ]
    quotas = raw.get("quotas")
    if isinstance(quotas, dict):
        # Accept both the on-disk shape (quotas: {provider: ...}) and a
        # nested shape that may be written by older settings versions.
        limits = quotas.get("limits") if isinstance(quotas.get("limits"), dict) else quotas
        config.quotas.limits = limits


# (section, field, minimum, maximum): a value outside the range, or not a
# number at all, would crash the app much later with an unhelpful error, so
# it is replaced by the built-in default and reported in the log.
_NUMBER_LIMITS = [
    ("engine", "chunk_duration_ms", 50, 2000),
    ("engine", "buffer_trimming_sec", 3, 60),
    ("engine", "beam_size", 1, 10),
    ("vad", "threshold", 0.05, 0.95),
    ("vad", "min_silence_ms", 100, 3000),
    ("vad", "padding_ms", 0, 500),
    ("vad", "min_speech_ms", 50, 2000),
    ("filters", "no_speech_threshold", 0.0, 1.0),
    ("filters", "same_output_threshold", 2, 50),
    ("filters", "anti_loop_ngram", 1, 10),
    ("translation", "context_window_sentences", 0, 20),
    ("translation", "context_max_chars", 100, 8000),
    ("translation", "max_request_chars", 50, 500),
    ("translation", "quota_headroom", 0.0, 0.5),
    ("translation", "partial_max_usage", 0.0, 1.0),
    ("overlay", "opacity", 0.1, 1.0),
    ("overlay", "font_size", 8, 96),
    ("overlay", "max_lines", 2, 8),
    ("overlay", "paragraph_pause_s", 0.5, 60.0),
    ("overlay", "fade_duration_ms", 0, 2000),
    ("overlay", "width", 0, 10000),
    ("overlay", "height", 0, 10000),
    ("audio", "sample_rate", 16000, 16000),   # Whisper and the VAD need 16 kHz
    ("audio", "ring_buffer_seconds", 5, 300),
    ("logging", "max_size_mb", 1, 500),
    ("logging", "backup_count", 0, 20),
]
_CHOICES = [
    ("overlay", "position", {"bottom-center", "bottom-left", "bottom-right",
                             "top-center", "top-left", "top-right", "custom"}),
    ("overlay", "text_direction", {"rtl", "ltr"}),
    ("audio", "capture_mode", {"device", "session", "auto"}),
    ("engine", "whisper_device", {"cuda", "cpu", "auto"}),
]


def validate_config(config: "EngineConfig") -> list[str]:
    """Replace unusable values by their defaults; return what was changed."""
    problems: list[str] = []

    def reset(section, name, value, allowed):
        default = getattr(type(section)(), name)
        problems.append(
            f"{type(section).__name__}.{name} = {value!r} is not valid "
            f"({allowed}); using {default!r}"
        )
        setattr(section, name, default)

    for section_name, name, low, high in _NUMBER_LIMITS:
        section = getattr(config, section_name)
        value = getattr(section, name)
        valid = (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and low <= value <= high
        )
        if not valid:
            reset(section, name, value, f"allowed {low}..{high}")
    for section_name, name, choices in _CHOICES:
        section = getattr(config, section_name)
        value = getattr(section, name)
        if value not in choices:
            reset(section, name, value, "one of " + ", ".join(sorted(choices)))
    for message in problems:
        logger.warning("Config: %s", message)
    return problems


def _read_yaml(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Invalid YAML in %s: %s -- ignoring", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("Ignoring %s: expected a YAML mapping", path)
        return None
    return raw


def load_config(
    path: Path | str | None = None,
    user_path: Path | str | None = None,
) -> EngineConfig:
    """Load config.yaml, then optional user overrides.

    ``user_path=None`` uses ``<state_dir>/settings.yaml``. Pass an explicit
    path to test isolated settings, or the same path as ``path`` to disable
    the second load.
    """
    config = EngineConfig()
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = _expand(path)

    if path.is_file():
        raw = _read_yaml(path)
        if raw is not None:
            _apply_raw_config(config, raw)
            logger.info("Config loaded from %s", path)
    else:
        logger.warning("Config file not found at %s -- using built-in defaults", path)

    if user_path is None:
        user_path = config.user_settings_path
    user_path = _expand(user_path)
    if user_path != path and user_path.is_file():
        raw_user = _read_yaml(user_path)
        if raw_user is not None:
            _apply_raw_config(config, raw_user)
            logger.info("User settings loaded from %s", user_path)

    validate_config(config)
    return config


def save_user_settings(
    config: EngineConfig,
    path: Path | str | None = None,
) -> bool:
    """Atomically persist UI-editable settings outside config.yaml.

    API keys are deliberately excluded: those belong in the Windows
    Credential Manager. Returns True on success.
    """
    target = _expand(path) if path is not None else config.user_settings_path
    payload = {
        "engine": {
            "whisper_model": config.engine.whisper_model,
            "whisper_device": config.engine.whisper_device,
        },
        "translation": {
            "providers": [
                dict(entry) for entry in config.translation.providers
                if isinstance(entry, dict)
            ],
        },
        "overlay": asdict(config.overlay),
        "hotkeys": asdict(config.hotkeys),
        "audio": {
            "capture_mode": config.audio.capture_mode,
            "auto_detect_apps": list(config.audio.auto_detect_apps),
            "selected_app": config.audio.selected_app,
        },
        "privacy": {
            "confidential_mode": config.privacy.confidential_mode,
            "log_translation_content": config.privacy.log_translation_content,
        },
    }

    tmp_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".settings-", suffix=".yaml.tmp", dir=target.parent
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
        logger.info("User settings saved to %s", target)
        return True
    except OSError:
        logger.exception("Could not save user settings to %s", target)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False
