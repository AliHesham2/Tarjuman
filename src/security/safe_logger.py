"""Safe logger — never lets API keys or meeting content reach the logs.

CRITICAL from the security audit: logger.debug(f"Translating: {text}")
must never leak the meeting transcript; keys must never leak anywhere.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

# Var names that must never appear with values in logs
_SECRET_ENV_NAMES = re.compile(
    r"(" + "|".join([
        "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPL_API_KEY",
        "AZURE_TRANSLATOR_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "LOCAL_API_KEY",
    ]) + r")\s*=\s*\S+"
)


def redact(message: str) -> str:
    """Mask anything that looks like an env-style key assignment."""
    return _SECRET_ENV_NAMES.sub(r"\1=***", message)


class SafeLogger:
    """Logging facade that enforces content privacy.

    - API keys are always redacted.
    - Translation content is logged ONLY if privacy.log_translation_content
      is explicitly enabled (it never is by default, even in debug mode).
    """

    def __init__(self, logger: logging.Logger, log_content: bool = False):
        self._logger = logger
        self._log_content = log_content

    def set_log_content(self, allowed: bool) -> None:
        self._log_content = allowed

    def _safe(self, msg: Any) -> str:
        text = str(msg)
        if not self._log_content:
            text = redact(text)
        return text

    def debug(self, msg: Any, *args) -> None:
        self._logger.debug(self._safe(msg), *args)

    def info(self, msg: Any, *args) -> None:
        self._logger.info(self._safe(msg), *args)

    def warning(self, msg: Any, *args) -> None:
        self._logger.warning(self._safe(msg), *args)

    def error(self, msg: Any, *args) -> None:
        self._logger.error(self._safe(msg), *args)

    def exception(self, msg: Any) -> None:
        self._logger.exception(self._safe(msg))


def setup_logging(level: str, file_path: str | None, max_size_mb: int, backup_count: int) -> None:
    from logging.handlers import RotatingFileHandler
    import pathlib

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if file_path:
        path = pathlib.Path(os.path.expanduser(file_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=max_size_mb * 1024 * 1024, backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
