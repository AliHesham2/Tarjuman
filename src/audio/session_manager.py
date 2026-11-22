"""Audio-session discovery for per-application capture.

Windows exposes every process that is rendering audio as an *audio session*.
``IAudioSessionManager2`` (through :mod:`pycaw`) can enumerate those sessions
and read their PID/state/volume, but it cannot return the session's samples.
Process-loopback capture is implemented separately in
:mod:`src.audio.process_capture`.

This module deliberately has no Qt dependency. The tray calls it on the UI
thread only when its menu is opened, which keeps COM callbacks away from the
audio and STT workers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

MEETING_APP_DEFAULTS = (
    "zoom.exe",
    "ms-teams.exe",
    "teams.exe",
    "webex.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "discord.exe",
    # media players
    "microsoft.media.player.exe",
    "vlc.exe",
    "wmplayer.exe",
    "mpc-hc64.exe",
    "potplayermini64.exe",
)


@dataclass(frozen=True)
class AudioSessionInfo:
    """One process that currently owns a Windows audio session."""

    pid: int
    name: str
    display_name: str = ""
    executable: str = ""
    is_active: bool = True

    @property
    def label(self) -> str:
        if self.display_name and self.display_name.lower() != self.name.lower():
            return f"{self.display_name} ({self.name})"
        return self.name

    @property
    def identity(self) -> str:
        """Stable-ish identity used when persisting a user choice.

        A PID is intentionally not used: apps restart and PIDs are reused.
        """
        return (self.executable or self.name).lower()

    @property
    def executable_name(self) -> str:
        return os.path.basename(self.executable or self.name).lower()


class AudioSessionManager:
    """Discover active Windows audio sessions.

    ``pycaw`` is imported lazily so the rest of the application still starts
    when the optional dependency is missing. In that case ``list_sessions``
    returns an empty list and ``available`` is false.
    """

    def __init__(
        self,
        meeting_apps: Iterable[str] | None = None,
        exclude_pid: int | None = None,
    ):
        self.meeting_apps = tuple(
            app.lower() for app in (meeting_apps or MEETING_APP_DEFAULTS)
        )
        self.exclude_pid = exclude_pid if exclude_pid is not None else os.getpid()

    # -- availability --------------------------------------------------------

    @staticmethod
    def available() -> bool:
        try:
            import pycaw  # noqa: F401
        except Exception:
            return False
        return True

    # -- enumeration ---------------------------------------------------------

    def list_sessions(self, active_only: bool = True) -> list[AudioSessionInfo]:
        """Return one entry per process, newest/canonical session first.

        Multiple audio sessions can belong to one process (for example a
        browser with several renderers). Process-loopback capture sees the
        whole process tree, so presenting duplicate PIDs would be misleading.
        """
        try:
            from pycaw.constants import AudioSessionState
            from pycaw.pycaw import AudioUtilities
        except ImportError:
            logger.debug("pycaw is not installed; audio-session discovery disabled")
            return []
        except Exception:
            logger.debug("pycaw could not be imported", exc_info=True)
            return []

        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception:
            logger.warning("Could not enumerate Windows audio sessions", exc_info=True)
            return []

        by_pid: dict[int, AudioSessionInfo] = {}
        for session in sessions:
            try:
                process = getattr(session, "Process", None)
            except Exception:
                logger.debug("Could not read audio session process", exc_info=True)
                continue
            if process is None:
                # System sounds have no process identity and cannot be
                # selected with process-loopback capture.
                continue

            try:
                pid = int(process.pid)
            except Exception:
                continue
            if pid <= 0 or pid == self.exclude_pid:
                continue

            try:
                name = str(process.name() or f"PID {pid}")
            except Exception:
                name = f"PID {pid}"

            try:
                state = session.State
            except Exception:
                state = None
            try:
                is_active = state == AudioSessionState.Active
            except Exception:
                is_active = bool(state)

            display_name = ""
            try:
                display_name = str(getattr(session, "DisplayName", "") or "")
            except Exception:
                display_name = ""

            executable = self._process_executable(pid, name)
            info = AudioSessionInfo(
                pid=pid,
                name=name,
                display_name=display_name,
                executable=executable,
                is_active=is_active,
            )
            previous = by_pid.get(pid)
            if previous is None or (not previous.is_active and info.is_active):
                by_pid[pid] = info

        result = list(by_pid.values())
        if active_only:
            result = [session for session in result if session.is_active]
        result.sort(key=lambda session: (not session.is_active, session.label.lower()))
        return result

    @staticmethod
    def _process_executable(pid: int, fallback: str) -> str:
        try:
            import psutil

            return str(psutil.Process(pid).exe())
        except Exception:
            return fallback

    # -- selection -----------------------------------------------------------

    def find_by_executable(
        self,
        executable: str,
        sessions: list[AudioSessionInfo] | None = None,
    ) -> AudioSessionInfo | None:
        wanted = os.path.basename(str(executable or "")).lower()
        if not wanted:
            return None
        for session in sessions if sessions is not None else self.list_sessions():
            if session.executable_name == wanted:
                return session
        return None

    def resolve_pid(
        self,
        executable: str,
        sessions: list[AudioSessionInfo] | None = None,
    ) -> int:
        session = self.find_by_executable(executable, sessions)
        return session.pid if session is not None else 0

    def auto_detect_meeting(
        self,
        sessions: list[AudioSessionInfo] | None = None,
    ) -> AudioSessionInfo | None:
        """Return the highest-priority active meeting app, if any.

        This is only a *proposal*. The caller must not silently switch an
        already-running pipeline just because this method finds a new app.
        """
        candidates = sessions if sessions is not None else self.list_sessions()
        priority = {name: index for index, name in enumerate(self.meeting_apps)}

        def rank(session: AudioSessionInfo) -> tuple[int, str]:
            names = {session.executable_name, session.name.lower()}
            order = min(
                (priority.get(name, len(priority)) for name in names),
                default=len(priority),
            )
            return order, session.label.lower()

        matches = [
            session
            for session in candidates
            if session.is_active
            and (
                session.executable_name in priority
                or session.name.lower() in priority
            )
        ]
        if not matches:
            return None
        return min(matches, key=rank)
