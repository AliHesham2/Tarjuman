"""System tray icon — Start/Stop, Settings, Exit + status/provider badges."""

from __future__ import annotations

import logging
import os

from src.pipeline.ipc import PipelineStatus

logger = logging.getLogger(__name__)

STATUS_LABELS = {
    PipelineStatus.IDLE: "Stopped",
    PipelineStatus.RECORDING: "Recording",
    PipelineStatus.PAUSED: "Paused",
    PipelineStatus.ERROR: "Error",
}

STATUS_COLORS = {
    PipelineStatus.IDLE: "#8a8f98",
    PipelineStatus.RECORDING: "#2ecc71",
    PipelineStatus.PAUSED: "#f1c40f",
    PipelineStatus.ERROR: "#e74c3c",
}


class TrayController:
    """Owns the QSystemTrayIcon so it can be created after QApplication."""

    def __init__(
        self,
        on_toggle_listening,
        on_open_settings,
        on_exit,
        on_toggle_click_through=None,
        on_toggle_stealth=None,
        click_through: bool = True,
        stealth: bool = True,
        on_select_audio_source=None,
        audio_sources_provider=None,
        current_audio_source=None,
        process_capture_available: bool = True,
    ):
        self.on_toggle_listening = on_toggle_listening
        self.on_open_settings = on_open_settings
        self.on_exit = on_exit
        self.on_toggle_click_through = on_toggle_click_through
        self.on_toggle_stealth = on_toggle_stealth
        self.click_through = click_through
        self.stealth = stealth
        self.on_select_audio_source = on_select_audio_source
        self.audio_sources_provider = audio_sources_provider
        self.current_audio_source = current_audio_source
        self.process_capture_available = bool(process_capture_available)
        self.tray = None
        self.menu = None
        self.audio_menu = None
        self._status = PipelineStatus.IDLE
        self._provider: str | None = None

    def _make_icon(self, color: str):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap

        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(5, 5, 22, 22)
        finally:
            painter.end()
        return QIcon(pixmap)

    def create(self):
        try:
            from PyQt6.QtGui import QAction
            from PyQt6.QtWidgets import QMenu, QSystemTrayIcon
        except ImportError:
            logger.error("PyQt6 not installed — tray unavailable")
            return None

        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray is not available")
            return None

        self.menu = QMenu()
        self.status_action = QAction("Status: Stopped", self.menu)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)

        self.provider_action = QAction("Provider: —", self.menu)
        self.provider_action.setEnabled(False)
        self.menu.addAction(self.provider_action)
        self.menu.addSeparator()

        self.toggle_action = QAction("Start listening", self.menu)
        self.toggle_action.triggered.connect(
            lambda _checked=False: self.on_toggle_listening()
        )
        self.menu.addAction(self.toggle_action)

        settings_action = QAction("Settings", self.menu)
        # QAction.triggered passes a bool. Never forward it to QDialog(parent).
        settings_action.triggered.connect(
            lambda _checked=False: self.on_open_settings()
        )
        self.menu.addAction(settings_action)

        self.audio_menu = self.menu.addMenu("Audio source")
        self.audio_menu.aboutToShow.connect(self.refresh_audio_sources)
        self.refresh_audio_sources()

        self.menu.addSeparator()

        self.click_through_action = QAction("Click-through overlay", self.menu)
        self.click_through_action.setCheckable(True)
        self.click_through_action.setChecked(bool(self.click_through))
        if self.on_toggle_click_through is not None:
            self.click_through_action.toggled.connect(
                self._on_click_through_toggled
            )
        else:
            self.click_through_action.setEnabled(False)
        self.menu.addAction(self.click_through_action)

        self.stealth_action = QAction("Stealth mode", self.menu)
        self.stealth_action.setCheckable(True)
        self.stealth_action.setChecked(bool(self.stealth))
        if self.on_toggle_stealth is not None:
            self.stealth_action.toggled.connect(self._on_stealth_toggled)
        else:
            self.stealth_action.setEnabled(False)
        self.menu.addAction(self.stealth_action)
        self.menu.addSeparator()

        exit_action = QAction("Exit", self.menu)
        exit_action.triggered.connect(lambda _checked=False: self.on_exit())
        self.menu.addAction(exit_action)

        self.tray = QSystemTrayIcon(self._make_icon(STATUS_COLORS[self._status]))
        self.tray.setContextMenu(self.menu)
        self.tray.show()
        self._refresh()
        return self.tray

    def _choose_audio(self, selection) -> None:
        if self.on_select_audio_source is None:
            return
        try:
            ok = self.on_select_audio_source(selection)
        except Exception as exc:
            logger.exception("Audio-source callback failed")
            self.show_message("liveTranslate", f"Could not switch audio source: {exc}")
            return
        if ok is False:
            self.show_message(
                "liveTranslate",
                "Could not switch audio source. The selected app may have stopped; "
                "system audio is still available.",
            )

    def refresh_audio_sources(self) -> None:
        """Rebuild the live submenu. Called on the Qt thread when it opens."""
        if self.audio_menu is None:
            return
        from PyQt6.QtGui import QAction, QActionGroup

        from src.audio.source_manager import AudioSelection

        self.audio_menu.clear()
        group = QActionGroup(self.audio_menu)
        group.setExclusive(True)
        current = self.current_audio_source
        current_mode = getattr(current, "mode", "device") if current else "device"

        def add_choice(text: str, selection, *, enabled: bool = True, tip: str = ""):
            action = QAction(text, self.audio_menu)
            action.setCheckable(True)
            action.setChecked(selection.mode == current_mode)
            action.setEnabled(bool(enabled))
            if tip:
                action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, chosen=selection: self._choose_audio(chosen)
            )
            group.addAction(action)
            self.audio_menu.addAction(action)
            return action

        add_choice(
            "All system audio",
            AudioSelection("device", label="All system audio"),
        )
        add_choice(
            "Auto-detect meeting app",
            AudioSelection("auto", label="Auto-detect meeting app"),
            enabled=bool(self.process_capture_available),
            tip="Propose an active meeting app; fall back to system audio.",
        )

        if not self.process_capture_available:
            unavailable = QAction("Per-app capture unavailable", self.audio_menu)
            unavailable.setEnabled(False)
            unavailable.setToolTip(
                "Needs Windows 10 build 20348+ and the process-audio-capture "
                "package (with its DLL)."
            )
            self.audio_menu.addAction(unavailable)
            return

        self.audio_menu.addSeparator()
        heading = QAction("Listen to one application:", self.audio_menu)
        heading.setEnabled(False)
        self.audio_menu.addAction(heading)
        sessions = []
        if self.audio_sources_provider is not None:
            try:
                sessions = list(self.audio_sources_provider() or [])
            except Exception:
                logger.warning("Could not refresh audio sessions", exc_info=True)
                sessions = []

        seen: set[int] = set()
        current_executable = os.path.basename(
            str(getattr(current, "executable", "") or "").lower()
        )
        for session in sessions:
            if session.pid in seen:
                continue
            seen.add(session.pid)
            selection = AudioSelection.from_session(session)
            # Apps that own an audio session but are silent right now (a paused
            # video) can be chosen too; they are listed after the ones playing.
            caption = session.label
            if not getattr(session, "is_active", True):
                caption += "  (not playing)"
            action = QAction(caption, self.audio_menu)
            action.setCheckable(True)
            action.setChecked(
                current_mode == "session"
                and (
                    getattr(current, "pid", 0) == session.pid
                    or (
                        current_executable
                        and current_executable
                        == os.path.basename(selection.executable.lower())
                    )
                )
            )
            action.setToolTip(session.executable or session.name)
            action.triggered.connect(
                lambda _checked=False, chosen=selection: self._choose_audio(chosen)
            )
            group.addAction(action)
            self.audio_menu.addAction(action)

        # The saved choice is always visible, even when its app is closed.
        listed = {
            os.path.basename(str(s.executable or s.name).lower()) for s in sessions
        }
        if current_mode == "session" and current_executable \
                and current_executable not in listed:
            missing = QAction(
                f"{os.path.basename(str(current.executable))}  (not running)",
                self.audio_menu,
            )
            missing.setCheckable(True)
            missing.setChecked(True)
            missing.setEnabled(False)
            missing.setToolTip("Falls back to all system audio until it starts.")
            group.addAction(missing)
            self.audio_menu.addAction(missing)

        if not sessions:
            empty = QAction(
                "No apps with audio yet - play something, then open this menu again",
                self.audio_menu,
            )
            empty.setEnabled(False)
            self.audio_menu.addAction(empty)

    def update_audio_source(self, selection) -> None:
        self.current_audio_source = selection
        if self.audio_menu is not None:
            self.refresh_audio_sources()

    def _on_click_through_toggled(self, checked: bool) -> None:
        self.click_through = bool(checked)
        if self.on_toggle_click_through is not None:
            self.on_toggle_click_through(bool(checked))

    def _on_stealth_toggled(self, checked: bool) -> None:
        self.stealth = bool(checked)
        if self.on_toggle_stealth is not None:
            self.on_toggle_stealth(bool(checked))

    def _refresh(self) -> None:
        label = STATUS_LABELS.get(self._status, str(self._status.value))
        provider = self._provider or "not selected"

        if self.tray is not None:
            self.tray.setIcon(self._make_icon(STATUS_COLORS[self._status]))
            self.tray.setToolTip(f"liveTranslate — {label} · {provider}")

        if hasattr(self, "status_action"):
            self.status_action.setText(f"Status: {label}")
        if hasattr(self, "provider_action"):
            self.provider_action.setText(f"Provider: {provider}")
        if hasattr(self, "toggle_action"):
            if self._status == PipelineStatus.RECORDING:
                text = "Pause listening"
            elif self._status == PipelineStatus.PAUSED:
                text = "Resume listening"
            elif self._status == PipelineStatus.ERROR:
                text = "Retry listening"
            else:
                text = "Start listening"
            self.toggle_action.setText(text)

    def update_status(self, status: PipelineStatus) -> None:
        self._status = status
        if self.tray is not None or hasattr(self, "status_action"):
            self._refresh()

    def update_provider(self, provider: str | None) -> None:
        self._provider = provider
        if self.tray is not None or hasattr(self, "provider_action"):
            self._refresh()

    def sync_overlay_actions(self, click_through: bool, stealth: bool) -> None:
        """Update check marks without firing the toggled callbacks."""
        for action, value in (
            (getattr(self, "click_through_action", None), click_through),
            (getattr(self, "stealth_action", None), stealth),
        ):
            if action is not None:
                action.blockSignals(True)
                action.setChecked(bool(value))
                action.blockSignals(False)
        self.click_through = bool(click_through)
        self.stealth = bool(stealth)

    def update_stealth(self, enabled: bool) -> None:
        self.sync_overlay_actions(self.click_through, enabled)

    def update_click_through(self, enabled: bool) -> None:
        self.sync_overlay_actions(enabled, self.stealth)

    def show_message(self, title: str, body: str) -> None:
        if self.tray is None:
            return
        try:
            from PyQt6.QtWidgets import QSystemTrayIcon
            icon = QSystemTrayIcon.MessageIcon.Information
        except ImportError:
            icon = 1  # QSystemTrayIcon.MessageIcon.Information
        self.tray.showMessage(title, body, icon, 5000)
