"""liveTranslate entry point.

Startup flow (plan.md):
  1. Windows Named Mutex — single instance protection
  2. Load .env + config.yaml
  3. Setup safe logging (never logs meeting content or keys)
  4. Build provider chain (auto-skip unconfigured/unavailable)
  5. Start Qt app + overlay + tray, poll UI queue
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading

# Keep this import BEFORE anything that loads PyQt6. PyQt6 ships an old
# MSVCP140.dll (14.26); if Qt loads it first, the later "import onnxruntime"
# (the voice detector) crashes the whole process with an access violation the
# moment listening starts. Loading onnxruntime first picks up the newer system
# runtime, which Qt then shares.
try:
    import onnxruntime  # noqa: F401
except Exception:  # missing/broken: the VAD falls back to its energy detector
    pass

from dotenv import load_dotenv

from src.config.settings import load_config, save_user_settings
from src.pipeline.coordinator import MainPipeline
from src.pipeline.ipc import MessageType, PipelineStatus
from src.security.safe_logger import setup_logging
from src.translation.engine import TranslationEngine
from src.translation.registry import ProviderRegistry
from src.ui import stealth
from src.ui.hotkeys import HOTKEY_ACTIONS, GlobalHotkeyManager
from src.ui.overlay import OverlayWindow
from src.ui.settings_dialog import SettingsDialogController
from src.ui.tray import TrayController

logger = logging.getLogger("livetranslate")


_crash_log = None  # kept open for the life of the process


def enable_crash_log() -> None:
    """Write native crashes (access violations) to ~/.liveTranslate/logs/crash.log.

    A windowed exe has no stderr, so a hard crash in a C extension otherwise
    leaves nothing but an exit code.
    """
    global _crash_log
    try:
        import faulthandler
        from pathlib import Path

        path = Path.home() / ".liveTranslate" / "logs" / "crash.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        _crash_log = open(path, "a", buffering=1)
        faulthandler.enable(_crash_log, all_threads=True)
    except Exception:
        pass


_mutex_handle = None  # kept for the whole run: closing it would free the name


def acquire_single_instance() -> bool:
    """Windows Named Mutex — a second launch must not start a second pipeline."""
    global _mutex_handle
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _mutex_handle = kernel32.CreateMutexW(None, False, "Global\\liveTranslateMutex")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def notify_already_running() -> None:
    """The exe has no console, so say it in a box instead of exiting silently."""
    message = "liveTranslate is already running - look for its icon near the clock."
    print(message)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "liveTranslate", 0x40)
        except Exception:
            logger.debug("Could not show the already-running notice", exc_info=True)


def install_sigint_handler(app) -> bool:
    """Quit the Qt event loop cleanly when the console receives Ctrl+C.

    Qt does not process Python signals on its own; the UI QTimer gives the
    interpreter regular bytecode slots in which this handler can run.
    """
    def _on_sigint(_signum, _frame):
        logger.info("Ctrl+C received")
        app.quit()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
        return True
    except (ValueError, OSError):
        logger.debug("SIGINT handler could not be installed", exc_info=True)
        return False


def open_first_run_settings(config) -> bool:
    """Let a first-run user add a provider key before the app can start.

    The translation pipeline needs at least one provider, but exiting before
    Qt starts traps new users: there is no way to open Settings and add the
    first key. This shows only the Settings dialog, then asks for a restart.
    """
    try:
        from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    except ImportError:
        return False

    try:
        app = QApplication.instance() or QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)
        controller = SettingsDialogController(config, engine=None)
        dialog = controller.show()
        if dialog is None:
            return False
        accepted = dialog.result() == QDialog.DialogCode.Accepted
        if accepted:
            QMessageBox.information(
                None,
                "liveTranslate",
                "Settings saved. Restart liveTranslate to start translating.",
            )
        return accepted
    except Exception:
        logger.exception("First-run settings dialog failed")
        return False


def main() -> int:
    enable_crash_log()
    # 1. single instance
    if not acquire_single_instance():
        notify_already_running()
        return 0

    # 2. config
    load_dotenv()
    config = load_config()

    # 3. logging (never logs translation content or API keys)
    setup_logging(
        level=config.logging.level,
        file_path=config.logging.file,
        max_size_mb=config.logging.max_size_mb,
        backup_count=config.logging.backup_count,
    )
    logger.info("liveTranslate starting...")

    # 4. provider chain
    registry = ProviderRegistry()
    providers, skipped = registry.build_chain(
        config.translation.providers,
        confidential_mode=config.privacy.confidential_mode,
    )
    if not providers:
        logger.warning("No translation providers available — opening first-run settings")
        print("No translation providers available. Opening Settings so you can add "
              "an API key or enable a local provider.")
        if open_first_run_settings(config):
            print("Settings saved. Restart liveTranslate to start translating.")
            return 0
        print("Setup cancelled. A provider is required before liveTranslate can start.")
        return 1

    engine = TranslationEngine(config, providers)
    pipeline = MainPipeline(config, engine)
    logger.info(
        "Per-app audio capture available: %s",
        getattr(pipeline.capture, "process_capture_available", False),
    )

    # 5. Qt UI
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    overlay = OverlayWindow(config)
    overlay.create()
    # A user-dragged overlay switches to "custom"; persist it immediately.
    overlay.on_moved = lambda: save_user_settings(config)

    hotkey_manager = GlobalHotkeyManager()
    settings_dialog = SettingsDialogController(config, engine=engine)

    starting = False
    # Worker threads have no Qt event loop, so QTimer.singleShot(0, ...) from
    # them never fires. They post callables here; the UI timer runs them.
    ui_calls: "queue.Queue" = queue.Queue()

    def _run_async(fn, name):
        threading.Thread(target=fn, name=name, daemon=True).start()

    def on_toggle_listening():
        nonlocal starting
        status = pipeline.status
        if status in (PipelineStatus.IDLE, PipelineStatus.ERROR):
            if starting:
                return
            starting = True

            def _do_start():
                # start/restart joins workers; never do that on the Qt thread.
                nonlocal starting
                try:
                    started = pipeline.start() if status == PipelineStatus.IDLE else pipeline.restart()
                    if not started:
                        # Tray messages must be shown on the Qt thread.
                        ui_calls.put(lambda: tray.show_message(
                            "liveTranslate",
                            "Failed to start: no loopback audio device. "
                            "Install VB-Cable and set it as default output.",
                        ))
                finally:
                    starting = False

            _run_async(_do_start, "pipeline-start")
        else:
            pipeline.toggle()

    def on_toggle_click_through(checked: bool):
        config.overlay.click_through = bool(checked)
        overlay.set_click_through(bool(checked))

    def on_toggle_stealth(checked: bool):
        config.overlay.stealth_mode = bool(checked)
        overlay.set_stealth(bool(checked))
        if checked and not stealth.is_supported():
            tray.show_message(
                "liveTranslate",
                "Stealth mode is not supported on this Windows version.",
            )

    def hotkey_toggle_overlay():
        overlay.toggle_visibility()

    def hotkey_clear_history():
        overlay.clear_text()
        engine.clear_context()

    def hotkey_open_settings():
        settings_dialog.show()

    def apply_hotkeys():
        bindings = {
            attr: getattr(config.hotkeys, attr, "")
            for attr, _label in HOTKEY_ACTIONS
        }
        callbacks = {
            "toggle_overlay": hotkey_toggle_overlay,
            "toggle_listening": on_toggle_listening,
            "clear_history": hotkey_clear_history,
            "open_settings": hotkey_open_settings,
        }
        errors = hotkey_manager.reload(bindings, callbacks)
        if errors:
            tray.show_message(
                "liveTranslate — hotkeys", "\n".join(errors)
            )

    def on_config_changed(cfg: object):
        overlay.apply_settings()
        overlay.set_stealth(config.overlay.stealth_mode)
        overlay.set_click_through(config.overlay.click_through)
        tray.sync_overlay_actions(
            config.overlay.click_through, config.overlay.stealth_mode
        )
        apply_hotkeys()
        if settings_dialog.last_restart_required:
            tray.show_message(
                "liveTranslate",
                "Restart the app to apply provider or STT model changes.",
            )

    settings_dialog.on_config_changed = on_config_changed

    def list_audio_sessions():
        # Every app that owns an audio session, playing or not (a paused video
        # keeps its session), so it can be picked before it makes a sound.
        provider = getattr(pipeline.capture, "list_sessions", None)
        return provider(active_only=False) if provider is not None else []

    def on_select_audio_source(selection):
        ok = pipeline.switch_audio_source(selection)
        if ok:
            config.audio.capture_mode = selection.mode
            config.audio.selected_app = (
                selection.executable if selection.mode == "session" else ""
            )
            save_user_settings(config)
        return ok

    def on_audio_source_changed(selection, reason):
        def apply():
            tray.update_audio_source(selection)
            if reason:
                tray.show_message("liveTranslate", reason)

        # Source callbacks can originate on the capture monitor thread.
        ui_calls.put(apply)

    shutdown_started = False

    def shutdown():
        """Idempotent graceful shutdown for tray, Ctrl+C and window close."""
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        logger.info("Shutting down...")
        try:
            hotkey_manager.stop()
        except Exception:
            logger.exception("Failed to unregister hotkeys")
        try:
            pipeline.stop()
        except Exception:
            logger.exception("Failed to stop pipeline cleanly")
        try:
            engine.shutdown()
        except Exception:
            logger.exception("Failed to shut down translation engine")

    def on_exit():
        logger.info("Exit requested")
        shutdown()
        app.quit()

    # The overlay's close button quits the whole app, exactly like the tray's Exit.
    overlay.on_close = on_exit

    tray = TrayController(
        on_toggle_listening=on_toggle_listening,
        on_open_settings=settings_dialog.show,
        on_exit=on_exit,
        on_toggle_click_through=on_toggle_click_through,
        on_toggle_stealth=on_toggle_stealth,
        click_through=config.overlay.click_through,
        stealth=config.overlay.stealth_mode,
        on_select_audio_source=on_select_audio_source,
        audio_sources_provider=list_audio_sessions,
        current_audio_source=pipeline.audio_source,
        process_capture_available=getattr(
            pipeline.capture, "process_capture_available", False
        ),
    )
    tray.create()
    if hasattr(pipeline.capture, "on_source_changed"):
        pipeline.capture.on_source_changed = on_audio_source_changed
    apply_hotkeys()
    app.aboutToQuit.connect(shutdown)

    # Ctrl+C is still useful when launched from a terminal. Qt's event loop
    # does not process Python signals on its own; the 50 ms UI timer below
    # gives the interpreter regular bytecode slots in which it can run.
    install_sigint_handler(app)

    # UI queue polling — the only bridge from worker threads to the GUI
    def poll_ui_queue():
        while True:
            try:
                call = ui_calls.get_nowait()
            except queue.Empty:
                break
            try:
                call()
            except Exception:
                logger.exception("UI call failed")
        for msg in pipeline.ui_queue.drain(limit=200):
            if msg.type == MessageType.TEXT_UPDATE:
                overlay.update_text(msg.payload["arabic"], is_partial=False)
            elif msg.type == MessageType.PARTIAL_UPDATE:
                overlay.update_text(msg.payload["arabic"], is_partial=True)
            elif msg.type == MessageType.STATUS_UPDATE:
                tray.update_status(PipelineStatus(msg.payload["status"]))
            elif msg.type == MessageType.PROVIDER_UPDATE:
                tray.update_provider(msg.payload.get("provider"))
            elif msg.type == MessageType.ERROR_EVENT:
                tray.show_message("liveTranslate error", msg.payload.get("error", ""))
            elif msg.type == MessageType.INFO_EVENT:
                tray.show_message("liveTranslate", msg.payload.get("text", ""))
            elif msg.type == MessageType.QUOTA_UPDATE:
                # Quota details are shown in Settings; keep the tray responsive
                # even when a future provider starts emitting live counters.
                logger.debug("Quota update: %s", msg.payload)

    timer = QTimer()
    timer.timeout.connect(poll_ui_queue)
    timer.start(50)  # ms

    tray.show_message(
        "liveTranslate",
        f"Ready. Translation chain: {', '.join(p.name for p in providers)}",
    )

    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        logger.info("Ctrl+C received")
        exit_code = 0
    finally:
        shutdown()

    logger.info("liveTranslate exited (code=%d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
