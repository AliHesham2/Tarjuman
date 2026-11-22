"""Settings dialog — providers, keys, model, overlay, hotkeys, privacy.

API keys entered here go straight to the Windows Credential Store (never
config.yaml, never plain text). If the credential backend is unavailable,
the key is used for this process only and the dialog says so.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping

from src.config.providers import GROQ_EXTRA_ACCOUNTS, KEY_ENV_VARS
from src.config.settings import EngineConfig, save_user_settings
from src.security.credential_store import CredentialStore
from src.ui import stealth
from src.ui.hotkeys import HOTKEY_ACTIONS, validate_hotkeys

logger = logging.getLogger(__name__)

KEY_PROVIDERS = {
    "groq": "Groq API key",
    "deepl": "DeepL API key",
}

PROVIDER_ORDER = ["groq", *GROQ_EXTRA_ACCOUNTS, "deepl"]

PROVIDER_LABELS = {
    "groq": "Groq",
    "deepl": "DeepL",
}

LOCAL_PROVIDERS: set[str] = set()   # cloud-only app: no offline providers
MODEL_SIZES = [
    "tiny.en", "base.en", "small.en", "tiny", "base", "small", "medium", "large-v3",
]
OVERLAY_POSITIONS = [
    ("bottom-center", "Bottom center"),
    ("bottom-left", "Bottom left"),
    ("bottom-right", "Bottom right"),
    ("top-center", "Top center"),
    ("top-left", "Top left"),
    ("top-right", "Top right"),
    ("custom", "Custom coordinates"),
]

DEFAULT_PROVIDER_ENTRIES = {
    "groq": {"name": "groq", "model": "qwen/qwen3.8-27b", "timeout_ms": 1500},
    "deepl": {"name": "deepl", "timeout_ms": 2000},
}


for _name in GROQ_EXTRA_ACCOUNTS:
    _number = _name.rsplit("_", 1)[1]
    KEY_PROVIDERS[_name] = f"Groq API key (account {_number})"
    PROVIDER_LABELS[_name] = f"Groq (account {_number})"
    DEFAULT_PROVIDER_ENTRIES[_name] = dict(
        DEFAULT_PROVIDER_ENTRIES["groq"], name=_name
    )

# Show the key fields in the same order as the provider chain.
KEY_PROVIDERS = {name: KEY_PROVIDERS[name] for name in PROVIDER_ORDER if name in KEY_PROVIDERS}


def provider_display_name(name: str) -> str:
    label = PROVIDER_LABELS.get(name, name)
    if name in LOCAL_PROVIDERS:
        return f"{label}  [local]"
    return label


def default_provider_entry(name: str) -> dict:
    return dict(DEFAULT_PROVIDER_ENTRIES.get(name, {"name": name}))


def normalize_provider_entries(entries: Iterable) -> list[dict]:
    """Deduplicate provider entries, preserving priority order."""
    result: list[dict] = []
    seen: set[str] = set()
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(dict(entry, name=name))
    return result


def build_provider_entries(
    current_entries: Iterable,
    ordered_names: Iterable[str],
    enabled_names: Iterable[str],
) -> list[dict]:
    """Rebuild the config list from priority order + checked providers."""
    existing = {
        entry["name"]: entry for entry in normalize_provider_entries(current_entries)
    }
    enabled = set(enabled_names)
    result: list[dict] = []
    seen: set[str] = set()
    for raw_name in ordered_names:
        name = str(raw_name).strip()
        if not name or name in seen or name not in enabled:
            continue
        seen.add(name)
        entry = dict(existing.get(name) or default_provider_entry(name))
        entry["name"] = name
        result.append(entry)
    return result


class SettingsDialogController:
    """Builds the dialog lazily; keeps Qt imports out of module scope."""

    def __init__(
        self,
        config: EngineConfig,
        on_config_changed=None,
        engine=None,
    ):
        self.config = config
        self.on_config_changed = on_config_changed
        self.engine = engine
        self.store = CredentialStore()
        self._dialog = None
        self._is_open = False
        self._keys_changed = False
        self._status_label = None
        self.last_restart_required = False

    # -- public API ----------------------------------------------------------

    def show(self, parent=None, *args, **kwargs):
        """Show the dialog. Ignores the bool QAction.triggered passes."""
        if isinstance(parent, bool):
            parent = None
        if self._is_open:
            return self._dialog

        dlg = self._build_dialog(parent)
        if dlg is None:
            return None
        self._is_open = True
        try:
            dlg.exec()
        finally:
            self._is_open = False
        return dlg

    def quota_summary(self, engine=None) -> str:
        engine = engine or self.engine
        if engine is None:
            return "Quota status is available after the engine starts."
        providers = list(getattr(engine, "providers", []) or [])
        if not providers:
            return "No providers in the active chain."

        lines: list[str] = []
        for provider in providers:
            name = getattr(provider, "name", str(provider))
            try:
                remaining = engine.router.remaining(name)
            except Exception:
                remaining = -1
            limits = self.config.quotas.limits.get(name, {}) or {}
            if "monthly_char_limit" in limits:
                unit = "characters"
            elif "daily_token_limit" in limits:
                unit = "requests (token cap tracked)"
            else:
                unit = "requests"
            if remaining < 0:
                lines.append(f"{name}: unlimited")
            elif remaining == 0:
                lines.append(f"{name}: exhausted ({unit})")
            else:
                lines.append(f"{name}: {remaining} {unit} remaining")
        return "\n".join(lines)

    # -- dialog construction ------------------------------------------------

    def _build_dialog(self, parent=None):
        try:
            from PyQt6.QtCore import Qt
            from PyQt6.QtGui import QFont, QFontDatabase, QKeySequence
            from PyQt6.QtWidgets import (
                QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                QDoubleSpinBox, QFontComboBox, QFormLayout, QHBoxLayout,
                QKeySequenceEdit, QLabel, QLineEdit, QListWidget,
                QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
                QTabWidget, QVBoxLayout, QWidget,
            )
            from src.ui.overlay import _load_bundled_fonts
        except ImportError:
            logger.error("PyQt6 not installed — settings dialog unavailable")
            return None

        # First-run Settings can be opened before the overlay exists, so make
        # sure the bundled Arabic font is visible to QFontComboBox too.
        _load_bundled_fonts(QFontDatabase)

        dlg = QDialog(parent)
        dlg.setWindowTitle("liveTranslate — Settings")
        dlg.setMinimumSize(640, 620)
        root = QVBoxLayout(dlg)
        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        # -- Providers / priority -------------------------------------------
        provider_tab = QWidget()
        provider_layout = QVBoxLayout(provider_tab)
        provider_layout.addWidget(QLabel(
            "Priority: the top checked provider is tried first. "
            "Unchecked providers are disabled."
        ))
        provider_list = QListWidget()
        provider_list.setMinimumHeight(240)
        self._provider_list = provider_list

        current_entries = normalize_provider_entries(
            self.config.translation.providers
        )
        seen_names: set[str] = set()

        def add_provider_item(name: str, enabled: bool) -> None:
            item = QListWidgetItem(provider_display_name(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
            )
            item.setToolTip(
                "Built-in local provider — no API key required"
                if name in LOCAL_PROVIDERS
                else "Requires a key in the API Keys tab or .env"
            )
            # Extra accounts sit right under their base provider (rotation order).
            base = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else ""
            position = provider_list.count()
            if base:
                rows = [
                    row for row in range(provider_list.count())
                    if str(provider_list.item(row).data(Qt.ItemDataRole.UserRole))
                    .split("_")[0] == base
                ]
                if rows:
                    position = rows[-1] + 1
            provider_list.insertItem(position, item)

        for entry in current_entries:
            seen_names.add(entry["name"])
            add_provider_item(entry["name"], True)
        for name in PROVIDER_ORDER:
            if name not in seen_names:
                add_provider_item(name, False)
        provider_layout.addWidget(provider_list, 1)

        move_row = QHBoxLayout()
        up_button = QPushButton("Move up")
        down_button = QPushButton("Move down")

        def move_selected(delta: int) -> None:
            row = provider_list.currentRow()
            target = row + delta
            if row < 0 or target < 0 or target >= provider_list.count():
                return
            item = provider_list.takeItem(row)
            provider_list.insertItem(target, item)
            provider_list.setCurrentRow(target)

        up_button.clicked.connect(lambda: move_selected(-1))
        down_button.clicked.connect(lambda: move_selected(1))
        move_row.addWidget(up_button)
        move_row.addWidget(down_button)
        move_row.addStretch(1)
        provider_layout.addLayout(move_row)
        tabs.addTab(provider_tab, "Providers")

        # -- API keys ---------------------------------------------------------
        keys_tab = QWidget()
        keys_form = QFormLayout(keys_tab)
        self._key_edits: dict[str, QLineEdit] = {}
        for name, label in KEY_PROVIDERS.items():
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            edit.setClearButtonEnabled(True)
            try:
                existing = self.store.get_key(name)
            except Exception:
                existing = None
            if existing:
                edit.setPlaceholderText("saved in Windows Credential Manager")
            else:
                edit.setPlaceholderText("paste key")
            save = QPushButton("Save")
            save.clicked.connect(
                lambda _checked=False, n=name, e=edit: self._save_key(n, e)
            )
            row.addWidget(edit, 1)
            row.addWidget(save)
            keys_form.addRow(label, row)
            self._key_edits[name] = edit
        keys_note = QLabel(
            "Keys are stored in the Windows Credential Manager. "
            "They are never written to config.yaml or settings.yaml."
        )
        keys_note.setWordWrap(True)
        keys_form.addRow(keys_note)
        tabs.addTab(keys_tab, "API Keys")
        if not current_entries:
            # First run: there is nothing useful on the priority tab yet.
            # Start where the user can add the first provider key.
            tabs.setCurrentWidget(keys_tab)

        # -- STT model --------------------------------------------------------
        speech_tab = QWidget()
        speech_form = QFormLayout(speech_tab)
        model_combo = QComboBox()
        model_combo.addItems(MODEL_SIZES)
        current_model = self.config.engine.whisper_model
        if current_model not in MODEL_SIZES:
            model_combo.addItem(current_model)
        model_combo.setCurrentText(current_model)
        speech_form.addRow("Whisper model", model_combo)

        device_combo = QComboBox()
        device_combo.addItems(["cuda", "cpu"])
        if self.config.engine.whisper_device not in ("cuda", "cpu"):
            device_combo.addItem(self.config.engine.whisper_device)
        device_combo.setCurrentText(self.config.engine.whisper_device)
        speech_form.addRow("Device", device_combo)
        speech_form.addRow(QLabel(
            "Model/device changes take effect after restart."
        ))
        tabs.addTab(speech_tab, "Speech")

        # -- Overlay ----------------------------------------------------------
        overlay_tab = QWidget()
        overlay_form = QFormLayout(overlay_tab)
        opacity = QDoubleSpinBox()
        opacity.setRange(0.1, 1.0)
        opacity.setSingleStep(0.05)
        opacity.setDecimals(2)
        opacity.setValue(self.config.overlay.opacity)
        overlay_form.addRow("Opacity", opacity)

        font_combo = QFontComboBox()
        font_combo.setCurrentFont(QFont(self.config.overlay.font_family))
        overlay_form.addRow("Arabic font", font_combo)

        font_size = QSpinBox()
        font_size.setRange(10, 64)
        font_size.setValue(self.config.overlay.font_size)
        overlay_form.addRow("Font size (pt)", font_size)

        position = QComboBox()
        for value, label in OVERLAY_POSITIONS:
            position.addItem(label, value)
        index = position.findData(self.config.overlay.position)
        position.setCurrentIndex(index if index >= 0 else 0)
        overlay_form.addRow("Position", position)

        custom_row = QWidget()
        custom_form = QFormLayout(custom_row)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_x = QSpinBox()
        custom_x.setRange(-10000, 10000)
        custom_x.setValue(self.config.overlay.custom_x)
        custom_y = QSpinBox()
        custom_y.setRange(-10000, 10000)
        custom_y.setValue(self.config.overlay.custom_y)
        custom_form.addRow("Custom X", custom_x)
        custom_form.addRow("Custom Y", custom_y)
        overlay_form.addRow("Custom position", custom_row)

        max_lines = QSpinBox()
        max_lines.setRange(2, 8)
        max_lines.setValue(max(2, self.config.overlay.max_lines))
        overlay_form.addRow("Visible lines (default height)", max_lines)

        fade = QSpinBox()
        fade.setRange(0, 1000)
        fade.setSingleStep(25)
        fade.setSuffix(" ms")
        fade.setValue(self.config.overlay.fade_duration_ms)
        overlay_form.addRow("Fade-in duration", fade)

        stealth_supported = stealth.is_supported()
        stealth_check = QCheckBox(
            "Stealth mode (hidden from screen share)"
            + ("" if stealth_supported else " — Windows 10 2004+ only")
        )
        stealth_check.setChecked(
            bool(self.config.overlay.stealth_mode and stealth_supported)
        )
        stealth_check.setEnabled(stealth_supported)
        overlay_form.addRow("", stealth_check)

        click_through = QCheckBox(
            "Click-through (hold Alt to drag the overlay)"
        )
        click_through.setChecked(self.config.overlay.click_through)
        overlay_form.addRow("", click_through)

        direction = QComboBox()
        direction.addItem("Right to left (Arabic)", "rtl")
        direction.addItem("Left to right", "ltr")
        direction.setCurrentIndex(0 if self.config.overlay.text_direction == "rtl" else 1)
        overlay_form.addRow("Text direction", direction)

        def sync_custom_visibility() -> None:
            custom_row.setVisible(position.currentData() == "custom")

        position.currentIndexChanged.connect(lambda _index: sync_custom_visibility())
        sync_custom_visibility()
        tabs.addTab(overlay_tab, "Overlay")

        # -- Hotkeys ----------------------------------------------------------
        hotkey_tab = QWidget()
        hotkey_form = QFormLayout(hotkey_tab)
        self._hotkey_edits: dict[str, QKeySequenceEdit] = {}
        for attr, label in HOTKEY_ACTIONS:
            edit = QKeySequenceEdit()
            edit.setKeySequence(QKeySequence(getattr(self.config.hotkeys, attr, "")))
            hotkey_form.addRow(label, edit)
            self._hotkey_edits[attr] = edit

        reset_hotkeys = QPushButton("Reset hotkeys to defaults")
        hotkey_form.addRow("", reset_hotkeys)

        def reset_hotkey_edits() -> None:
            from src.config.settings import HotkeySettings
            defaults = HotkeySettings()
            for attr, edit in self._hotkey_edits.items():
                edit.setKeySequence(QKeySequence(getattr(defaults, attr)))

        reset_hotkeys.clicked.connect(lambda: reset_hotkey_edits())
        hotkey_form.addRow(QLabel(
            "Use Ctrl, Alt, Shift or Win plus a key. Clear a field to disable it."
        ))
        tabs.addTab(hotkey_tab, "Hotkeys")

        # -- Privacy & quotas -------------------------------------------------
        privacy_tab = QWidget()
        privacy_form = QFormLayout(privacy_tab)
        log_content = QCheckBox("Log translation content (off by default)")
        log_content.setChecked(self.config.privacy.log_translation_content)
        privacy_form.addRow("", log_content)

        quota_label = QLabel(self.quota_summary())
        quota_label.setWordWrap(True)
        quota_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        refresh_quota = QPushButton("Refresh quota status")
        refresh_quota.clicked.connect(
            lambda: quota_label.setText(self.quota_summary())
        )
        privacy_form.addRow("Quotas", quota_label)
        privacy_form.addRow("", refresh_quota)
        tabs.addTab(privacy_tab, "Privacy && Quotas")  # && = a literal ampersand

        # -- status + buttons -------------------------------------------------
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(dlg.reject)
        root.addWidget(buttons)

        def on_accept() -> None:
            hotkey_values = {
                attr: edit.keySequence().toString(
                    QKeySequence.SequenceFormat.PortableText
                ).strip()
                for attr, edit in self._hotkey_edits.items()
            }
            errors = validate_hotkeys(hotkey_values)
            if errors:
                QMessageBox.warning(
                    dlg, "Invalid hotkeys", "\n".join(errors)
                )
                return

            ordered_names: list[str] = []
            enabled_names: set[str] = set()
            for row in range(provider_list.count()):
                item = provider_list.item(row)
                name = str(item.data(Qt.ItemDataRole.UserRole))
                ordered_names.append(name)
                if item.checkState() == Qt.CheckState.Checked:
                    enabled_names.add(name)

            new_entries = build_provider_entries(
                self.config.translation.providers, ordered_names, enabled_names
            )
            providers_changed = (
                new_entries != self.config.translation.providers
            )
            old_model = self.config.engine.whisper_model
            old_device = self.config.engine.whisper_device
            old_log_content = self.config.privacy.log_translation_content

            self.config.translation.providers = new_entries
            self.config.engine.whisper_model = model_combo.currentText().strip()
            self.config.engine.whisper_device = device_combo.currentText().strip()

            self.config.overlay.opacity = opacity.value()
            self.config.overlay.font_family = font_combo.currentFont().family()
            self.config.overlay.font_size = font_size.value()
            self.config.overlay.position = str(position.currentData())
            self.config.overlay.custom_x = custom_x.value()
            self.config.overlay.custom_y = custom_y.value()
            self.config.overlay.max_lines = max_lines.value()
            self.config.overlay.fade_duration_ms = fade.value()
            self.config.overlay.stealth_mode = stealth_check.isChecked()
            self.config.overlay.click_through = click_through.isChecked()
            self.config.overlay.text_direction = str(direction.currentData())

            self.config.privacy.log_translation_content = log_content.isChecked()

            for attr, sequence in hotkey_values.items():
                setattr(self.config.hotkeys, attr, sequence)

            self.last_restart_required = (
                providers_changed
                or old_model != self.config.engine.whisper_model
                or old_device != self.config.engine.whisper_device
                or old_log_content != self.config.privacy.log_translation_content
                or self._keys_changed
            )
            self._keys_changed = False

            if not save_user_settings(self.config):
                QMessageBox.warning(
                    dlg, "Settings",
                    "Settings could not be saved to disk; they apply for "
                    "this session only.",
                )

            if self.on_config_changed:
                try:
                    self.on_config_changed(self.config)
                except Exception:
                    logger.exception("on_config_changed callback failed")
                    QMessageBox.warning(
                        dlg, "Settings",
                        "The settings were saved, but applying them "
                        "at runtime failed. See the log for details.",
                    )
            if self.last_restart_required:
                QMessageBox.information(
                    dlg,
                    "Restart required",
                    "Provider, key and speech-model changes take effect "
                    "after you restart liveTranslate.",
                )
            dlg.accept()

        buttons.accepted.connect(on_accept)
        self._dialog = dlg
        return dlg

    def _save_key(self, provider: str, edit) -> None:
        key = edit.text().strip()
        if not key:
            self._set_status("Enter an API key first.", error=True)
            return

        stored = self.store.set_key(provider, key)
        env_name = KEY_ENV_VARS.get(provider)
        if env_name:
            # Make the key available to a provider rebuilt in this process.
            os.environ[env_name] = key
        self._keys_changed = True
        edit.clear()

        # Saving a key is a strong signal that the provider should be used.
        provider_list = getattr(self, "_provider_list", None)
        if provider_list is not None:
            try:
                from PyQt6.QtCore import Qt
            except ImportError:
                Qt = None
            if Qt is not None:
                for row in range(provider_list.count()):
                    item = provider_list.item(row)
                    if item.data(Qt.ItemDataRole.UserRole) == provider:
                        item.setCheckState(Qt.CheckState.Checked)
                        break

        if stored:
            self._set_status(
                f"{provider}: key saved in Windows Credential Manager."
            )
        else:
            self._set_status(
                f"{provider}: credential store unavailable — key is "
                "active for this session only.",
                error=True,
            )

    def _set_status(self, text: str, error: bool = False) -> None:
        if self._status_label is None:
            return
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            "color: #c0392b;" if error else "color: #27864a;"
        )
