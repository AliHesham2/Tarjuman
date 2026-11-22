"""PyQt6 frameless RTL overlay window.

Always-on-top, click-through, stealth-capable, Arabic RTL with a
glassmorphism background and fade-in per new sentence. Hold Alt to drag
the overlay even while clicks normally pass through to the meeting app.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src.config.settings import PROJECT_ROOT, EngineConfig
from src.ui import stealth, win_effects

logger = logging.getLogger(__name__)


def _now() -> float:
    """Clock for paragraph pauses (a function so tests can control it)."""
    return time.monotonic()


def _font_asset_dirs() -> list[Path]:
    dirs: list[Path] = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        dirs.append(Path(bundle) / "assets" / "fonts")
    dirs.append(PROJECT_ROOT / "assets" / "fonts")
    return dirs


def _load_bundled_fonts(font_db) -> set[str]:
    """Register every bundled .ttf/.otf and return its family names."""
    families: set[str] = set()
    seen: set[Path] = set()
    for folder in _font_asset_dirs():
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".ttf", ".otf"} or path in seen:
                continue
            seen.add(path)
            try:
                font_id = font_db.addApplicationFont(str(path))
                if font_id >= 0:
                    families.update(font_db.applicationFontFamilies(font_id))
                else:
                    logger.warning("Could not load bundled font %s", path.name)
            except Exception:
                logger.exception("Could not load bundled font %s", path)
    if families:
        logger.info("Loaded bundled Arabic font(s): %s", ", ".join(sorted(families)))
    return families


def _choose_font_family(requested: str, loaded: set[str], font_db) -> str:
    """Prefer the requested family, then a bundled font, then Qt fallback."""
    try:
        installed = set(font_db.families())
    except Exception:
        installed = set()
    if requested in installed or requested in loaded:
        return requested
    if loaded:
        # Prefer a Naskh/Sans Arabic face if the bundle contains several.
        return sorted(loaded, key=lambda f: ("Naskh" not in f, f))[0]
    return requested


class OverlayWindow:
    """Owns the QWidget so the module can be imported without Qt installed."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.widget = None  # created in create() once QApplication exists
        self._acrylic = False
        # Set by the app (main.py) to persist a user-initiated move.
        self.on_moved = None
        self.on_close = None

    def create(self):
        """Build the actual window. Must run after QApplication exists."""
        try:
            from PyQt6.QtCore import (
                QEasingCurve, QPoint, QPropertyAnimation, QRect, QRectF, Qt,
                QTimer,
            )
            from PyQt6.QtGui import (
                QColor, QCursor, QFont, QFontDatabase, QFontMetrics, QPainter,
                QPen,
            )
            from PyQt6.QtWidgets import (
                QApplication, QFrame, QGraphicsOpacityEffect, QLabel,
                QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
            )
        except ImportError:
            logger.error("PyQt6 not installed — overlay unavailable")
            return None

        overlay_cfg = self.config.overlay
        loaded_fonts = _load_bundled_fonts(QFontDatabase)
        self._font_family = _choose_font_family(
            overlay_cfg.font_family, loaded_fonts, QFontDatabase
        )

        # Layout constants (pixels).  The top strip drags the window, the two
        # bottom corners resize it, and the right-hand scrollbar scrolls it.
        # Those are the only areas that stop being click-through on hover.
        HANDLE_H = 22
        GRIP = 22
        MIN_W = 240
        HISTORY_LIMIT = 200   # labels kept for scrolling back (memory bound)
        PARAGRAPH_MAX_CHARS = 700   # a paragraph this long is closed
        BTN_W, BTN_H = 26, 18  # hide / show / close buttons in the top strip
        CLOSE_ARM_MS = 2500    # the second click must follow within this time

        class _Overlay(QWidget):
            def __init__(inner):
                super().__init__()
                inner.config = self.config
                inner._loaded_fonts = loaded_fonts
                inner._font_family = self._font_family
                inner._labels: list[QLabel] = []
                inner._partial_label = None   # the label that currently ends in a draft
                inner._para = None            # the paragraph text is flowing into
                inner._animations: list[QPropertyAnimation] = []
                inner._drag_pos = None
                inner._dragging = False
                inner._resize = None        # (start global pos, geometry, corner)
                inner._resizing = False
                inner._handle_h = HANDLE_H
                inner._collapsed = False    # folded down to the top strip
                inner._expanded_height = None
                inner._close_armed = False
                inner._follow = True        # keep the newest line in view
                inner._click_through_enabled = False
                inner._native_click_through = False
                inner._alt_down = False

                inner.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                    | Qt.WindowType.WindowDoesNotAcceptFocus
                )
                inner.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
                inner.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
                # Custom QWidget subclasses need this for the stylesheet
                # background (the glass panel) to be painted.
                inner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                inner.setLayoutDirection(
                    Qt.LayoutDirection.RightToLeft
                    if overlay_cfg.text_direction == "rtl"
                    else Qt.LayoutDirection.LeftToRight
                )
                inner.setWindowOpacity(max(0.05, min(1.0, overlay_cfg.opacity)))
                inner.setMouseTracking(True)

                inner._font = QFont(inner._font_family, overlay_cfg.font_size)
                inner._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

                # Scrollable history: every sentence stays, newest at the bottom.
                inner._content = QWidget()
                inner._content.setObjectName("overlayContent")
                inner._content.setStyleSheet(
                    "#overlayContent { background: transparent; }"
                )
                inner.layout_v = QVBoxLayout(inner._content)
                inner.layout_v.setContentsMargins(18, 6, 18, 6)
                inner.layout_v.setSpacing(5)

                inner._scroll = QScrollArea(inner)
                inner._scroll.setWidget(inner._content)
                inner._scroll.setWidgetResizable(True)
                inner._scroll.setFrameShape(QFrame.Shape.NoFrame)
                inner._scroll.setHorizontalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )
                inner._scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded
                )
                inner._scroll.setStyleSheet(
                    "QScrollArea { background: transparent; border: none; }"
                    "QScrollArea > QWidget > QWidget { background: transparent; }"
                    "QScrollBar:vertical { width: 10px; background: transparent;"
                    " margin: 2px; }"
                    "QScrollBar::handle:vertical { background: rgba(255,255,255,110);"
                    " border-radius: 3px; min-height: 28px; }"
                    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical"
                    " { height: 0; }"
                    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical"
                    " { background: transparent; }"
                )
                inner._scroll.viewport().setAutoFillBackground(False)
                bar = inner._scroll.verticalScrollBar()
                bar.valueChanged.connect(inner._on_scrolled)
                bar.rangeChanged.connect(inner._on_range_changed)

                root = QVBoxLayout(inner)
                root.setContentsMargins(0, HANDLE_H, 0, GRIP)
                root.setSpacing(0)
                root.addWidget(inner._scroll)

                inner._btn_hide = inner._make_button(
                    "\u2013", "Hide the text (fold to the top strip)")
                inner._btn_show = inner._make_button("\u25a1", "Show the text")
                inner._btn_close = inner._make_button(
                    "\u00d7", "Close liveTranslate")
                inner._btn_hide.clicked.connect(inner.collapse)
                inner._btn_show.clicked.connect(inner.expand)
                inner._btn_close.clicked.connect(inner._on_close_clicked)
                inner._close_timer = QTimer(inner)
                inner._close_timer.setSingleShot(True)
                inner._close_timer.setInterval(CLOSE_ARM_MS)
                inner._close_timer.timeout.connect(inner._disarm_close)

                inner._update_min_size()
                inner._initial_size()
                inner._position()
                inner._sync_buttons()

                # Alt is a global key, so poll it while click-through is on.
                inner._alt_timer = QTimer(inner)
                inner._alt_timer.setInterval(50)
                inner._alt_timer.timeout.connect(inner._sync_click_through)
                inner._alt_timer.start()

            # -- painting -------------------------------------------------
            def paintEvent(inner, event):
                """Paint the glass panel explicitly.

                A stylesheet background is unreliable on a custom top-level
                QWidget with WA_TranslucentBackground, so keep the panel in a
                paintEvent and leave the window itself transparent outside
                the rounded corners.  The drag handle and resize grips are
                drawn here too, faintly, so they are discoverable.
                """
                painter = QPainter(inner)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                rect = QRectF(inner.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
                painter.setBrush(QColor(18, 18, 26, 190))
                painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
                painter.drawRoundedRect(rect, 14.0, 14.0)

                # Drag handle: a small pill centred in the top strip.
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255, 70))
                painter.drawRoundedRect(
                    QRectF(inner.width() / 2.0 - 18.0, 9.0, 36.0, 4.0), 2.0, 2.0
                )

                if inner._collapsed:
                    return

                # Resize grips: three diagonal strokes in each bottom corner.
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1.5))
                w, h = inner.width(), inner.height()
                for step in (6, 11, 16):
                    painter.drawLine(w - 4, h - 4 - step, w - 4 - step, h - 4)
                    painter.drawLine(4, h - 4 - step, 4 + step, h - 4)

            # -- geometry -------------------------------------------------
            def _screen_geometry(inner):
                screen = inner.screen() or QApplication.primaryScreen()
                if screen is None:
                    return None
                return screen.availableGeometry()

            def _position(inner):
                geo = inner._screen_geometry()
                if geo is None:
                    return
                margin = 40
                pos = inner.config.overlay.position
                w, h = inner.width(), inner.height()
                if pos == "custom":
                    x = inner.config.overlay.custom_x
                    y = inner.config.overlay.custom_y
                elif pos == "bottom-center":
                    x = geo.center().x() - w // 2
                    y = geo.bottom() - h - margin
                elif pos == "bottom-left":
                    x = geo.left() + margin
                    y = geo.bottom() - h - margin
                elif pos == "bottom-right":
                    x = geo.right() - w - margin
                    y = geo.bottom() - h - margin
                elif pos == "top-center":
                    x = geo.center().x() - w // 2
                    y = geo.top() + margin
                elif pos == "top-left":
                    x = geo.left() + margin
                    y = geo.top() + margin
                else:  # top-right
                    x = geo.right() - w - margin
                    y = geo.top() + margin
                inner.move(x, y)

            def _min_height(inner) -> int:
                """Room for the handle, the grips and at least TWO text lines."""
                lines = 2 * QFontMetrics(inner._font).lineSpacing()
                spacing = inner.layout_v.spacing()
                margins = inner.layout_v.contentsMargins()
                return (
                    HANDLE_H + GRIP + lines + spacing
                    + margins.top() + margins.bottom()
                )

            def _update_min_size(inner) -> None:
                height = HANDLE_H if inner._collapsed else inner._min_height()
                inner.setMinimumSize(MIN_W, height)

            def _initial_size(inner) -> None:
                """Saved size if the user resized before, else a readable default."""
                cfg = inner.config.overlay
                geo = inner._screen_geometry()
                if cfg.width > 0:
                    width = cfg.width
                elif geo is None:
                    width = 596
                else:
                    width = min(
                        max(360, min(760, int(geo.width() * 0.55))) + 36,
                        max(MIN_W, geo.width() - 80),
                    )
                if cfg.height > 0:
                    height = cfg.height
                else:
                    visible = max(2, int(cfg.max_lines))
                    fm = QFontMetrics(inner._font)
                    margins = inner.layout_v.contentsMargins()
                    height = (
                        HANDLE_H + GRIP + margins.top() + margins.bottom()
                        + visible * fm.lineSpacing()
                        + (visible - 1) * inner.layout_v.spacing()
                    )
                if geo is not None:
                    width = min(width, geo.width() - 20)
                    height = min(height, geo.height() - 20)
                inner.resize(
                    max(inner.minimumWidth(), width),
                    max(inner.minimumHeight(), height),
                )

            # -- scrolling --------------------------------------------------
            def _on_scrolled(inner, value: int) -> None:
                bar = inner._scroll.verticalScrollBar()
                # Scrolled away from the bottom = the user is reading history:
                # stop following until they come back down.
                inner._follow = value >= bar.maximum() - 4

            def _on_range_changed(inner, _minimum: int, maximum: int) -> None:
                if inner._follow:
                    inner._scroll.verticalScrollBar().setValue(maximum)

            def scroll_to_latest(inner) -> None:
                inner._follow = True
                bar = inner._scroll.verticalScrollBar()
                bar.setValue(bar.maximum())

            # -- text -----------------------------------------------------
            def _style_label(inner, label: QLabel, is_partial: bool) -> None:
                label.setStyleSheet(
                    "color: rgba(235,235,245,185); background: transparent;"
                    "border: none;"
                    if is_partial else
                    "color: white; background: transparent; border: none;"
                )

            def _text_alignment(inner):
                # AlignAbsolute: in a right-to-left window Qt would otherwise
                # mirror AlignRight into "left", and every short line (the last
                # line of a paragraph) would sit on the wrong side.
                horizontal = (
                    Qt.AlignmentFlag.AlignRight
                    if inner.config.overlay.text_direction == "rtl"
                    else Qt.AlignmentFlag.AlignLeft
                )
                return (
                    horizontal
                    | Qt.AlignmentFlag.AlignAbsolute
                    | Qt.AlignmentFlag.AlignVCenter
                )

            def _make_label(inner, text: str, is_partial: bool) -> QLabel:
                label = QLabel(text)
                label.setFont(inner._font)
                label.setWordWrap(True)
                label.setAlignment(inner._text_alignment())
                label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
                label.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
                )
                inner._style_label(label, is_partial)
                return label

            def _fade_in(inner, label: QLabel) -> None:
                duration = int(inner.config.overlay.fade_duration_ms)
                if duration <= 0:
                    return
                effect = QGraphicsOpacityEffect(label)
                label.setGraphicsEffect(effect)
                anim = QPropertyAnimation(effect, b"opacity", inner)
                anim.setDuration(duration)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
                anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                anim.finished.connect(
                    lambda a=anim: inner._animations.remove(a)
                    if a in inner._animations else None
                )
                inner._animations.append(anim)
                anim.start()

            def _trim(inner) -> None:
                """Bound memory: drop the oldest lines beyond the history cap."""
                while len(inner._labels) > HISTORY_LIMIT:
                    old = inner._labels.pop(0)
                    if old is inner._partial_label:
                        inner._partial_label = None
                    inner.layout_v.removeWidget(old)
                    old.deleteLater()

            def _render_paragraph(inner, para: dict) -> None:
                """Confirmed text, then the unconfirmed draft, in one label."""
                label = para["label"]
                text = (para["confirmed"] + " " + para["draft"]).strip()
                label.setText(text)
                has_draft = bool(para["draft"])
                inner._style_label(label, has_draft)
                inner._partial_label = label if has_draft else None

            def update_text(inner, arabic: str, is_partial: bool = False) -> None:
                """Flow translated text into the current paragraph.

                Translations arrive in small pieces (a few words each). They are
                joined into one paragraph that fills the window width and wraps
                at its edge, instead of one short line per piece. A draft
                (``is_partial``) replaces the previous draft at the end of the
                paragraph; a confirmed piece is appended to the paragraph. A new
                paragraph starts after a pause or when the paragraph is long.
                """
                text = (arabic or "").strip()
                if not text:
                    return

                now = _now()
                para = inner._para
                if para is not None and (
                    para["label"] not in inner._labels
                    or now - para["time"] > float(inner.config.overlay.paragraph_pause_s)
                    or len(para["confirmed"]) > PARAGRAPH_MAX_CHARS
                ):
                    para = inner._para = None            # close it

                if para is None:
                    label = inner._make_label("", True)
                    inner.layout_v.addWidget(label)
                    inner._labels.append(label)
                    inner._fade_in(label)
                    para = inner._para = {
                        "label": label, "confirmed": "", "draft": "", "time": now,
                    }

                if is_partial:
                    para["draft"] = text
                else:
                    para["confirmed"] = (para["confirmed"] + " " + text).strip()
                    para["draft"] = ""
                para["time"] = now
                inner._render_paragraph(para)

                # The scrollbar's rangeChanged handler follows the newest line
                # once the layout has grown, unless the user scrolled up.
                inner._trim()

            def clear_text(inner) -> None:
                for anim in inner._animations:
                    anim.stop()
                inner._animations.clear()
                for label in inner._labels:
                    inner.layout_v.removeWidget(label)
                    label.deleteLater()
                inner._labels.clear()
                inner._partial_label = None
                inner._para = None
                inner._follow = True

            def apply_settings(inner) -> None:
                cfg = inner.config.overlay
                inner.setWindowOpacity(max(0.05, min(1.0, cfg.opacity)))
                inner.setLayoutDirection(
                    Qt.LayoutDirection.RightToLeft
                    if cfg.text_direction == "rtl"
                    else Qt.LayoutDirection.LeftToRight
                )
                inner._font_family = _choose_font_family(
                    cfg.font_family, inner._loaded_fonts, QFontDatabase
                )
                inner._font = QFont(inner._font_family, cfg.font_size)
                inner._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
                for label in inner._labels:
                    label.setFont(inner._font)
                    label.setAlignment(inner._text_alignment())
                inner._trim()
                inner._update_min_size()
                # Keep the user's size, but never below the two-line minimum.
                inner.resize(
                    max(inner.width(), inner.minimumWidth()),
                    max(inner.height(), inner.minimumHeight()),
                )
                inner._position()

            # -- native click-through / drag / resize -----------------------
            def _set_native_click_through(inner, enable: bool) -> None:
                if enable == inner._native_click_through:
                    return
                if win_effects.set_click_through(int(inner.winId()), enable):
                    inner._native_click_through = enable

            def _interactive_zones(inner) -> list:
                """Screen rectangles that must receive the mouse (not pass it on)."""
                g = inner.geometry()
                zones = [QRect(g.left(), g.top(), g.width(), HANDLE_H)]
                if not inner._collapsed:
                    zones += [
                        QRect(g.left(), g.bottom() - GRIP + 1, GRIP, GRIP),
                        QRect(g.right() - GRIP + 1, g.bottom() - GRIP + 1,
                              GRIP, GRIP),
                    ]
                bar = inner._scroll.verticalScrollBar()
                if not inner._collapsed and bar.isVisible():
                    top_left = bar.mapToGlobal(QPoint(0, 0))
                    zones.append(
                        QRect(top_left.x(), top_left.y(), bar.width(), bar.height())
                    )
                return zones

            def _sync_click_through(inner) -> None:
                if (
                    not inner._click_through_enabled
                    or inner._dragging
                    or inner._resizing
                ):
                    return
                alt = win_effects.alt_is_down()
                cursor = QCursor.pos()
                over_zone = any(z.contains(cursor) for z in inner._interactive_zones())
                inner._alt_down = alt
                inner._set_native_click_through(not (alt or over_zone))

            def set_click_through_enabled(inner, enable: bool) -> None:
                inner._click_through_enabled = bool(enable)
                if not enable:
                    inner._set_native_click_through(False)
                else:
                    inner._set_native_click_through(not win_effects.alt_is_down())

            def _hit(inner, pos) -> str | None:
                x, y = pos.x(), pos.y()
                w, h = inner.width(), inner.height()
                if not inner._collapsed and y >= h - GRIP:
                    if x <= GRIP:
                        return "grip_left"
                    if x >= w - GRIP:
                        return "grip_right"
                if y <= HANDLE_H:
                    return "move"
                return None

            def _persist_geometry(inner) -> None:
                """Remember where the user put the window and how big it is."""
                cfg = inner.config.overlay
                cfg.position = "custom"
                cfg.custom_x = inner.x()
                cfg.custom_y = inner.y()
                cfg.width = inner.width()
                # A folded window must not save its thin strip as "the size".
                cfg.height = (
                    inner._expanded_height
                    if inner._collapsed and inner._expanded_height
                    else inner.height()
                )
                try:
                    cb = getattr(inner, "on_moved", None)
                    if cb:
                        cb()
                except Exception:
                    logger.exception("Overlay on_moved callback failed")

            def mousePressEvent(inner, event):
                if event.button() != Qt.MouseButton.LeftButton:
                    super().mousePressEvent(event)
                    return
                alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                hit = inner._hit(event.position())
                if alt or hit == "move":
                    # The timer normally clears click-through before the click
                    # arrives; doing it here also covers a very fast click.
                    inner._set_native_click_through(False)
                    inner._drag_pos = (
                        event.globalPosition().toPoint()
                        - inner.frameGeometry().topLeft()
                    )
                    inner._dragging = True
                    event.accept()
                    return
                if hit in ("grip_left", "grip_right"):
                    inner._set_native_click_through(False)
                    inner._resize = (
                        event.globalPosition().toPoint(), inner.geometry(), hit
                    )
                    inner._resizing = True
                    event.accept()
                    return
                super().mousePressEvent(event)

            def mouseMoveEvent(inner, event):
                if not event.buttons() & Qt.MouseButton.LeftButton:
                    super().mouseMoveEvent(event)
                    return
                if inner._drag_pos is not None:
                    inner.move(
                        event.globalPosition().toPoint() - inner._drag_pos
                    )
                    event.accept()
                    return
                if inner._resize is not None:
                    start, geom, corner = inner._resize
                    delta = event.globalPosition().toPoint() - start
                    height = max(inner.minimumHeight(), geom.height() + delta.y())
                    if corner == "grip_right":
                        width = max(inner.minimumWidth(), geom.width() + delta.x())
                        left = geom.x()
                    else:  # the left corner keeps the right edge fixed
                        width = max(inner.minimumWidth(), geom.width() - delta.x())
                        left = geom.x() + geom.width() - width
                    inner.setGeometry(left, geom.y(), width, height)
                    event.accept()
                    return
                super().mouseMoveEvent(event)

            def mouseReleaseEvent(inner, event):
                if event.button() == Qt.MouseButton.LeftButton:
                    changed = inner._dragging or inner._resizing
                    inner._drag_pos = None
                    inner._dragging = False
                    inner._resize = None
                    inner._resizing = False
                    if changed:
                        # Persist, otherwise the next _position() snaps the
                        # window back to where the config says it should be.
                        inner._persist_geometry()
                    event.accept()
                    return
                super().mouseReleaseEvent(event)

            # -- hide / show / close buttons -------------------------------
            def _make_button(inner, text: str, tip: str):
                button = QPushButton(text, inner)
                button.setToolTip(tip)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.setFixedSize(BTN_W, BTN_H)
                # The window is right-to-left; the button captions are English.
                button.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
                button.setStyleSheet(
                    "QPushButton { color: rgba(255,255,255,210);"
                    " background: rgba(255,255,255,26); border: none;"
                    " border-radius: 4px; font-size: 15px; font-weight: 600;"
                    " padding: 0; }"
                    "QPushButton:hover { background: rgba(255,255,255,70); }"
                    "QPushButton:disabled { color: rgba(255,255,255,70);"
                    " background: transparent; }"
                    "QPushButton[armed=\"true\"] {"
                    " background: rgba(214,60,60,230); color: white; }"
                )
                return button

            def _layout_buttons(inner) -> None:
                """Hide, Show, Close at the top-right, whatever the text direction."""
                if getattr(inner, "_btn_close", None) is None:
                    return
                y = (HANDLE_H - BTN_H) // 2
                x = inner.width() - 10
                for button in (inner._btn_close, inner._btn_show, inner._btn_hide):
                    x -= button.width()
                    button.move(x, y)
                    x -= 4

            def resizeEvent(inner, event):
                super().resizeEvent(event)
                inner._layout_buttons()

            def _sync_buttons(inner) -> None:
                inner._btn_hide.setEnabled(not inner._collapsed)
                inner._btn_show.setEnabled(inner._collapsed)
                inner._layout_buttons()
                inner.update()

            def collapse(inner) -> None:
                """Fold the window to the top strip; Show brings it back."""
                if inner._collapsed:
                    return
                inner._expanded_height = inner.height()
                inner._collapsed = True
                inner._scroll.setVisible(False)
                inner._update_min_size()
                inner.resize(inner.width(), HANDLE_H)
                inner._sync_buttons()

            def expand(inner) -> None:
                if not inner._collapsed:
                    return
                inner._collapsed = False
                inner._scroll.setVisible(True)
                inner._update_min_size()
                inner.resize(
                    inner.width(),
                    max(inner._expanded_height or 0, inner.minimumHeight()),
                )
                inner._sync_buttons()

            def _repolish_close(inner) -> None:
                style = inner._btn_close.style()
                style.unpolish(inner._btn_close)
                style.polish(inner._btn_close)

            def _on_close_clicked(inner) -> None:
                """Two clicks to quit: a stray click must not end the meeting."""
                if not inner._close_armed:
                    inner._close_armed = True
                    # A trailing U+200E (left-to-right mark) keeps the RTL window from moving
                    # the neutral "?" to the left of "Close".
                    inner._btn_close.setText("Close?‎")
                    inner._btn_close.setFixedWidth(56)
                    inner._btn_close.setProperty("armed", "true")
                    inner._repolish_close()
                    inner._layout_buttons()
                    inner._close_timer.start()
                    return
                inner._close_timer.stop()
                inner._disarm_close()
                try:
                    cb = getattr(inner, "on_close", None)
                    if cb:
                        cb()
                except Exception:
                    logger.exception("Overlay on_close callback failed")

            def _disarm_close(inner) -> None:
                inner._close_armed = False
                inner._btn_close.setText("\u00d7")
                inner._btn_close.setFixedWidth(BTN_W)
                inner._btn_close.setProperty("armed", "false")
                inner._repolish_close()
                inner._layout_buttons()

            def wheelEvent(inner, event):
                """Scroll the history when the wheel is over the handle strip."""
                bar = inner._scroll.verticalScrollBar()
                bar.setValue(bar.value() - event.angleDelta().y())
                event.accept()

        self.widget = _Overlay()
        self.widget.on_moved = lambda: self.on_moved() if self.on_moved else None
        self.widget.on_close = lambda: self.on_close() if self.on_close else None
        self._apply_window_effects()
        self.widget.show()
        self.set_click_through(self.config.overlay.click_through)
        self.apply_stealth(self.config.overlay.stealth_mode)
        return self.widget

    # -- wrapper API --------------------------------------------------------

    def _apply_window_effects(self) -> None:
        if self.widget is None:
            return
        try:
            hwnd = int(self.widget.winId())
        except Exception:
            return
        if hwnd:
            self._acrylic = win_effects.apply_acrylic(hwnd)

    def apply_stealth(self, enable: bool) -> None:
        if self.widget is None:
            return
        try:
            hwnd = int(self.widget.winId())
        except Exception:
            return
        if enable:
            if not stealth.apply_stealth(hwnd):
                logger.warning(
                    "Stealth mode unavailable — overlay WILL appear in screen share"
                )
        else:
            stealth.remove_stealth(hwnd)

    def set_stealth(self, enable: bool) -> None:
        self.apply_stealth(enable)

    def set_click_through(self, enable: bool) -> None:
        if self.widget is not None:
            self.widget.set_click_through_enabled(enable)

    def apply_settings(self) -> None:
        """Apply config.overlay changes to an existing widget."""
        if self.widget is None:
            return
        self.widget.apply_settings()
        self.set_click_through(self.config.overlay.click_through)
        self.apply_stealth(self.config.overlay.stealth_mode)

    def update_text(self, arabic: str, is_partial: bool = False) -> None:
        if self.widget is not None:
            self.widget.update_text(arabic, is_partial)

    def clear_text(self) -> None:
        if self.widget is not None:
            self.widget.clear_text()

    def toggle_visibility(self) -> bool:
        """Show/hide the overlay; returns True when it is now visible."""
        if self.widget is None:
            return False
        if self.widget.isVisible():
            self.widget.hide()
            return False
        self.widget.show()
        self.widget.raise_()
        self.widget._position()
        self.apply_stealth(self.config.overlay.stealth_mode)
        self.set_click_through(self.config.overlay.click_through)
        return True

    def is_visible(self) -> bool:
        return bool(self.widget is not None and self.widget.isVisible())
