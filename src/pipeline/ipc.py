"""IPC between the Python engine threads and the Qt UI thread.

Typed messages through thread-safe Queues. The Qt main thread polls
the UI queue with a QTimer (every 50ms) — no cross-thread GUI calls.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from enum import Enum


class MessageType(str, Enum):
    TEXT_UPDATE = "text_update"        # confirmed translation → overlay
    PARTIAL_UPDATE = "partial_update"  # hypothesis translation → overlay
    STATUS_UPDATE = "status_update"    # recording / paused / error
    ERROR_EVENT = "error_event"
    QUOTA_UPDATE = "quota_update"      # remaining quota per provider
    PROVIDER_UPDATE = "provider_update"  # active provider (tray badge)
    INFO_EVENT = "info_event"          # plain notice for the tray (not an error)


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Message:
    type: MessageType
    payload: dict = field(default_factory=dict)


@dataclass
class TextUpdate(Message):
    type: MessageType = MessageType.TEXT_UPDATE
    arabic: str = ""
    english: str = ""

    def __post_init__(self) -> None:
        self.payload.setdefault("arabic", self.arabic)
        self.payload.setdefault("english", self.english)


@dataclass
class StatusUpdate(Message):
    type: MessageType = MessageType.STATUS_UPDATE
    status: PipelineStatus = PipelineStatus.IDLE
    detail: str = ""

    def __post_init__(self) -> None:
        self.payload.setdefault("status", self.status.value)
        self.payload.setdefault("detail", self.detail)


@dataclass
class ErrorEvent(Message):
    type: MessageType = MessageType.ERROR_EVENT
    error: str = ""

    def __post_init__(self) -> None:
        self.payload.setdefault("error", self.error)


@dataclass
class QuotaUpdate(Message):
    type: MessageType = MessageType.QUOTA_UPDATE
    quotas: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.payload.setdefault("quotas", self.quotas)


class UIQueue(queue.Queue):
    """Bounded UI queue with non-blocking, drop-oldest semantics.

    Worker threads must never block on a slow Qt repaint. ``drain`` is for
    the UI timer: it returns at most ``limit`` messages in FIFO order.
    """

    def put_message(self, msg: Message) -> None:
        put_message(self, msg)

    def drain(self, limit: int | None = None) -> list[Message]:
        messages: list[Message] = []
        while limit is None or len(messages) < limit:
            try:
                messages.append(self.get_nowait())
            except queue.Empty:
                break
        return messages


def make_ui_queue() -> "queue.Queue[Message]":
    return UIQueue(maxsize=200)


def put_message(q: "queue.Queue[Message]", msg: Message) -> None:
    """Non-blocking put — drop the oldest message if the UI is lagging."""
    try:
        q.put_nowait(msg)
    except queue.Full:
        try:
            q.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass
