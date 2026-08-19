from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Reminder:
    id: str
    user_id: int
    chat_id: int
    text: str
    due_at: str
    created_at: str
    status: str = "active"
    recurrence_frequency: str = "none"
    recurrence_interval: int = 1
    recurrence_end_at: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Reminder":
        return cls(
            id=str(value["id"]),
            user_id=int(value["user_id"]),
            chat_id=int(value["chat_id"]),
            text=str(value["text"]),
            due_at=str(value["due_at"]),
            created_at=str(value["created_at"]),
            status=str(value.get("status", "active")),
            recurrence_frequency=str(value.get("recurrence_frequency", "none")),
            recurrence_interval=int(value.get("recurrence_interval", 1)),
            recurrence_end_at=(
                str(value["recurrence_end_at"])
                if value.get("recurrence_end_at") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def due_datetime(self) -> datetime:
        return datetime.fromisoformat(self.due_at)

    @property
    def recurrence_end_datetime(self) -> datetime | None:
        if self.recurrence_end_at is None:
            return None
        return datetime.fromisoformat(self.recurrence_end_at)


@dataclass(frozen=True, slots=True)
class UserSettings:
    user_id: int
    chat_id: int
    daily_time: str
    daily_enabled: bool = True
    last_daily_sent_on: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserSettings":
        return cls(
            user_id=int(value["user_id"]),
            chat_id=int(value["chat_id"]),
            daily_time=str(value["daily_time"]),
            daily_enabled=bool(value.get("daily_enabled", True)),
            last_daily_sent_on=(
                str(value["last_daily_sent_on"])
                if value.get("last_daily_sent_on") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
