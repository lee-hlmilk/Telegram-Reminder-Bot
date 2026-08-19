from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from .models import Reminder, UserSettings


class JsonReminderStore:
    """Small JSON repository with atomic writes for the local prototype."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_unlocked(self) -> list[Reminder]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid reminder database: {self.path}") from exc
        if not isinstance(raw, list):
            raise RuntimeError(f"Reminder database must contain a JSON list: {self.path}")
        return [Reminder.from_dict(item) for item in raw]

    def _write_unlocked(self, reminders: list[Reminder]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    [reminder.to_dict() for reminder in reminders],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def add(self, reminder: Reminder) -> None:
        with self._lock:
            reminders = self._read_unlocked()
            reminders.append(reminder)
            self._write_unlocked(reminders)

    def list_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        with self._lock:
            reminders = self._read_unlocked()
        current = now or datetime.now().astimezone()
        return sorted(
            (
                reminder
                for reminder in reminders
                if reminder.user_id == user_id
                and reminder.status == "active"
                and reminder.due_datetime > current
            ),
            key=lambda reminder: reminder.due_datetime,
        )

    def list_old_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        with self._lock:
            reminders = self._read_unlocked()
        current = now or datetime.now().astimezone()
        return sorted(
            (
                reminder
                for reminder in reminders
                if reminder.user_id == user_id and reminder.due_datetime <= current
            ),
            key=lambda reminder: reminder.due_datetime,
            reverse=True,
        )

    def get_for_user(self, reminder_id: str, user_id: int) -> Reminder | None:
        with self._lock:
            return next(
                (
                    item
                    for item in self._read_unlocked()
                    if item.id == reminder_id and item.user_id == user_id
                ),
                None,
            )

    def list_active(self) -> list[Reminder]:
        with self._lock:
            reminders = self._read_unlocked()
        return sorted(
            reminders,
            key=lambda reminder: reminder.due_datetime,
        )

    def purge_old(self, now: datetime | None = None) -> int:
        """Permanently remove reminders at least seven days past their deadline."""
        current = now or datetime.now().astimezone()
        cutoff = current - timedelta(days=7)
        with self._lock:
            reminders = self._read_unlocked()
            kept = [
                item
                for item in reminders
                if (
                    item.status == "active"
                    and item.recurrence_frequency != "none"
                    and (
                        item.recurrence_end_datetime is None
                        or item.recurrence_end_datetime > cutoff
                    )
                )
                or item.due_datetime > cutoff
            ]
            removed = len(reminders) - len(kept)
            if removed:
                self._write_unlocked(kept)
            return removed

    def delete_for_user(self, reminder_id: str, user_id: int) -> bool:
        with self._lock:
            reminders = self._read_unlocked()
            kept = [
                reminder
                for reminder in reminders
                if not (reminder.id == reminder_id and reminder.user_id == user_id)
            ]
            if len(kept) == len(reminders):
                return False
            self._write_unlocked(kept)
            return True

    def delete_active_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        """Delete and return only this user's upcoming active reminders."""
        current = now or datetime.now().astimezone()
        with self._lock:
            reminders = self._read_unlocked()
            removed = [
                reminder
                for reminder in reminders
                if reminder.user_id == user_id
                and reminder.status == "active"
                and reminder.due_datetime > current
            ]
            if not removed:
                return []
            removed_ids = {reminder.id for reminder in removed}
            self._write_unlocked(
                [reminder for reminder in reminders if reminder.id not in removed_ids]
            )
            return removed

    def set_status_for_user(self, reminder_id: str, user_id: int, status: str) -> bool:
        with self._lock:
            reminders = self._read_unlocked()
            changed = False
            updated: list[Reminder] = []
            for reminder in reminders:
                if reminder.id == reminder_id and reminder.user_id == user_id:
                    updated.append(
                        Reminder(
                            id=reminder.id,
                            user_id=reminder.user_id,
                            chat_id=reminder.chat_id,
                            text=reminder.text,
                            due_at=reminder.due_at,
                            created_at=reminder.created_at,
                            status=status,
                            recurrence_frequency=reminder.recurrence_frequency,
                            recurrence_interval=reminder.recurrence_interval,
                            recurrence_end_at=reminder.recurrence_end_at,
                        )
                    )
                    changed = True
                else:
                    updated.append(reminder)
            if changed:
                self._write_unlocked(updated)
            return changed

    def update_deadline_for_user(
        self, reminder_id: str, user_id: int, due_at: str
    ) -> Reminder | None:
        with self._lock:
            reminders = self._read_unlocked()
            updated_reminder = None
            updated_items: list[Reminder] = []
            for reminder in reminders:
                if reminder.id == reminder_id and reminder.user_id == user_id:
                    updated_reminder = Reminder(
                        id=reminder.id,
                        user_id=reminder.user_id,
                        chat_id=reminder.chat_id,
                        text=reminder.text,
                        due_at=due_at,
                        created_at=reminder.created_at,
                        status="active",
                        recurrence_frequency=reminder.recurrence_frequency,
                        recurrence_interval=reminder.recurrence_interval,
                        recurrence_end_at=reminder.recurrence_end_at,
                    )
                    updated_items.append(updated_reminder)
                else:
                    updated_items.append(reminder)
            if updated_reminder is not None:
                self._write_unlocked(updated_items)
            return updated_reminder

    def id_exists(self, reminder_id: str) -> bool:
        with self._lock:
            return any(item.id == reminder_id for item in self._read_unlocked())


class JsonUserSettingsStore:
    """Per-user delivery preferences stored in a small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_unlocked(self) -> list[UserSettings]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid user settings database: {self.path}") from exc
        if not isinstance(raw, list):
            raise RuntimeError(f"User settings database must contain a JSON list: {self.path}")
        return [UserSettings.from_dict(item) for item in raw]

    def _write_unlocked(self, settings: list[UserSettings]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    [item.to_dict() for item in settings],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def get(self, user_id: int) -> UserSettings | None:
        with self._lock:
            return next(
                (item for item in self._read_unlocked() if item.user_id == user_id),
                None,
            )

    def list_enabled(self) -> list[UserSettings]:
        with self._lock:
            return [item for item in self._read_unlocked() if item.daily_enabled]

    def save(self, value: UserSettings) -> None:
        with self._lock:
            settings = [
                item for item in self._read_unlocked() if item.user_id != value.user_id
            ]
            settings.append(value)
            self._write_unlocked(settings)
