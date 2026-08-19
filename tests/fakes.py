from __future__ import annotations

from datetime import datetime, timedelta

from reminder_bot.models import Reminder, UserSettings


class InMemoryReminderStore:
    def __init__(self) -> None:
        self.items: list[Reminder] = []

    def add(self, reminder: Reminder) -> None:
        self.items.append(reminder)

    def list_for_user(self, user_id: int, now: datetime | None = None) -> list[Reminder]:
        current = now or datetime.now().astimezone()
        return sorted(
            [item for item in self.items if item.user_id == user_id and item.status == "active" and item.due_datetime > current],
            key=lambda item: item.due_datetime,
        )

    def list_old_for_user(self, user_id: int, now: datetime | None = None) -> list[Reminder]:
        current = now or datetime.now().astimezone()
        return sorted(
            [item for item in self.items if item.user_id == user_id and (item.status in {"completed", "expired"} or item.due_datetime <= current)],
            key=lambda item: item.due_datetime,
            reverse=True,
        )

    def delete_for_user(self, reminder_id: str, user_id: int) -> bool:
        before = len(self.items)
        self.items = [item for item in self.items if not (item.id == reminder_id and item.user_id == user_id)]
        return len(self.items) != before

    def delete_active_for_user(self, user_id: int, now: datetime | None = None) -> list[Reminder]:
        current = now or datetime.now().astimezone()
        removed = [item for item in self.items if item.user_id == user_id and item.status == "active" and item.due_datetime > current]
        removed_ids = {item.id for item in removed}
        self.items = [item for item in self.items if item.id not in removed_ids]
        return removed

    def update_deadline_for_user(self, reminder_id: str, user_id: int, due_at: str) -> Reminder | None:
        for index, reminder in enumerate(self.items):
            if reminder.id == reminder_id and reminder.user_id == user_id:
                updated = Reminder(
                    id=reminder.id, user_id=reminder.user_id, chat_id=reminder.chat_id,
                    text=reminder.text, due_at=due_at, created_at=reminder.created_at,
                    status="active", recurrence_frequency=reminder.recurrence_frequency,
                    recurrence_interval=reminder.recurrence_interval,
                    recurrence_end_at=reminder.recurrence_end_at,
                )
                self.items[index] = updated
                return updated
        return None

    def id_exists(self, reminder_id: str) -> bool:
        return any(item.id == reminder_id for item in self.items)

    def purge_old(self, now: datetime | None = None) -> int:
        cutoff = (now or datetime.now().astimezone()) - timedelta(days=7)
        kept = [item for item in self.items if item.due_datetime > cutoff]
        removed = len(self.items) - len(kept)
        self.items = kept
        return removed


class InMemoryUserSettingsStore:
    def __init__(self) -> None:
        self.items: dict[int, UserSettings] = {}

    def get(self, user_id: int) -> UserSettings | None:
        return self.items.get(user_id)

    def list_enabled(self) -> list[UserSettings]:
        return [item for item in self.items.values() if item.daily_enabled]

    def save(self, value: UserSettings) -> None:
        self.items[value.user_id] = value
