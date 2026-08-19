from __future__ import annotations

import secrets
import calendar
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import Reminder
from .storage import JsonReminderStore


class ReminderInputError(ValueError):
    pass


def get_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Unknown BOT_TIMEZONE: {name}") from exc


def parse_add_arguments(arguments: list[str], timezone: ZoneInfo) -> tuple[datetime, str]:
    if len(arguments) < 2:
        raise ReminderInputError("Usage: /add YYYY-MM-DD [HH:MM] reminder text")

    try:
        due_date = datetime.strptime(arguments[0], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ReminderInputError("Date must use YYYY-MM-DD, for example 2026-08-25.") from exc

    text_start = 1
    due_time = time(23, 59)
    if len(arguments) >= 3 and ":" in arguments[1]:
        try:
            due_time = datetime.strptime(arguments[1], "%H:%M").time()
        except ValueError as exc:
            raise ReminderInputError("Time must use 24-hour HH:MM, for example 14:30.") from exc
        text_start = 2

    reminder_text = " ".join(arguments[text_start:]).strip()
    if not reminder_text:
        raise ReminderInputError("Please include what you want to be reminded about.")
    if len(reminder_text) > 1000:
        raise ReminderInputError("Reminder text must be 1,000 characters or fewer.")

    return datetime.combine(due_date, due_time, timezone), reminder_text


def parse_daily_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ReminderInputError(
            "Daily reminder time must use 24-hour HH:MM, for example /daily 08:00."
        ) from exc


class ReminderService:
    def __init__(self, store: JsonReminderStore, timezone: ZoneInfo) -> None:
        self.store = store
        self.timezone = timezone

    def create(
        self,
        *,
        user_id: int,
        chat_id: int,
        arguments: list[str],
        now: datetime | None = None,
    ) -> Reminder:
        due_at, text = parse_add_arguments(arguments, self.timezone)
        current_time = now or datetime.now(self.timezone)
        if due_at <= current_time:
            raise ReminderInputError("The reminder time must be in the future.")

        return self.create_at(
            user_id=user_id,
            chat_id=chat_id,
            due_at=due_at,
            text=text,
            now=current_time,
        )

    def create_at(
        self,
        *,
        user_id: int,
        chat_id: int,
        due_at: datetime,
        text: str,
        recurrence_frequency: str = "none",
        recurrence_interval: int = 1,
        recurrence_end_at: datetime | None = None,
        now: datetime | None = None,
    ) -> Reminder:
        current_time = now or datetime.now(self.timezone)
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=self.timezone)
        else:
            due_at = due_at.astimezone(self.timezone)
        if due_at <= current_time:
            raise ReminderInputError("The reminder time must be in the future.")
        text = text.strip()
        if not text:
            raise ReminderInputError("Please include what you want to be reminded about.")
        if len(text) > 1000:
            raise ReminderInputError("Reminder text must be 1,000 characters or fewer.")
        if recurrence_frequency not in {"none", "daily", "weekly", "monthly", "yearly"}:
            raise ReminderInputError("That repeating schedule is not supported.")
        if recurrence_interval < 1:
            raise ReminderInputError("The repeat interval must be at least one.")
        if recurrence_end_at is not None:
            if recurrence_end_at.tzinfo is None:
                recurrence_end_at = recurrence_end_at.replace(tzinfo=self.timezone)
            else:
                recurrence_end_at = recurrence_end_at.astimezone(self.timezone)
            if recurrence_end_at < due_at:
                raise ReminderInputError("The repeat end must follow its first reminder.")

        reminder_id = self._new_id()
        reminder = Reminder(
            id=reminder_id,
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            due_at=due_at.isoformat(),
            created_at=current_time.isoformat(),
            recurrence_frequency=recurrence_frequency,
            recurrence_interval=recurrence_interval,
            recurrence_end_at=(
                recurrence_end_at.isoformat() if recurrence_end_at is not None else None
            ),
        )
        self.store.add(reminder)
        return reminder

    def update_deadline(
        self,
        *,
        reminder: Reminder,
        user_id: int,
        due_at: datetime,
        now: datetime | None = None,
    ) -> Reminder:
        current_time = now or datetime.now(self.timezone)
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=self.timezone)
        else:
            due_at = due_at.astimezone(self.timezone)
        if due_at <= current_time:
            raise ReminderInputError("The new reminder time must be in the future.")
        if (
            reminder.recurrence_end_datetime is not None
            and due_at > reminder.recurrence_end_datetime
        ):
            raise ReminderInputError(
                "The new deadline falls after this reminder's repeat end date."
            )
        updated = self.store.update_deadline_for_user(
            reminder.id, user_id, due_at.isoformat()
        )
        if updated is None:
            raise ReminderInputError("That reminder could not be found.")
        return updated

    def advance_recurrence(self, reminder: Reminder) -> Reminder | None:
        if reminder.recurrence_frequency == "none":
            return None
        next_due = next_recurrence_datetime(
            reminder.due_datetime,
            reminder.recurrence_frequency,
            reminder.recurrence_interval,
        )
        end_at = reminder.recurrence_end_datetime
        if end_at is not None and next_due > end_at:
            return None
        return self.store.update_deadline_for_user(
            reminder.id, reminder.user_id, next_due.isoformat()
        )

    def _new_id(self) -> str:
        for _ in range(20):
            candidate = secrets.token_hex(3)
            if not self.store.id_exists(candidate):
                return candidate
        raise RuntimeError("Unable to generate a unique reminder ID")


def next_recurrence_datetime(
    current: datetime, frequency: str, interval: int
) -> datetime:
    if frequency == "daily":
        return current + timedelta(days=interval)
    if frequency == "weekly":
        return current + timedelta(weeks=interval)
    if frequency == "monthly":
        month_index = current.year * 12 + current.month - 1 + interval
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        day = min(current.day, calendar.monthrange(year, month)[1])
        return current.replace(year=year, month=month, day=day)
    if frequency == "yearly":
        year = current.year + interval
        day = min(current.day, calendar.monthrange(year, current.month)[1])
        return current.replace(year=year, day=day)
    raise ReminderInputError("That repeating schedule is not supported.")
