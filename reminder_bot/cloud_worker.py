from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from telegram.ext import Application

from .models import Reminder
from .service import (
    ReminderService,
    get_timezone,
    next_recurrence_datetime,
    parse_daily_time,
)

LOGGER = logging.getLogger(__name__)


def _details(reminder: Reminder) -> str:
    lines: list[str] = []
    if reminder.note:
        lines.append(f"📝 {reminder.note}")
    lines.extend(
        f"{'✅' if item.completed else '☐'} {item.text}"
        for item in reminder.checklist
    )
    return ("\n" + "\n".join(lines)) if lines else ""


def next_future_occurrence(
    reminder: Reminder, now: datetime, timezone: ZoneInfo
) -> datetime | None:
    """Advance past missed occurrences without flooding the user after downtime."""
    if reminder.recurrence_frequency == "none":
        return None
    candidate = next_recurrence_datetime(
        reminder.due_datetime.astimezone(timezone),
        reminder.recurrence_frequency,
        reminder.recurrence_interval,
    )
    end_at = reminder.recurrence_end_datetime
    while candidate <= now:
        if end_at is not None and candidate > end_at:
            return None
        candidate = next_recurrence_datetime(
            candidate,
            reminder.recurrence_frequency,
            reminder.recurrence_interval,
        )
    if end_at is not None and candidate > end_at:
        return None
    return candidate


async def process_cloud_work(
    application: Application,
    service: ReminderService,
    settings_store: Any,
    conversation_store: Any,
) -> dict[str, int]:
    """Deliver due work once; Firestore claims protect overlapping invocations."""
    now = datetime.now(service.timezone)
    reminder_store = service.store
    upcoming_sent = 0
    failed = 0
    for warning_minutes in (60, 10):
        upcoming = reminder_store.claim_upcoming(
            now, lead_minutes=warning_minutes
        )
        for reminder in upcoming:
            setting = settings_store.get(reminder.user_id)
            user_timezone = get_timezone(
                setting.timezone if setting else service.timezone.key
            )
            due = reminder.due_datetime.astimezone(user_timezone)
            if warning_minutes == 60:
                heading = "⏳ Reminder in 1 hour"
                urgency = "You have one hour left to get ready."
            else:
                heading = "🚨 Reminder due very soon"
                urgency = "Less than 10 minutes left — time to act!"
            try:
                await application.bot.send_message(
                    chat_id=reminder.chat_id,
                    text=(
                        f"{heading}\n\n"
                        f"📌 {reminder.text}\n"
                        f"⏰ Due at {due:%H:%M}\n"
                        f"{urgency}"
                        f"{_details(reminder)}"
                    ),
                )
                upcoming_sent += 1
            except Exception:
                failed += 1
                LOGGER.exception(
                    "Could not deliver %s-minute warning %s",
                    warning_minutes,
                    reminder.id,
                )
                reminder_store.release_upcoming_claim(
                    reminder, warning_minutes
                )

    claimed = reminder_store.claim_due(now)
    sent = 0
    for reminder in claimed:
        setting = settings_store.get(reminder.user_id)
        user_timezone = get_timezone(
            setting.timezone if setting else service.timezone.key
        )
        due = reminder.due_datetime.astimezone(user_timezone)
        try:
            await application.bot.send_message(
                chat_id=reminder.chat_id,
                text=(
                    f"🔔 Reminder due now\n\n📌 {reminder.text}\n"
                    f"⏰ {due:%H:%M}{_details(reminder)}"
                ),
            )
        except Exception:
            failed += 1
            LOGGER.exception("Could not deliver reminder %s", reminder.id)
            reminder_store.release_claim(reminder.id)
            continue
        reminder_store.finish_delivery(
            reminder, next_future_occurrence(reminder, now, user_timezone)
        )
        sent += 1

    summaries = 0
    for setting in settings_store.list_enabled():
        user_timezone = get_timezone(setting.timezone)
        local_now = now.astimezone(user_timezone)
        date_key = local_now.date().isoformat()
        if local_now.time().replace(tzinfo=None) < parse_daily_time(setting.daily_time):
            continue
        claimed_setting = settings_store.claim_daily_summary(
            setting.user_id, date_key
        )
        if claimed_setting is None:
            continue
        try:
            # Imported lazily to avoid a startup module cycle.
            from bot import format_daily_summary

            await application.bot.send_message(
                chat_id=claimed_setting.chat_id,
                text=format_daily_summary(
                    service,
                    claimed_setting.user_id,
                    user_timezone,
                    now,
                ),
                parse_mode="MarkdownV2",
            )
            summaries += 1
        except Exception:
            failed += 1
            LOGGER.exception(
                "Could not deliver daily summary for user %s", setting.user_id
            )
            settings_store.release_daily_summary(setting.user_id, date_key)

    purged = reminder_store.purge_old(now)
    sessions_purged = conversation_store.purge_expired(now)
    return {
        "advance_warnings_sent": upcoming_sent,
        "reminders_sent": sent,
        "summaries_sent": summaries,
        "purged": purged,
        "conversation_sessions_purged": sessions_purged,
        "failed": failed,
    }
