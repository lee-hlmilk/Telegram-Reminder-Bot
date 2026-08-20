from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from telegram.ext import Application

from .models import Reminder
from .service import ReminderService, next_recurrence_datetime, parse_daily_time

LOGGER = logging.getLogger(__name__)


def next_future_occurrence(reminder: Reminder, now: datetime) -> datetime | None:
    """Advance past missed occurrences without flooding the user after downtime."""
    if reminder.recurrence_frequency == "none":
        return None
    candidate = next_recurrence_datetime(
        reminder.due_datetime,
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
) -> dict[str, int]:
    """Deliver due work once; Firestore claims protect overlapping invocations."""
    now = datetime.now(service.timezone)
    reminder_store = service.store
    warning_minutes = int(os.getenv("REMINDER_WARNING_MINUTES", "60"))
    if warning_minutes < 1 or warning_minutes > 10080:
        raise RuntimeError("REMINDER_WARNING_MINUTES must be between 1 and 10080")

    upcoming = reminder_store.claim_upcoming(now, lead_minutes=warning_minutes)
    upcoming_sent = 0
    failed = 0
    for reminder in upcoming:
        due = reminder.due_datetime.astimezone(service.timezone)
        try:
            await application.bot.send_message(
                chat_id=reminder.chat_id,
                text=(
                    "⏳ Upcoming reminder\n\n"
                    f"📌 {reminder.text}\n"
                    f"⏰ Due at {due:%H:%M}"
                ),
            )
            upcoming_sent += 1
        except Exception:
            failed += 1
            LOGGER.exception("Could not deliver advance warning %s", reminder.id)
            reminder_store.release_upcoming_claim(reminder)

    claimed = reminder_store.claim_due(now)
    sent = 0
    for reminder in claimed:
        try:
            await application.bot.send_message(
                chat_id=reminder.chat_id,
                text=f"🔔 Reminder\n\n📌 {reminder.text}",
            )
        except Exception:
            failed += 1
            LOGGER.exception("Could not deliver reminder %s", reminder.id)
            reminder_store.release_claim(reminder.id)
            continue
        reminder_store.finish_delivery(
            reminder, next_future_occurrence(reminder, now)
        )
        sent += 1

    summaries = 0
    date_key = now.date().isoformat()
    for setting in settings_store.list_enabled():
        if now.time().replace(tzinfo=None) < parse_daily_time(setting.daily_time):
            continue
        claimed_setting = settings_store.claim_daily_summary(
            setting.user_id, date_key
        )
        if claimed_setting is None:
            continue
        try:
            # Imported lazily to avoid a startup module cycle.
            from bot import format_reminder_list

            await application.bot.send_message(
                chat_id=claimed_setting.chat_id,
                text=format_reminder_list(
                    service, claimed_setting.user_id, "☀️ Your daily reminders"
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
    return {
        "advance_warnings_sent": upcoming_sent,
        "reminders_sent": sent,
        "summaries_sent": summaries,
        "purged": purged,
        "failed": failed,
    }
