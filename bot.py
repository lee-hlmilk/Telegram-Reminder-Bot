from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

from reminder_bot.conversation import find_matching_reminders
from reminder_bot.llm import (
    IntentInterpretationError,
    LLMUnavailableError,
    OpenAIIntentInterpreter,
)
from reminder_bot.models import UserSettings
from reminder_bot.service import (
    ReminderInputError,
    ReminderService,
    get_timezone,
    parse_daily_time,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

BOT_COMMANDS = [
    BotCommand("start", "meet your reminder assistant"),
]

WELCOME_TEXT = """Hey! I’m Remi 👋

I’m your personal reminder assistant. You don’t need to remember commands—just talk to me normally.

You could say:
• “Remind me tomorrow at 3pm to call Mum.”
• “What do I have coming up?”
• “I finished my science homework.”
• “Clear all my reminders.”
• “Send my daily reminder at 8:30am.”
• “Turn off my daily reminder.”

Your daily summary starts enabled at 08:00. Tell me anytime if you’d prefer another time."""


def confirmation_decision(message: str) -> bool | None:
    normalized = re.sub(r"[^a-z ]", "", message.lower()).strip()
    if re.match(r"^(no|n|cancel|stop|never ?mind|dont|do not)\b", normalized):
        return False
    if re.match(
        r"^(yes|y|yeah|yep|confirm|sure|ok|okay|do it|go ahead|proceed)\b",
        normalized,
    ):
        return True
    return None


def _append_reminder_details(lines: list[str], reminder: Any) -> None:
    if reminder.note:
        lines.append(f"📝 _{escape_markdown(reminder.note, version=2)}_")
    for item in reminder.checklist:
        marker = "✅" if item.completed else "☐"
        lines.append(f"{marker} {escape_markdown(item.text, version=2)}")


def format_reminder_list(
    service: ReminderService,
    user_id: int,
    title: str,
    timezone: ZoneInfo | None = None,
    reminders: list[Any] | None = None,
) -> str:
    reminders = reminders if reminders is not None else service.store.list_for_user(user_id)
    if not reminders:
        if title != "📋 Your reminders":
            return "_No active reminders matched that topic\\._"
        return "_You have no active reminders\\._"

    display_timezone = timezone or service.timezone
    lines = [f"*{escape_markdown(title, version=2)}*"]
    for index, reminder in enumerate(reminders, start=1):
        due = reminder.due_datetime.astimezone(display_timezone)
        lines.extend(
            [
                "",
                f"*{index} · {escape_markdown(reminder.text, version=2)}*",
                f"📅 _{due:%a, %d %b %Y}_",
                f"⏰ _{due:%H:%M}_",
            ]
        )
        if reminder.recurrence_frequency != "none":
            interval = reminder.recurrence_interval
            unit = {
                "daily": "day",
                "weekly": "week",
                "monthly": "month",
                "yearly": "year",
            }[reminder.recurrence_frequency]
            repeat = (
                f"🔁 _Every {unit}"
                if interval == 1
                else f"🔁 _Every {interval} {unit}s"
            )
            if reminder.recurrence_end_datetime is not None:
                recurrence_end = reminder.recurrence_end_datetime.astimezone(
                    display_timezone
                )
                repeat += f" until {recurrence_end:%d %b %Y}"
            repeat += "_"
            lines.append(repeat)
        _append_reminder_details(lines, reminder)
    return "\n".join(lines)


def format_old_reminder_list(
    service: ReminderService, user_id: int, timezone: ZoneInfo | None = None
) -> str:
    reminders = service.store.list_old_for_user(user_id)
    if not reminders:
        return "_You have no old reminders\\._"

    display_timezone = timezone or service.timezone
    lines = [
        "*🗃 Your old reminders*",
        "_Automatically removed 7 days after the deadline\\._",
    ]
    for index, reminder in enumerate(reminders, start=1):
        due = reminder.due_datetime.astimezone(display_timezone)
        label = "✅ Completed" if reminder.status == "completed" else "⌛ Expired"
        lines.extend(
            [
                "",
                f"*{index} · {label}*",
                escape_markdown(reminder.text, version=2),
                f"📅 _{due:%a, %d %b %Y}_",
                f"⏰ _{due:%H:%M}_",
            ]
        )
        _append_reminder_details(lines, reminder)
    return "\n".join(lines)


def format_daily_summary(
    service: ReminderService,
    user_id: int,
    timezone: ZoneInfo,
    now: datetime,
) -> str:
    local_now = now.astimezone(timezone)
    today = local_now.date()
    tomorrow = today + timedelta(days=1)
    week_end = today + timedelta(days=7)
    active = service.store.list_for_user(user_id, now)
    overdue = [
        item for item in service.store.list_old_for_user(user_id, now)
        if item.status != "completed" and item.due_datetime <= now
    ]
    groups: list[tuple[str, list[Any]]] = [
        ("🚨 Overdue", overdue),
        ("📍 Today", [item for item in active if item.due_datetime.astimezone(timezone).date() == today]),
        ("🌤 Tomorrow", [item for item in active if item.due_datetime.astimezone(timezone).date() == tomorrow]),
        ("📆 Next 7 days", [item for item in active if tomorrow < item.due_datetime.astimezone(timezone).date() <= week_end]),
        ("🗓 Later", [item for item in active if item.due_datetime.astimezone(timezone).date() > week_end]),
    ]
    if not any(items for _, items in groups):
        return "*☀️ Daily summary*\n\n_You’re all clear — no reminders need your attention\\._"
    lines = ["*☀️ Daily summary*", f"_{escape_markdown(timezone.key, version=2)}_"]
    for heading, items in groups:
        if not items:
            continue
        lines.extend(["", f"*{escape_markdown(heading, version=2)}*"])
        for reminder in items:
            due = reminder.due_datetime.astimezone(timezone)
            lines.append(
                f"• *{escape_markdown(reminder.text, version=2)}* — _{due:%a %H:%M}_"
            )
            _append_reminder_details(lines, reminder)
    return "\n".join(lines)


def build_application(
    token: str,
    service: ReminderService,
    settings_store: Any,
    conversation_store: Any,
    default_daily_time: str = "08:00",
    llm_interpreter: OpenAIIntentInterpreter | None = None,
) -> Application:
    if llm_interpreter is None:
        raise RuntimeError("An OpenAI intent interpreter is required.")
    application = Application.builder().token(token).build()

    def ensure_daily_default(user_id: int, chat_id: int) -> UserSettings:
        existing = settings_store.get(user_id)
        if existing is not None:
            return existing
        setting = UserSettings(
            user_id=user_id,
            chat_id=chat_id,
            daily_time=default_daily_time,
            daily_enabled=True,
        )
        settings_store.save(setting)
        return setting

    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user and update.effective_chat:
            ensure_daily_default(update.effective_user.id, update.effective_chat.id)
        if update.effective_message:
            await update.effective_message.reply_text(WELCOME_TEXT)

    async def retired_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_message:
            await update.effective_message.reply_text(
                "You don’t need commands with me anymore—just tell me what you’d "
                "like in your own words. For example: “What do I have coming up?”"
            )

    async def conversation_message(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (
            not update.effective_user
            or not update.effective_chat
            or not update.effective_message
            or not update.effective_message.text
        ):
            return
        user_id = update.effective_user.id
        message = update.effective_message.text
        user_setting = ensure_daily_default(user_id, update.effective_chat.id)
        user_timezone = get_timezone(user_setting.timezone)
        history = conversation_store.history(user_id)
        pending = conversation_store.confirmation(user_id)
        conversation_store.append(user_id, "user", message)

        async def respond(text: str, **kwargs: Any) -> None:
            await update.effective_message.reply_text(text, **kwargs)
            conversation_store.append(user_id, "assistant", text)

        if pending is not None:
            decision = confirmation_decision(message)
            if decision is False:
                conversation_store.clear_confirmation(user_id)
                await respond("Okay, cancelled — I didn’t delete anything.")
                return
            if decision is True:
                conversation_store.clear_confirmation(user_id)
                if pending.get("action") == "delete_all":
                    removed = service.store.delete_active_for_user(
                        user_id, datetime.now(service.timezone)
                    )
                    noun = "reminder" if len(removed) == 1 else "reminders"
                    await respond(f"🗑 Done — I cleared {len(removed)} active {noun}.")
                    return
                if pending.get("action") == "delete":
                    reminder_id = str(pending.get("reminder_id", ""))
                    deleted = service.store.delete_for_user(reminder_id, user_id)
                    if deleted:
                        await respond(f"🗑 Removed “{pending.get('title', 'reminder')}”.")
                    else:
                        await respond("That reminder is no longer active, so there was nothing to delete.")
                    return
            # A new request implicitly abandons the old confirmation.
            conversation_store.clear_confirmation(user_id)
        try:
            intent = await llm_interpreter.interpret(
                message, history=history, timezone=user_timezone
            )
        except LLMUnavailableError:
            LOGGER.exception("OpenAI intent interpretation failed")
            await respond(
                "I can’t reach the language service right now. Please try again shortly."
            )
            return
        except IntentInterpretationError as exc:
            try:
                question = await llm_interpreter.clarification(
                    message, history, str(exc)
                )
            except (IntentInterpretationError, LLMUnavailableError):
                question = "What reminder, date, or time would you like me to clarify?"
            await respond(question)
            return

        if intent.action == "create" and intent.due_at:
            try:
                reminder = service.create_at(
                    user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    due_at=intent.due_at,
                    text=intent.title,
                    recurrence_frequency=intent.recurrence_frequency,
                    recurrence_interval=intent.recurrence_interval,
                    recurrence_end_at=intent.recurrence_end_at,
                    note=intent.note,
                    checklist_items=intent.checklist_items,
                )
            except ReminderInputError as exc:
                await respond(f"⚠️ {exc}")
                return
            due = reminder.due_datetime.astimezone(user_timezone)
            await respond(
                "✅ Reminder created!\n\n"
                f"📌 {reminder.text}\n"
                f"📅 {due:%d %b %Y}\n"
                f"⏰ {due:%H:%M}"
                + (
                    f"\n🔁 Repeats {reminder.recurrence_frequency}"
                    if reminder.recurrence_frequency != "none"
                    else ""
                )
                + (
                    "\n🛑 Until "
                    f"{reminder.recurrence_end_datetime.astimezone(user_timezone):%d %b %Y}"
                    if reminder.recurrence_end_datetime is not None
                    else ""
                )
                + (f"\n📝 {reminder.note}" if reminder.note else "")
                + (
                    "\n" + "\n".join(f"☐ {item.text}" for item in reminder.checklist)
                    if reminder.checklist
                    else ""
                )
            )
            return

        if intent.action == "list":
            matching = None
            heading = "📋 Your reminders"
            if intent.title:
                matching = find_matching_reminders(service, user_id, intent.title)
                heading = f"📋 Reminders matching “{intent.title}”"
            await respond(
                format_reminder_list(
                    service,
                    user_id,
                    heading,
                    user_timezone,
                    reminders=matching,
                ),
                parse_mode="MarkdownV2",
            )
            return

        if intent.action == "old":
            await respond(
                format_old_reminder_list(service, user_id, user_timezone),
                parse_mode="MarkdownV2",
            )
            return

        if intent.action == "settings":
            setting = ensure_daily_default(
                update.effective_user.id, update.effective_chat.id
            )
            if setting.daily_enabled:
                reply = (
                    f"Your daily reminder is on and arrives at {setting.daily_time} "
                    f"({setting.timezone})."
                )
            else:
                reply = (
                    "Your daily reminder is currently turned off. "
                    f"Your timezone is {setting.timezone}."
                )
            await respond(reply)
            return

        if intent.action == "set_timezone":
            try:
                new_timezone = get_timezone(intent.timezone_name)
            except RuntimeError:
                await respond(
                    "I couldn’t identify that timezone. Try a city such as Singapore, "
                    "Tokyo, London, or New York."
                )
                return
            current = settings_store.get(user_id) or user_setting
            setting = UserSettings(
                user_id=user_id,
                chat_id=update.effective_chat.id,
                daily_time=current.daily_time,
                daily_enabled=current.daily_enabled,
                last_daily_sent_on=current.last_daily_sent_on,
                timezone=new_timezone.key,
            )
            settings_store.save(setting)
            await respond(f"✅ I’ll now use {new_timezone.key} for your reminders.")
            return

        if intent.action == "delete_all":
            active = service.store.list_for_user(user_id)
            if not active:
                await respond("Your active reminder list is already empty.")
                return
            conversation_store.set_confirmation(
                user_id, action="delete_all", count=len(active)
            )
            noun = "reminder" if len(active) == 1 else "reminders"
            await respond(
                f"This will permanently delete {len(active)} active {noun}. "
                "Reply yes to confirm or no to cancel."
            )
            return

        if intent.action == "set_daily":
            if intent.daily_enabled is False:
                current = settings_store.get(update.effective_user.id)
                setting = UserSettings(
                    user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    daily_time=current.daily_time if current else default_daily_time,
                    daily_enabled=False,
                    last_daily_sent_on=(current.last_daily_sent_on if current else None),
                    timezone=(current.timezone if current else user_timezone.key),
                )
                settings_store.save(setting)
                await respond("Daily reminders are disabled.")
                return
            try:
                parsed_time = parse_daily_time(intent.daily_time or "")
            except ReminderInputError:
                await respond(
                    "I understood the daily setting, but not its time. Try “Send my "
                    "daily reminders at 8:30am.”"
                )
                return
            current = settings_store.get(update.effective_user.id)
            setting = UserSettings(
                user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                daily_time=parsed_time.strftime("%H:%M"),
                daily_enabled=True,
                last_daily_sent_on=(current.last_daily_sent_on if current else None),
                timezone=(current.timezone if current else user_timezone.key),
            )
            settings_store.save(setting)
            await respond(
                f"✅ Daily reminders will be sent at {setting.daily_time} "
                f"({setting.timezone})."
            )
            return

        if intent.action == "chat":
            await respond(intent.reply)
            return

        if intent.action in {
            "delete", "complete", "update", "add_note",
            "add_checklist_item", "check_item",
        }:
            matches = find_matching_reminders(
                service, update.effective_user.id, intent.title
            )
            if not matches:
                await respond(
                    f'I couldn’t find an active reminder matching “{intent.title}”. '
                    "Which reminder did you mean?"
                )
                return
            if len(matches) > 1:
                choices = [
                    "I found several matching reminders. Tell me which one you mean "
                    "using its title or date:"
                ]
                choices.extend(
                    f"• {item.text} — "
                    f"{item.due_datetime.astimezone(user_timezone):%d %b}"
                    for item in matches
                )
                await respond("\n".join(choices))
                return

            reminder = matches[0]
            if intent.action == "delete":
                conversation_store.set_confirmation(
                    user_id,
                    action="delete",
                    reminder_id=reminder.id,
                    title=reminder.text,
                )
                await respond(
                    f"Permanently delete “{reminder.text}”? "
                    "Reply yes to confirm or no to cancel."
                )
                return
            elif intent.action == "complete":
                service.store.set_status_for_user(
                    reminder.id, update.effective_user.id, "completed"
                )
                reply = f"✅ Marked “{reminder.text}” as completed."
            elif intent.action == "add_note":
                updated = service.store.set_note_for_user(
                    reminder.id, user_id, intent.note
                )
                if updated is None:
                    await respond("I couldn’t update that reminder’s note.")
                    return
                reply = f"📝 Added the note to “{reminder.text}”."
            elif intent.action == "check_item":
                completed = service.store.complete_checklist_item_for_user(
                    reminder.id, user_id, intent.checklist_item
                )
                if completed is None:
                    await respond(
                        f'I couldn’t find one checklist item matching “{intent.checklist_item}”. '
                        "Which item did you complete?"
                    )
                    return
                reply = f"✅ Checked off “{completed[1]}” in “{reminder.text}”."
            elif intent.action == "add_checklist_item":
                updated = service.store.add_checklist_item_for_user(
                    reminder.id, user_id, intent.checklist_item
                )
                if updated is None:
                    await respond("I couldn’t add that checklist item.")
                    return
                reply = (
                    f"☐ Added “{intent.checklist_item}” to “{reminder.text}”."
                )
            else:
                try:
                    updated = service.update_deadline(
                        reminder=reminder,
                        user_id=update.effective_user.id,
                        due_at=intent.due_at,
                    )
                except ReminderInputError as exc:
                    await respond(f"⚠️ {exc}")
                    return
                reply = (
                    f"✅ I moved “{updated.text}” to "
                    f"{updated.due_datetime.astimezone(user_timezone):%d %b %Y at %H:%M}."
                )
            await respond(reply)
            return

        if intent.action == "unknown":
            await respond(intent.reply)
            return

        await respond(
            "What would you like me to clarify about the reminder or schedule?"
        )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.COMMAND, retired_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, conversation_message))

    return application
