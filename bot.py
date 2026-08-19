from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

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


def format_reminder_list(service: ReminderService, user_id: int, title: str) -> str:
    reminders = service.store.list_for_user(user_id)
    if not reminders:
        return "_You have no active reminders\\._"

    lines = [f"*{escape_markdown(title, version=2)}*"]
    for index, reminder in enumerate(reminders, start=1):
        due = reminder.due_datetime
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
                repeat += f" until {reminder.recurrence_end_datetime:%d %b %Y}"
            repeat += "_"
            lines.append(repeat)
    return "\n".join(lines)


def format_old_reminder_list(service: ReminderService, user_id: int) -> str:
    reminders = service.store.list_old_for_user(user_id)
    if not reminders:
        return "_You have no old reminders\\._"

    lines = [
        "*🗃 Your old reminders*",
        "_Automatically removed 7 days after the deadline\\._",
    ]
    for index, reminder in enumerate(reminders, start=1):
        due = reminder.due_datetime
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
    return "\n".join(lines)


def build_application(
    token: str,
    service: ReminderService,
    settings_store: Any,
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
        ensure_daily_default(update.effective_user.id, update.effective_chat.id)
        try:
            intent = await llm_interpreter.interpret(update.effective_message.text)
        except LLMUnavailableError:
            LOGGER.exception("OpenAI intent interpretation failed")
            await update.effective_message.reply_text(
                "I can’t reach the language service right now. Please try again shortly."
            )
            return
        except IntentInterpretationError:
            await update.effective_message.reply_text(
                "I’m not confident I understood that. Please rephrase it with the "
                "reminder, date, and time you want."
            )
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
                )
            except ReminderInputError as exc:
                await update.effective_message.reply_text(f"⚠️ {exc}")
                return
            due = reminder.due_datetime
            await update.effective_message.reply_text(
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
                    f"\n🛑 Until {reminder.recurrence_end_datetime:%d %b %Y}"
                    if reminder.recurrence_end_datetime is not None
                    else ""
                )
            )
            return

        if intent.action == "list":
            await update.effective_message.reply_text(
                format_reminder_list(
                    service, update.effective_user.id, "📋 Your reminders"
                ),
                parse_mode="MarkdownV2",
            )
            return

        if intent.action == "old":
            await update.effective_message.reply_text(
                format_old_reminder_list(service, update.effective_user.id),
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
                    f"({service.timezone.key})."
                )
            else:
                reply = "Your daily reminder is currently turned off."
            await update.effective_message.reply_text(reply)
            return

        if intent.action == "delete_all":
            removed = service.store.delete_active_for_user(
                update.effective_user.id, datetime.now(service.timezone)
            )
            if removed:
                noun = "reminder" if len(removed) == 1 else "reminders"
                await update.effective_message.reply_text(
                    f"🗑 Done — I cleared {len(removed)} active {noun}."
                )
            else:
                await update.effective_message.reply_text(
                    "Your active reminder list is already empty."
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
                )
                settings_store.save(setting)
                await update.effective_message.reply_text("Daily reminders are disabled.")
                return
            try:
                parsed_time = parse_daily_time(intent.daily_time or "")
            except ReminderInputError:
                await update.effective_message.reply_text(
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
            )
            settings_store.save(setting)
            await update.effective_message.reply_text(
                f"✅ Daily reminders will be sent at {setting.daily_time} "
                f"({service.timezone.key})."
            )
            return

        if intent.action == "chat":
            await update.effective_message.reply_text(intent.reply)
            return

        if intent.action in {"delete", "complete", "update"}:
            matches = find_matching_reminders(
                service, update.effective_user.id, intent.title
            )
            if not matches:
                await update.effective_message.reply_text(
                    f'I could not find an active reminder matching “{intent.title}”.'
                )
                return
            if len(matches) > 1:
                choices = [
                    "I found several matching reminders. Tell me which one you mean "
                    "using its title or date:"
                ]
                choices.extend(
                    f"• {item.text} — {item.due_datetime:%d %b}"
                    for item in matches
                )
                await update.effective_message.reply_text("\n".join(choices))
                return

            reminder = matches[0]
            if intent.action == "delete":
                service.store.delete_for_user(reminder.id, update.effective_user.id)
                reply = f"🗑 Removed “{reminder.text}”."
            elif intent.action == "complete":
                service.store.set_status_for_user(
                    reminder.id, update.effective_user.id, "completed"
                )
                reply = f"✅ Marked “{reminder.text}” as completed."
            else:
                try:
                    updated = service.update_deadline(
                        reminder=reminder,
                        user_id=update.effective_user.id,
                        due_at=intent.due_at,
                    )
                except ReminderInputError as exc:
                    await update.effective_message.reply_text(f"⚠️ {exc}")
                    return
                reply = (
                    f"✅ I moved “{updated.text}” to "
                    f"{updated.due_datetime:%d %b %Y at %H:%M}."
                )
            await update.effective_message.reply_text(reply)
            return

        await update.effective_message.reply_text(
            "I didn’t quite catch that. I’m best at reminders and schedules—try "
            "telling me what to remember and when, or ask what you have coming up."
        )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.COMMAND, retired_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, conversation_message))

    return application
