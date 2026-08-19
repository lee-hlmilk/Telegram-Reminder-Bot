from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

from reminder_bot.conversation import find_matching_reminders
from reminder_bot.llm import (
    IntentInterpretationError,
    LLMUnavailableError,
    OpenAIIntentInterpreter,
)
from reminder_bot.models import Reminder, UserSettings
from reminder_bot.service import (
    ReminderInputError,
    ReminderService,
    get_timezone,
    parse_daily_time,
)
from reminder_bot.storage import JsonReminderStore, JsonUserSettingsStore

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
    settings_store: JsonUserSettingsStore,
    default_daily_time: str = "08:00",
    llm_interpreter: OpenAIIntentInterpreter | None = None,
) -> Application:
    if llm_interpreter is None:
        raise RuntimeError("An OpenAI intent interpreter is required.")
    async def register_command_menu(application: Application) -> None:
        await application.bot.set_my_commands(BOT_COMMANDS)

    application = (
        Application.builder()
        .token(token)
        .post_init(register_command_menu)
        .build()
    )

    async def daily_digest(context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = int(context.job.data["user_id"])
        setting = settings_store.get(user_id)
        if not setting or not setting.daily_enabled:
            return
        await context.bot.send_message(
            chat_id=setting.chat_id,
            text=format_reminder_list(service, user_id, "☀️ Your daily reminders"),
            parse_mode="MarkdownV2",
        )

    async def deadline_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = int(context.job.data["user_id"])
        reminder_id = str(context.job.data["reminder_id"])
        reminder = service.store.get_for_user(reminder_id, user_id)
        if reminder is None:
            return
        if reminder.status == "active":
            await context.bot.send_message(
                chat_id=reminder.chat_id,
                text=f"🔔 Reminder\n\n📌 {reminder.text}",
            )
            next_reminder = service.advance_recurrence(reminder)
            if next_reminder is not None:
                schedule_deadline(next_reminder)
            else:
                service.store.set_status_for_user(reminder.id, user_id, "expired")

    async def purge_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = int(context.job.data["user_id"])
        reminder_id = str(context.job.data["reminder_id"])
        reminder = service.store.get_for_user(reminder_id, user_id)
        if reminder is None:
            return
        if reminder.due_datetime + timedelta(days=7) <= datetime.now(service.timezone):
            service.store.delete_for_user(reminder_id, user_id)

    def schedule_deadline(reminder: Reminder) -> None:
        if application.job_queue is None:
            raise RuntimeError("JobQueue is unavailable. Install the job-queue dependency.")
        job_name = f"reminder:{reminder.id}"
        for job in application.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        if reminder.due_datetime <= datetime.now(service.timezone):
            return
        application.job_queue.run_once(
            deadline_reminder,
            when=reminder.due_datetime,
            name=job_name,
            data={"user_id": reminder.user_id, "reminder_id": reminder.id},
        )

    def schedule_purge(reminder: Reminder) -> None:
        if application.job_queue is None:
            raise RuntimeError("JobQueue is unavailable. Install the job-queue dependency.")
        if (
            reminder.recurrence_frequency != "none"
            and reminder.recurrence_end_datetime is None
        ):
            return
        purge_base = reminder.recurrence_end_datetime or reminder.due_datetime
        purge_at = purge_base + timedelta(days=7)
        if purge_at <= datetime.now(service.timezone):
            return
        job_name = f"purge:{reminder.id}"
        for job in application.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        application.job_queue.run_once(
            purge_reminder,
            when=purge_at,
            name=job_name,
            data={"user_id": reminder.user_id, "reminder_id": reminder.id},
        )

    def cancel_reminder_jobs(reminder: Reminder) -> None:
        if application.job_queue is None:
            return
        for job_name in (f"reminder:{reminder.id}", f"purge:{reminder.id}"):
            for job in application.job_queue.get_jobs_by_name(job_name):
                job.schedule_removal()

    def schedule_daily(setting: UserSettings) -> None:
        if application.job_queue is None:
            raise RuntimeError("JobQueue is unavailable. Install the job-queue dependency.")
        job_name = f"daily-digest:{setting.user_id}"
        for job in application.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        if setting.daily_enabled:
            delivery_time = parse_daily_time(setting.daily_time).replace(
                tzinfo=service.timezone
            )
            application.job_queue.run_daily(
                daily_digest,
                time=delivery_time,
                name=job_name,
                data={"user_id": setting.user_id},
            )

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
        schedule_daily(setting)
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
            schedule_deadline(reminder)
            schedule_purge(reminder)
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
            for reminder in removed:
                cancel_reminder_jobs(reminder)
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
                )
                settings_store.save(setting)
                schedule_daily(setting)
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
            setting = UserSettings(
                user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                daily_time=parsed_time.strftime("%H:%M"),
                daily_enabled=True,
            )
            settings_store.save(setting)
            schedule_daily(setting)
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
                    f"• {item.text} — {item.due_datetime:%d %b} — {item.id}"
                    for item in matches
                )
                await update.effective_message.reply_text("\n".join(choices))
                return

            reminder = matches[0]
            if intent.action == "delete":
                service.store.delete_for_user(reminder.id, update.effective_user.id)
                cancel_reminder_jobs(reminder)
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
                cancel_reminder_jobs(reminder)
                schedule_deadline(updated)
                schedule_purge(updated)
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

    # Migrate existing reminder owners who predate per-user settings. Explicit
    # settings, including disabled ones, are never overwritten.
    for reminder in service.store.list_active():
        if settings_store.get(reminder.user_id) is None:
            settings_store.save(
                UserSettings(
                    user_id=reminder.user_id,
                    chat_id=reminder.chat_id,
                    daily_time=default_daily_time,
                    daily_enabled=True,
                )
            )

    for setting in settings_store.list_enabled():
        schedule_daily(setting)
    service.store.purge_old(datetime.now(service.timezone))
    for reminder in service.store.list_active():
        schedule_deadline(reminder)
        schedule_purge(reminder)
    return application


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env and add it.")

    store_path = Path(os.getenv("REMINDERS_FILE", "data/reminders.json"))
    settings_path = Path(os.getenv("USER_SETTINGS_FILE", "data/user_settings.json"))
    timezone = get_timezone(os.getenv("BOT_TIMEZONE", "Asia/Singapore"))
    default_daily_time = os.getenv("DEFAULT_DAILY_TIME", "08:00").strip()
    parse_daily_time(default_daily_time)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment.")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    llm_interpreter = OpenAIIntentInterpreter(
        api_key=openai_api_key,
        model=openai_model,
        timezone=timezone,
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20")),
    )
    LOGGER.info("OpenAI natural-language interpretation enabled: %s", openai_model)
    service = ReminderService(JsonReminderStore(store_path), timezone)
    LOGGER.info("Reminder bot is running")
    build_application(
        token,
        service,
        JsonUserSettingsStore(settings_path),
        default_daily_time,
        llm_interpreter,
    ).run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
