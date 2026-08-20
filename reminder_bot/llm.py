from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationIntent


class IntentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "create",
        "delete",
        "delete_all",
        "complete",
        "update",
        "list",
        "old",
        "set_daily",
        "settings",
        "chat",
        "unknown",
    ]
    title: str
    due_at: str | None
    trigger_phrase: str | None
    daily_time: str | None
    daily_enabled: bool | None
    reply: str
    recurrence_frequency: Literal["none", "daily", "weekly", "monthly", "yearly"]
    recurrence_interval: int = Field(ge=1, le=365)
    recurrence_end_at: str | None
    confidence: float = Field(ge=0, le=1)


class ClarificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class IntentInterpretationError(RuntimeError):
    """The message could not safely be converted into an action."""


class LLMUnavailableError(IntentInterpretationError):
    """The hosted language service could not be reached."""


class OpenAIIntentInterpreter:
    """Translate natural language into data; never execute reminder operations."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timezone: ZoneInfo,
        timeout_seconds: float = 20,
        minimum_confidence: float = 0.65,
    ) -> None:
        self.model = model
        self.timezone = timezone
        self.minimum_confidence = minimum_confidence
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )

    async def interpret(
        self,
        message: str,
        now: datetime | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ConversationIntent:
        current = now or datetime.now(self.timezone)
        try:
            output = await asyncio.to_thread(
                self._request, message, current, history or []
            )
        except OpenAIError as exc:
            raise LLMUnavailableError("OpenAI request failed") from exc
        return self._to_intent(output, current)

    def _request(
        self,
        message: str,
        current: datetime,
        history: list[dict[str, str]],
    ) -> IntentOutput:
        instructions = self._instructions(current)
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": instructions},
                *history,
                {"role": "user", "content": message},
            ],
            text_format=IntentOutput,
        )
        if response.output_parsed is None:
            raise IntentInterpretationError("The model did not return a usable intent")
        return response.output_parsed

    async def clarification(
        self,
        message: str,
        history: list[dict[str, str]],
        reason: str,
    ) -> str:
        try:
            output = await asyncio.to_thread(
                self._clarification_request, message, history, reason
            )
        except OpenAIError as exc:
            raise LLMUnavailableError("OpenAI clarification request failed") from exc
        question = output.question.strip()
        if not question:
            raise IntentInterpretationError("The clarification was empty")
        return question

    def _clarification_request(
        self,
        message: str,
        history: list[dict[str, str]],
        reason: str,
    ) -> ClarificationOutput:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are Remi, a reminder assistant. The latest request could "
                        "not be executed safely. Ask exactly one short, specific question "
                        "that obtains only the missing or ambiguous reminder detail. "
                        "Use recent conversation context. Do not claim an action occurred. "
                        f"Validation reason: {reason}"
                    ),
                },
                *history,
                {"role": "user", "content": message},
            ],
            text_format=ClarificationOutput,
        )
        if response.output_parsed is None:
            raise IntentInterpretationError("The model did not return a clarification")
        return response.output_parsed

    def _instructions(self, current: datetime) -> str:
        return (
            "Classify the user's message for a reminder bot. Do not execute actions. "
            "Interpret natural, casual English by meaning rather than requiring command "
            "words. Use recent conversation turns to resolve follow-ups such as 'make "
            "that 5pm', 'move it to tomorrow', or 'delete that one', but never invent "
            "context that is absent. Use create to add a reminder; delete to permanently remove one named "
            "reminder; delete_all to remove every active reminder belonging to the user; "
            "complete to mark one named reminder completed; update to change the date or "
            "time of one named reminder; list to show upcoming active "
            "reminders; old to show reminders past their deadline; set_daily to enable, "
            "change, or disable the daily summary; settings to show the current daily "
            "summary time and whether it is enabled. "
            "Repeating reminders are create actions with recurrence_frequency set to "
            "daily, weekly, monthly, or yearly and recurrence_interval set to the number "
            "of those units between occurrences. due_at is the first occurrence. Resolve "
            "recurrence_end_at to ISO 8601 when the user supplies an ending condition such "
            "as 'for the next 3 years'; otherwise it is null. A normal one-time reminder "
            "uses recurrence_frequency='none', recurrence_interval=1, and a null end. "
            "Every action other than create must also use recurrence_frequency='none', "
            "recurrence_interval=1, and a null recurrence_end_at. "
            "CRITICAL DISTINCTION: set_daily only changes the user's summary of their full "
            "reminder list and requires words such as summary, digest, list, or 'my daily "
            "reminder time'. Wording such as 'remind me every day', 'every week', 'weekly', "
            "or 'each month' creates one repeating reminder and must never use set_daily. "
            "Example: 'remind me to text Lee Hongliang at 12pm every day for the next 3 "
            "years' means create, title='Text Lee Hongliang', due_at is the next 12:00, "
            "recurrence_frequency='daily', recurrence_interval=1, and recurrence_end_at "
            "is three years from now. 'Remind me at 12pm at least once every week to update "
            "him about my lunch for the next 3 years' is a weekly repeating create action, "
            "not a daily-summary setting. "
            "Use chat for greetings, thanks, farewells, introductions, simple small talk, "
            "or questions about what this reminder assistant can do. You are Remi, a "
            "warm, calm, lightly playful personal reminder assistant. For chat, write a "
            "natural reply of at most three short sentences in reply. Stay focused on "
            "reminders and scheduling; if asked for unrelated knowledge or unsupported "
            "actions, respond briefly and steer back to what you can do. Never claim an "
            "action happened when action is chat. For every non-chat action except unknown, reply must "
            "be an empty string because the application will acknowledge it after it is "
            "safely completed. Examples: 'hi', 'hello', 'hey there', 'good morning', 'thanks', "
            "'how are you?', and 'what can you do?' mean chat. "
            "Questions such as 'when is my daily reminder?', 'what time is my daily "
            "summary?', and 'are daily reminders on?' mean settings. Use unknown when a "
            "supported request is missing or has conflicting details. For unknown, put "
            "one short and focused clarification question in reply. "
            "A request to tell, show, read, list, or describe the user's reminders means "
            "list even when it is phrased as a short question or does not contain the word "
            "list. Exact examples 'tell me my reminders', 'show me my reminders', 'what "
            "are my reminders?', 'what is in my reminders?', 'list my reminders', 'what do "
            "I have coming up?', 'show my schedule', 'anything on my reminder list?', and "
            "'what have I got planned?' all mean list with high confidence. These are not "
            "chat, settings, or unknown. 'Show my old reminders', 'what did I miss?', and "
            "'what reminders have passed?' mean old. "
            "Only use delete_all when the user explicitly applies a comprehensive word "
            "such as all, every, clear, empty, or wipe to their reminders. Examples: "
            "'delete all my reminders', 'clear my reminders', 'empty my reminder list', "
            "and 'wipe every upcoming reminder' mean delete_all. For delete_all, title "
            "must be empty. A request to remove a named topic means delete, not delete_all. "
            "For update, title is only the existing reminder's identifying subject, due_at "
            "is the new deadline, and trigger_phrase is the user's exact new date/time "
            "phrase. 'Change the deadline for IDP assignment to 24 Aug' means update, "
            "title='IDP assignment', trigger_phrase='24 Aug', and due_at is 24 Aug at "
            "23:59. 'Move my dentist reminder to next Monday at 2pm' also means update. "
            "For create, separately extract (1) the delivery "
            "trigger into trigger_phrase and due_at and (2) the complete reminder subject "
            "into title. Resolve due_at to ISO 8601. A date without a time defaults to "
            "23:59. A time without a date means its next occurrence. "
            "A scheduling expression attached to 'remind me', such as 'tomorrow', "
            "'in 50 minutes', or 'at 4pm', controls delivery. A date that describes the "
            "thing being remembered must remain in the title, especially a date after "
            "'for', 'about', 'that', or the task object. Never silently drop meaningful "
            "subject details from the title. If two delivery times are genuinely plausible, "
            "use unknown with low confidence instead of guessing. Examples: "
            "'send me a reminder tomorrow to book Chiikawa tickets for 22 Aug' means "
            "trigger_phrase='tomorrow', due_at is tomorrow at 23:59, and title is "
            "'Book Chiikawa tickets for 22 Aug'. "
            "'remind me on 22 Aug to book Chiikawa tickets' means trigger_phrase='on 22 "
            "Aug', due_at is 22 Aug at 23:59, and title is 'Book Chiikawa tickets'. "
            "'remind me tomorrow at 4pm that the concert is on Friday' means the trigger "
            "is tomorrow at 4pm and the title keeps 'The concert is on Friday'. "
            "For delete/complete, title is the search "
            "phrase. For set_daily, return daily_enabled and return daily_time as HH:MM "
            "when enabling. Before returning, re-check that the selected action captures "
            "the user's whole request and that no reminder details were lost. Reply is "
            "always a string. Use null for nullable fields irrelevant to the action, "
            "including trigger_phrase for actions other than create or update. Do not invent "
            "missing dates, titles, or intentions. "
            f"Current datetime: {current.isoformat()}. Timezone: {self.timezone.key}."
        )

    def _to_intent(
        self, value: IntentOutput, current: datetime | None = None
    ) -> ConversationIntent:
        # Read-only requests are safe to honor at a lower confidence than mutations.
        minimum_confidence = (
            0.4
            if value.action in {"chat", "list", "old", "settings", "unknown"}
            else self.minimum_confidence
        )
        if not math.isfinite(value.confidence) or value.confidence < minimum_confidence:
            raise IntentInterpretationError("The request was too ambiguous")

        title = value.title.strip()
        trigger_phrase = (
            value.trigger_phrase.strip() if value.trigger_phrase is not None else None
        )
        due_at = None
        if value.due_at:
            try:
                due_at = datetime.fromisoformat(value.due_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise IntentInterpretationError("The deadline was invalid") from exc
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=self.timezone)
            else:
                due_at = due_at.astimezone(self.timezone)
        recurrence_end_at = None
        if value.recurrence_end_at:
            try:
                recurrence_end_at = datetime.fromisoformat(
                    value.recurrence_end_at.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise IntentInterpretationError("The recurrence end was invalid") from exc
            if recurrence_end_at.tzinfo is None:
                recurrence_end_at = recurrence_end_at.replace(tzinfo=self.timezone)
            else:
                recurrence_end_at = recurrence_end_at.astimezone(self.timezone)
        if due_at is not None and trigger_phrase:
            due_at = _correct_named_weekday(
                trigger_phrase,
                current or datetime.now(self.timezone),
                due_at,
                self.timezone,
            )

        if value.action == "create" and (
            not title or due_at is None or not trigger_phrase
        ):
            raise IntentInterpretationError(
                "The reminder needs a distinct title and delivery time"
            )
        if value.action == "create" and value.recurrence_frequency != "none":
            if recurrence_end_at is not None and recurrence_end_at < due_at:
                raise IntentInterpretationError(
                    "The repeating reminder ends before its first occurrence"
                )
        if value.action in {"delete", "complete"} and not title:
            raise IntentInterpretationError("The reminder description is missing")
        if value.action == "update" and (
            not title or due_at is None or not trigger_phrase
        ):
            raise IntentInterpretationError(
                "Updating a reminder needs its title and a new deadline"
            )
        if value.action == "set_daily":
            if value.daily_enabled is None:
                raise IntentInterpretationError("The daily setting is incomplete")
            if value.daily_enabled and not value.daily_time:
                raise IntentInterpretationError("The daily reminder time is missing")
        reply = value.reply.strip()
        if value.action == "chat" and not reply:
            raise IntentInterpretationError("The conversational reply is missing")
        if value.action == "unknown" and not reply:
            raise IntentInterpretationError("The clarification question is missing")

        return ConversationIntent(
            action=value.action,
            title=title,
            due_at=due_at,
            trigger_phrase=trigger_phrase,
            daily_time=value.daily_time,
            daily_enabled=value.daily_enabled,
            reply=reply,
            recurrence_frequency=value.recurrence_frequency,
            recurrence_interval=value.recurrence_interval,
            recurrence_end_at=recurrence_end_at,
        )


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _correct_named_weekday(
    trigger_phrase: str,
    current: datetime,
    model_due: datetime,
    timezone: ZoneInfo,
) -> datetime:
    """Resolve named weekdays locally instead of trusting model date arithmetic."""
    match = re.search(
        r"\b(?:(this|next|coming)\s+)?"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        trigger_phrase,
        re.IGNORECASE,
    )
    if match is None:
        return model_due

    local_now = current.astimezone(timezone)
    qualifier = (match.group(1) or "").lower()
    target_weekday = _WEEKDAYS[match.group(2).lower()]
    days_ahead = (target_weekday - local_now.weekday()) % 7
    if days_ahead == 0 and (
        qualifier == "next" or model_due.time() <= local_now.time()
    ):
        days_ahead = 7
    target_date = (local_now + timedelta(days=days_ahead)).date()
    return datetime.combine(target_date, model_due.time(), timezone)
