from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationIntent


LOGGER = logging.getLogger(__name__)


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
        "set_timezone",
        "add_note",
        "add_checklist_item",
        "check_item",
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
    note: str | None = None
    checklist_items: list[str] = Field(default_factory=list)
    checklist_item: str | None = None
    timezone_name: str | None = None


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
        timezone: ZoneInfo | None = None,
    ) -> ConversationIntent:
        fast_intent = self._fast_intent(message)
        if fast_intent is not None:
            LOGGER.info("Intent fast path action=%s input_tokens=0", fast_intent.action)
            return fast_intent
        effective_timezone = timezone or self.timezone
        current = now or datetime.now(effective_timezone)
        try:
            output = await asyncio.to_thread(
                self._request, message, current, history or [], effective_timezone
            )
        except OpenAIError as exc:
            raise LLMUnavailableError("OpenAI request failed") from exc
        return self._to_intent(output, current, effective_timezone)

    @staticmethod
    def _fast_intent(message: str) -> ConversationIntent | None:
        """Resolve only exact, context-free requests without an OpenAI call."""
        normalized = " ".join(re.findall(r"[a-z0-9]+", message.lower()))

        named_delete = re.fullmatch(
            r"(?:please )?(?:delete|remove) my (.+?) reminder(?: please)?",
            normalized,
        )
        if named_delete is not None:
            return ConversationIntent(action="delete", title=named_delete.group(1))

        filtered_list = re.fullmatch(
            r"(?:what (.+?) reminders are there|(?:show|tell|list)(?: me)? "
            r"(?:all )?my (.+?) reminders)(?: please)?",
            normalized,
        )
        if filtered_list is not None:
            topic = filtered_list.group(1) or filtered_list.group(2)
            return ConversationIntent(action="list", title=topic)

        patterns: tuple[tuple[str, str], ...] = (
            (
                "list",
                r"(?:please )?(?:(?:show|tell|list)(?: me)?(?: all)?(?: of)? my "
                r"(?:active |upcoming )?reminders|what (?:are|is in) my reminders|"
                r"what do i have coming up|show me my schedule)(?: please)?",
            ),
            (
                "old",
                r"(?:please )?(?:show|tell|list)(?: me)? my (?:old|past|expired) "
                r"reminders(?: please)?|what did i miss|what reminders have passed",
            ),
            (
                "settings",
                r"(?:please )?(?:show|tell) me my (?:reminder )?settings(?: please)?|"
                r"what time is my daily (?:reminder|summary)|when is my daily reminder|"
                r"are daily reminders on|what timezone am i using",
            ),
            (
                "delete_all",
                r"(?:please )?(?:delete|clear|remove|empty|wipe)(?: all| every)? "
                r"(?:of )?my (?:active )?reminders(?: please)?",
            ),
        )
        for action, pattern in patterns:
            if re.fullmatch(pattern, normalized):
                return ConversationIntent(action=action)

        chat_replies = {
            "hi": "Hi! What can I help you remember? 😊",
            "hello": "Hello! What can I help you remember? 😊",
            "hey": "Hey! What can I help you remember? 😊",
            "good morning": "Good morning! What can I help you remember today? ☀️",
            "good afternoon": "Good afternoon! What can I help you remember?",
            "good evening": "Good evening! What can I help you remember?",
            "thanks": "You’re welcome! 😊",
            "thank you": "You’re welcome! 😊",
            "how are you": "I’m doing well—and ready to keep you on schedule. 😊",
        }
        reply = chat_replies.get(normalized)
        if reply is not None:
            return ConversationIntent(action="chat", reply=reply)
        return None

    def _request(
        self,
        message: str,
        current: datetime,
        history: list[dict[str, str]],
        timezone: ZoneInfo,
    ) -> IntentOutput:
        instructions = self._instructions(current, timezone)
        selected_history = self._select_history(message, history)
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": instructions},
                {
                    "role": "system",
                    "content": (
                        f"Now: {current.isoformat()}. User timezone: {timezone.key}."
                    ),
                },
                *selected_history,
                {"role": "user", "content": message},
            ],
            text_format=IntentOutput,
            prompt_cache_key="remi-intent-v2",
        )
        self._log_usage(response, len(selected_history))
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
                *history[-8:],
                {"role": "user", "content": message},
            ],
            text_format=ClarificationOutput,
            prompt_cache_key="remi-clarification-v1",
        )
        self._log_usage(response, min(len(history), 8))
        if response.output_parsed is None:
            raise IntentInterpretationError("The model did not return a clarification")
        return response.output_parsed

    def _instructions(
        self, current: datetime, timezone: ZoneInfo | None = None
    ) -> str:
        return (
            "You classify natural messages for Remi; never execute actions. Actions: "
            "create=new reminder; update=change named reminder deadline; delete=remove "
            "one; delete_all=clear every active reminder; complete=finish one; list=active "
            "list; old=past list; set_daily=enable/change/disable the full-list daily "
            "summary; settings=show summary/timezone settings; set_timezone=change user "
            "timezone; add_note; add_checklist_item; check_item; chat; unknown. Resolve "
            "follow-ups from supplied history (for example 'make that 5pm') but never "
            "invent missing context.\n"
            "DATES: Resolve due_at as ISO 8601 in the supplied timezone. Date-only => "
            "23:59; time-only => next occurrence. For create/update, trigger_phrase is the "
            "exact scheduling phrase. Keep dates describing the subject in title. Never "
            "silently drop meaningful subject details. Example: 'send me a reminder "
            "tomorrow to book Chiikawa tickets for 22 Aug' => create, "
            "trigger_phrase='tomorrow', title='Book Chiikawa tickets for 22 Aug'. 'remind "
            "me on 22 Aug to book tickets' uses 22 Aug as delivery. For update, title is "
            "only the existing subject: 'change IDP assignment to 24 Aug' => update, "
            "title='IDP assignment', trigger_phrase='24 Aug'. If timing is genuinely "
            "ambiguous, use unknown rather than guessing.\n"
            "RECURRENCE: 'remind me every day/week/month/year' is create and must never "
            "use set_daily. Set recurrence_frequency, interval, first due_at, and optional "
            "recurrence_end_at. Other actions use frequency=none, interval=1, null end. "
            "set_daily applies only to the user's summary/digest/list delivery time.\n"
            "FIELDS: create may extract note and bullet/numbered checklist_items. add_note, "
            "add_checklist_item and check_item require a named reminder and their matching "
            "field. set_timezone returns an IANA name. set_daily returns daily_enabled and "
            "HH:MM when enabled. delete/complete use title as the search phrase. delete_all "
            "requires explicit all/every/clear/empty/wipe and an empty title.\n"
            "LANGUAGE: tell/show/read/list/describe my reminders, including 'tell me my "
            "reminders', 'show me my reminders', 'what are my reminders?', and 'what is "
            "in my reminders?' => list with empty title. Topic-filtered requests such as "
            "'what Kahoot reminders are there?' or 'show all my Kahoot reminders' => list "
            "with title='Kahoot'. 'delete my Kahoot reminder' => delete with title='Kahoot'; "
            "do not ask for a date or more detail merely because the title is broad. "
            "'what did I miss?' => old. 'when is my daily "
            "reminder?' or timezone/status questions => settings. Greetings, thanks and "
            "capability questions => chat. For chat, reply as warm, calm, lightly playful "
            "Remi in at most 3 short sentences, focused on reminders. unknown asks exactly "
            "one focused clarification question. All other actions have empty reply. Use "
            "null for irrelevant nullable fields and do a final completeness check."
        )

    @staticmethod
    def _select_history(
        message: str, history: list[dict[str, str]], limit: int = 8
    ) -> list[dict[str, str]]:
        """Only pay for history when the latest message appears context-dependent."""
        normalized = message.lower().strip()
        referential = re.search(
            r"\b(it|that|this|those|them|one|ones|previous|earlier|instead|actually)\b",
            normalized,
        )
        follow_up = re.match(
            r"^(make|move|change|reschedule|delete|remove|complete|mark|add|yes|no)\b",
            normalized,
        )
        fragment = len(normalized.split()) <= 5 and bool(
            re.search(r"\b(today|tomorrow|tonight|next|at|am|pm|morning|evening)\b|\d", normalized)
        )
        if not (referential or follow_up or fragment):
            return []
        return history[-limit:]

    @staticmethod
    def _log_usage(response: object, history_turns: int) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "input_tokens_details", None)
        LOGGER.info(
            "OpenAI usage input=%s cached=%s output=%s history_turns=%s",
            getattr(usage, "input_tokens", None),
            getattr(details, "cached_tokens", 0) if details else 0,
            getattr(usage, "output_tokens", None),
            history_turns,
        )

    def _to_intent(
        self,
        value: IntentOutput,
        current: datetime | None = None,
        timezone: ZoneInfo | None = None,
    ) -> ConversationIntent:
        effective_timezone = timezone or self.timezone
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
                due_at = due_at.replace(tzinfo=effective_timezone)
            else:
                due_at = due_at.astimezone(effective_timezone)
        recurrence_end_at = None
        if value.recurrence_end_at:
            try:
                recurrence_end_at = datetime.fromisoformat(
                    value.recurrence_end_at.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise IntentInterpretationError("The recurrence end was invalid") from exc
            if recurrence_end_at.tzinfo is None:
                recurrence_end_at = recurrence_end_at.replace(tzinfo=effective_timezone)
            else:
                recurrence_end_at = recurrence_end_at.astimezone(effective_timezone)
        if due_at is not None and trigger_phrase:
            due_at = _correct_named_weekday(
                trigger_phrase,
                current or datetime.now(effective_timezone),
                due_at,
                effective_timezone,
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
        timezone_name = (value.timezone_name or "").strip()
        if value.action == "set_timezone" and not timezone_name:
            raise IntentInterpretationError("The timezone is missing")
        note = (value.note or "").strip()
        checklist_item = (value.checklist_item or "").strip()
        if value.action == "add_note" and (not title or not note):
            raise IntentInterpretationError("Adding a note needs a reminder and note text")
        if value.action == "check_item" and (not title or not checklist_item):
            raise IntentInterpretationError(
                "Completing a checklist item needs a reminder and item"
            )
        if value.action == "add_checklist_item" and (not title or not checklist_item):
            raise IntentInterpretationError(
                "Adding a checklist item needs a reminder and item"
            )
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
            note=note,
            checklist_items=tuple(item.strip() for item in value.checklist_items if item.strip()),
            checklist_item=checklist_item,
            timezone_name=timezone_name,
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
