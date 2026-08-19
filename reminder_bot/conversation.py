from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

from .models import Reminder
from .service import ReminderService


@dataclass(frozen=True, slots=True)
class ConversationIntent:
    action: str
    title: str = ""
    due_at: datetime | None = None
    trigger_phrase: str | None = None
    daily_time: str | None = None
    daily_enabled: bool | None = None
    reply: str = ""
    recurrence_frequency: str = "none"
    recurrence_interval: int = 1
    recurrence_end_at: datetime | None = None


def find_matching_reminders(
    service: ReminderService,
    user_id: int,
    query: str,
) -> list[Reminder]:
    """Match only the requesting user's upcoming reminders."""
    normalized_query = _normalize(query)
    if not normalized_query:
        return []

    query_words = set(normalized_query.split())
    scored: list[tuple[float, Reminder]] = []
    for reminder in service.store.list_for_user(user_id):
        candidate = _normalize(reminder.text)
        candidate_words = set(candidate.split())
        contains = normalized_query in candidate or candidate in normalized_query
        overlap = len(query_words & candidate_words) / max(len(query_words), 1)
        similarity = SequenceMatcher(None, normalized_query, candidate).ratio()
        score = max(
            1.0 if candidate == normalized_query else 0.0,
            0.9 if contains else 0.0,
            overlap,
            similarity,
        )
        if score >= 0.55:
            scored.append((score, reminder))

    scored.sort(key=lambda item: (-item[0], item[1].due_datetime))
    if not scored:
        return []
    best = scored[0][0]
    return [reminder for score, reminder in scored if score >= best - 0.08]


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
