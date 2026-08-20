from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from .models import Reminder, UserSettings


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _reminder_from_document(document: Any) -> Reminder:
    value = document.to_dict() or {}
    due_at = _as_datetime(value["due_at"])
    created_at = _as_datetime(value["created_at"])
    recurrence_end = _as_datetime(value.get("recurrence_end_at"))
    if due_at is None or created_at is None:
        raise RuntimeError(f"Reminder {document.id} has invalid timestamps")
    return Reminder(
        id=document.id,
        user_id=int(value["user_id"]),
        chat_id=int(value["chat_id"]),
        text=str(value["text"]),
        due_at=due_at.isoformat(),
        created_at=created_at.isoformat(),
        status=str(value.get("status", "active")),
        recurrence_frequency=str(value.get("recurrence_frequency", "none")),
        recurrence_interval=int(value.get("recurrence_interval", 1)),
        recurrence_end_at=(recurrence_end.isoformat() if recurrence_end else None),
    )


def _purge_at(reminder: Reminder) -> datetime | None:
    if reminder.recurrence_frequency != "none":
        if reminder.recurrence_end_datetime is None:
            return None
        return reminder.recurrence_end_datetime + timedelta(days=7)
    return reminder.due_datetime + timedelta(days=7)


def _reminder_data(reminder: Reminder) -> dict[str, Any]:
    return {
        "user_id": reminder.user_id,
        "chat_id": reminder.chat_id,
        "text": reminder.text,
        "due_at": reminder.due_datetime,
        "created_at": datetime.fromisoformat(reminder.created_at),
        "status": reminder.status,
        "recurrence_frequency": reminder.recurrence_frequency,
        "recurrence_interval": reminder.recurrence_interval,
        "recurrence_end_at": reminder.recurrence_end_datetime,
        "purge_at": _purge_at(reminder),
        "lease_until": None,
        "advance_notice_for": None,
    }


class FirestoreReminderStore:
    """Firestore reminder repository for concurrent, disposable Cloud Run instances."""

    def __init__(
        self,
        client: firestore.Client | None = None,
        collection: str = "reminders",
    ) -> None:
        self.client = client or firestore.Client()
        self.collection = self.client.collection(collection)

    def add(self, reminder: Reminder) -> None:
        self.collection.document(reminder.id).create(_reminder_data(reminder))

    def list_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        current = now or datetime.now().astimezone()
        documents = self.collection.where(
            filter=FieldFilter("user_id", "==", user_id)
        ).stream()
        return sorted(
            (
                reminder
                for reminder in map(_reminder_from_document, documents)
                if reminder.status == "active" and reminder.due_datetime > current
            ),
            key=lambda item: item.due_datetime,
        )

    def list_old_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        current = now or datetime.now().astimezone()
        documents = self.collection.where(
            filter=FieldFilter("user_id", "==", user_id)
        ).stream()
        return sorted(
            (
                reminder
                for reminder in map(_reminder_from_document, documents)
                if reminder.status in {"completed", "expired"}
                or reminder.due_datetime <= current
            ),
            key=lambda item: item.due_datetime,
            reverse=True,
        )

    def purge_old(self, now: datetime | None = None) -> int:
        current = now or datetime.now().astimezone()
        documents = list(
            self.collection.where(
                filter=FieldFilter("purge_at", "<=", current)
            ).stream()
        )
        for start in range(0, len(documents), 400):
            batch = self.client.batch()
            for document in documents[start : start + 400]:
                batch.delete(document.reference)
            batch.commit()
        return len(documents)

    def delete_for_user(self, reminder_id: str, user_id: int) -> bool:
        reference = self.collection.document(reminder_id)
        transaction = self.client.transaction()

        @firestore.transactional
        def delete(transaction: Any) -> bool:
            document = reference.get(transaction=transaction)
            if not document.exists or int(document.get("user_id")) != user_id:
                return False
            transaction.delete(reference)
            return True

        return delete(transaction)

    def delete_active_for_user(
        self, user_id: int, now: datetime | None = None
    ) -> list[Reminder]:
        reminders = self.list_for_user(user_id, now)
        for start in range(0, len(reminders), 400):
            batch = self.client.batch()
            for reminder in reminders[start : start + 400]:
                batch.delete(self.collection.document(reminder.id))
            batch.commit()
        return reminders

    def set_status_for_user(self, reminder_id: str, user_id: int, status: str) -> bool:
        reference = self.collection.document(reminder_id)
        transaction = self.client.transaction()

        @firestore.transactional
        def update(transaction: Any) -> bool:
            document = reference.get(transaction=transaction)
            if not document.exists or int(document.get("user_id")) != user_id:
                return False
            updates: dict[str, Any] = {"status": status, "lease_until": None}
            if status == "completed":
                due_at = _as_datetime(document.get("due_at"))
                if due_at is not None:
                    updates["purge_at"] = due_at + timedelta(days=7)
            transaction.update(reference, updates)
            return True

        return update(transaction)

    def update_deadline_for_user(
        self, reminder_id: str, user_id: int, due_at: str
    ) -> Reminder | None:
        reference = self.collection.document(reminder_id)
        transaction = self.client.transaction()
        new_due = datetime.fromisoformat(due_at)

        @firestore.transactional
        def update(transaction: Any) -> Reminder | None:
            document = reference.get(transaction=transaction)
            if not document.exists or int(document.get("user_id")) != user_id:
                return None
            reminder = _reminder_from_document(document)
            updated = Reminder(
                id=reminder.id,
                user_id=reminder.user_id,
                chat_id=reminder.chat_id,
                text=reminder.text,
                due_at=new_due.isoformat(),
                created_at=reminder.created_at,
                status="active",
                recurrence_frequency=reminder.recurrence_frequency,
                recurrence_interval=reminder.recurrence_interval,
                recurrence_end_at=reminder.recurrence_end_at,
            )
            transaction.update(
                reference,
                {
                    "due_at": new_due,
                    "status": "active",
                    "purge_at": _purge_at(updated),
                    "lease_until": None,
                    "advance_notice_for": None,
                },
            )
            return updated

        return update(transaction)

    def id_exists(self, reminder_id: str) -> bool:
        return self.collection.document(reminder_id).get().exists

    def claim_due(
        self,
        now: datetime,
        *,
        limit: int = 100,
        lease_minutes: int = 5,
    ) -> list[Reminder]:
        active = list(
            self.collection.where(filter=FieldFilter("status", "==", "active"))
            .where(filter=FieldFilter("due_at", "<=", now))
            .limit(limit)
            .stream()
        )
        remaining = max(0, limit - len(active))
        expired_leases = []
        if remaining:
            expired_leases = list(
                self.collection.where(
                    filter=FieldFilter("status", "==", "processing")
                )
                .where(filter=FieldFilter("lease_until", "<=", now))
                .limit(remaining)
                .stream()
            )

        claimed: list[Reminder] = []
        lease_until = now + timedelta(minutes=lease_minutes)
        for snapshot in [*active, *expired_leases]:
            reference = snapshot.reference
            transaction = self.client.transaction()

            @firestore.transactional
            def claim(transaction: Any) -> Reminder | None:
                current = reference.get(transaction=transaction)
                if not current.exists:
                    return None
                value = current.to_dict() or {}
                status = value.get("status")
                current_lease = _as_datetime(value.get("lease_until"))
                due_at = _as_datetime(value.get("due_at"))
                eligible = status == "active" and due_at is not None and due_at <= now
                recoverable = (
                    status == "processing"
                    and current_lease is not None
                    and current_lease <= now
                )
                if not eligible and not recoverable:
                    return None
                transaction.update(
                    reference,
                    {"status": "processing", "lease_until": lease_until},
                )
                return _reminder_from_document(current)

            reminder = claim(transaction)
            if reminder is not None:
                claimed.append(reminder)
        return claimed

    def claim_upcoming(
        self,
        now: datetime,
        *,
        lead_minutes: int = 60,
        limit: int = 100,
    ) -> list[Reminder]:
        """Claim reminders entering their advance-warning window exactly once."""
        warning_cutoff = now + timedelta(minutes=lead_minutes)
        candidates = list(
            self.collection.where(filter=FieldFilter("status", "==", "active"))
            .where(filter=FieldFilter("due_at", ">", now))
            .where(filter=FieldFilter("due_at", "<=", warning_cutoff))
            .limit(limit)
            .stream()
        )

        claimed: list[Reminder] = []
        for snapshot in candidates:
            reference = snapshot.reference
            transaction = self.client.transaction()

            @firestore.transactional
            def claim(transaction: Any) -> Reminder | None:
                current = reference.get(transaction=transaction)
                if not current.exists:
                    return None
                value = current.to_dict() or {}
                due_at = _as_datetime(value.get("due_at"))
                already_sent_for = _as_datetime(value.get("advance_notice_for"))
                if (
                    value.get("status") != "active"
                    or due_at is None
                    or due_at <= now
                    or due_at > warning_cutoff
                    or already_sent_for == due_at
                ):
                    return None
                transaction.update(reference, {"advance_notice_for": due_at})
                return _reminder_from_document(current)

            reminder = claim(transaction)
            if reminder is not None:
                claimed.append(reminder)
        return claimed

    def release_upcoming_claim(self, reminder: Reminder) -> None:
        """Allow a failed advance warning to be retried on the next worker run."""
        reference = self.collection.document(reminder.id)
        transaction = self.client.transaction()

        @firestore.transactional
        def release(transaction: Any) -> None:
            current = reference.get(transaction=transaction)
            if not current.exists:
                return
            sent_for = _as_datetime(current.get("advance_notice_for"))
            if sent_for == reminder.due_datetime:
                transaction.update(reference, {"advance_notice_for": None})

        release(transaction)

    def finish_delivery(
        self, reminder: Reminder, next_due: datetime | None
    ) -> None:
        reference = self.collection.document(reminder.id)
        if next_due is None:
            reference.update({"status": "expired", "lease_until": None})
            return
        updated = Reminder(
            id=reminder.id,
            user_id=reminder.user_id,
            chat_id=reminder.chat_id,
            text=reminder.text,
            due_at=next_due.isoformat(),
            created_at=reminder.created_at,
            status="active",
            recurrence_frequency=reminder.recurrence_frequency,
            recurrence_interval=reminder.recurrence_interval,
            recurrence_end_at=reminder.recurrence_end_at,
        )
        reference.update(
            {
                "due_at": next_due,
                "status": "active",
                "lease_until": None,
                "purge_at": _purge_at(updated),
                "advance_notice_for": None,
            }
        )

    def release_claim(self, reminder_id: str) -> None:
        self.collection.document(reminder_id).update(
            {"status": "active", "lease_until": None}
        )


class FirestoreUserSettingsStore:
    def __init__(
        self,
        client: firestore.Client | None = None,
        collection: str = "user_settings",
    ) -> None:
        self.client = client or firestore.Client()
        self.collection = self.client.collection(collection)

    def get(self, user_id: int) -> UserSettings | None:
        document = self.collection.document(str(user_id)).get()
        if not document.exists:
            return None
        value = document.to_dict() or {}
        return UserSettings.from_dict(value)

    def list_enabled(self) -> list[UserSettings]:
        documents = self.collection.where(
            filter=FieldFilter("daily_enabled", "==", True)
        ).stream()
        return [UserSettings.from_dict(document.to_dict() or {}) for document in documents]

    def save(self, value: UserSettings) -> None:
        self.collection.document(str(value.user_id)).set(value.to_dict())

    def claim_daily_summary(self, user_id: int, date_key: str) -> UserSettings | None:
        reference = self.collection.document(str(user_id))
        transaction = self.client.transaction()

        @firestore.transactional
        def claim(transaction: Any) -> UserSettings | None:
            document = reference.get(transaction=transaction)
            if not document.exists:
                return None
            value = document.to_dict() or {}
            if not value.get("daily_enabled", True):
                return None
            if value.get("last_daily_sent_on") == date_key:
                return None
            transaction.update(reference, {"last_daily_sent_on": date_key})
            return UserSettings.from_dict(value)

        return claim(transaction)

    def release_daily_summary(self, user_id: int, date_key: str) -> None:
        reference = self.collection.document(str(user_id))
        transaction = self.client.transaction()

        @firestore.transactional
        def release(transaction: Any) -> None:
            document = reference.get(transaction=transaction)
            if document.exists and document.get("last_daily_sent_on") == date_key:
                transaction.update(reference, {"last_daily_sent_on": None})

        release(transaction)
