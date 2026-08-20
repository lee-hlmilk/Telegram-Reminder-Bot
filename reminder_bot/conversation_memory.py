from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter


class FirestoreConversationStore:
    """Short-lived chat context and destructive-action confirmations."""

    def __init__(
        self,
        client: firestore.Client,
        collection: str = "conversation_sessions",
        history_limit: int = 8,
    ) -> None:
        self.client = client
        self.collection = client.collection(collection)
        self.history_limit = history_limit

    def history(self, user_id: int) -> list[dict[str, str]]:
        snapshot = self.collection.document(str(user_id)).get()
        if not snapshot.exists:
            return []
        value = snapshot.to_dict() or {}
        updated_at = value.get("updated_at")
        if isinstance(updated_at, datetime) and updated_at < datetime.now(timezone.utc) - timedelta(hours=24):
            snapshot.reference.delete()
            return []
        turns = value.get("history", [])
        if not isinstance(turns, list):
            return []
        return [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in turns[-self.history_limit :]
            if isinstance(turn, dict)
            and turn.get("role") in {"user", "assistant"}
            and turn.get("content")
        ]

    def append(self, user_id: int, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be user or assistant")
        reference = self.collection.document(str(user_id))
        transaction = self.client.transaction()

        @firestore.transactional
        def save(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            value = snapshot.to_dict() if snapshot.exists else {}
            turns = list((value or {}).get("history", []))
            turns.append({"role": role, "content": content[:2000]})
            transaction.set(
                reference,
                {
                    "history": turns[-self.history_limit :],
                    "updated_at": firestore.SERVER_TIMESTAMP,
                    "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
                },
                merge=True,
            )

        save(transaction)

    def set_confirmation(
        self,
        user_id: int,
        *,
        action: str,
        title: str = "",
        reminder_id: str | None = None,
        count: int | None = None,
    ) -> None:
        self.collection.document(str(user_id)).set(
            {
                "pending_confirmation": {
                    "action": action,
                    "title": title,
                    "reminder_id": reminder_id,
                    "count": count,
                },
                "confirmation_expires_at": datetime.now(timezone.utc)
                + timedelta(minutes=10),
                "updated_at": firestore.SERVER_TIMESTAMP,
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
            },
            merge=True,
        )

    def confirmation(self, user_id: int) -> dict[str, Any] | None:
        snapshot = self.collection.document(str(user_id)).get()
        if not snapshot.exists:
            return None
        value = snapshot.to_dict() or {}
        pending = value.get("pending_confirmation")
        expires_at = value.get("confirmation_expires_at")
        if not isinstance(pending, dict):
            return None
        if not isinstance(expires_at, datetime) or expires_at <= datetime.now(timezone.utc):
            self.clear_confirmation(user_id)
            return None
        return pending

    def clear_confirmation(self, user_id: int) -> None:
        self.collection.document(str(user_id)).set(
            {
                "pending_confirmation": firestore.DELETE_FIELD,
                "confirmation_expires_at": firestore.DELETE_FIELD,
            },
            merge=True,
        )

    def purge_expired(self, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        documents = list(
            self.collection.where(
                filter=FieldFilter("expires_at", "<=", current)
            ).stream()
        )
        for start in range(0, len(documents), 400):
            batch = self.client.batch()
            for document in documents[start : start + 400]:
                batch.delete(document.reference)
            batch.commit()
        return len(documents)
