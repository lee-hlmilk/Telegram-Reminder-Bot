import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from bot import confirmation_decision
from reminder_bot.models import UserSettings
from reminder_bot.models import Reminder
from reminder_bot.llm import IntentInterpretationError, IntentOutput, OpenAIIntentInterpreter
from reminder_bot.conversation import find_matching_reminders
from reminder_bot.service import (
    ReminderInputError,
    ReminderService,
    next_recurrence_datetime,
    parse_add_arguments,
    parse_daily_time,
)
from tests.fakes import InMemoryReminderStore, InMemoryUserSettingsStore


SGT = ZoneInfo("Asia/Singapore")


class ReminderCoreTests(unittest.TestCase):
    def test_confirmation_language_is_understood(self) -> None:
        self.assertIs(confirmation_decision("yes please"), True)
        self.assertIs(confirmation_decision("go ahead!"), True)
        self.assertIs(confirmation_decision("never mind"), False)
        self.assertIsNone(confirmation_decision("remind me tomorrow"))

    def test_firestore_utc_timestamp_converts_back_to_singapore_time(self) -> None:
        reminder = Reminder(
            id="abc123",
            user_id=1,
            chat_id=1,
            text="Submit assignment",
            due_at="2026-08-21T15:59:00+00:00",
            created_at="2026-08-19T12:00:00+00:00",
        )
        local_due = reminder.due_datetime.astimezone(SGT)
        self.assertEqual(local_due.strftime("%Y-%m-%d %H:%M"), "2026-08-21 23:59")

    def test_existing_user_settings_default_to_singapore_timezone(self) -> None:
        setting = UserSettings.from_dict(
            {"user_id": 1, "chat_id": 1, "daily_time": "08:00"}
        )
        self.assertEqual(setting.timezone, "Asia/Singapore")

    def test_reminder_can_store_notes_and_checklists(self) -> None:
        store = InMemoryReminderStore()
        service = ReminderService(store, SGT)
        reminder = service.create_at(
            user_id=1,
            chat_id=1,
            due_at=datetime(2026, 8, 21, 17, 0, tzinfo=SGT),
            text="Prepare for trip",
            note="Bring the blue suitcase",
            checklist_items=("Passport", "Charger"),
            now=datetime(2026, 8, 20, 12, 0, tzinfo=SGT),
        )
        self.assertEqual(reminder.note, "Bring the blue suitcase")
        self.assertEqual([item.text for item in reminder.checklist], ["Passport", "Charger"])
        completed = store.complete_checklist_item_for_user(
            reminder.id, 1, "passport"
        )
        self.assertEqual(completed[1], "Passport")

    def test_intent_uses_the_users_timezone(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        tokyo = ZoneInfo("Asia/Tokyo")
        intent = interpreter._to_intent(
            IntentOutput(
                action="create",
                title="Call home",
                due_at="2026-08-21T17:00:00",
                trigger_phrase="tomorrow at 5pm",
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.95,
            ),
            datetime(2026, 8, 20, 12, 0, tzinfo=tokyo),
            tokyo,
        )
        self.assertEqual(intent.due_at.utcoffset(), tokyo.utcoffset(intent.due_at))

    def test_llm_output_is_validated_before_becoming_an_intent(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="create",
                title="Call Mum",
                due_at="2026-08-20T16:00:00+08:00",
                trigger_phrase="tomorrow at 4pm",
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.9,
            )
        )
        self.assertEqual(intent.action, "create")
        self.assertEqual(intent.title, "Call Mum")
        self.assertEqual(intent.due_at.isoformat(), "2026-08-20T16:00:00+08:00")

    def test_low_confidence_llm_output_is_rejected(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        with self.assertRaises(IntentInterpretationError):
            interpreter._to_intent(
                IntentOutput(
                    action="delete",
                    title="homework",
                    due_at=None,
                    trigger_phrase=None,
                    daily_time=None,
                    daily_enabled=None,
                    reply="",
                    recurrence_frequency="none",
                    recurrence_interval=1,
                    recurrence_end_at=None,
                    confidence=0.2,
                )
            )

    def test_prompt_separates_delivery_date_from_subject_date(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        prompt = interpreter._instructions(datetime(2026, 8, 19, 19, 59, tzinfo=SGT))
        self.assertIn("trigger_phrase='tomorrow'", prompt)
        self.assertIn("Book Chiikawa tickets for 22 Aug", prompt)
        self.assertIn("Never silently drop meaningful subject details", prompt)

    def test_prompt_understands_natural_lists_and_bulk_deletion(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        prompt = interpreter._instructions(datetime(2026, 8, 19, 19, 59, tzinfo=SGT))
        self.assertIn("tell me my reminders", prompt)
        self.assertIn("show me my reminders", prompt)
        self.assertIn("what are my reminders?", prompt)
        self.assertIn("what is in my reminders?", prompt)
        self.assertIn("clear my reminders", prompt)
        self.assertIn("delete_all", prompt)
        self.assertIn("when is my daily reminder?", prompt)
        self.assertIn("make that 5pm", prompt)

    def test_unknown_intent_returns_a_clarification_question(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="unknown",
                title="",
                due_at=None,
                trigger_phrase=None,
                daily_time=None,
                daily_enabled=None,
                reply="What time should I remind you?",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.2,
            )
        )
        self.assertEqual(intent.reply, "What time should I remind you?")

    def test_read_only_list_intent_accepts_moderate_confidence(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="list",
                title="",
                due_at=None,
                trigger_phrase=None,
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.5,
            )
        )
        self.assertEqual(intent.action, "list")

    def test_prompt_separates_repeating_reminder_from_daily_summary(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        prompt = interpreter._instructions(datetime(2026, 8, 19, 20, 27, tzinfo=SGT))
        self.assertIn("must never use set_daily", prompt)
        self.assertIn("remind me every day/week/month/year", prompt)
        self.assertIn("summary/digest/list", prompt)

    def test_history_is_omitted_for_standalone_requests(self) -> None:
        history = [
            {"role": "user", "content": "Remind me about homework"},
            {"role": "assistant", "content": "What time?"},
        ]
        self.assertEqual(
            OpenAIIntentInterpreter._select_history("Show me my reminders", history),
            [],
        )

    def test_contextual_follow_up_uses_only_four_recent_turns(self) -> None:
        history = [
            {"role": "user", "content": f"message {index}"} for index in range(7)
        ]
        selected = OpenAIIntentInterpreter._select_history(
            "Actually make that tomorrow at 5pm", history
        )
        self.assertEqual(selected, history[-4:])

    def test_fast_path_handles_common_read_only_requests(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        self.assertEqual(interpreter._fast_intent("Show me my reminders").action, "list")
        self.assertEqual(interpreter._fast_intent("What did I miss?").action, "old")
        self.assertEqual(
            interpreter._fast_intent("What timezone am I using?").action, "settings"
        )

    def test_fast_path_handles_greetings_and_confirmed_bulk_clear_intent(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        greeting = interpreter._fast_intent("Hello!")
        self.assertEqual(greeting.action, "chat")
        self.assertTrue(greeting.reply)
        self.assertEqual(
            interpreter._fast_intent("Clear all my reminders").action, "delete_all"
        )

    def test_fast_path_defers_complex_requests_to_openai(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        self.assertIsNone(
            interpreter._fast_intent("Remind me tomorrow at 5pm to call Mum")
        )
        self.assertIsNone(
            interpreter._fast_intent("Delete my science homework reminder")
        )

    def test_chat_intent_has_a_personality_reply(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="chat",
                title="",
                due_at=None,
                trigger_phrase=None,
                daily_time=None,
                daily_enabled=None,
                reply="Hey! What can I help you remember?",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.95,
            )
        )
        self.assertEqual(intent.action, "chat")
        self.assertEqual(intent.reply, "Hey! What can I help you remember?")

    def test_safe_small_talk_can_use_a_lower_confidence_threshold(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="chat",
                title="",
                due_at=None,
                trigger_phrase=None,
                daily_time=None,
                daily_enabled=None,
                reply="Hi! What can I help you remember?",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.5,
            )
        )
        self.assertEqual(intent.action, "chat")

    def test_next_monday_is_resolved_by_backend_not_model_arithmetic(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="create",
                title="IDP assignment",
                due_at="2026-08-28T23:59:00+08:00",
                trigger_phrase="next monday",
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.95,
            ),
            datetime(2026, 8, 19, 20, 22, tzinfo=SGT),
        )
        self.assertEqual(intent.due_at.isoformat(), "2026-08-24T23:59:00+08:00")

    def test_update_intent_extracts_subject_and_new_deadline(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="update",
                title="IDP assignment",
                due_at="2026-08-24T23:59:00+08:00",
                trigger_phrase="24 aug",
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="none",
                recurrence_interval=1,
                recurrence_end_at=None,
                confidence=0.95,
            ),
            datetime(2026, 8, 19, 20, 23, tzinfo=SGT),
        )
        self.assertEqual(intent.action, "update")
        self.assertEqual(intent.title, "IDP assignment")
        self.assertEqual(intent.due_at.isoformat(), "2026-08-24T23:59:00+08:00")

    def test_repeating_intent_keeps_recurrence_separate_from_daily_setting(self) -> None:
        interpreter = OpenAIIntentInterpreter(
            api_key="test-key", model="test", timezone=SGT
        )
        intent = interpreter._to_intent(
            IntentOutput(
                action="create",
                title="Text Lee Hongliang",
                due_at="2026-08-20T12:00:00+08:00",
                trigger_phrase="at 12pm every day",
                daily_time=None,
                daily_enabled=None,
                reply="",
                recurrence_frequency="daily",
                recurrence_interval=1,
                recurrence_end_at="2029-08-19T23:59:00+08:00",
                confidence=0.98,
            ),
            datetime(2026, 8, 19, 20, 27, tzinfo=SGT),
        )
        self.assertEqual(intent.action, "create")
        self.assertEqual(intent.recurrence_frequency, "daily")
        self.assertIsNone(intent.daily_time)

    def test_recurrence_advances_daily_weekly_and_monthly(self) -> None:
        start = datetime(2026, 1, 31, 12, 0, tzinfo=SGT)
        self.assertEqual(next_recurrence_datetime(start, "daily", 1).day, 1)
        self.assertEqual(next_recurrence_datetime(start, "weekly", 1).day, 7)
        monthly = next_recurrence_datetime(start, "monthly", 1)
        self.assertEqual((monthly.month, monthly.day), (2, 28))

    def test_conversation_matches_homework_for_completion(self) -> None:
        service = ReminderService(InMemoryReminderStore(), SGT)
        reminder = service.create(
            user_id=7,
            chat_id=7,
            arguments=["2026-08-25", "Finish science homework"],
            now=datetime(2026, 8, 19, 12, 0, tzinfo=SGT),
        )
        self.assertEqual(
            find_matching_reminders(service, 7, "science homework"), [reminder]
        )

    def test_daily_time_parser(self) -> None:
        self.assertEqual(parse_daily_time("08:30").strftime("%H:%M"), "08:30")
        with self.assertRaises(ReminderInputError):
            parse_daily_time("25:00")

    def test_user_daily_settings_are_independent(self) -> None:
        store = InMemoryUserSettingsStore()
        first = UserSettings(1, 101, "08:00")
        second = UserSettings(2, 202, "18:30")
        store.save(first)
        store.save(second)
        self.assertEqual(store.get(1), first)
        self.assertEqual(store.get(2), second)
        store.save(UserSettings(1, 101, "08:00", daily_enabled=False))
        self.assertEqual(store.list_enabled(), [second])

    def test_parser_accepts_date_with_default_time(self) -> None:
        due, text = parse_add_arguments(["2026-08-25", "Submit", "assignment"], SGT)
        self.assertEqual(due.isoformat(), "2026-08-25T23:59:00+08:00")
        self.assertEqual(text, "Submit assignment")

    def test_parser_accepts_explicit_time(self) -> None:
        due, text = parse_add_arguments(["2026-09-01", "14:30", "Dentist"], SGT)
        self.assertEqual(due.isoformat(), "2026-09-01T14:30:00+08:00")
        self.assertEqual(text, "Dentist")

    def test_create_list_and_delete_are_scoped_to_user(self) -> None:
        store = InMemoryReminderStore()
        service = ReminderService(store, SGT)
        now = datetime(2026, 8, 19, 12, 0, tzinfo=SGT)
        created = service.create(
            user_id=1,
            chat_id=10,
            arguments=["2026-08-25", "Task"],
            now=now,
        )
        self.assertEqual([created], store.list_for_user(1))
        self.assertEqual([], store.list_for_user(2))
        self.assertFalse(store.delete_for_user(created.id, 2))
        self.assertTrue(store.delete_for_user(created.id, 1))
        self.assertEqual([], store.list_for_user(1))

    def test_delete_all_only_removes_requesting_users_active_reminders(self) -> None:
        store = InMemoryReminderStore()
        service = ReminderService(store, SGT)
        now = datetime(2026, 8, 19, 12, 0, tzinfo=SGT)
        first = service.create(
            user_id=1, chat_id=10, arguments=["2026-08-25", "First"], now=now
        )
        second = service.create(
            user_id=1, chat_id=10, arguments=["2026-08-26", "Second"], now=now
        )
        other = service.create(
            user_id=2, chat_id=20, arguments=["2026-08-27", "Other"], now=now
        )
        removed = store.delete_active_for_user(1, now)
        self.assertEqual({item.id for item in removed}, {first.id, second.id})
        self.assertEqual([], store.list_for_user(1, now))
        self.assertEqual([other], store.list_for_user(2, now))

    def test_deadline_update_is_user_scoped(self) -> None:
        store = InMemoryReminderStore()
        service = ReminderService(store, SGT)
        now = datetime(2026, 8, 19, 12, 0, tzinfo=SGT)
        reminder = service.create(
            user_id=1,
            chat_id=10,
            arguments=["2026-08-28", "IDP assignment"],
            now=now,
        )
        updated = service.update_deadline(
            reminder=reminder,
            user_id=1,
            due_at=datetime(2026, 8, 24, 23, 59, tzinfo=SGT),
            now=now,
        )
        self.assertEqual(updated.id, reminder.id)
        self.assertEqual(updated.text, "IDP assignment")
        self.assertEqual(updated.due_datetime.day, 24)
        self.assertIsNone(
            store.update_deadline_for_user(
                reminder.id, 2, "2026-08-25T23:59:00+08:00"
            )
        )

    def test_rejects_past_reminder(self) -> None:
        service = ReminderService(InMemoryReminderStore(), SGT)
        with self.assertRaises(ReminderInputError):
            service.create(
                user_id=1,
                chat_id=1,
                arguments=["2026-08-18", "Too late"],
                now=datetime(2026, 8, 19, 12, 0, tzinfo=SGT),
            )

    def test_old_list_and_seven_day_purge(self) -> None:
        store = InMemoryReminderStore()
        service = ReminderService(store, SGT)
        old = service.create(
            user_id=1,
            chat_id=1,
            arguments=["2026-08-20", "Old task"],
            now=datetime(2026, 8, 19, 12, 0, tzinfo=SGT),
        )
        future = service.create(
            user_id=1,
            chat_id=1,
            arguments=["2026-09-20", "Future task"],
            now=datetime(2026, 8, 19, 12, 0, tzinfo=SGT),
        )
        six_days_later = datetime(2026, 8, 26, 12, 0, tzinfo=SGT)
        eight_days_later = datetime(2026, 8, 28, 12, 0, tzinfo=SGT)
        self.assertEqual(store.list_old_for_user(1, six_days_later), [old])
        self.assertEqual(store.list_for_user(1, six_days_later), [future])
        self.assertEqual(store.purge_old(six_days_later), 0)
        self.assertEqual(store.purge_old(eight_days_later), 1)
        self.assertEqual(store.list_for_user(1, eight_days_later), [future])


if __name__ == "__main__":
    unittest.main()
