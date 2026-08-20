# Remi — Telegram Reminder Bot

Remi is a conversation-first Telegram reminder assistant deployed on Google
Cloud. Users talk naturally to create, list, change, complete, and delete
one-time or repeating reminders. `/start` is the only exposed command.

## Production architecture

- **Cloud Run** receives Telegram webhooks and scales to zero while idle.
- **Cloud Firestore** stores reminders and per-user daily-summary preferences.
- **Cloud Firestore** also keeps up to eight recent conversation turns for 24
  hours and pending deletion confirmations for 10 minutes.
- **Cloud Scheduler** calls `/jobs/process` once per minute to deliver due work.
- **OpenAI Responses API** converts natural language into validated actions.
- **Secret Manager** supplies the Telegram token, OpenAI key, and webhook secret.

The application has no polling mode and does not use local JSON storage.

## Supported conversations

- `Remind me tomorrow at 3pm to call Mum`
- `Remind me every Friday at noon to send my update`
- `What do I have coming up?`
- `Show me my old reminders`
- `Move my IDP assignment to next Monday`
- `I finished my science homework`
- `Clear my reminders`
- `Send my daily summary at 8:30am`
- `Turn off my daily summary`

Remi can use short follow-ups such as `make that 5pm` or `move it to tomorrow`.
When details are ambiguous, it asks a focused question instead of failing.
Deleting one reminder or clearing all reminders requires an explicit yes/no
confirmation.

Daily summaries are enabled for new users at `08:00` by default. Completed and
expired reminders move to the old list and are purged seven days after their
deadline.

Each reminder sends advance warnings 60 minutes and 10 minutes before its
deadline, followed by a final notification at the deadline.

## Required environment variables

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_WEBHOOK_SECRET
OPENAI_API_KEY
GOOGLE_CLOUD_PROJECT
FIRESTORE_DATABASE=(default)
CLOUD_RUN_AUDIENCE
SCHEDULER_SERVICE_ACCOUNT
BOT_TIMEZONE=Asia/Singapore
DEFAULT_DAILY_TIME=08:00
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=20
```

In production, supply secret values through Google Secret Manager rather than a
checked-in `.env` file.

## Entry points

- `cloud_app.py` exposes `/health`, `/telegram/webhook`, and `/jobs/process`.
- `bot.py` contains Telegram conversation handling and message formatting.
- `reminder_bot/firestore_storage.py` contains Firestore repositories.
- `reminder_bot/cloud_worker.py` delivers reminders, summaries, and cleanup.

Cloud Run starts the service using the command defined in `Dockerfile`.

Redeploy saved changes from Cloud Shell with:

```bash
bash deploy.sh
```

## Tests

Install the dependencies, then run:

```bash
python -m unittest discover -s tests -v
```
