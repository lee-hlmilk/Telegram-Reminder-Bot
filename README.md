# Telegram Reminder Bot

A small first version of a personal reminder bot. It supports registering,
listing, and deleting reminders, with JSON persistence and Singapore time.

## Conversation-first interface

`/start` introduces Remi and gives conversational examples. It is the only
Telegram command exposed by the bot. Every reminder, list, deletion, completion,
and daily-summary setting is handled through ordinary conversation.

## Conversational messages

Commands are optional for common actions. The bot understands messages such as:

- `Remind me on 20 August that a new Marvel movie is coming out`
- `Remind me tomorrow at 3pm to call Mum`
- `Remind me in 50 minutes to do something`
- `Remind me at 4:40pm to say hi`
- `Send me a reminder tomorrow to book Chiikawa tickets for 22 Aug`
- `Remove my reminder for science homework`
- `I have completed my science homework`
- `Change the deadline for IDP assignment to 24 Aug`
- `Move my dentist reminder to next Monday at 2pm`
- `Remind me every day at 12pm to text Lee for the next 3 years`
- `Remind me every week to send my lunch update`
- `Show me my reminders`
- `What is in my reminders?`
- `What do I have coming up?`
- `Show me my old reminders`
- `Delete all my reminders`
- `Clear my reminders`

Dates without a year use the next matching date. Dates without a time default
to `23:59`. The bot separates the delivery date from dates that are part of the
reminder subject. For example, the Chiikawa message above is delivered tomorrow,
while `for 22 Aug` is retained in its title. To deliver it on 22 August instead,
say `Remind me on 22 Aug to book Chiikawa tickets`. Completing a reminder changes
its status to `completed`, so it no
longer appears in active reminders without destroying its history. If several
reminders match a description, the bot lists them instead of guessing.
Natural variations are classified by meaning rather than exact command wording.
Explicit requests to clear or delete all reminders remove only that user's active
reminders; old reminders keep their normal seven-day purge lifecycle.
Each created reminder is also scheduled as a one-time Telegram message at its
deadline. Future deadline jobs are restored when the bot restarts.

Once a deadline passes, the reminder leaves the active list and appears in the
old reminder list.
Only reminders in this old list are eligible for permanent deletion. Each one is
purged seven days after its own deadline; pending reminders are never purged.

## Run locally

1. Create a bot with Telegram's BotFather and copy its token.
2. Create a virtual environment and install `requirements.txt`.
3. Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN`.
4. Run `python bot.py`.

The first reminder creates `data/reminders.json`, and daily preferences are
stored in `data/user_settings.json`. Both files are intentionally ignored by Git
because they contain user data. Every user's setting is stored independently.
Daily summaries are enabled by default at `08:00` Singapore time for new users.
Existing reminder owners without a saved preference are enabled automatically on
startup. An explicitly disabled daily preference is preserved.

## OpenAI conversational interpretation

All ordinary conversational messages are interpreted through the OpenAI
Responses API using Structured Outputs. There is no local-model or deterministic
language fallback. Python remains responsible for validating the structured
result, finding reminders owned by the user, changing storage, scheduling
messages, and purging old data.

1. Create an OpenAI API key. API billing is separate from ChatGPT plans.
2. Set these values in `.env` locally:

   ```text
   OPENAI_API_KEY=your-secret-api-key
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TIMEOUT_SECONDS=20
   ```

3. Restart the bot.

For Cloud Run, store `OPENAI_API_KEY` in Google Secret Manager and expose it to
the service as an environment variable. Never commit the key or include it in a
container image. The model returns only an action, reminder title, due date,
daily-summary setting, and confidence score; it never receives database access.

Conversational daily settings are also supported, for example `Send my daily
reminders at 8:30am` and `Turn off my daily reminders`.

Repeating reminders support daily, weekly, monthly, and yearly intervals with an
optional end date. They are separate from the daily summary preference: `remind
me every day` creates a repeating reminder, while `send my daily summary at 8am`
changes the summary setting.

Remi can also respond briefly to greetings, thanks, farewells, and questions
about its capabilities while remaining focused on reminders and scheduling.

## Tests

Run `python -m unittest discover -s tests -v`.

## Next milestone

Add the reminder engine that sends messages at individual stored deadlines,
followed by seven-day overdue cleanup. Daily digests are already scheduled at
each user's chosen time. The bot, service, and storage layers are separated so
JSON can later be replaced with PostgreSQL.
