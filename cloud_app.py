from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import firestore
from google.oauth2 import id_token
from telegram import Update

from bot import BOT_COMMANDS, build_application
from reminder_bot.cloud_worker import process_cloud_work
from reminder_bot.conversation_memory import FirestoreConversationStore
from reminder_bot.firestore_storage import (
    FirestoreReminderStore,
    FirestoreUserSettingsStore,
)
from reminder_bot.llm import OpenAIIntentInterpreter
from reminder_bot.service import ReminderService, get_timezone, parse_daily_time

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing from the environment")
    return value


TELEGRAM_BOT_TOKEN = _required("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = _required("OPENAI_API_KEY")
TELEGRAM_WEBHOOK_SECRET = _required("TELEGRAM_WEBHOOK_SECRET")
BOT_TIMEZONE = get_timezone(os.getenv("BOT_TIMEZONE", "Asia/Singapore"))
DEFAULT_DAILY_TIME = os.getenv("DEFAULT_DAILY_TIME", "08:00").strip()
parse_daily_time(DEFAULT_DAILY_TIME)

firestore_options: dict[str, str] = {}
if os.getenv("GOOGLE_CLOUD_PROJECT"):
    firestore_options["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
if os.getenv("FIRESTORE_DATABASE"):
    firestore_options["database"] = os.environ["FIRESTORE_DATABASE"]
firestore_client = firestore.Client(**firestore_options)
reminder_store = FirestoreReminderStore(firestore_client)
settings_store = FirestoreUserSettingsStore(firestore_client)
conversation_store = FirestoreConversationStore(firestore_client)
service = ReminderService(reminder_store, BOT_TIMEZONE)
interpreter = OpenAIIntentInterpreter(
    api_key=OPENAI_API_KEY,
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
    timezone=BOT_TIMEZONE,
    timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20")),
)
telegram_application = build_application(
    TELEGRAM_BOT_TOKEN,
    service,
    settings_store,
    conversation_store,
    DEFAULT_DAILY_TIME,
    interpreter,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await telegram_application.initialize()
    await telegram_application.bot.set_my_commands(BOT_COMMANDS)
    LOGGER.info("Cloud reminder bot initialized")
    try:
        yield
    finally:
        await telegram_application.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, bool]:
    if not x_telegram_bot_api_secret_token or not secrets.compare_digest(
        x_telegram_bot_api_secret_token, TELEGRAM_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")
    update = Update.de_json(await request.json(), telegram_application.bot)
    await telegram_application.process_update(update)
    return {"ok": True}


def _verify_scheduler(authorization: str | None) -> None:
    expected_email = _required("SCHEDULER_SERVICE_ACCOUNT")
    audience = _required("CLOUD_RUN_AUDIENCE")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing scheduler identity")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = id_token.verify_oauth2_token(
            token, GoogleAuthRequest(), audience=audience
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid scheduler identity") from exc
    if claims.get("email") != expected_email or not claims.get("email_verified", False):
        raise HTTPException(status_code=403, detail="Unexpected scheduler identity")


@app.post("/jobs/process")
async def scheduled_work(authorization: str | None = Header(default=None)) -> dict[str, int]:
    _verify_scheduler(authorization)
    return await process_cloud_work(
        telegram_application, service, settings_store, conversation_store
    )
