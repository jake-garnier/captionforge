"""Telegram Bot API admin operations for the membership-sync flow.

Three primitives needed:
  - approve_join_request(chat_id, user_id) — used by the listener service
  - decline_join_request(chat_id, user_id) — used for 7d-expiry of pending requests
  - kick_user(chat_id, user_id) — used by sync_patreon_telegram on lapse

The bot is the existing `telegram_bots.{niche}` row; it must already be admin
of the channel with `can_invite_users` + `can_restrict_members` rights.

429 responses honour `parameters.retry_after`. Other failures return False
(callers convert to sync_alerts).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from database.db import get_db_context
from database.models import TelegramBot

logger = logging.getLogger(__name__)


def _bot_token_for(niche: str) -> Optional[str]:
    with get_db_context() as db:
        bot = db.query(TelegramBot).filter(
            TelegramBot.niche == niche,
            TelegramBot.is_enabled.is_(True),
        ).first()
        return bot.bot_token if bot else None


def _post(token: str, method: str, payload: dict, *, max_attempts: int = 3) -> dict:
    """Single Bot API POST with retry on 429.

    Returns the decoded `result` on success. Raises on non-429 failure.
    """
    url = f"https://api.telegram.org/bot{token}/{method}"
    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            resp = httpx.post(url, json=payload, timeout=30.0)
        except httpx.HTTPError as e:
            last_error = e
            time.sleep(2 ** attempt)
            continue
        try:
            data = resp.json()
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)
            continue
        if data.get("ok"):
            return data.get("result", {})
        # Honor flood-wait
        params = data.get("parameters") or {}
        retry_after = params.get("retry_after")
        if resp.status_code == 429 and retry_after:
            sleep_s = int(retry_after) + 1
            logger.warning("Telegram %s 429 — sleeping %ds", method, sleep_s)
            time.sleep(sleep_s)
            continue
        # Non-429 failure: don't retry blindly
        raise RuntimeError(
            f"Telegram {method} failed: status={resp.status_code} "
            f"description={data.get('description')!r} "
            f"error_code={data.get('error_code')} payload={payload}"
        )
    raise RuntimeError(f"Telegram {method} exhausted retries: {last_error}")


def approve_join_request(niche: str, chat_id: str | int, user_id: int) -> bool:
    token = _bot_token_for(niche)
    if not token:
        logger.error("approve_join_request: no enabled bot for niche=%s", niche)
        return False
    try:
        _post(token, "approveChatJoinRequest", {"chat_id": chat_id, "user_id": user_id})
        return True
    except Exception:
        logger.exception("approve_join_request(chat=%s, user=%s) failed", chat_id, user_id)
        return False


def decline_join_request(niche: str, chat_id: str | int, user_id: int) -> bool:
    token = _bot_token_for(niche)
    if not token:
        return False
    try:
        _post(token, "declineChatJoinRequest", {"chat_id": chat_id, "user_id": user_id})
        return True
    except Exception:
        logger.exception("decline_join_request(chat=%s, user=%s) failed", chat_id, user_id)
        return False


def kick_user(niche: str, chat_id: str | int, user_id: int) -> bool:
    """Kick a user but allow rejoin later (per Q7 — re-pledges should work).

    Implementation: banChatMember then unbanChatMember(only_if_banned=true).
    """
    token = _bot_token_for(niche)
    if not token:
        return False
    try:
        _post(token, "banChatMember", {"chat_id": chat_id, "user_id": user_id})
    except Exception:
        logger.exception("banChatMember(chat=%s, user=%s) failed", chat_id, user_id)
        return False
    try:
        _post(token, "unbanChatMember", {
            "chat_id": chat_id,
            "user_id": user_id,
            "only_if_banned": True,
        })
    except Exception:
        # Ban worked but unban didn't — they're permanently banned. Log loudly
        # but still return True since they were removed from the channel.
        logger.exception(
            "unbanChatMember(chat=%s, user=%s) failed — user is still banned, "
            "manual intervention required if they re-pledge",
            chat_id, user_id,
        )
    return True
