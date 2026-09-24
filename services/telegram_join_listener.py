"""Long-poll Telegram chat_join_request updates and approve matched patrons.

Runs as its own docker-compose service (single instance — Telegram getUpdates
is single-consumer per bot).

Loop, per enabled bot in telegram_bots:
  1. getUpdates(offset, timeout=30, allowed_updates=["chat_join_request"])
  2. For each update:
       - Upsert telegram_join_requests row (status=pending).
       - Look up patreon_subscribers by claimed_telegram_username + active_patron.
       - Match → approveChatJoinRequest, mark approved, bind telegram_user_id.
       - No match → leave pending (hourly sync task re-checks).
  3. Persist offset in Redis (key telegram:join_listener:offset:{niche}).

Errors raise sync_alerts and the loop continues — Telegram retains updates for
24 hours, so transient failures don't lose data.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import redis

# Ensure we can import siblings when run as `python -m services.telegram_join_listener`.
sys.path.insert(0, "/app")

from config.settings import settings  # noqa: E402
from database.db import get_db_context  # noqa: E402
from database.models import (  # noqa: E402
    PatreonSubscriber,
    SyncAlert,
    TelegramBot,
    TelegramChannel,
    TelegramJoinRequest,
)
from publishers.telegram_admin import approve_join_request  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("telegram_join_listener")

LONG_POLL_TIMEOUT = 30  # seconds
LOOP_BACKOFF_ON_ERROR = 5  # seconds


def _redis() -> redis.Redis:
    return redis.from_url(settings.celery_broker_url)


def _offset_key(niche: str) -> str:
    return f"telegram:join_listener:offset:{niche}"


def _raise_alert(niche: str, category: str, message: str, severity: str = "error", context: dict | None = None):
    try:
        with get_db_context() as db:
            db.add(SyncAlert(
                niche=niche,
                severity=severity,
                category=category,
                message=message,
                context=context or {},
            ))
    except Exception:
        logger.exception("Failed to record sync_alert(%s/%s)", category, niche)


def _get_updates(token: str, offset: Optional[int]) -> list[dict]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "timeout": LONG_POLL_TIMEOUT,
        "allowed_updates": '["chat_join_request"]',
    }
    if offset is not None:
        params["offset"] = offset
    # httpx timeout slightly larger than the long-poll window.
    resp = httpx.get(url, params=params, timeout=LONG_POLL_TIMEOUT + 10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates failed: {data}")
    return data.get("result") or []


def _expected_chat_id(niche: str) -> Optional[str]:
    with get_db_context() as db:
        ch = db.query(TelegramChannel).filter(
            TelegramChannel.niche == niche,
            TelegramChannel.is_enabled.is_(True),
        ).first()
        return ch.channel_id if ch else None


def _handle_join_request(niche: str, payload: dict):
    """Process a single chat_join_request update."""
    chat = payload.get("chat") or {}
    user = payload.get("from") or {}
    chat_id = str(chat.get("id"))
    user_id = user.get("id")
    if not chat_id or not user_id:
        logger.warning("[%s] join_request missing chat/user — skipping: %s", niche, payload)
        return

    username = user.get("username")
    first_name = user.get("first_name")
    last_name = user.get("last_name")

    # If the channel id from this update doesn't match the configured one for
    # the niche, log + alert but still record the request.
    expected = _expected_chat_id(niche)
    if expected and expected != chat_id:
        _raise_alert(
            niche, "telegram_api_failed",
            f"Got join request for unexpected chat_id={chat_id} (expected {expected})",
            severity="warning",
            context={"chat_id": chat_id, "expected": expected, "user_id": user_id},
        )

    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        existing = db.query(TelegramJoinRequest).filter(
            TelegramJoinRequest.chat_id == chat_id,
            TelegramJoinRequest.telegram_user_id == user_id,
        ).first()
        if existing:
            # Refresh fields in case the user changed @username between requests.
            existing.telegram_username = username
            existing.first_name = first_name
            existing.last_name = last_name
            if existing.status != "pending":
                # Status was previously approved/declined/expired but Telegram
                # is sending us a fresh request — flip back to pending.
                existing.status = "pending"
                existing.resolved_at = None
                existing.matched_patreon_user_id = None
                existing.decline_reason = None
                existing.received_at = now
            req = existing
        else:
            req = TelegramJoinRequest(
                niche=niche,
                chat_id=chat_id,
                telegram_user_id=user_id,
                telegram_username=username,
                first_name=first_name,
                last_name=last_name,
                status="pending",
                received_at=now,
            )
            db.add(req)
            db.flush()

        # Try to match immediately. Without a public username we can't.
        if not username:
            logger.info("[%s] join request from user_id=%s has no public username — leaving pending",
                        niche, user_id)
            return

        sub = db.query(PatreonSubscriber).filter(
            PatreonSubscriber.niche == niche,
            PatreonSubscriber.patron_status == "active_patron",
            PatreonSubscriber.claimed_telegram_username == username,
        ).first()
        if not sub:
            logger.info("[%s] join request @%s has no matching active patron yet — leaving pending",
                        niche, username)
            return

        if approve_join_request(niche, chat_id, user_id):
            req.status = "approved"
            req.resolved_at = now
            req.matched_patreon_user_id = sub.patreon_user_id
            sub.telegram_user_id = user_id
            sub.telegram_state = "in_channel"
            logger.info("[%s] approved join request @%s (patreon_user=%s)",
                        niche, username, sub.patreon_user_id)
        else:
            _raise_alert(
                niche, "telegram_api_failed",
                f"approveChatJoinRequest failed for user_id={user_id} @{username}",
                severity="error",
                context={"telegram_user_id": user_id, "username": username},
            )


def _run_listener_for(bot_row: TelegramBot):
    """Poll one niche's bot. Loops forever; only returns on fatal errors."""
    niche = bot_row.niche
    token = bot_row.bot_token
    r = _redis()

    raw = r.get(_offset_key(niche))
    offset = int(raw) if raw else None
    logger.info("[%s] listener starting (offset=%s)", niche, offset)

    while True:
        # Heartbeat — refreshed every poll iteration. The UI status endpoint
        # treats >120s without a beat as "listener stalled".
        try:
            r.set(f"telegram:join_listener:heartbeat:{niche}", int(time.time()), ex=300)
        except Exception:
            pass
        try:
            updates = _get_updates(token, offset)
        except Exception as e:
            logger.exception("[%s] getUpdates failed: %s", niche, e)
            _raise_alert(
                niche, "telegram_api_failed",
                f"getUpdates failed: {e}",
                severity="error",
            )
            time.sleep(LOOP_BACKOFF_ON_ERROR)
            continue

        for upd in updates:
            offset = upd["update_id"] + 1
            cjr = upd.get("chat_join_request")
            if not cjr:
                # Shouldn't happen given allowed_updates filter, but just in case.
                continue
            try:
                _handle_join_request(niche, cjr)
            except Exception:
                logger.exception("[%s] handler failed for update %s", niche, upd.get("update_id"))
                _raise_alert(
                    niche, "telegram_api_failed",
                    f"Listener handler raised on update {upd.get('update_id')}",
                    severity="error",
                    context={"update_id": upd.get("update_id")},
                )

        # Persist offset only after the batch is processed so a crash mid-batch
        # replays the unprocessed updates instead of losing them.
        if offset is not None:
            r.set(_offset_key(niche), offset)


def main():
    niche_filter = os.environ.get("TELEGRAM_LISTENER_NICHE", "").strip().lower()

    with get_db_context() as db:
        q = db.query(TelegramBot).filter(TelegramBot.is_enabled.is_(True))
        if niche_filter:
            q = q.filter(TelegramBot.niche == niche_filter)
        bots = q.all()
        # Pull values out before session closes.
        bot_specs = [(b.niche, b.bot_token) for b in bots]

    if not bot_specs:
        logger.error("No enabled telegram_bots rows found (niche_filter=%r) — exiting", niche_filter)
        sys.exit(1)

    if len(bot_specs) > 1:
        logger.error(
            "Listener only supports one bot per process (got %d). "
            "Restart with TELEGRAM_LISTENER_NICHE set to the target niche.",
            len(bot_specs),
        )
        sys.exit(1)

    niche, token = bot_specs[0]

    class _Spec:
        pass
    spec = _Spec()
    spec.niche = niche
    spec.bot_token = token
    _run_listener_for(spec)


if __name__ == "__main__":
    main()
