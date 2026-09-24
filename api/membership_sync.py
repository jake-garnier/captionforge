"""Patreon→Telegram membership sync API + Membership tab data source.

Surfaces:
  - GET  /membership-sync/status            — overall pipeline health
  - GET  /membership-sync/alerts            — open / all sync_alerts rows
  - POST /membership-sync/alerts/{id}/resolve
  - GET  /membership-sync/subscribers       — active patrons + telegram state
  - GET  /membership-sync/join-requests     — pending / all chat_join_request entries
  - POST /membership-sync/join-requests/{id}/approve  — manual override
  - POST /membership-sync/join-requests/{id}/decline  — manual override
  - POST /membership-sync/sync/trigger      — fire the celery task on demand
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from config.settings import settings
from database.db import get_db
from database.models import (
    PatreonSubscriber,
    SyncAlert,
    TelegramBot,
    TelegramChannel,
    TelegramJoinRequest,
)
from publishers.telegram_admin import (
    approve_join_request as tg_approve,
    decline_join_request as tg_decline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/membership-sync", tags=["membership-sync"])


def _redis() -> redis.Redis:
    return redis.from_url(settings.celery_broker_url)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


@router.get("/status")
def get_status(niche: str, db: Session = Depends(get_db)) -> dict:
    """Aggregate health signal for the Membership tab top bar."""
    now = int(time.time())
    r = _redis()

    # Patreon configuration: env vars + a cheap /identity probe.
    upper = niche.upper()
    has_client = bool(os.environ.get(f"PATREON_{upper}_CLIENT_ID"))
    has_tokens = bool(os.environ.get(f"PATREON_{upper}_ACCESS_TOKEN"))
    post_id = os.environ.get(f"PATREON_{upper}_POST_ID") or ""
    campaign_id = os.environ.get(f"PATREON_{upper}_CAMPAIGN_ID") or ""

    patreon_ok: Optional[bool] = None
    patreon_error: Optional[str] = None
    if has_client and has_tokens:
        try:
            from publishers.patreon_api import PatreonAPI
            with PatreonAPI(niche) as api:
                # The campaign-id call is the cheapest token-exercising request we have.
                api.get_campaign_id()
                patreon_ok = True
        except Exception as e:
            patreon_ok = False
            patreon_error = str(e)[:200]

    # Telegram bot/channel config.
    bot = db.query(TelegramBot).filter(TelegramBot.niche == niche).first()
    channel = db.query(TelegramChannel).filter(TelegramChannel.niche == niche).first()

    # Listener heartbeat (set by services/telegram_join_listener.py every poll).
    raw_beat = r.get(f"telegram:join_listener:heartbeat:{niche}")
    last_beat = int(raw_beat) if raw_beat else None
    listener_ok = bool(last_beat and (now - last_beat) < 120)

    # Last successful celery sync run.
    raw_run = r.get(f"sync:patreon_telegram:last_run:{niche}")
    last_run = int(raw_run) if raw_run else None

    # Counters (cheap, single-table queries).
    open_alerts = db.query(SyncAlert).filter(SyncAlert.status == "open").count()
    pending_requests = db.query(TelegramJoinRequest).filter(
        TelegramJoinRequest.niche == niche,
        TelegramJoinRequest.status == "pending",
    ).count()
    active_patrons = db.query(PatreonSubscriber).filter(
        PatreonSubscriber.niche == niche,
        PatreonSubscriber.patron_status == "active_patron",
    ).count()
    in_channel = db.query(PatreonSubscriber).filter(
        PatreonSubscriber.niche == niche,
        PatreonSubscriber.telegram_state == "in_channel",
    ).count()

    return {
        "niche": niche,
        "patreon": {
            "credentials_present": has_client and has_tokens,
            "ok": patreon_ok,
            "error": patreon_error,
            "campaign_id": campaign_id or None,
            "post_id": post_id or None,
        },
        "telegram": {
            "bot_configured": bool(bot),
            "bot_username": bot.bot_username if bot else None,
            "channel_configured": bool(channel),
            "channel_id": channel.channel_id if channel else None,
            "listener_running": listener_ok,
            "listener_last_heartbeat": last_beat,
            "listener_seconds_since_heartbeat": (now - last_beat) if last_beat else None,
        },
        "sync": {
            "last_run_at": last_run,
            "last_run_seconds_ago": (now - last_run) if last_run else None,
        },
        "counters": {
            "open_alerts": open_alerts,
            "pending_join_requests": pending_requests,
            "active_patrons": active_patrons,
            "in_channel": in_channel,
        },
    }


@router.get("/alerts")
def list_alerts(
    status: str = Query("open", regex="^(open|resolved|all)$"),
    niche: Optional[str] = None,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
) -> dict:
    q = db.query(SyncAlert)
    if status != "all":
        q = q.filter(SyncAlert.status == status)
    if niche:
        q = q.filter(SyncAlert.niche == niche)
    rows = q.order_by(desc(SyncAlert.created_at)).limit(limit).all()
    return {
        "alerts": [
            {
                "id": a.id,
                "niche": a.niche,
                "severity": a.severity,
                "category": a.category,
                "message": a.message,
                "context": a.context,
                "status": a.status,
                "created_at": _iso(a.created_at),
                "resolved_at": _iso(a.resolved_at),
            }
            for a in rows
        ]
    }


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)) -> dict:
    a = db.query(SyncAlert).filter(SyncAlert.id == alert_id).first()
    if not a:
        raise HTTPException(404, "alert not found")
    a.status = "resolved"
    a.resolved_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": a.id, "status": a.status}


@router.get("/subscribers")
def list_subscribers(
    niche: str,
    status: Optional[str] = Query(None, regex="^(active_patron|declined_patron|former_patron)$"),
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
) -> dict:
    q = db.query(PatreonSubscriber).filter(PatreonSubscriber.niche == niche)
    if status:
        q = q.filter(PatreonSubscriber.patron_status == status)
    rows = q.order_by(desc(PatreonSubscriber.last_synced_at)).limit(limit).all()
    return {
        "subscribers": [
            {
                "id": s.id,
                "patreon_user_id": s.patreon_user_id,
                "full_name": s.full_name,
                "email": s.email,
                "patron_status": s.patron_status,
                "claimed_telegram_username": s.claimed_telegram_username,
                "claimed_at": _iso(s.claimed_at),
                "comment_id": s.comment_id,
                "comment_last_modified": _iso(s.comment_last_modified),
                "telegram_user_id": s.telegram_user_id,
                "telegram_state": s.telegram_state,
                "last_charge_status": s.last_charge_status,
                "last_charge_date": _iso(s.last_charge_date),
                "last_synced_at": _iso(s.last_synced_at),
            }
            for s in rows
        ]
    }


@router.get("/join-requests")
def list_join_requests(
    niche: str,
    status: str = Query("pending", regex="^(pending|approved|declined|expired|all)$"),
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
) -> dict:
    q = db.query(TelegramJoinRequest).filter(TelegramJoinRequest.niche == niche)
    if status != "all":
        q = q.filter(TelegramJoinRequest.status == status)
    rows = q.order_by(desc(TelegramJoinRequest.received_at)).limit(limit).all()
    return {
        "join_requests": [
            {
                "id": jr.id,
                "telegram_user_id": jr.telegram_user_id,
                "telegram_username": jr.telegram_username,
                "first_name": jr.first_name,
                "last_name": jr.last_name,
                "chat_id": jr.chat_id,
                "status": jr.status,
                "received_at": _iso(jr.received_at),
                "resolved_at": _iso(jr.resolved_at),
                "matched_patreon_user_id": jr.matched_patreon_user_id,
                "decline_reason": jr.decline_reason,
            }
            for jr in rows
        ]
    }


@router.post("/join-requests/{request_id}/approve")
def approve_request_manually(request_id: int, db: Session = Depends(get_db)) -> dict:
    jr = db.query(TelegramJoinRequest).filter(TelegramJoinRequest.id == request_id).first()
    if not jr:
        raise HTTPException(404, "join request not found")
    if jr.status != "pending":
        raise HTTPException(400, f"request is already {jr.status}")
    if not tg_approve(jr.niche, jr.chat_id, jr.telegram_user_id):
        raise HTTPException(502, "Telegram approveChatJoinRequest failed (see logs)")
    jr.status = "approved"
    jr.resolved_at = datetime.now(timezone.utc)
    jr.decline_reason = None
    # Best-effort: bind to the patreon_subscribers row if we can find one.
    sub = db.query(PatreonSubscriber).filter(
        PatreonSubscriber.niche == jr.niche,
        PatreonSubscriber.claimed_telegram_username == jr.telegram_username,
    ).first()
    if sub:
        jr.matched_patreon_user_id = sub.patreon_user_id
        sub.telegram_user_id = jr.telegram_user_id
        sub.telegram_state = "in_channel"
    db.commit()
    return {"id": jr.id, "status": jr.status, "matched_patreon_user_id": jr.matched_patreon_user_id}


@router.post("/join-requests/{request_id}/decline")
def decline_request_manually(request_id: int, db: Session = Depends(get_db)) -> dict:
    jr = db.query(TelegramJoinRequest).filter(TelegramJoinRequest.id == request_id).first()
    if not jr:
        raise HTTPException(404, "join request not found")
    if jr.status != "pending":
        raise HTTPException(400, f"request is already {jr.status}")
    if not tg_decline(jr.niche, jr.chat_id, jr.telegram_user_id):
        raise HTTPException(502, "Telegram declineChatJoinRequest failed (see logs)")
    jr.status = "declined"
    jr.resolved_at = datetime.now(timezone.utc)
    jr.decline_reason = "manually declined via Membership tab"
    db.commit()
    return {"id": jr.id, "status": jr.status}


@router.post("/sync/trigger")
def trigger_sync(niche: str) -> dict:
    """Fire the hourly task immediately (Celery, async)."""
    from tasks.patreon_telegram_sync import sync_patreon_telegram
    async_result = sync_patreon_telegram.apply_async(args=(niche,), queue="maintenance")
    return {"task_id": async_result.id, "niche": niche, "status": "queued"}
