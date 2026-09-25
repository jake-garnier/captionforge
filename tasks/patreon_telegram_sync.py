"""Hourly Patreon→Telegram membership sync (per niche).

Pipeline (hourly at :25 on the maintenance queue):
  1. Pull the Patreon active-member list, upsert into patreon_subscribers.
     Anyone present in our DB but missing from the API response is flipped
     to former_patron.
  2. Playwright-scrape comments on the configured Patreon post (PATREON_{NICHE}_POST_ID).
     Parse the first valid Telegram @username out of each. Attach it to the
     matching patreon_subscribers row (joined by patreon_user_id from the
     comment's `commenter` relationship). Unparseable comments raise a
     comment_parse_failed sync_alert.
  3. Re-evaluate pending join requests against the freshly-updated subscribers
     table. Any active patron whose claimed @username matches gets approved.
  4. Kick subscribers whose status is now declined_patron or former_patron and
     who are currently in_channel. Reuse-friendly: ban then immediate unban
     (only_if_banned=True) so they can rejoin if they re-pledge.
  5. Expire pending join requests older than 7 days with no match — decline
     them and raise pending_request_expired alerts.

All errors land in sync_alerts; this task swallows exceptions per-step rather
than letting one failure mode kill the entire run.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import settings
from database.db import get_db_context
from database.models import (
    PatreonSubscriber,
    SyncAlert,
    TelegramChannel,
    TelegramJoinRequest,
)
from publishers.patreon_api import PatreonAPI, PatreonAuthError
from publishers.patreon_comment_scraper import (
    PatreonScrapeError,
    scrape_post_comments,
)
from publishers.telegram_admin import (
    decline_join_request,
    kick_user,
    approve_join_request,
)
from publishers.telegram_username_parser import parse_telegram_username
from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS = 30 * 60  # plenty of headroom for an hourly schedule
PENDING_REQUEST_TTL = timedelta(days=7)


def _redis() -> redis.Redis:
    return redis.from_url(settings.celery_broker_url)


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _raise_alert(niche: str, category: str, message: str, severity: str = "error", context: dict | None = None):
    """Best-effort sync_alerts insert."""
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


def _channel_chat_id(niche: str) -> Optional[str]:
    """Look up the configured Telegram channel id for this niche."""
    with get_db_context() as db:
        ch = db.query(TelegramChannel).filter(
            TelegramChannel.niche == niche,
            TelegramChannel.is_enabled.is_(True),
        ).first()
        return ch.channel_id if ch else None


def _sync_members(niche: str) -> dict:
    """Pull every member of the campaign and upsert into patreon_subscribers."""
    with PatreonAPI(niche) as api:
        members = list(api.list_active_members())

    seen_user_ids: set[str] = set()
    upserted = 0
    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        for m in members:
            seen_user_ids.add(m.user_id)
            stmt = pg_insert(PatreonSubscriber).values(
                niche=niche,
                patreon_user_id=m.user_id,
                patreon_member_id=m.member_id,
                full_name=m.full_name,
                email=m.email,
                patron_status=m.patron_status,
                pledge_relationship_start=_parse_iso(m.pledge_relationship_start),
                last_charge_status=m.last_charge_status,
                last_charge_date=_parse_iso(m.last_charge_date),
                last_synced_at=now,
            ).on_conflict_do_update(
                index_elements=["niche", "patreon_user_id"],
                set_={
                    "patreon_member_id": m.member_id,
                    "full_name": m.full_name,
                    "email": m.email,
                    "patron_status": m.patron_status,
                    "pledge_relationship_start": _parse_iso(m.pledge_relationship_start),
                    "last_charge_status": m.last_charge_status,
                    "last_charge_date": _parse_iso(m.last_charge_date),
                    "last_synced_at": now,
                },
            )
            db.execute(stmt)
            upserted += 1

        stale = []
        if seen_user_ids:
            stale = db.query(PatreonSubscriber).filter(
                PatreonSubscriber.niche == niche,
                ~PatreonSubscriber.patreon_user_id.in_(seen_user_ids),
                PatreonSubscriber.patron_status != "former_patron",
            ).all()
            for s in stale:
                s.patron_status = "former_patron"
                s.last_synced_at = now

    return {
        "upserted": upserted,
        "marked_former": len(stale),
        "total_seen": len(seen_user_ids),
    }


def _sync_comments(niche: str) -> dict:
    """Scrape the configured post's comments and bind @usernames to subscribers."""
    post_id = os.environ.get(f"PATREON_{niche.upper()}_POST_ID") or ""
    if not post_id:
        _raise_alert(
            niche, "patreon_scrape_failed",
            f"PATREON_{niche.upper()}_POST_ID is not set; skipping comment scrape",
            severity="warning",
        )
        return {"skipped": True, "reason": "no_post_id"}

    comments = scrape_post_comments(post_id, niche)

    parsed_count = 0
    unparsed_count = 0
    bound = 0
    skipped_unknown_user = 0
    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        for c in comments:
            username = parse_telegram_username(c.body)
            if not username:
                unparsed_count += 1
                _raise_alert(
                    niche, "comment_parse_failed",
                    f"Could not parse a Telegram @username from comment {c.comment_id}",
                    severity="warning",
                    context={
                        "comment_id": c.comment_id,
                        "patreon_user_id": c.patreon_user_id,
                        "body_preview": (c.body or "")[:200],
                    },
                )
                continue

            parsed_count += 1
            sub = db.query(PatreonSubscriber).filter(
                PatreonSubscriber.niche == niche,
                PatreonSubscriber.patreon_user_id == c.patreon_user_id,
            ).first()
            if not sub:
                # Commenter has never been seen as a patron (yet?). The next
                # member-sync run will create the row — for now skip silently.
                skipped_unknown_user += 1
                continue

            # Only update if anything actually changed (so updated_at stays meaningful).
            changed = (
                sub.claimed_telegram_username != username
                or sub.comment_id != c.comment_id
                or sub.comment_last_modified != c.last_modified
            )
            if changed:
                sub.claimed_telegram_username = username
                sub.comment_id = c.comment_id
                sub.comment_last_modified = c.last_modified
                if not sub.claimed_at:
                    sub.claimed_at = now
                bound += 1

    return {
        "scraped": len(comments),
        "parsed": parsed_count,
        "unparsed": unparsed_count,
        "bound": bound,
        "skipped_unknown_user": skipped_unknown_user,
    }


def _process_pending_requests(niche: str) -> dict:
    """Approve any pending join request that now matches an active patron.

    Also expires requests older than PENDING_REQUEST_TTL with no match.
    """
    chat_id = _channel_chat_id(niche)
    if not chat_id:
        _raise_alert(
            niche, "telegram_api_failed",
            "No enabled telegram_channels row for niche — cannot reconcile join requests",
            severity="error",
        )
        return {"skipped": True, "reason": "no_channel"}

    approved = 0
    expired = 0
    declined = 0
    now = datetime.now(timezone.utc)
    cutoff = now - PENDING_REQUEST_TTL

    with get_db_context() as db:
        pending = db.query(TelegramJoinRequest).filter(
            TelegramJoinRequest.niche == niche,
            TelegramJoinRequest.status == "pending",
        ).all()

        for req in pending:
            if not req.telegram_username:
                # User has no public Telegram username — we can never match
                # by claimed @username. Wait for manual resolution; only
                # expire it when it ages out.
                if req.received_at and req.received_at < cutoff:
                    if decline_join_request(niche, chat_id, req.telegram_user_id):
                        req.status = "expired"
                        req.resolved_at = now
                        req.decline_reason = "no public Telegram username, expired after 7d"
                        expired += 1
                        _raise_alert(
                            niche, "pending_request_expired",
                            f"Declined stale join request from user {req.telegram_user_id} (no public username)",
                            severity="info",
                            context={"telegram_user_id": req.telegram_user_id},
                        )
                continue

            sub = db.query(PatreonSubscriber).filter(
                PatreonSubscriber.niche == niche,
                PatreonSubscriber.patron_status == "active_patron",
                PatreonSubscriber.claimed_telegram_username == req.telegram_username,
            ).first()
            if sub:
                if approve_join_request(niche, chat_id, req.telegram_user_id):
                    req.status = "approved"
                    req.resolved_at = now
                    req.matched_patreon_user_id = sub.patreon_user_id
                    sub.telegram_user_id = req.telegram_user_id
                    sub.telegram_state = "in_channel"
                    approved += 1
                else:
                    _raise_alert(
                        niche, "telegram_api_failed",
                        f"approveChatJoinRequest failed for user {req.telegram_user_id}",
                        severity="error",
                        context={"telegram_user_id": req.telegram_user_id, "username": req.telegram_username},
                    )
                continue

            # No match. Expire if too old.
            if req.received_at and req.received_at < cutoff:
                if decline_join_request(niche, chat_id, req.telegram_user_id):
                    req.status = "expired"
                    req.resolved_at = now
                    req.decline_reason = "no matching active patron found within 7 days"
                    expired += 1
                    _raise_alert(
                        niche, "pending_request_expired",
                        f"Declined stale join request @{req.telegram_username} after 7d",
                        severity="info",
                        context={
                            "telegram_user_id": req.telegram_user_id,
                            "username": req.telegram_username,
                        },
                    )

    return {"approved": approved, "expired": expired, "declined": declined}


def _process_kicks(niche: str) -> dict:
    """Kick subscribers whose pledge has lapsed."""
    chat_id = _channel_chat_id(niche)
    if not chat_id:
        return {"skipped": True, "reason": "no_channel"}

    kicked = 0
    failures = 0
    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        lapsed = db.query(PatreonSubscriber).filter(
            PatreonSubscriber.niche == niche,
            PatreonSubscriber.telegram_state == "in_channel",
            PatreonSubscriber.patron_status.in_(["former_patron", "declined_patron"]),
            PatreonSubscriber.telegram_user_id.is_not(None),
        ).all()
        for sub in lapsed:
            ok = kick_user(niche, chat_id, sub.telegram_user_id)
            if ok:
                sub.telegram_state = "kicked"
                sub.last_synced_at = now
                kicked += 1
            else:
                failures += 1
                _raise_alert(
                    niche, "telegram_api_failed",
                    f"Failed to kick lapsed patron tg_user_id={sub.telegram_user_id}",
                    severity="error",
                    context={
                        "patreon_user_id": sub.patreon_user_id,
                        "telegram_user_id": sub.telegram_user_id,
                        "patron_status": sub.patron_status,
                    },
                )

    return {"kicked": kicked, "failures": failures}


@celery_app.task(name="tasks.patreon_telegram_sync.sync_patreon_telegram", queue="maintenance")
def sync_patreon_telegram(niche: str) -> dict:
    """Top-level hourly orchestrator. Redis-locked to prevent overlap."""
    r = _redis()
    lock_key = f"sync:patreon_telegram:{niche}"
    if not r.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS):
        logger.info("[%s] another sync run holds the lock — skipping", niche)
        return {"status": "locked"}

    try:
        logger.info("[%s] sync_patreon_telegram starting", niche)
        result: dict = {"niche": niche}

        try:
            result["members"] = _sync_members(niche)
        except PatreonAuthError as e:
            logger.exception("[%s] Patreon API failed", niche)
            _raise_alert(niche, "patreon_api_failed", f"Member sync failed: {e}")
            result["members"] = {"error": str(e)}

        try:
            result["comments"] = _sync_comments(niche)
        except PatreonScrapeError as e:
            logger.exception("[%s] Patreon comment scrape failed", niche)
            _raise_alert(niche, "patreon_scrape_failed", f"Comment scrape failed: {e}")
            result["comments"] = {"error": str(e)}

        try:
            result["join_requests"] = _process_pending_requests(niche)
        except Exception as e:
            logger.exception("[%s] join-request reconciliation failed", niche)
            _raise_alert(niche, "telegram_api_failed", f"Join-request processing failed: {e}")
            result["join_requests"] = {"error": str(e)}

        try:
            result["kicks"] = _process_kicks(niche)
        except Exception as e:
            logger.exception("[%s] kick processing failed", niche)
            _raise_alert(niche, "telegram_api_failed", f"Kick processing failed: {e}")
            result["kicks"] = {"error": str(e)}

        logger.info("[%s] sync_patreon_telegram done: %s", niche, result)
        # Heartbeat for UI status — clears the "no successful sync" warning.
        try:
            r.set(f"sync:patreon_telegram:last_run:{niche}", int(datetime.now(timezone.utc).timestamp()))
        except Exception:
            pass
        return result
    finally:
        r.delete(lock_key)
