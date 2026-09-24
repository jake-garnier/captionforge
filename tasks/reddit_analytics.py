"""
Reddit analytics tasks — backs the /analytics tab.

Periodically scrapes:
  - /user/<username>/about.json for karma + suspension state
  - /user/<username>/submitted.json for the post feed

Writes to:
  - reddit_account_snapshots  (one row per fetch)
  - reddit_posts              (one row per (Reddit post id) ever seen)
  - reddit_post_stats         (one row per fetch per post)

Auto-discovers tracked accounts via NICHE_CONFIGS[*].postpone_reddit_username
so adding/removing niche accounts in config flows through without code
changes here. Skips niches with no Reddit account configured.

Scraping goes through the existing RedditJsonScraper + proxy pool, keying
proxy assignments as "user:<username>" so each account is sticky-ish.
With our 8-proxy pool & 66 subreddit assignments, in practice all user
keys share one fallback proxy — that's fine for v1 cadence.

Cadence (set up in tasks/celery_app.py beat schedule):
  - dispatch-reddit-account-snapshots   every 15 min
  - dispatch-reddit-post-stats-fresh    every 15 min (posts < 48h old)
  - dispatch-reddit-post-stats-aged     daily 02:30 UTC (posts >= 48h old)
  - detect-removed-reddit-posts         hourly
Suspended accounts drop to daily cadence automatically.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import func, and_

from tasks.celery_app import celery_app
from scrapers.reddit_json_scraper import RedditJsonScraper
from database.db import get_db_context
from database.models import (
    RedditAccountSnapshot,
    RedditPost,
    RedditPostStat,
    ComposedVideo,
)
from config.automation_config import get_automation_config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_proxy(username: str) -> Optional[str]:
    """Return a proxy URL for this username, assigning one if needed.

    We key proxy_pool assignments as ``user:<username>`` (prefix prevents
    collision with real subreddit names). When the pool is exhausted it
    falls back to its first proxy — that's fine for analytics.
    """
    try:
        from utils.proxy_pool import get_proxy_for_subreddit as pool_get_proxy
        return pool_get_proxy(f"user:{username}", auto_assign=True)
    except Exception as e:
        logger.debug(f"Could not get proxy for user:{username}: {e}")
        return None


def _tracked_usernames() -> List[str]:
    """Pull the list of Reddit usernames to track from automation config.

    Reads NicheConfig.postpone_reddit_username for every enabled niche.
    Falls back to the global postpone default for niches that don't
    override. Empty usernames are skipped.
    """
    config = get_automation_config()
    usernames = set()
    for niche in config.get_enabled_niches():
        name = niche.postpone_reddit_username or config.postpone.reddit_username
        if name:
            usernames.add(name)
    return sorted(usernames)


def _canonical_media_url(url: Optional[str]) -> Optional[str]:
    """Normalize a hosted video URL for matching.

    Strip protocol, lowercase, drop trailing slash, drop query string.
    Both ComposedVideo.hosted_url and the URL stored on a Reddit post
    can vary slightly; canonicalize before joining.
    """
    if not url:
        return None
    s = url.strip().lower()
    # Strip protocol
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Drop fragment + query
    for sep in ("#", "?"):
        if sep in s:
            s = s.split(sep, 1)[0]
    # Drop trailing slash
    s = s.rstrip("/")
    # Strip www.
    if s.startswith("www."):
        s = s[4:]
    return s or None


def _find_composed_video_id(db, link_url: Optional[str]) -> Optional[int]:
    """Match a Reddit post's link URL back to our composed_videos row.

    Joins on canonicalized hosted_url. None if no match.
    """
    target = _canonical_media_url(link_url)
    if not target:
        return None

    # Pull a narrow candidate set then match in Python. SQL LIKE is
    # awkward because case + trailing-slash variations exist. The
    # candidate set is small (one row per composed video) — fine.
    rows = (
        db.query(ComposedVideo.id, ComposedVideo.hosted_url)
        .filter(ComposedVideo.hosted_url.isnot(None))
        .all()
    )
    for video_id, hosted_url in rows:
        if _canonical_media_url(hosted_url) == target:
            return video_id
    return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(queue="scraping")
def refresh_account_snapshot(username: str) -> dict:
    """Fetch /user/<u>/about.json + scrape follower count from the
    rendered profile page, append a row to reddit_account_snapshots.

    The follower scrape uses Playwright and is allowed to fail without
    blocking the karma data — if it errors, subscribers gets stored as
    NULL and the rest of the snapshot still lands.

    Always inserts a row — even for suspended accounts — so the UI can
    show 'we checked at T and it was still suspended'. Callers should
    rate-limit themselves; this task does not.
    """
    proxy = _user_proxy(username)
    if proxy:
        logger.info(f"Snapshot {username} via proxy {proxy.split('@')[-1]}")
    else:
        logger.info(f"Snapshot {username} direct (no proxy)")

    with RedditJsonScraper(use_proxy=False, proxy_url=proxy, check_blocks=True) as scraper:
        about = scraper.get_user_about(username)

    if about is None:
        logger.warning(f"Snapshot {username}: scrape failed; skipping insert")
        return {"status": "error", "username": username, "error": "scrape_failed"}

    created_utc = about.get("created_utc")
    created_dt = None
    if created_utc is not None:
        try:
            created_dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            created_dt = None

    # Follower count via Playwright. Reddit's JSON returns 0 for everyone;
    # the real number is rendered client-side on the profile page.
    # Wrapped in try/except so this never blocks the karma insert.
    subscribers = None
    if not about.get("is_suspended", False):
        try:
            from scrapers.reddit_profile_scraper import RedditProfileScraper
            with RedditProfileScraper() as pscraper:
                subscribers = pscraper.get_follower_count(username)
            logger.info(f"Snapshot {username}: followers={subscribers}")
        except Exception as e:
            logger.warning(f"Snapshot {username}: follower scrape failed: {e}")

    with get_db_context() as db:
        snap = RedditAccountSnapshot(
            username=username,
            total_karma=about.get("total_karma"),
            link_karma=about.get("link_karma"),
            comment_karma=about.get("comment_karma"),
            awardee_karma=about.get("awardee_karma"),
            awarder_karma=about.get("awarder_karma"),
            is_suspended=bool(about.get("is_suspended", False)),
            account_created_utc=created_dt,
            verified_email=about.get("verified_email"),
            subscribers=subscribers,
            raw_about=about.get("raw"),
        )
        db.add(snap)
        db.commit()
        snap_id = snap.id

    return {
        "status": "ok",
        "username": username,
        "snapshot_id": snap_id,
        "is_suspended": bool(about.get("is_suspended", False)),
        "total_karma": about.get("total_karma"),
        "subscribers": subscribers,
    }


@celery_app.task(queue="scraping")
def refresh_account_posts(username: str, full_history: bool = False) -> dict:
    """Paginate /user/<u>/submitted.json and upsert reddit_posts +
    insert a stats row for every observed post.

    full_history=True: paginate until Reddit returns no `after` cursor
      (Reddit caps at ~1000 posts). Used for the initial backfill.
    full_history=False (default): stop after we see a post we've
      already recorded a stats row for in the last 30 minutes —
      avoids dragging the entire history every cycle.
    """
    proxy = _user_proxy(username)
    if proxy:
        logger.info(f"Posts {username} via proxy {proxy.split('@')[-1]} (full={full_history})")
    else:
        logger.info(f"Posts {username} direct (full={full_history})")

    after: Optional[str] = None
    pages = 0
    posts_seen = 0
    posts_inserted = 0
    stats_inserted = 0
    stop_threshold = _now_utc() - timedelta(minutes=30)

    with RedditJsonScraper(use_proxy=False, proxy_url=proxy, check_blocks=True) as scraper:
        while True:
            pages += 1
            posts, next_after = scraper.get_user_submitted(username, after=after, limit=100)
            if not posts:
                break

            with get_db_context() as db:
                hit_recent = False
                for p in posts:
                    posts_seen += 1
                    reddit_post_id = p["id"]

                    # Upsert reddit_posts
                    existing = (
                        db.query(RedditPost)
                        .filter_by(reddit_post_id=reddit_post_id)
                        .first()
                    )
                    if existing is None:
                        composed_video_id = _find_composed_video_id(db, p.get("url"))
                        new_post = RedditPost(
                            reddit_post_id=reddit_post_id,
                            username=username,
                            subreddit=p["subreddit"],
                            title=p.get("title") or "",
                            link_url=p.get("url"),
                            permalink=p.get("permalink") or "",
                            created_utc=p.get("created_utc"),
                            is_video=p.get("is_video", False),
                            composed_video_id=composed_video_id,
                        )
                        db.add(new_post)
                        posts_inserted += 1
                    else:
                        existing.last_seen_at = _now_utc()
                        # Clear removed_at if the post is visible again
                        if existing.removed_at is not None:
                            existing.removed_at = None
                            existing.removed_reason = None

                    # Always insert a stats row when score is present
                    if p.get("score") is not None:
                        stat = RedditPostStat(
                            reddit_post_id=reddit_post_id,
                            score=p.get("score"),
                            num_comments=p.get("num_comments"),
                            upvote_ratio=p.get("upvote_ratio"),
                        )
                        db.add(stat)
                        stats_inserted += 1

                    # Bail early if we've already got a recent stats row
                    # for this post (and we're not in full-history mode).
                    if not full_history and existing is not None:
                        most_recent = (
                            db.query(func.max(RedditPostStat.fetched_at))
                            .filter(RedditPostStat.reddit_post_id == reddit_post_id)
                            .scalar()
                        )
                        if most_recent and most_recent > stop_threshold:
                            hit_recent = True
                db.commit()

            if hit_recent and not full_history:
                logger.info(f"Posts {username}: hit recent post on page {pages}, stopping early")
                break
            after = next_after
            if not after:
                break
            # Polite delay between pages
            time.sleep(2.5)

    return {
        "status": "ok",
        "username": username,
        "pages": pages,
        "posts_seen": posts_seen,
        "posts_inserted": posts_inserted,
        "stats_inserted": stats_inserted,
        "full_history": full_history,
    }


@celery_app.task(queue="scraping")
def refresh_post_stats_aged(username: str) -> dict:
    """Refresh stats for posts >= 48h old but < 30 days old.

    Daily task. The fresh-posts path already covers <48h every 15min,
    so this catches the slow-cooling tail. Posts older than 30 days
    are considered settled and not re-checked.
    """
    cutoff_fresh = _now_utc() - timedelta(hours=48)
    cutoff_settled = _now_utc() - timedelta(days=30)

    with get_db_context() as db:
        rows = (
            db.query(RedditPost)
            .filter(
                RedditPost.username == username,
                RedditPost.removed_at.is_(None),
                RedditPost.created_utc < cutoff_fresh,
                RedditPost.created_utc >= cutoff_settled,
            )
            .all()
        )
        target_ids = [r.reddit_post_id for r in rows]

    if not target_ids:
        return {"status": "ok", "username": username, "refreshed": 0}

    proxy = _user_proxy(username)
    refreshed = 0
    with RedditJsonScraper(use_proxy=False, proxy_url=proxy, check_blocks=True) as scraper:
        for post_id in target_ids:
            # We don't know the subreddit easily here without another
            # lookup, but get_post_by_id works without it.
            data = scraper.get_post_by_id(post_id)
            if not data:
                continue
            with get_db_context() as db:
                stat = RedditPostStat(
                    reddit_post_id=post_id,
                    score=data.get("score"),
                    num_comments=data.get("num_comments"),
                    upvote_ratio=data.get("upvote_ratio"),
                )
                db.add(stat)
                db.commit()
            refreshed += 1
            time.sleep(2.0)

    return {"status": "ok", "username": username, "refreshed": refreshed}


@celery_app.task(queue="scraping")
def detect_removed_posts(username: str) -> dict:
    """Mark posts as removed when Reddit no longer returns them.

    Scrapes the user's submitted feed (first 2 pages, ~200 posts).
    Any tracked post for this user that is NOT in the result AND was
    last seen >24h ago is marked removed.

    Doesn't try to fetch the permalink to get a reason — that's a
    separate slower task and Reddit often hides the reason anyway.
    """
    proxy = _user_proxy(username)
    seen_ids = set()
    with RedditJsonScraper(use_proxy=False, proxy_url=proxy, check_blocks=True) as scraper:
        after = None
        for _ in range(2):
            posts, after = scraper.get_user_submitted(username, after=after, limit=100)
            if not posts:
                break
            for p in posts:
                seen_ids.add(p["id"])
            if not after:
                break
            time.sleep(2.5)

    if not seen_ids:
        # Empty result is suspicious — could be suspension OR full
        # block. Don't mark everything removed in that case.
        return {"status": "skipped", "username": username, "reason": "empty_feed"}

    stale_cutoff = _now_utc() - timedelta(hours=24)
    with get_db_context() as db:
        rows = (
            db.query(RedditPost)
            .filter(
                RedditPost.username == username,
                RedditPost.removed_at.is_(None),
                RedditPost.last_seen_at < stale_cutoff,
            )
            .all()
        )
        marked = 0
        for row in rows:
            if row.reddit_post_id not in seen_ids:
                row.removed_at = _now_utc()
                row.removed_reason = "missing_from_feed"
                marked += 1
        db.commit()

    return {"status": "ok", "username": username, "marked_removed": marked}


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

@celery_app.task(queue="maintenance")
def dispatch_account_snapshots() -> dict:
    """Fan out refresh_account_snapshot for every tracked account.

    Suspended accounts (per the most recent snapshot) drop to once-daily
    cadence here — they're skipped on every dispatch except the 02:30 one.
    """
    usernames = _tracked_usernames()
    now = _now_utc()
    is_daily_window = now.hour == 2 and now.minute < 45

    dispatched = []
    skipped = []
    with get_db_context() as db:
        for u in usernames:
            # Find latest snapshot
            latest = (
                db.query(RedditAccountSnapshot)
                .filter(RedditAccountSnapshot.username == u)
                .order_by(RedditAccountSnapshot.fetched_at.desc())
                .first()
            )
            if latest and latest.is_suspended and not is_daily_window:
                skipped.append(u)
                continue
            refresh_account_snapshot.delay(u)
            dispatched.append(u)

    return {"dispatched": dispatched, "skipped_suspended": skipped}


@celery_app.task(queue="maintenance")
def dispatch_post_stats_fresh() -> dict:
    """Fan out refresh_account_posts for every tracked account.

    Only the incremental path runs here. Initial backfill must be
    explicitly triggered via refresh_account_posts.delay(u, full_history=True).
    """
    usernames = _tracked_usernames()
    for u in usernames:
        # Skip if the most recent snapshot says suspended
        with get_db_context() as db:
            latest = (
                db.query(RedditAccountSnapshot)
                .filter(RedditAccountSnapshot.username == u)
                .order_by(RedditAccountSnapshot.fetched_at.desc())
                .first()
            )
            if latest and latest.is_suspended:
                continue
        refresh_account_posts.delay(u, full_history=False)

    return {"dispatched": usernames}


@celery_app.task(queue="maintenance")
def dispatch_post_stats_aged() -> dict:
    """Fan out refresh_post_stats_aged for every tracked account."""
    for u in _tracked_usernames():
        refresh_post_stats_aged.delay(u)
    return {"dispatched": _tracked_usernames()}


@celery_app.task(queue="maintenance")
def dispatch_detect_removed() -> dict:
    """Fan out detect_removed_posts for every tracked account."""
    for u in _tracked_usernames():
        detect_removed_posts.delay(u)
    return {"dispatched": _tracked_usernames()}


# ---------------------------------------------------------------------------
# One-off: link historical RedditPost rows back to ComposedVideo by URL.
# Useful after the initial backfill — first-time inserts already link,
# but if hosted_url got updated on a composed_video after the reddit
# post was first observed, we can re-link here.
# ---------------------------------------------------------------------------

@celery_app.task(queue="maintenance")
def relink_posts_to_composed_videos(limit: int = 2000) -> dict:
    """Match reddit_posts.link_url against composed_videos.hosted_url.

    Only touches rows where composed_video_id IS NULL. Safe to re-run.
    """
    updated = 0
    with get_db_context() as db:
        rows = (
            db.query(RedditPost)
            .filter(RedditPost.composed_video_id.is_(None))
            .filter(RedditPost.link_url.isnot(None))
            .limit(limit)
            .all()
        )
        for row in rows:
            cvid = _find_composed_video_id(db, row.link_url)
            if cvid:
                row.composed_video_id = cvid
                updated += 1
        db.commit()
    return {"checked": len(rows), "updated": updated}
