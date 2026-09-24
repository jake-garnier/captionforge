"""
Reddit Analytics API.

Read-only endpoints backing the /analytics page. All data comes from
the reddit_account_snapshots / reddit_posts / reddit_post_stats tables,
populated by tasks/reddit_analytics.py.

The HTML lives inside the gallery dashboard as the "Analytics" tab —
see api/gallery_with_dashboard.html. This module is API-only.

Routes:
  GET  /analytics/accounts              → leaderboard (latest snapshot per account)
  GET  /analytics/accounts/{u}          → drilldown bundle for one account
  GET  /analytics/accounts/{u}/karma-history?range=30d
  GET  /analytics/accounts/{u}/posts?sort=score|date|comments&page=N&page_size=50
  GET  /analytics/accounts/{u}/subreddit-breakdown?range=30d
  GET  /analytics/posts/{rid}           → single post detail with caption link
  GET  /analytics/posts/{rid}/history   → score trajectory for a post
  POST /analytics/accounts/{u}/backfill → fire off refresh_account_posts full_history

Time ranges are passed as "7d" / "30d" / "90d" / "all" (default 30d).
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, and_, desc, distinct
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import (
    RedditAccountSnapshot,
    RedditPost,
    RedditPostStat,
    ComposedVideo,
    GeneratedCaption,
)
from config.automation_config import get_automation_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_range(range_str: str) -> Optional[datetime]:
    """'30d' → datetime 30 days ago in UTC. 'all' → None."""
    if not range_str or range_str.lower() == "all":
        return None
    if range_str.endswith("d") and range_str[:-1].isdigit():
        days = int(range_str[:-1])
        return datetime.now(timezone.utc) - timedelta(days=days)
    raise HTTPException(400, f"Invalid range: {range_str!r}. Use 7d, 30d, 90d, or all.")


def _niche_for_username(username: str) -> Optional[str]:
    """Look up which niche a Reddit username is configured for."""
    config = get_automation_config()
    for niche in config.get_enabled_niches():
        name = niche.postpone_reddit_username or config.postpone.reddit_username
        if name == username:
            return niche.name
    return None


def _tracked_usernames() -> List[str]:
    """Same logic as tasks/reddit_analytics._tracked_usernames."""
    config = get_automation_config()
    usernames = set()
    for niche in config.get_enabled_niches():
        name = niche.postpone_reddit_username or config.postpone.reddit_username
        if name:
            usernames.add(name)
    return sorted(usernames)


def _latest_snapshot(db: Session, username: str) -> Optional[RedditAccountSnapshot]:
    return (
        db.query(RedditAccountSnapshot)
        .filter(RedditAccountSnapshot.username == username)
        .order_by(RedditAccountSnapshot.fetched_at.desc())
        .first()
    )


def _latest_stat_for_post(db: Session, reddit_post_id: str) -> Optional[RedditPostStat]:
    return (
        db.query(RedditPostStat)
        .filter(RedditPostStat.reddit_post_id == reddit_post_id)
        .order_by(RedditPostStat.fetched_at.desc())
        .first()
    )


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

@router.get("/accounts")
def list_accounts(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List of tracked Reddit accounts with their latest snapshot.

    Includes accounts that are configured in NICHE_CONFIGS but have
    not been scraped yet — those rows show no snapshot data.
    """
    out = []
    for username in _tracked_usernames():
        snap = _latest_snapshot(db, username)

        # 30-day karma delta
        delta_total = None
        if snap is not None:
            old_snap = (
                db.query(RedditAccountSnapshot)
                .filter(
                    RedditAccountSnapshot.username == username,
                    RedditAccountSnapshot.fetched_at <= datetime.now(timezone.utc) - timedelta(days=30),
                )
                .order_by(RedditAccountSnapshot.fetched_at.desc())
                .first()
            )
            if old_snap is not None and snap.total_karma is not None and old_snap.total_karma is not None:
                delta_total = snap.total_karma - old_snap.total_karma

        # Total tracked posts
        post_count = (
            db.query(func.count(RedditPost.id))
            .filter(RedditPost.username == username)
            .scalar()
        ) or 0

        # 30d aggregates (avg score, best score)
        thirty_ago = datetime.now(timezone.utc) - timedelta(days=30)
        agg = (
            db.query(
                func.avg(RedditPostStat.score).label("avg_score"),
                func.max(RedditPostStat.score).label("best_score"),
                func.count(distinct(RedditPostStat.reddit_post_id)).label("posts_30d"),
            )
            .join(RedditPost, RedditPost.reddit_post_id == RedditPostStat.reddit_post_id)
            .filter(RedditPost.username == username)
            .filter(RedditPost.created_utc >= thirty_ago)
            .one()
        )

        out.append({
            "username": username,
            "niche": _niche_for_username(username),
            "total_karma": snap.total_karma if snap else None,
            "link_karma": snap.link_karma if snap else None,
            "comment_karma": snap.comment_karma if snap else None,
            "karma_delta_30d": delta_total,
            "subscribers": snap.subscribers if snap else None,
            "is_suspended": snap.is_suspended if snap else False,
            "account_created_utc": snap.account_created_utc.isoformat() if snap and snap.account_created_utc else None,
            "last_fetched_at": snap.fetched_at.isoformat() if snap else None,
            "total_posts": post_count,
            "posts_30d": int(agg.posts_30d or 0),
            "avg_score_30d": float(agg.avg_score) if agg.avg_score is not None else None,
            "best_score_30d": int(agg.best_score) if agg.best_score is not None else None,
            "has_data": snap is not None,
        })
    return out


@router.get("/accounts/{username}")
def get_account_drilldown(username: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    snap = _latest_snapshot(db, username)
    if snap is None:
        # Still surface tracked-but-unscraped accounts; just no metrics.
        if username not in _tracked_usernames():
            raise HTTPException(404, f"Account {username!r} is not tracked")

    post_count = (
        db.query(func.count(RedditPost.id))
        .filter(RedditPost.username == username)
        .scalar()
    ) or 0
    removed_count = (
        db.query(func.count(RedditPost.id))
        .filter(RedditPost.username == username, RedditPost.removed_at.isnot(None))
        .scalar()
    ) or 0

    return {
        "username": username,
        "niche": _niche_for_username(username),
        "snapshot": _serialize_snapshot(snap) if snap else None,
        "total_posts": post_count,
        "removed_posts": removed_count,
    }


@router.get("/accounts/{username}/karma-history")
def karma_history(
    username: str,
    range: str = Query("30d"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    since = _parse_range(range)
    q = (
        db.query(RedditAccountSnapshot)
        .filter(RedditAccountSnapshot.username == username)
    )
    if since:
        q = q.filter(RedditAccountSnapshot.fetched_at >= since)
    rows = q.order_by(RedditAccountSnapshot.fetched_at.asc()).all()

    return {
        "username": username,
        "range": range,
        "points": [
            {
                "t": r.fetched_at.isoformat(),
                "total_karma": r.total_karma,
                "link_karma": r.link_karma,
                "comment_karma": r.comment_karma,
                "subscribers": r.subscribers,
                "is_suspended": r.is_suspended,
            }
            for r in rows
        ],
    }


@router.get("/accounts/{username}/posts")
def account_posts(
    username: str,
    sort: str = Query("date", regex="^(date|score|comments|ratio)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Paginated post list for an account with latest-stat per post.

    Sort by date (created_utc desc), score, comments, or ratio. The
    score/comments/ratio sort joins on the most recent stat row per post.
    """
    # Subquery: most recent stat per post
    latest_stat_subq = (
        db.query(
            RedditPostStat.reddit_post_id,
            func.max(RedditPostStat.fetched_at).label("max_fetched"),
        )
        .group_by(RedditPostStat.reddit_post_id)
        .subquery()
    )
    latest_stat_join = (
        db.query(RedditPostStat)
        .join(
            latest_stat_subq,
            and_(
                RedditPostStat.reddit_post_id == latest_stat_subq.c.reddit_post_id,
                RedditPostStat.fetched_at == latest_stat_subq.c.max_fetched,
            ),
        )
        .subquery()
    )

    # Select named columns so we can unpack by attribute (no positional
    # ambiguity from joining a subquery onto an ORM model).
    q = (
        db.query(
            RedditPost.reddit_post_id.label("reddit_post_id"),
            RedditPost.subreddit.label("subreddit"),
            RedditPost.title.label("title"),
            RedditPost.link_url.label("link_url"),
            RedditPost.permalink.label("permalink"),
            RedditPost.created_utc.label("created_utc"),
            RedditPost.is_video.label("is_video"),
            RedditPost.composed_video_id.label("composed_video_id"),
            RedditPost.removed_at.label("removed_at"),
            RedditPost.removed_reason.label("removed_reason"),
            latest_stat_join.c.score.label("score"),
            latest_stat_join.c.num_comments.label("num_comments"),
            latest_stat_join.c.upvote_ratio.label("upvote_ratio"),
        )
        .outerjoin(
            latest_stat_join,
            latest_stat_join.c.reddit_post_id == RedditPost.reddit_post_id,
        )
        .filter(RedditPost.username == username)
    )

    if sort == "date":
        q = q.order_by(desc(RedditPost.created_utc))
    elif sort == "score":
        q = q.order_by(desc(latest_stat_join.c.score))
    elif sort == "comments":
        q = q.order_by(desc(latest_stat_join.c.num_comments))
    elif sort == "ratio":
        q = q.order_by(desc(latest_stat_join.c.upvote_ratio))

    total = db.query(func.count(RedditPost.id)).filter(RedditPost.username == username).scalar() or 0

    rows = q.offset((page - 1) * page_size).limit(page_size).all()

    items = []
    for r in rows:
        items.append({
            "reddit_post_id": r.reddit_post_id,
            "subreddit": r.subreddit,
            "title": r.title,
            "link_url": r.link_url,
            "permalink": r.permalink,
            "created_utc": r.created_utc.isoformat() if r.created_utc else None,
            "is_video": r.is_video,
            "composed_video_id": r.composed_video_id,
            "removed_at": r.removed_at.isoformat() if r.removed_at else None,
            "removed_reason": r.removed_reason,
            "score": r.score,
            "num_comments": r.num_comments,
            "upvote_ratio": float(r.upvote_ratio) if r.upvote_ratio is not None else None,
        })

    return {
        "username": username,
        "total": total,
        "page": page,
        "page_size": page_size,
        "sort": sort,
        "items": items,
    }


@router.get("/accounts/{username}/subreddit-breakdown")
def subreddit_breakdown(
    username: str,
    range: str = Query("30d"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Per-subreddit aggregates for this account: post count, avg/best score."""
    since = _parse_range(range)

    # Latest stat per post
    latest_stat_subq = (
        db.query(
            RedditPostStat.reddit_post_id,
            func.max(RedditPostStat.fetched_at).label("max_fetched"),
        )
        .group_by(RedditPostStat.reddit_post_id)
        .subquery()
    )
    latest_score = (
        db.query(
            RedditPostStat.reddit_post_id,
            RedditPostStat.score,
            RedditPostStat.num_comments,
            RedditPostStat.upvote_ratio,
        )
        .join(
            latest_stat_subq,
            and_(
                RedditPostStat.reddit_post_id == latest_stat_subq.c.reddit_post_id,
                RedditPostStat.fetched_at == latest_stat_subq.c.max_fetched,
            ),
        )
        .subquery()
    )

    q = (
        db.query(
            RedditPost.subreddit,
            func.count(RedditPost.id).label("post_count"),
            func.avg(latest_score.c.score).label("avg_score"),
            func.max(latest_score.c.score).label("best_score"),
            func.avg(latest_score.c.upvote_ratio).label("avg_ratio"),
            func.avg(latest_score.c.num_comments).label("avg_comments"),
        )
        .outerjoin(latest_score, latest_score.c.reddit_post_id == RedditPost.reddit_post_id)
        .filter(RedditPost.username == username)
    )
    if since:
        q = q.filter(RedditPost.created_utc >= since)

    rows = q.group_by(RedditPost.subreddit).order_by(desc("avg_score")).all()

    return [
        {
            "subreddit": r.subreddit,
            "post_count": int(r.post_count or 0),
            "avg_score": float(r.avg_score) if r.avg_score is not None else None,
            "best_score": int(r.best_score) if r.best_score is not None else None,
            "avg_ratio": float(r.avg_ratio) if r.avg_ratio is not None else None,
            "avg_comments": float(r.avg_comments) if r.avg_comments is not None else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Single post
# ---------------------------------------------------------------------------

@router.get("/posts/{reddit_post_id}")
def post_detail(reddit_post_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    post = (
        db.query(RedditPost)
        .filter(RedditPost.reddit_post_id == reddit_post_id)
        .first()
    )
    if post is None:
        raise HTTPException(404, f"Reddit post {reddit_post_id!r} not tracked")

    latest = _latest_stat_for_post(db, reddit_post_id)

    # Pull linked caption text if available
    caption_text = None
    composed_thumb_url = None
    if post.composed_video_id:
        cv = db.query(ComposedVideo).filter_by(id=post.composed_video_id).first()
        if cv:
            caption_text = cv.edited_caption_text
            if not caption_text and cv.generated_caption_id:
                gc = db.query(GeneratedCaption).filter_by(id=cv.generated_caption_id).first()
                if gc:
                    caption_text = gc.caption_text

    return {
        "reddit_post_id": reddit_post_id,
        "username": post.username,
        "subreddit": post.subreddit,
        "title": post.title,
        "link_url": post.link_url,
        "permalink": post.permalink,
        "created_utc": post.created_utc.isoformat() if post.created_utc else None,
        "is_video": post.is_video,
        "composed_video_id": post.composed_video_id,
        "caption_text": caption_text,
        "removed_at": post.removed_at.isoformat() if post.removed_at else None,
        "removed_reason": post.removed_reason,
        "current_score": latest.score if latest else None,
        "current_comments": latest.num_comments if latest else None,
        "current_ratio": float(latest.upvote_ratio) if latest and latest.upvote_ratio is not None else None,
        "last_stat_at": latest.fetched_at.isoformat() if latest else None,
    }


@router.get("/posts/{reddit_post_id}/history")
def post_history(reddit_post_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    rows = (
        db.query(RedditPostStat)
        .filter(RedditPostStat.reddit_post_id == reddit_post_id)
        .order_by(RedditPostStat.fetched_at.asc())
        .all()
    )
    return {
        "reddit_post_id": reddit_post_id,
        "points": [
            {
                "t": r.fetched_at.isoformat(),
                "score": r.score,
                "comments": r.num_comments,
                "ratio": float(r.upvote_ratio) if r.upvote_ratio is not None else None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Admin actions
# ---------------------------------------------------------------------------

@router.post("/accounts/{username}/backfill")
def trigger_backfill(username: str) -> Dict[str, Any]:
    """Trigger a one-shot full-history pull for an account."""
    if username not in _tracked_usernames():
        raise HTTPException(404, f"Account {username!r} is not tracked")

    from tasks.reddit_analytics import refresh_account_posts
    result = refresh_account_posts.delay(username, full_history=True)

    return {
        "status": "dispatched",
        "username": username,
        "task_id": result.id,
        "message": "Backfill in progress. Refresh in 1-2 minutes for full history.",
    }


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _serialize_snapshot(snap: RedditAccountSnapshot) -> Dict[str, Any]:
    return {
        "fetched_at": snap.fetched_at.isoformat(),
        "total_karma": snap.total_karma,
        "link_karma": snap.link_karma,
        "comment_karma": snap.comment_karma,
        "awardee_karma": snap.awardee_karma,
        "awarder_karma": snap.awarder_karma,
        "subscribers": snap.subscribers,
        "is_suspended": snap.is_suspended,
        "account_created_utc": snap.account_created_utc.isoformat() if snap.account_created_utc else None,
        "verified_email": snap.verified_email,
    }
