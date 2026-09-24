"""
Reddit Background Videos API Router
Manages Reddit subreddits scraped for background videos (not captions).
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database.db import get_db
from database.models import RedditBackgroundSubreddit, BackgroundVideo
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
from config.settings import settings
import redis
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/background-videos/reddit", tags=["reddit-background-videos"])

REDDIT_BG_ENABLED_KEY = "reddit_bg:enabled"

SCRAPE_STAGES = ['top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new']


def get_redis_client():
    return redis.from_url(settings.celery_broker_url)


# --- Pydantic Models ---

class RedditBgSubredditCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    enabled: bool = True
    min_score: int = Field(default=100, ge=0)
    min_duration: int = Field(default=10, ge=1)
    max_duration: int = Field(default=60, ge=10)
    batch_size: int = Field(default=10, ge=1, le=50)

class RedditBgSubredditUpdate(BaseModel):
    enabled: Optional[bool] = None
    min_score: Optional[int] = Field(default=None, ge=0)
    min_duration: Optional[int] = Field(default=None, ge=1)
    max_duration: Optional[int] = Field(default=None, ge=10)
    batch_size: Optional[int] = Field(default=None, ge=1, le=50)


# --- Helper ---

def _sub_to_status(sub: RedditBackgroundSubreddit) -> dict:
    hours_since = None
    if sub.last_scrape_at:
        delta = datetime.now(timezone.utc) - sub.last_scrape_at.replace(tzinfo=timezone.utc)
        hours_since = round(delta.total_seconds() / 3600, 1)

    return {
        "name": sub.subreddit,
        "enabled": sub.enabled,
        "min_score": sub.min_score,
        "min_duration": sub.min_duration,
        "max_duration": sub.max_duration,
        "batch_size": sub.batch_size,
        "scrape_stage": sub.scrape_stage,
        "posts_scraped": sub.posts_scraped or 0,
        "videos_downloaded": sub.videos_downloaded or 0,
        "videos_failed": sub.videos_failed or 0,
        "hours_since_scrape": hours_since,
        "created_at": sub.created_at.isoformat() if sub.created_at else None,
    }


# --- Subreddit CRUD ---

@router.get("/subreddits/")
def list_reddit_bg_subreddits(db: Session = Depends(get_db)):
    """List all Reddit background subreddits with status."""
    subs = db.query(RedditBackgroundSubreddit).order_by(
        RedditBackgroundSubreddit.subreddit
    ).all()
    return [_sub_to_status(s) for s in subs]


@router.post("/subreddits/")
def add_reddit_bg_subreddit(config: RedditBgSubredditCreate, db: Session = Depends(get_db)):
    """Add a new Reddit subreddit for background video scraping."""
    name = config.name.lower().strip().lstrip("r/")

    existing = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Subreddit '{name}' already configured")

    sub = RedditBackgroundSubreddit(
        subreddit=name,
        enabled=config.enabled,
        min_score=config.min_score,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        batch_size=config.batch_size,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    return {"message": f"Added r/{name}", "subreddit": _sub_to_status(sub)}


@router.put("/subreddits/{name}")
def update_reddit_bg_subreddit(
    name: str, update: RedditBgSubredditUpdate, db: Session = Depends(get_db)
):
    """Update settings for a Reddit background subreddit."""
    sub = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name.lower()).first()
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subreddit '{name}' not found")

    if update.enabled is not None:
        sub.enabled = update.enabled
    if update.min_score is not None:
        sub.min_score = update.min_score
    if update.min_duration is not None:
        sub.min_duration = update.min_duration
    if update.max_duration is not None:
        sub.max_duration = update.max_duration
    if update.batch_size is not None:
        sub.batch_size = update.batch_size

    db.commit()
    return {"message": f"Updated r/{name}", "subreddit": _sub_to_status(sub)}


@router.delete("/subreddits/{name}")
def delete_reddit_bg_subreddit(name: str, db: Session = Depends(get_db)):
    """Remove a Reddit background subreddit (does not delete downloaded videos)."""
    sub = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name.lower()).first()
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subreddit '{name}' not found")

    db.delete(sub)
    db.commit()
    return {"message": f"Deleted r/{name}"}


@router.post("/subreddits/{name}/toggle")
def toggle_reddit_bg_subreddit(name: str, db: Session = Depends(get_db)):
    """Toggle a subreddit's enabled status."""
    sub = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name.lower()).first()
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subreddit '{name}' not found")

    sub.enabled = not sub.enabled
    db.commit()
    return {"message": f"r/{name} {'enabled' if sub.enabled else 'disabled'}", "enabled": sub.enabled}


@router.post("/subreddits/{name}/reset")
def reset_reddit_bg_subreddit(
    name: str,
    stage: str = Query(default="top_all"),
    db: Session = Depends(get_db),
):
    """Reset scraping progress for a subreddit to a specific stage."""
    if stage not in SCRAPE_STAGES:
        raise HTTPException(status_code=400, detail=f"Invalid stage. Must be one of: {SCRAPE_STAGES}")

    sub = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name.lower()).first()
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subreddit '{name}' not found")

    sub.scrape_stage = stage
    sub.last_pagination_url = None
    sub.posts_scraped = 0
    db.commit()
    return {"message": f"Reset r/{name} to stage '{stage}'"}


@router.post("/subreddits/{name}/scrape")
def trigger_reddit_bg_scrape(name: str, db: Session = Depends(get_db)):
    """Manually trigger a scrape for a specific subreddit."""
    sub = db.query(RedditBackgroundSubreddit).filter_by(subreddit=name.lower()).first()
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subreddit '{name}' not found")

    from tasks.reddit_background_scraping import scrape_reddit_background_subreddit
    task = scrape_reddit_background_subreddit.delay(sub.subreddit, batch_size=sub.batch_size or 10)

    return {
        "message": f"Triggered scrape for r/{name}",
        "task_id": task.id,
        "stage": sub.scrape_stage,
    }


@router.get("/subreddits/stages")
def list_scrape_stages():
    """List available scraping stages."""
    return {"stages": SCRAPE_STAGES}


# --- Scraper Control ---

@router.get("/control/status")
def get_reddit_bg_status(db: Session = Depends(get_db)):
    """Get Reddit background scraper status and stats."""
    r = get_redis_client()
    enabled = r.get(REDDIT_BG_ENABLED_KEY) == b"1"

    total_subs = db.query(RedditBackgroundSubreddit).count()
    enabled_subs = db.query(RedditBackgroundSubreddit).filter_by(enabled=True).count()

    total_videos = db.query(BackgroundVideo).filter_by(source_type='reddit').count()
    total_size = db.query(
        func.coalesce(func.sum(BackgroundVideo.file_size_bytes), 0)
    ).filter(BackgroundVideo.source_type == 'reddit').scalar()

    return {
        "enabled": enabled,
        "total_subreddits": total_subs,
        "enabled_subreddits": enabled_subs,
        "total_videos": total_videos,
        "total_size_gb": round((total_size or 0) / (1024 ** 3), 2),
    }


@router.post("/control/enable")
def enable_reddit_bg_scraper():
    """Enable Reddit background video scraping."""
    r = get_redis_client()
    r.set(REDDIT_BG_ENABLED_KEY, "1")
    return {"message": "Reddit background scraping enabled", "enabled": True}


@router.post("/control/disable")
def disable_reddit_bg_scraper():
    """Disable Reddit background video scraping."""
    r = get_redis_client()
    r.set(REDDIT_BG_ENABLED_KEY, "0")
    return {"message": "Reddit background scraping disabled", "enabled": False}
