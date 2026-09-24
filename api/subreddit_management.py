"""
API endpoints for managing subreddit configurations.

All configuration is stored in the database (scraping_progress table).
No config file needed - manage subreddits via API or web UI.

Progressive depth scraping stages:
- top_all -> top_year -> top_month -> top_week -> top_day -> new
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from sqlalchemy.orm import Session
from database.db import get_db
from database.models import ScrapingProgress
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subreddits", tags=["subreddit-management"])

# Valid scrape stages (must match tasks/incremental_scraping.py)
SCRAPE_STAGES = ['top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new']

# Stage-specific min_score thresholds
STAGE_MIN_SCORES = {
    'top_all': 300,
    'top_year': 300,
    'top_month': 300,
    'top_week': 300,
    'top_day': 300,
    'new': 25,
}


class SubredditConfig(BaseModel):
    """Subreddit configuration model"""
    name: str = Field(..., min_length=1, max_length=50)
    enabled: bool = True
    min_score: int = Field(default=300, ge=0)
    batch_size: int = Field(default=25, ge=1, le=100)
    description: Optional[str] = None
    scrape_stage: str = Field(default="top_all", description="Starting scrape stage")


class SubredditUpdate(BaseModel):
    """Model for updating subreddit settings"""
    enabled: Optional[bool] = None
    min_score: Optional[int] = Field(default=None, ge=0)
    batch_size: Optional[int] = Field(default=None, ge=1, le=100)
    description: Optional[str] = None


class SubredditStatus(BaseModel):
    """Full subreddit status including config and scraping progress"""
    name: str
    enabled: bool
    min_score: int
    batch_size: int
    description: Optional[str]
    scrape_stage: str = "top_all"
    stage_min_score: int = 300  # Min score for current stage
    posts_scraped: int = 0
    videos_downloaded: int = 0
    videos_failed: int = 0
    success_rate: float = 0.0
    last_post_score: Optional[int] = None
    scraping_active: bool = False
    hours_since_scrape: Optional[float] = None


def _progress_to_status(progress: ScrapingProgress) -> SubredditStatus:
    """Convert a ScrapingProgress record to SubredditStatus"""
    hours_since = None
    if progress.last_scrape_at:
        now = datetime.utcnow()
        if progress.last_scrape_at.tzinfo:
            now = datetime.now(progress.last_scrape_at.tzinfo)
        hours_since = round((now - progress.last_scrape_at).total_seconds() / 3600, 1)

    success_rate = 0.0
    if progress.posts_scraped > 0:
        success_rate = round((progress.videos_downloaded / progress.posts_scraped * 100), 1)

    # Get current scrape stage and its min_score
    scrape_stage = getattr(progress, 'scrape_stage', None) or 'top_all'
    stage_min_score = STAGE_MIN_SCORES.get(scrape_stage, 300)

    return SubredditStatus(
        name=progress.subreddit,
        enabled=progress.scraping_active,
        min_score=progress.target_min_score or 300,
        batch_size=getattr(progress, 'batch_size', None) or 25,
        description=getattr(progress, 'description', None),
        scrape_stage=scrape_stage,
        stage_min_score=stage_min_score,
        posts_scraped=progress.posts_scraped or 0,
        videos_downloaded=progress.videos_downloaded or 0,
        videos_failed=progress.videos_failed or 0,
        success_rate=success_rate,
        last_post_score=progress.last_post_score,
        scraping_active=progress.scraping_active,
        hours_since_scrape=hours_since
    )


@router.get("/", response_model=List[SubredditStatus])
def list_subreddits(db: Session = Depends(get_db)):
    """List all subreddits from database."""
    try:
        progress_records = db.query(ScrapingProgress).order_by(ScrapingProgress.subreddit).all()
        return [_progress_to_status(p) for p in progress_records]
    except Exception as e:
        logger.error(f"Error listing subreddits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=SubredditStatus)
def add_subreddit(subreddit: SubredditConfig, db: Session = Depends(get_db)):
    """Add a new subreddit to the database."""
    try:
        subreddit_name = subreddit.name.lower()

        # Validate scrape stage
        if subreddit.scrape_stage not in SCRAPE_STAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scrape_stage '{subreddit.scrape_stage}'. Valid stages: {SCRAPE_STAGES}"
            )

        # Check if already exists
        existing = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Subreddit '{subreddit_name}' already exists")

        # Create new record
        new_progress = ScrapingProgress(
            subreddit=subreddit_name,
            description=subreddit.description or f"Scraper for r/{subreddit_name}",
            batch_size=subreddit.batch_size,
            target_min_score=subreddit.min_score,
            scrape_stage=subreddit.scrape_stage,
            scraping_active=subreddit.enabled
        )
        db.add(new_progress)
        db.commit()
        db.refresh(new_progress)

        logger.info(f"Added new subreddit: {subreddit_name} (stage: {subreddit.scrape_stage})")
        return _progress_to_status(new_progress)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding subreddit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{subreddit_name}", response_model=SubredditStatus)
def update_subreddit(subreddit_name: str, update: SubredditUpdate, db: Session = Depends(get_db)):
    """Update subreddit settings."""
    try:
        subreddit_name = subreddit_name.lower()
        progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

        if not progress:
            raise HTTPException(status_code=404, detail=f"Subreddit '{subreddit_name}' not found")

        # Update fields
        if update.enabled is not None:
            progress.scraping_active = update.enabled
        if update.min_score is not None:
            progress.target_min_score = update.min_score
        if update.batch_size is not None:
            progress.batch_size = update.batch_size
        if update.description is not None:
            progress.description = update.description

        db.commit()
        db.refresh(progress)

        logger.info(f"Updated subreddit: {subreddit_name}")
        return _progress_to_status(progress)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating subreddit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{subreddit_name}")
def delete_subreddit(subreddit_name: str, delete_data: bool = False, db: Session = Depends(get_db)):
    """Remove a subreddit. Optionally delete all scraped data."""
    try:
        subreddit_name = subreddit_name.lower()
        progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

        if not progress:
            raise HTTPException(status_code=404, detail=f"Subreddit '{subreddit_name}' not found")

        deleted_videos = 0

        if delete_data:
            # Delete videos and captions
            from database.models import Video, ScrapedCaption
            videos = db.query(Video).filter_by(source_subreddit=subreddit_name).all()
            for video in videos:
                db.query(ScrapedCaption).filter_by(video_id=video.id).delete()
                db.delete(video)
                deleted_videos += 1

        # Delete the progress record
        db.delete(progress)
        db.commit()

        logger.info(f"Deleted subreddit: {subreddit_name} (data deleted: {delete_data})")

        return {
            "message": f"Subreddit '{subreddit_name}' removed",
            "deleted_videos": deleted_videos
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting subreddit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{subreddit_name}/toggle")
def toggle_subreddit(subreddit_name: str, db: Session = Depends(get_db)):
    """Toggle a subreddit's enabled status."""
    try:
        subreddit_name = subreddit_name.lower()
        progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

        if not progress:
            raise HTTPException(status_code=404, detail=f"Subreddit '{subreddit_name}' not found")

        # Toggle
        progress.scraping_active = not progress.scraping_active
        db.commit()

        return {
            "subreddit": subreddit_name,
            "enabled": progress.scraping_active,
            "message": f"Subreddit {'enabled' if progress.scraping_active else 'disabled'}"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling subreddit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{subreddit_name}/reset")
def reset_subreddit_progress(
    subreddit_name: str,
    stage: Optional[str] = "top_all",
    db: Session = Depends(get_db)
):
    """
    Reset scraping progress for a subreddit (restart from beginning or specific stage).

    Args:
        subreddit_name: Name of the subreddit to reset
        stage: Stage to reset to (default: 'top_all')
               Valid stages: top_all, top_year, top_month, top_week, top_day, new
    """
    try:
        subreddit_name = subreddit_name.lower()

        # Validate stage
        if stage not in SCRAPE_STAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage '{stage}'. Valid stages: {SCRAPE_STAGES}"
            )

        progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

        if not progress:
            raise HTTPException(status_code=404, detail=f"Subreddit '{subreddit_name}' not found")

        old_stage = getattr(progress, 'scrape_stage', None) or 'top_all'

        # Reset progress fields
        progress.scrape_stage = stage
        progress.last_post_id = None
        progress.last_post_score = None
        progress.last_pagination_url = None
        progress.posts_scraped = 0
        # Note: videos_downloaded is cumulative, don't reset
        progress.videos_failed = 0
        progress.scraping_active = True
        db.commit()

        logger.info(f"Reset scraping progress for {subreddit_name}: '{old_stage}' -> '{stage}'")

        return {
            "subreddit": subreddit_name,
            "previous_stage": old_stage,
            "new_stage": stage,
            "message": f"Scraping progress reset to stage '{stage}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stages")
def list_scrape_stages():
    """List all available scrape stages and their configurations."""
    return {
        "stages": SCRAPE_STAGES,
        "stage_configs": STAGE_MIN_SCORES,
        "description": "Progression: top_all -> top_year -> top_month -> top_week -> top_day -> new"
    }


@router.post("/{subreddit_name}/scrape")
def trigger_subreddit_scrape(subreddit_name: str, batch_size: Optional[int] = None, db: Session = Depends(get_db)):
    """Manually trigger a scrape for a specific subreddit."""
    try:
        subreddit_name = subreddit_name.lower()
        progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

        if not progress:
            raise HTTPException(status_code=404, detail=f"Subreddit '{subreddit_name}' not found")

        # Use stored batch_size if not provided
        actual_batch_size = batch_size or getattr(progress, 'batch_size', None) or 25
        min_score = progress.target_min_score or 300

        # Trigger the scrape task
        from tasks.incremental_scraping import incremental_scrape_subreddit
        task = incremental_scrape_subreddit.delay(subreddit_name, actual_batch_size, min_score)

        return {
            "task_id": task.id,
            "subreddit": subreddit_name,
            "batch_size": actual_batch_size,
            "min_score": min_score,
            "message": f"Scrape task queued for r/{subreddit_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering scrape: {e}")
        raise HTTPException(status_code=500, detail=str(e))
