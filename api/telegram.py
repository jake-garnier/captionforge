"""
Telegram API - Bot and channel management, video publishing.

Endpoints:
- /telegram/bots - CRUD for Telegram bots (per niche)
- /telegram/channels - CRUD for Telegram channels (per niche)
- /telegram/publish - Publish composed videos to Telegram
- /telegram/jobs - View publish job history
"""

import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.automation_config import get_automation_config
from database.db import get_db
from database.models import (
    TelegramBot,
    TelegramChannel,
    TelegramPublishJob,
    ComposedVideo,
    BackgroundVideo,
)
from publishers.telegram_publisher import TelegramPublisher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


# ============================================================================
# Pydantic Models
# ============================================================================

class BotCreate(BaseModel):
    niche: str
    bot_username: str
    bot_token: str
    bot_name: Optional[str] = None


class BotUpdate(BaseModel):
    bot_username: Optional[str] = None
    bot_token: Optional[str] = None
    bot_name: Optional[str] = None
    is_enabled: Optional[bool] = None


class ChannelCreate(BaseModel):
    niche: str
    channel_id: str
    channel_username: Optional[str] = None
    channel_name: Optional[str] = None


class ChannelUpdate(BaseModel):
    channel_id: Optional[str] = None
    channel_username: Optional[str] = None
    channel_name: Optional[str] = None
    is_enabled: Optional[bool] = None


class PublishRequest(BaseModel):
    """Schedule (default) or immediately publish a composed video to Telegram."""
    publish_immediately: bool = False  # If True, post now instead of scheduling
    scheduled_date: Optional[str] = None  # YYYY-MM-DD; defaults to next free day
    posting_hour_utc: Optional[int] = None
    posting_minute: Optional[int] = None


# ============================================================================
# Bot Management
# ============================================================================

@router.get("/bots")
def list_bots(db: Session = Depends(get_db)):
    """List all configured Telegram bots."""
    bots = db.query(TelegramBot).order_by(TelegramBot.niche).all()
    return [
        {
            "id": bot.id,
            "niche": bot.niche,
            "bot_username": bot.bot_username,
            "bot_name": bot.bot_name,
            "is_enabled": bot.is_enabled,
            "created_at": bot.created_at.isoformat() if bot.created_at else None,
            # Don't expose full token, just last 4 chars
            "token_hint": f"...{bot.bot_token[-4:]}" if bot.bot_token else None,
        }
        for bot in bots
    ]


@router.post("/bots")
async def create_or_update_bot(data: BotCreate, db: Session = Depends(get_db)):
    """
    Create or update a Telegram bot for a niche.

    Verifies the bot token is valid before saving.
    """
    # Verify bot token
    try:
        publisher = TelegramPublisher(data.bot_token)
        bot_info = await publisher.verify_bot()
    except Exception as e:
        raise HTTPException(400, f"Invalid bot token: {str(e)}")

    # Check if bot already exists for this niche
    existing = db.query(TelegramBot).filter_by(niche=data.niche).first()

    if existing:
        existing.bot_username = data.bot_username or bot_info.get("username")
        existing.bot_token = data.bot_token
        existing.bot_name = data.bot_name or bot_info.get("first_name")
        existing.updated_at = datetime.utcnow()
        db.commit()
        return {
            "status": "updated",
            "id": existing.id,
            "bot_info": bot_info,
        }
    else:
        bot = TelegramBot(
            niche=data.niche,
            bot_username=data.bot_username or bot_info.get("username"),
            bot_token=data.bot_token,
            bot_name=data.bot_name or bot_info.get("first_name"),
        )
        db.add(bot)
        db.commit()
        db.refresh(bot)
        return {
            "status": "created",
            "id": bot.id,
            "bot_info": bot_info,
        }


@router.get("/bots/{niche}")
def get_bot(niche: str, db: Session = Depends(get_db)):
    """Get bot configuration for a niche."""
    bot = db.query(TelegramBot).filter_by(niche=niche).first()
    if not bot:
        raise HTTPException(404, f"No bot configured for niche: {niche}")

    return {
        "id": bot.id,
        "niche": bot.niche,
        "bot_username": bot.bot_username,
        "bot_name": bot.bot_name,
        "is_enabled": bot.is_enabled,
        "token_hint": f"...{bot.bot_token[-4:]}" if bot.bot_token else None,
    }


@router.delete("/bots/{niche}")
def delete_bot(niche: str, db: Session = Depends(get_db)):
    """Delete a bot configuration."""
    bot = db.query(TelegramBot).filter_by(niche=niche).first()
    if not bot:
        raise HTTPException(404, f"No bot configured for niche: {niche}")

    db.delete(bot)
    db.commit()
    return {"status": "deleted", "niche": niche}


@router.post("/bots/{niche}/verify")
async def verify_bot(niche: str, db: Session = Depends(get_db)):
    """Verify a bot token is valid and get bot info."""
    bot = db.query(TelegramBot).filter_by(niche=niche).first()
    if not bot:
        raise HTTPException(404, f"No bot configured for niche: {niche}")

    try:
        publisher = TelegramPublisher(bot.bot_token)
        bot_info = await publisher.verify_bot()
        return {
            "status": "valid",
            "bot_info": bot_info,
        }
    except Exception as e:
        return {
            "status": "invalid",
            "error": str(e),
        }


@router.post("/bots/{niche}/discover-channels")
async def discover_channels(niche: str, db: Session = Depends(get_db)):
    """
    Discover channel IDs from recent bot updates.

    The bot must have received at least one message in the channel
    (add bot as admin, then post something).
    """
    bot = db.query(TelegramBot).filter_by(niche=niche).first()
    if not bot:
        raise HTTPException(404, f"No bot configured for niche: {niche}")

    try:
        publisher = TelegramPublisher(bot.bot_token)
        channels = await publisher.discover_channel_id()
        return {
            "status": "ok",
            "channels": channels,
            "hint": "If no channels found, make sure the bot is added as admin to the channel and post a message there.",
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to discover channels: {str(e)}")


# ============================================================================
# Channel Management
# ============================================================================

@router.get("/channels")
def list_channels(db: Session = Depends(get_db)):
    """List all configured Telegram channels."""
    channels = db.query(TelegramChannel).order_by(TelegramChannel.niche).all()
    return [
        {
            "id": ch.id,
            "niche": ch.niche,
            "channel_id": ch.channel_id,
            "channel_username": ch.channel_username,
            "channel_name": ch.channel_name,
            "is_enabled": ch.is_enabled,
            "bot_id": ch.bot_id,
            "created_at": ch.created_at.isoformat() if ch.created_at else None,
        }
        for ch in channels
    ]


@router.post("/channels")
async def create_or_update_channel(data: ChannelCreate, db: Session = Depends(get_db)):
    """
    Create or update a Telegram channel for a niche.

    Attempts to get channel info to verify it's valid.
    """
    # Find bot for this niche to verify channel
    bot = db.query(TelegramBot).filter_by(niche=data.niche).first()

    channel_info = None
    if bot:
        try:
            publisher = TelegramPublisher(bot.bot_token)
            channel_info = await publisher.get_chat_info(data.channel_id)
        except Exception as e:
            logger.warning(f"Could not verify channel {data.channel_id}: {e}")

    # Check if channel already exists for this niche
    existing = db.query(TelegramChannel).filter_by(niche=data.niche).first()

    if existing:
        existing.channel_id = data.channel_id
        existing.channel_username = data.channel_username or (channel_info.get("username") if channel_info else None)
        existing.channel_name = data.channel_name or (channel_info.get("title") if channel_info else None)
        if bot:
            existing.bot_id = bot.id
        existing.updated_at = datetime.utcnow()
        db.commit()
        return {
            "status": "updated",
            "id": existing.id,
            "channel_info": channel_info,
        }
    else:
        channel = TelegramChannel(
            niche=data.niche,
            channel_id=data.channel_id,
            channel_username=data.channel_username or (channel_info.get("username") if channel_info else None),
            channel_name=data.channel_name or (channel_info.get("title") if channel_info else None),
            bot_id=bot.id if bot else None,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)
        return {
            "status": "created",
            "id": channel.id,
            "channel_info": channel_info,
        }


@router.get("/channels/{niche}")
def get_channel(niche: str, db: Session = Depends(get_db)):
    """Get channel configuration for a niche."""
    channel = db.query(TelegramChannel).filter_by(niche=niche).first()
    if not channel:
        raise HTTPException(404, f"No channel configured for niche: {niche}")

    return {
        "id": channel.id,
        "niche": channel.niche,
        "channel_id": channel.channel_id,
        "channel_username": channel.channel_username,
        "channel_name": channel.channel_name,
        "is_enabled": channel.is_enabled,
        "bot_id": channel.bot_id,
    }


@router.delete("/channels/{niche}")
def delete_channel(niche: str, db: Session = Depends(get_db)):
    """Delete a channel configuration."""
    channel = db.query(TelegramChannel).filter_by(niche=niche).first()
    if not channel:
        raise HTTPException(404, f"No channel configured for niche: {niche}")

    db.delete(channel)
    db.commit()
    return {"status": "deleted", "niche": niche}


@router.post("/channels/{niche}/test")
async def test_channel(niche: str, db: Session = Depends(get_db)):
    """Send a test message to verify channel is configured correctly."""
    channel = db.query(TelegramChannel).filter_by(niche=niche).first()
    if not channel:
        raise HTTPException(404, f"No channel configured for niche: {niche}")

    bot = db.query(TelegramBot).filter_by(niche=niche).first()
    if not bot:
        raise HTTPException(400, f"No bot configured for niche: {niche}")

    try:
        publisher = TelegramPublisher(bot.bot_token)
        result = await publisher.send_message(
            channel.channel_id,
            "🎬 Test message from captions bot!"
        )
        return {
            "status": "ok",
            "message_id": result["message_id"],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to send test message: {str(e)}")


# ============================================================================
# Publishing
# ============================================================================

@router.post("/publish/{composed_video_id}")
async def publish_to_telegram(
    composed_video_id: int,
    data: PublishRequest = None,
    db: Session = Depends(get_db)
):
    """
    Schedule (default) or immediately publish a composed video to its
    niche's Telegram channel.

    Default behavior schedules for the next open day for the niche (one
    post per niche per day, building a backlog). Pass publish_immediately=True
    to dispatch the publish task right away.
    """
    if data is None:
        data = PublishRequest()

    video = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
    if not video:
        raise HTTPException(404, "Composed video not found")
    if not video.niche:
        raise HTTPException(400, "Video has no niche assigned")

    channel = db.query(TelegramChannel).filter_by(
        niche=video.niche,
        is_enabled=True
    ).first()
    if not channel:
        raise HTTPException(400, f"No Telegram channel configured for niche: {video.niche}")

    bot = db.query(TelegramBot).filter_by(
        niche=video.niche,
        is_enabled=True
    ).first()
    if not bot:
        raise HTTPException(400, f"No Telegram bot configured for niche: {video.niche}")

    # Block stacking multiple in-flight or queued Telegram jobs on one video.
    existing = db.query(TelegramPublishJob).filter(
        TelegramPublishJob.composed_video_id == composed_video_id,
        TelegramPublishJob.status.in_(["scheduled", "pending", "uploading"]),
    ).first()
    if existing:
        raise HTTPException(
            400,
            f"Video already has an active Telegram publish job (id={existing.id}, status={existing.status})",
        )

    if data.publish_immediately:
        job = TelegramPublishJob(
            composed_video_id=composed_video_id,
            niche=video.niche,
            channel_id=channel.id,
            bot_id=bot.id,
            status="pending",
            caption_text="",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        from tasks.telegram_tasks import publish_to_telegram_task
        task = publish_to_telegram_task.delay(job.id)
        job.celery_task_id = task.id
        db.commit()

        return {
            "status": "queued",
            "job_id": job.id,
            "task_id": task.id,
            "niche": video.niche,
            "channel": channel.channel_name or channel.channel_id,
        }

    # Schedule for next open day for this niche
    if data.scheduled_date:
        try:
            target_date = date.fromisoformat(data.scheduled_date)
        except ValueError:
            raise HTTPException(400, "Invalid scheduled_date format. Use YYYY-MM-DD")
    else:
        target_date = _next_open_telegram_date(video.niche, db)

    post_time = _telegram_post_time_for(
        target_date,
        hour=data.posting_hour_utc,
        minute=data.posting_minute,
    )

    job = TelegramPublishJob(
        composed_video_id=composed_video_id,
        niche=video.niche,
        channel_id=channel.id,
        bot_id=bot.id,
        status="scheduled",
        caption_text="",
        scheduled_date=post_time,
        base_post_time=post_time,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "status": "scheduled",
        "job_id": job.id,
        "niche": video.niche,
        "channel": channel.channel_name or channel.channel_id,
        "scheduled_date": target_date.isoformat(),
        "post_time": post_time.isoformat(),
    }


@router.post("/jobs/{job_id}/publish-now")
async def publish_telegram_now(job_id: int, db: Session = Depends(get_db)):
    """Skip the schedule and dispatch a Telegram job immediately."""
    job = db.query(TelegramPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("scheduled", "pending", "failed"):
        raise HTTPException(400, f"Cannot publish job with status '{job.status}'")

    if job.status == "failed":
        job.error_message = None
    job.status = "pending"
    db.commit()

    from tasks.telegram_tasks import publish_to_telegram_task
    task = publish_to_telegram_task.delay(job.id)
    job.celery_task_id = task.id
    db.commit()

    return {"status": "dispatched", "job_id": job.id, "task_id": task.id}


# ============================================================================
# Scheduling Helpers
# ============================================================================

def _next_open_telegram_date(niche: str, db: Session) -> date:
    """Find the next day (>= today) with no scheduled/pending/uploading/completed
    Telegram job for this niche.

    Today counts as available if we haven't already posted today and no
    in-flight job is reserving today's slot — that way the first post of the
    day lands today instead of getting pushed to tomorrow.
    """
    taken: set = set()
    rows = db.query(TelegramPublishJob).filter(
        TelegramPublishJob.niche == niche,
        TelegramPublishJob.status.in_(["scheduled", "pending", "uploading", "completed"]),
    ).all()
    for row in rows:
        # For completed jobs, prefer the actual posted timestamp;
        # for in-flight jobs, the scheduled date is what reserves the slot.
        d = row.completed_at or row.scheduled_date or row.base_post_time
        if d:
            taken.add(d.date() if hasattr(d, "date") else d)

    target = date.today()
    while target in taken:
        target += timedelta(days=1)
    return target


def _telegram_post_time_for(target_date: date, hour: Optional[int] = None, minute: Optional[int] = None) -> datetime:
    """Build a UTC datetime at the configured posting hour for the given date."""
    cfg = get_automation_config().telegram_schedule
    h = hour if hour is not None else cfg.default_posting_hour_utc
    m = minute if minute is not None else cfg.default_posting_minute
    return datetime(target_date.year, target_date.month, target_date.day, h, m, tzinfo=timezone.utc)


def _telegram_fetch_bg_info(db: Session, jobs) -> dict:
    """Bulk-load background-video source info for a list of TelegramPublishJobs.

    Returns: composed_video_id -> {bg_source_url, bg_source_type, bg_subreddit}.
    Avoids N+1 round-trips when listing many jobs.
    """
    composed_ids = [j.composed_video_id for j in jobs if j.composed_video_id]
    if not composed_ids:
        return {}

    rows = (
        db.query(
            ComposedVideo.id,
            BackgroundVideo.source_url,
            BackgroundVideo.source_type,
            BackgroundVideo.reddit_subreddit,
        )
        .join(BackgroundVideo, ComposedVideo.background_video_id == BackgroundVideo.id)
        .filter(ComposedVideo.id.in_(composed_ids))
        .all()
    )
    return {
        composed_id: {
            "bg_source_url": source_url,
            "bg_source_type": source_type,
            "bg_subreddit": subreddit,
        }
        for composed_id, source_url, source_type, subreddit in rows
    }


@router.get("/jobs")
def list_jobs(
    status: Optional[str] = None,
    niche: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """List Telegram publish jobs."""
    query = db.query(TelegramPublishJob)

    if status:
        query = query.filter_by(status=status)
    if niche:
        query = query.filter_by(niche=niche)

    jobs = query.order_by(TelegramPublishJob.created_at.desc()).limit(limit).all()
    bg_lookup = _telegram_fetch_bg_info(db, jobs)

    return [
        {
            "id": job.id,
            "composed_video_id": job.composed_video_id,
            "niche": job.niche,
            "status": job.status,
            "telegram_message_id": job.telegram_message_id,
            "telegram_post_url": job.telegram_post_url,
            "error_message": job.error_message,
            "scheduled_date": job.scheduled_date.isoformat() if job.scheduled_date else None,
            "base_post_time": job.base_post_time.isoformat() if job.base_post_time else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            **(bg_lookup.get(job.composed_video_id) or {}),
        }
        for job in jobs
    ]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific publish job."""
    job = db.query(TelegramPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "id": job.id,
        "composed_video_id": job.composed_video_id,
        "niche": job.niche,
        "status": job.status,
        "telegram_message_id": job.telegram_message_id,
        "telegram_post_url": job.telegram_post_url,
        "caption_text": job.caption_text,
        "error_message": job.error_message,
        "celery_task_id": job.celery_task_id,
        "scheduled_date": job.scheduled_date.isoformat() if job.scheduled_date else None,
        "base_post_time": job.base_post_time.isoformat() if job.base_post_time else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: int, db: Session = Depends(get_db)):
    """Retry a failed publish job."""
    job = db.query(TelegramPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status not in ("failed", "cancelled"):
        raise HTTPException(400, f"Cannot retry job with status: {job.status}")

    job.status = "pending"
    job.error_message = None
    job.completed_at = None
    db.commit()

    from tasks.telegram_tasks import publish_to_telegram_task
    task = publish_to_telegram_task.delay(job.id)

    job.celery_task_id = task.id
    db.commit()

    return {"status": "retrying", "job_id": job.id, "task_id": task.id}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a publish job."""
    job = db.query(TelegramPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    db.delete(job)
    db.commit()
    return {"status": "deleted", "job_id": job_id}


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get Telegram publishing statistics."""
    from sqlalchemy import func

    total_jobs = db.query(func.count(TelegramPublishJob.id)).scalar()
    completed_jobs = db.query(func.count(TelegramPublishJob.id)).filter_by(status="completed").scalar()
    failed_jobs = db.query(func.count(TelegramPublishJob.id)).filter_by(status="failed").scalar()
    pending_jobs = db.query(func.count(TelegramPublishJob.id)).filter_by(status="pending").scalar()

    # Stats by niche
    by_niche = db.query(
        TelegramPublishJob.niche,
        func.count(TelegramPublishJob.id)
    ).filter_by(status="completed").group_by(TelegramPublishJob.niche).all()

    return {
        "total_jobs": total_jobs,
        "completed": completed_jobs,
        "failed": failed_jobs,
        "pending": pending_jobs,
        "by_niche": {f: c for f, c in by_niche},
    }
