"""
Telegram Scraping API - Manage Telegram channel scraping for training data.

Endpoints:
- /telegram/scrape/auth/* - Authentication flow (phone + code verification)
- /telegram/scrape/channels - CRUD for channels to scrape
- /telegram/scrape/control - Enable/disable scheduled scraping
- /telegram/scrape/{channel_id}/* - Manual scrape triggers, progress
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import TelegramScrapeChannel, ScrapingProgress
from scrapers.telegram_scraper import TelegramScraperAuth, TelegramScraper
from tasks.telegram_scraping import (
    is_telegram_scraper_enabled,
    set_telegram_scraper_enabled,
    scrape_telegram_channel_manual,
)
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram/scrape", tags=["telegram-scraping"])


# ============================================================================
# Pydantic Models
# ============================================================================

class AuthStartRequest(BaseModel):
    phone: str  # Phone number in international format (+1234567890)


class AuthVerifyRequest(BaseModel):
    phone: str
    code: str
    phone_code_hash: str
    password: Optional[str] = None  # For 2FA


class ChannelCreate(BaseModel):
    channel_id: str  # Channel ID or @username
    channel_name: Optional[str] = None
    batch_size: int = 25


class ChannelUpdate(BaseModel):
    channel_name: Optional[str] = None
    batch_size: Optional[int] = None
    is_enabled: Optional[bool] = None


# ============================================================================
# Authentication
# ============================================================================

@router.get("/auth/status")
async def auth_status():
    """Check if Telegram is authenticated."""
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        return {
            "status": "not_configured",
            "message": "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in environment"
        }

    auth = TelegramScraperAuth()
    is_auth = await auth.is_authenticated()

    return {
        "status": "authenticated" if is_auth else "not_authenticated",
        "session_path": settings.TELEGRAM_SESSION_PATH
    }


@router.post("/auth/start")
async def auth_start(data: AuthStartRequest):
    """
    Start authentication by sending verification code.

    Args:
        phone: Phone number in international format (+1234567890)

    Returns:
        phone_code_hash to use in verify step
    """
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        raise HTTPException(
            400,
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in environment"
        )

    auth = TelegramScraperAuth()
    result = await auth.start_auth(data.phone)

    if result["status"] == "error":
        raise HTTPException(400, result["message"])

    return result


@router.post("/auth/verify")
async def auth_verify(data: AuthVerifyRequest):
    """
    Complete authentication by verifying the code.

    Args:
        phone: Same phone number used in start
        code: Verification code received
        phone_code_hash: Hash from start response
        password: 2FA password (if enabled)
    """
    auth = TelegramScraperAuth()
    result = await auth.verify_code(
        phone=data.phone,
        code=data.code,
        phone_code_hash=data.phone_code_hash,
        password=data.password
    )

    if result["status"] == "error":
        raise HTTPException(400, result["message"])

    return result


@router.get("/auth/dialogs")
async def get_dialogs():
    """
    Get all dialogs (chats/channels) accessible to the authenticated user.
    Useful for discovering channel IDs.
    """
    auth = TelegramScraperAuth()
    if not await auth.is_authenticated():
        raise HTTPException(401, "Not authenticated. Complete authentication first.")

    async with TelegramScraper() as scraper:
        dialogs = await scraper.get_dialogs()

    # Filter to just channels and groups
    channels_and_groups = [
        d for d in dialogs
        if d["type"] in ("channel", "supergroup", "group", "gigagroup")
    ]

    return {
        "dialogs": channels_and_groups,
        "total": len(dialogs),
        "channels_and_groups": len(channels_and_groups)
    }


# ============================================================================
# Scraper Control
# ============================================================================

@router.get("/control/status")
def get_control_status():
    """Get Telegram scraper enabled status."""
    return {
        "enabled": is_telegram_scraper_enabled(),
        "api_configured": bool(settings.TELEGRAM_API_ID and settings.TELEGRAM_API_HASH)
    }


@router.post("/control/enable")
def enable_scraper():
    """Enable scheduled Telegram scraping."""
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        raise HTTPException(
            400,
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set before enabling"
        )

    set_telegram_scraper_enabled(True)
    return {"status": "enabled"}


@router.post("/control/disable")
def disable_scraper():
    """Disable scheduled Telegram scraping."""
    set_telegram_scraper_enabled(False)
    return {"status": "disabled"}


# ============================================================================
# Channel Management
# ============================================================================

@router.get("/channels")
def list_channels(db: Session = Depends(get_db)):
    """List all configured channels to scrape."""
    channels = db.query(TelegramScrapeChannel).order_by(
        TelegramScrapeChannel.channel_name
    ).all()

    result = []
    for ch in channels:
        # Get scraping progress
        progress_key = f"telegram:{ch.channel_id}"
        progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()

        result.append({
            "id": ch.id,
            "channel_id": ch.channel_id,
            "channel_username": ch.channel_username,
            "channel_name": ch.channel_name,
            "is_enabled": ch.is_enabled,
            "batch_size": ch.batch_size,
            "last_scrape_at": ch.last_scrape_at.isoformat() if ch.last_scrape_at else None,
            "videos_downloaded": progress.videos_downloaded if progress else 0,
            "posts_scraped": progress.posts_scraped if progress else 0,
        })

    return result


@router.post("/channels")
async def add_channel(data: ChannelCreate, db: Session = Depends(get_db)):
    """
    Add a channel to scrape.

    Can use channel ID (e.g., -1001234567890) or @username.
    """
    # Check if already exists
    existing = db.query(TelegramScrapeChannel).filter_by(
        channel_id=data.channel_id
    ).first()
    if existing:
        raise HTTPException(400, f"Channel already configured: {data.channel_id}")

    # Try to get channel info if authenticated
    channel_info = None
    auth = TelegramScraperAuth()
    if await auth.is_authenticated():
        try:
            async with TelegramScraper() as scraper:
                channel_info = await scraper.get_channel_info(data.channel_id)
        except Exception as e:
            logger.warning(f"Could not get channel info: {e}")

    # Create channel record
    channel = TelegramScrapeChannel(
        channel_id=data.channel_id,
        channel_username=channel_info.get("username") if channel_info else None,
        channel_name=data.channel_name or (channel_info.get("title") if channel_info else None),
        batch_size=data.batch_size,
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)

    return {
        "status": "created",
        "id": channel.id,
        "channel_id": channel.channel_id,
        "channel_name": channel.channel_name,
        "channel_info": channel_info
    }


@router.get("/channels/{channel_id}")
def get_channel(channel_id: str, db: Session = Depends(get_db)):
    """Get channel details and scraping progress."""
    channel = db.query(TelegramScrapeChannel).filter_by(channel_id=channel_id).first()
    if not channel:
        raise HTTPException(404, f"Channel not found: {channel_id}")

    # Get progress
    progress_key = f"telegram:{channel_id}"
    progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()

    return {
        "id": channel.id,
        "channel_id": channel.channel_id,
        "channel_username": channel.channel_username,
        "channel_name": channel.channel_name,
        "is_enabled": channel.is_enabled,
        "batch_size": channel.batch_size,
        "created_at": channel.created_at.isoformat() if channel.created_at else None,
        "last_scrape_at": channel.last_scrape_at.isoformat() if channel.last_scrape_at else None,
        "progress": {
            "videos_downloaded": progress.videos_downloaded if progress else 0,
            "posts_scraped": progress.posts_scraped if progress else 0,
            "videos_failed": progress.videos_failed if progress else 0,
            "last_post_id": progress.last_post_id if progress else None,
        } if progress else None
    }


@router.put("/channels/{channel_id}")
def update_channel(channel_id: str, data: ChannelUpdate, db: Session = Depends(get_db)):
    """Update channel configuration."""
    channel = db.query(TelegramScrapeChannel).filter_by(channel_id=channel_id).first()
    if not channel:
        raise HTTPException(404, f"Channel not found: {channel_id}")

    if data.channel_name is not None:
        channel.channel_name = data.channel_name
    if data.batch_size is not None:
        channel.batch_size = data.batch_size
    if data.is_enabled is not None:
        channel.is_enabled = data.is_enabled

    channel.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "updated", "channel_id": channel_id}


@router.delete("/channels/{channel_id}")
def delete_channel(
    channel_id: str,
    delete_progress: bool = Query(default=False),
    db: Session = Depends(get_db)
):
    """Delete a channel configuration."""
    channel = db.query(TelegramScrapeChannel).filter_by(channel_id=channel_id).first()
    if not channel:
        raise HTTPException(404, f"Channel not found: {channel_id}")

    db.delete(channel)

    # Optionally delete progress
    if delete_progress:
        progress_key = f"telegram:{channel_id}"
        progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()
        if progress:
            db.delete(progress)

    db.commit()

    return {"status": "deleted", "channel_id": channel_id}


@router.post("/channels/{channel_id}/toggle")
def toggle_channel(channel_id: str, db: Session = Depends(get_db)):
    """Toggle channel enabled status."""
    channel = db.query(TelegramScrapeChannel).filter_by(channel_id=channel_id).first()
    if not channel:
        raise HTTPException(404, f"Channel not found: {channel_id}")

    channel.is_enabled = not channel.is_enabled
    channel.updated_at = datetime.utcnow()
    db.commit()

    return {
        "status": "toggled",
        "channel_id": channel_id,
        "is_enabled": channel.is_enabled
    }


# ============================================================================
# Manual Scraping
# ============================================================================

@router.post("/channels/{channel_id}/scrape")
async def trigger_scrape(
    channel_id: str,
    batch_size: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Manually trigger a scrape of a channel."""
    # Verify channel exists
    channel = db.query(TelegramScrapeChannel).filter_by(channel_id=channel_id).first()
    if not channel:
        raise HTTPException(404, f"Channel not found: {channel_id}")

    # Check authentication
    auth = TelegramScraperAuth()
    if not await auth.is_authenticated():
        raise HTTPException(401, "Not authenticated. Complete authentication first.")

    # Trigger scrape task
    task = scrape_telegram_channel_manual.delay(channel_id, batch_size)

    return {
        "status": "queued",
        "task_id": task.id,
        "channel_id": channel_id,
        "batch_size": batch_size
    }


@router.get("/channels/{channel_id}/progress")
def get_progress(channel_id: str, db: Session = Depends(get_db)):
    """Get scraping progress for a channel."""
    progress_key = f"telegram:{channel_id}"
    progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()

    if not progress:
        return {
            "channel_id": channel_id,
            "status": "not_started",
            "videos_downloaded": 0,
            "posts_scraped": 0
        }

    return {
        "channel_id": channel_id,
        "status": "active" if progress.scraping_active else "paused",
        "videos_downloaded": progress.videos_downloaded,
        "posts_scraped": progress.posts_scraped,
        "videos_failed": progress.videos_failed,
        "last_post_id": progress.last_post_id,
        "last_scrape_at": progress.last_scrape_at.isoformat() if progress.last_scrape_at else None
    }


@router.post("/channels/{channel_id}/reset")
def reset_progress(channel_id: str, db: Session = Depends(get_db)):
    """Reset scraping progress for a channel (start from beginning)."""
    progress_key = f"telegram:{channel_id}"
    progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()

    if progress:
        progress.last_post_id = None
        progress.last_pagination_url = None
        progress.posts_scraped = 0
        progress.videos_downloaded = 0
        progress.videos_failed = 0
        progress.updated_at = datetime.utcnow()
        db.commit()

    return {"status": "reset", "channel_id": channel_id}


# ============================================================================
# Stats
# ============================================================================

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get Telegram scraping statistics."""
    from sqlalchemy import func

    # Count channels
    total_channels = db.query(func.count(TelegramScrapeChannel.id)).scalar()
    enabled_channels = db.query(func.count(TelegramScrapeChannel.id)).filter_by(
        is_enabled=True
    ).scalar()

    # Sum videos from all Telegram progress entries
    telegram_progress = db.query(ScrapingProgress).filter(
        ScrapingProgress.subreddit.like("telegram:%")
    ).all()

    total_videos = sum(p.videos_downloaded or 0 for p in telegram_progress)
    total_posts = sum(p.posts_scraped or 0 for p in telegram_progress)

    return {
        "total_channels": total_channels,
        "enabled_channels": enabled_channels,
        "total_videos_downloaded": total_videos,
        "total_posts_scraped": total_posts,
        "scraper_enabled": is_telegram_scraper_enabled(),
    }
