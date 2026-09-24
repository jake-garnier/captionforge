"""
Telegram scraping Celery tasks.

Tasks for scraping videos from private Telegram channels using Telethon.
"""

import logging
from datetime import datetime
from typing import Optional

import redis
from celery import group

from tasks.celery_app import celery_app
from database.db import get_db_context
from database.models import TelegramScrapeChannel, Video, ScrapingProgress
from scrapers.telegram_scraper import (
    is_authenticated_sync,
    get_media_messages_sync,
    download_media_sync,
)
from utils.extraction_queue import publish_video_downloaded
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis client for control flags
redis_client = redis.Redis.from_url(settings.REDIS_URL)

# Redis keys
TELEGRAM_SCRAPER_ENABLED_KEY = "telegram:scraper:enabled"


def is_telegram_scraper_enabled() -> bool:
    """Check if Telegram scraping is enabled via Redis flag."""
    value = redis_client.get(TELEGRAM_SCRAPER_ENABLED_KEY)
    return value == b"true" if value else False


def set_telegram_scraper_enabled(enabled: bool):
    """Set Telegram scraping enabled state."""
    redis_client.set(TELEGRAM_SCRAPER_ENABLED_KEY, "true" if enabled else "false")


@celery_app.task(queue="maintenance")
def dispatch_telegram_scrapers():
    """
    Dispatcher task - checks enabled channels and triggers scraping.

    Runs on schedule via Celery Beat.
    Checks if Telegram scraping is enabled and authenticated.
    """
    # Check if scraping is enabled
    if not is_telegram_scraper_enabled():
        logger.debug("Telegram scraping is disabled")
        return {"status": "disabled"}

    # Check if authenticated
    if not is_authenticated_sync():
        logger.warning("Telegram not authenticated - skipping scrape")
        return {"status": "not_authenticated"}

    # Get enabled channels
    with get_db_context() as db:
        channels = db.query(TelegramScrapeChannel).filter_by(is_enabled=True).all()

        if not channels:
            logger.debug("No enabled Telegram channels to scrape")
            return {"status": "no_channels"}

        # Create scraping tasks for each channel
        tasks = []
        for channel in channels:
            tasks.append(
                scrape_telegram_channel.s(
                    channel_id=channel.channel_id,
                    batch_size=channel.batch_size
                )
            )

        # Execute in parallel
        if tasks:
            job = group(tasks)
            job.apply_async()

            return {
                "status": "dispatched",
                "channels": len(tasks)
            }

    return {"status": "ok"}


@celery_app.task(bind=True, queue="scraping", max_retries=3)
def scrape_telegram_channel(self, channel_id: str, batch_size: int = 25):
    """
    Incrementally scrape a Telegram channel for videos.

    Args:
        channel_id: Telegram channel ID or @username
        batch_size: Number of messages to fetch per batch

    Returns:
        Dict with scraping results
    """
    logger.info(f"Starting Telegram scrape for channel: {channel_id}")

    # Create progress key for this channel
    progress_key = f"telegram:{channel_id}"

    with get_db_context() as db:
        # Get or create scraping progress
        progress = db.query(ScrapingProgress).filter_by(subreddit=progress_key).first()

        if not progress:
            progress = ScrapingProgress(
                subreddit=progress_key,
                batch_size=batch_size,
                scrape_stage="new",  # Telegram always scrapes newest first
                target_min_score=0,  # No score filtering for Telegram
            )
            db.add(progress)
            db.commit()
            db.refresh(progress)

        # Get last scraped message ID (or 0 to start from latest)
        min_id = int(progress.last_post_id) if progress.last_post_id else 0

        try:
            # Fetch new messages
            messages, next_offset = get_media_messages_sync(
                channel_id=channel_id,
                limit=batch_size,
                min_id=min_id,
            )

            if not messages:
                logger.info(f"No new messages in {channel_id}")
                return {
                    "status": "no_new_messages",
                    "channel_id": channel_id,
                }

            logger.info(f"Found {len(messages)} media messages in {channel_id}")

            # Track results
            downloaded = 0
            skipped = 0
            failed = 0
            highest_msg_id = min_id

            # Process each message
            for msg in messages:
                msg_id = msg["id"]
                highest_msg_id = max(highest_msg_id, msg_id)

                # Create unique source_post_id for deduplication
                source_post_id = f"tg_{channel_id}_{msg_id}"

                # Check if already downloaded
                existing = db.query(Video).filter_by(source_post_id=source_post_id).first()
                if existing:
                    logger.debug(f"Skipping already downloaded: {source_post_id}")
                    skipped += 1
                    continue

                # Download the media
                result = download_media_sync(channel_id, msg_id)

                if not result:
                    logger.warning(f"Failed to download message {msg_id}")
                    failed += 1
                    continue

                # Check for duplicate by file hash
                if result["file_hash"]:
                    hash_exists = db.query(Video).filter_by(
                        file_hash=result["file_hash"]
                    ).first()
                    if hash_exists:
                        logger.debug(f"Duplicate file hash: {result['file_hash']}")
                        # Clean up downloaded file
                        try:
                            import os
                            os.remove(result["path"])
                        except:
                            pass
                        skipped += 1
                        continue

                # Create video record
                video = Video(
                    source_url=f"https://t.me/c/{channel_id}/{msg_id}",
                    storage_path=result["path"],
                    file_hash=result["file_hash"],
                    file_size_bytes=result["file_size"],
                    source_subreddit=progress_key,  # Use progress key as source
                    source_post_id=source_post_id,
                    media_type=result["media_type"],
                    upvotes=result.get("views", 0),
                    processing_status="downloaded",
                    gallery_added_at=datetime.utcnow(),
                )
                db.add(video)
                db.commit()
                db.refresh(video)

                # Queue for caption extraction
                publish_video_downloaded(video.id, {
                    "source": "telegram",
                    "channel_id": channel_id,
                    "message_id": msg_id,
                })

                downloaded += 1
                logger.info(f"Downloaded: {source_post_id} ({result['media_type']})")

            # Update progress
            progress.last_post_id = str(highest_msg_id)
            progress.posts_scraped = (progress.posts_scraped or 0) + len(messages)
            progress.videos_downloaded = (progress.videos_downloaded or 0) + downloaded
            progress.videos_failed = (progress.videos_failed or 0) + failed
            progress.last_scrape_at = datetime.utcnow()
            db.commit()

            # Update channel last_scrape_at
            channel = db.query(TelegramScrapeChannel).filter_by(
                channel_id=channel_id
            ).first()
            if channel:
                channel.last_scrape_at = datetime.utcnow()
                db.commit()

            return {
                "status": "completed",
                "channel_id": channel_id,
                "messages_processed": len(messages),
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "last_message_id": highest_msg_id,
            }

        except Exception as e:
            logger.error(f"Telegram scrape failed for {channel_id}: {e}")

            # Retry on transient errors
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise self.retry(exc=e, countdown=60)

            return {
                "status": "error",
                "channel_id": channel_id,
                "error": str(e),
            }


@celery_app.task(queue="scraping")
def scrape_telegram_channel_manual(channel_id: str, batch_size: int = 25):
    """
    Manual trigger for scraping a specific channel.
    Same as scrape_telegram_channel but without retry logic.
    """
    return scrape_telegram_channel(channel_id, batch_size)
