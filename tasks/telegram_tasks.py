"""
Telegram publishing Celery tasks.

Tasks for posting composed videos to Telegram channels. Jobs land in the
table with status='scheduled' and a base_post_time; the beat dispatcher
promotes them to 'pending' and calls publish_to_telegram_task when their
slot arrives.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from tasks.celery_app import celery_app
from database.db import get_db_context
from database.models import (
    TelegramBot,
    TelegramChannel,
    TelegramPublishJob,
    ComposedVideo,
)
from publishers.telegram_publisher import TelegramPublisher

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, queue="publishing", max_retries=3)
def publish_to_telegram_task(self, job_id: int):
    """
    Publish a composed video to Telegram.

    Args:
        job_id: TelegramPublishJob ID

    Returns:
        Dict with status and post URL
    """
    with get_db_context() as db:
        job = db.query(TelegramPublishJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Telegram publish job {job_id} not found")
            return {"error": "Job not found"}

        if job.status == "completed":
            logger.info(f"Job {job_id} already completed")
            return {"status": "already_completed", "url": job.telegram_post_url}

        # Accept both "scheduled" (publish-now override) and "pending" (dispatched).
        if job.status not in ("scheduled", "pending"):
            logger.warning(f"Telegram job {job_id} not eligible to publish (status={job.status})")
            return {"error": f"Job is not pending: {job.status}"}

        job.status = "uploading"
        db.commit()

        try:
            # Get video, channel, and bot
            video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
            if not video:
                raise Exception(f"Composed video {job.composed_video_id} not found")

            channel = db.query(TelegramChannel).filter_by(id=job.channel_id).first()
            if not channel:
                raise Exception(f"Channel {job.channel_id} not found")

            bot = db.query(TelegramBot).filter_by(id=job.bot_id).first()
            if not bot:
                raise Exception(f"Bot {job.bot_id} not found")

            # Publish video (no caption - the video itself contains the caption overlay)
            logger.info(f"Publishing video {video.id} to Telegram channel {channel.channel_name or channel.channel_id}")

            publisher = TelegramPublisher(bot.bot_token)
            result = asyncio.run(publisher.post_video(
                chat_id=channel.channel_id,
                video_path=video.storage_path,
                caption="",  # No text caption - video has caption overlay
            ))

            # Extract message ID and build URL
            message_id = result["message_id"]
            chat_info = result["chat"]

            # Build post URL
            if chat_info.get("username"):
                post_url = publisher.get_post_url(chat_info["username"], message_id)
            else:
                post_url = publisher.get_post_url_private(channel.channel_id, message_id)

            # Update job
            job.telegram_message_id = message_id
            job.telegram_post_url = post_url
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"Video published to Telegram: {post_url}")

            return {
                "status": "completed",
                "job_id": job_id,
                "message_id": message_id,
                "url": post_url,
            }

        except Exception as e:
            logger.error(f"Failed to publish to Telegram: {e}")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()

            # Retry on transient errors
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise self.retry(exc=e, countdown=60)

            return {"status": "failed", "error": str(e)}


@celery_app.task(name="tasks.telegram_tasks.process_pending_telegram_schedules", queue="maintenance")
def process_pending_telegram_schedules_task():
    """
    Beat task: find TelegramPublishJobs scheduled for posting within the next
    24 hours and dispatch publish_to_telegram_task for each.

    Runs every 10 minutes via Celery Beat. Mirrors postpone_tasks.process_pending_schedules.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)

    dispatched = 0
    with get_db_context() as db:
        scheduled_jobs = db.query(TelegramPublishJob).filter(
            TelegramPublishJob.status == "scheduled",
        ).all()

        for job in scheduled_jobs:
            if not job.base_post_time:
                continue
            post_time = job.base_post_time if job.base_post_time.tzinfo else job.base_post_time.replace(tzinfo=timezone.utc)
            if post_time > cutoff:
                continue

            # Promote to pending and dispatch
            job.status = "pending"
            db.commit()

            task = publish_to_telegram_task.delay(job.id)
            job.celery_task_id = task.id
            db.commit()
            dispatched += 1
            logger.info(f"Dispatched Telegram job {job.id} (scheduled {post_time.isoformat()})")

    if dispatched:
        logger.info(f"Dispatched {dispatched} pending Telegram schedule jobs")
    return {"status": "ok", "dispatched": dispatched}
