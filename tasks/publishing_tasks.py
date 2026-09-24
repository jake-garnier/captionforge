"""
Video Publishing Celery Tasks

Orchestrates the Reddit publishing workflow:
1. Publish the composed video to the media host (self-hosted URL by default,
   see publishers/media_host.py) unless the job already has a hosted_url
2. Post the hosted link to the Reddit profile
3. Wait 30 minutes
4. Crosspost to all subreddits for the niche

Reddit poster can be configured via REDDIT_POSTER_TYPE env var:
- "playwright" (default): Uses browser automation, only needs username/password
- "praw": Uses PRAW library, requires API credentials
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List

from .celery_app import celery_app
from database.db import get_db_context
from database.models import (
    VideoPublishJob, RedditCrosspost, ComposedVideo
)
from config.automation_config import get_automation_config

logger = logging.getLogger(__name__)


def get_reddit_poster():
    """
    Factory function to get the appropriate Reddit poster.

    Reads REDDIT_POSTER_TYPE env var:
    - "playwright": Use browser automation (no API key needed)
    - "praw" (default): Use PRAW library (requires API credentials)

    Returns:
        RedditPoster or RedditPosterPlaywright instance
    """
    poster_type = os.environ.get("REDDIT_POSTER_TYPE", "playwright").lower()

    if poster_type == "playwright":
        from publishers import RedditPosterPlaywright
        logger.info("Using Playwright-based Reddit poster")
        return RedditPosterPlaywright(headless=True)
    else:
        from publishers import RedditPoster
        logger.info("Using PRAW-based Reddit poster")
        return RedditPoster()


def get_subreddits_for_niche(niche: str) -> List[str]:
    """Get list of subreddits for a niche from automation config."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if niche_config:
        return niche_config.subreddits
    return []


@celery_app.task(bind=True, name='tasks.publishing_tasks.publish_video', queue='publishing')
def publish_video_task(self, job_id: int):
    """
    Main publish workflow task.

    Orchestrates: media host → Reddit profile post → scheduled crosspost
    """
    logger.info(f"Starting publish workflow for job {job_id}")

    with get_db_context() as db:
        job = db.query(VideoPublishJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Publish job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        # Get the composed video
        video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
        if not video:
            job.status = "failed"
            job.error_message = "Composed video not found"
            job.error_stage = "validation"
            db.commit()
            return {"status": "error", "message": "Composed video not found"}

        if not video.storage_path or not os.path.exists(video.storage_path):
            job.status = "failed"
            job.error_message = f"Video file not found: {video.storage_path}"
            job.error_stage = "validation"
            db.commit()
            return {"status": "error", "message": "Video file not found"}

        try:
            job.started_at = datetime.utcnow()

            # Step 1: Make the video publicly reachable via the media host
            if not job.hosted_url:
                job.status = "hosting_media"
                db.commit()

                from publishers.media_host import get_media_host, MediaHostError
                try:
                    host = get_media_host()
                    hosted = host.publish(
                        video_path=video.storage_path,
                        title=job.title,
                        composed_video_id=video.id,
                    )
                except MediaHostError as e:
                    job.status = "failed"
                    job.error_message = str(e)
                    job.error_stage = "media_host"
                    db.commit()
                    return {"status": "error", "stage": "media_host", "message": job.error_message}

                job.hosted_url = hosted.url
                job.hosted_media_id = hosted.media_id
                job.hosted_at = datetime.utcnow()
                video.hosted_url = hosted.url
                db.commit()
                logger.info(f"Hosted via '{host.name}': {hosted.url}")

            job.status = "posting_profile"
            db.commit()

            # Step 2: Post to Reddit profile
            logger.info(f"Posting to Reddit profile: {job.title}")
            reddit_result = _post_to_profile(job.title, job.hosted_url)

            if not reddit_result["success"]:
                job.status = "failed"
                job.error_message = reddit_result.get("error", "Reddit post failed")
                job.error_stage = "profile"
                db.commit()
                return {"status": "error", "stage": "profile", "message": job.error_message}

            # Save Reddit profile post info
            job.profile_post_id = reddit_result["post_id"]
            job.profile_post_url = reddit_result["post_url"]
            job.profile_posted_at = datetime.utcnow()
            db.commit()

            logger.info(f"Posted to profile: {job.profile_post_url}")

            # Step 3: Create crosspost records and schedule delayed task
            subreddits = get_subreddits_for_niche(job.niche) if job.niche else []

            if subreddits:
                job.status = "waiting_crosspost"
                job.crosspost_scheduled_at = datetime.utcnow() + timedelta(minutes=job.crosspost_delay_minutes)
                db.commit()

                # Create crosspost records
                for subreddit in subreddits:
                    crosspost = RedditCrosspost(
                        publish_job_id=job.id,
                        subreddit=subreddit,
                        status="pending"
                    )
                    db.add(crosspost)
                db.commit()

                # Schedule crosspost task
                logger.info(f"Scheduling crosspost for {job.crosspost_scheduled_at}")
                crosspost_video_task.apply_async(
                    args=[job.id],
                    eta=job.crosspost_scheduled_at
                )

                return {
                    "status": "waiting_crosspost",
                    "job_id": job.id,
                    "hosted_url": job.hosted_url,
                    "profile_post_url": job.profile_post_url,
                    "crosspost_at": job.crosspost_scheduled_at.isoformat(),
                    "subreddits": subreddits
                }
            else:
                # No subreddits to crosspost to, we're done
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                video.is_published = True
                db.commit()

                logger.info(f"Publish job {job_id} completed (no crossposts)")
                return {
                    "status": "completed",
                    "job_id": job.id,
                    "hosted_url": job.hosted_url,
                    "profile_post_url": job.profile_post_url
                }

        except Exception as e:
            logger.error(f"Publish job {job_id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.publishing_tasks.crosspost_video', queue='publishing')
def crosspost_video_task(self, job_id: int):
    """
    Crosspost video to all target subreddits.

    Called 30 minutes after profile post.
    """
    logger.info(f"Starting crosspost for job {job_id}")

    with get_db_context() as db:
        job = db.query(VideoPublishJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Publish job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        if not job.profile_post_id:
            job.status = "failed"
            job.error_message = "No profile post to crosspost from"
            job.error_stage = "crosspost"
            db.commit()
            return {"status": "error", "message": "No profile post"}

        try:
            job.status = "crossposting"
            job.crosspost_started_at = datetime.utcnow()
            db.commit()

            # Get pending crossposts
            crossposts = db.query(RedditCrosspost).filter_by(
                publish_job_id=job.id,
                status="pending"
            ).all()

            if not crossposts:
                logger.info(f"No pending crossposts for job {job_id}")
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                db.commit()
                return {"status": "completed", "message": "No crossposts needed"}

            # Perform crossposts
            poster = get_reddit_poster()

            if not poster.connect():
                job.status = "failed"
                job.error_message = "Failed to connect to Reddit"
                job.error_stage = "crosspost"
                db.commit()
                return {"status": "error", "message": "Reddit connection failed"}

            success_count = 0
            fail_count = 0

            for crosspost in crossposts:
                crosspost.status = "posting"
                db.commit()

                result = poster.crosspost_to_subreddit(
                    source_post_id=job.profile_post_id,
                    subreddit=crosspost.subreddit,
                    title=job.title
                )

                if result.success:
                    crosspost.status = "posted"
                    crosspost.post_id = result.post_id
                    crosspost.post_url = result.post_url
                    crosspost.posted_at = datetime.utcnow()
                    success_count += 1
                    logger.info(f"Crossposted to r/{crosspost.subreddit}")
                else:
                    crosspost.status = "failed"
                    crosspost.error_message = result.error
                    crosspost.retry_count += 1
                    fail_count += 1
                    logger.warning(f"Crosspost to r/{crosspost.subreddit} failed: {result.error}")

                db.commit()

                # Rate limiting between posts
                import time
                time.sleep(5)

            # Update job status
            job.crosspost_completed_at = datetime.utcnow()

            if fail_count == 0:
                job.status = "completed"
            elif success_count > 0:
                job.status = "completed"  # Partial success is still completion
            else:
                job.status = "failed"
                job.error_stage = "crosspost"

            job.completed_at = datetime.utcnow()

            # Mark video as published
            video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
            if video:
                video.is_published = True

            db.commit()

            logger.info(f"Crosspost complete: {success_count} success, {fail_count} failed")

            return {
                "status": "completed",
                "job_id": job.id,
                "success_count": success_count,
                "fail_count": fail_count
            }

        except Exception as e:
            logger.error(f"Crosspost job {job_id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.error_stage = "crosspost"
            db.commit()
            return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.publishing_tasks.retry_failed_crossposts', queue='publishing')
def retry_failed_crossposts_task(self, job_id: int):
    """
    Retry failed crossposts for a job.
    """
    logger.info(f"Retrying failed crossposts for job {job_id}")

    with get_db_context() as db:
        job = db.query(VideoPublishJob).filter_by(id=job_id).first()
        if not job:
            return {"status": "error", "message": "Job not found"}

        # Get failed crossposts
        failed = db.query(RedditCrosspost).filter_by(
            publish_job_id=job.id,
            status="failed"
        ).filter(RedditCrosspost.retry_count < 3).all()

        if not failed:
            return {"status": "completed", "message": "No failed crossposts to retry"}

        poster = get_reddit_poster()

        if not poster.connect():
            return {"status": "error", "message": "Reddit connection failed"}

        success_count = 0
        for crosspost in failed:
            crosspost.status = "posting"
            db.commit()

            result = poster.crosspost_to_subreddit(
                source_post_id=job.profile_post_id,
                subreddit=crosspost.subreddit,
                title=job.title
            )

            if result.success:
                crosspost.status = "posted"
                crosspost.post_id = result.post_id
                crosspost.post_url = result.post_url
                crosspost.posted_at = datetime.utcnow()
                success_count += 1
            else:
                crosspost.status = "failed"
                crosspost.error_message = result.error
                crosspost.retry_count += 1

            db.commit()
            import time
            time.sleep(5)

        return {"status": "completed", "retried": len(failed), "success": success_count}


def _post_to_profile(title: str, url: str) -> dict:
    """
    Post link to Reddit profile.

    Returns dict with success, post_id, post_url, or error.
    """
    poster = get_reddit_poster()

    if not poster.connect():
        return {"success": False, "error": "Failed to connect to Reddit"}

    try:
        result = poster.post_to_profile(
            title=title,
            url=url
        )

        if result.success:
            return {
                "success": True,
                "post_id": result.post_id,
                "post_url": result.post_url
            }
        else:
            return {"success": False, "error": result.error}
    finally:
        # Close browser if using Playwright
        if hasattr(poster, 'close'):
            poster.close()


@celery_app.task(name='tasks.publishing_tasks.check_pending_crossposts', queue='publishing')
def check_pending_crossposts_task():
    """
    Periodic task to check for scheduled crossposts that are due.

    In case the ETA-scheduled task was missed (worker restart, etc.)
    """
    logger.info("Checking for pending crossposts")

    with get_db_context() as db:
        now = datetime.utcnow()

        # Find jobs waiting for crosspost that are past their scheduled time
        pending_jobs = db.query(VideoPublishJob).filter(
            VideoPublishJob.status == "waiting_crosspost",
            VideoPublishJob.crosspost_scheduled_at <= now
        ).all()

        for job in pending_jobs:
            logger.info(f"Found overdue crosspost for job {job.id}, triggering now")
            crosspost_video_task.delay(job.id)

        return {"checked": len(pending_jobs)}
