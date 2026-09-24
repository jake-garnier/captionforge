"""
Celery tasks for Patreon publishing.

Handles video upload and post creation on Patreon. Jobs land in the table
with status='scheduled' and a base_post_time; the beat dispatcher promotes
them to 'pending' and calls publish_to_patreon_task when their slot arrives.
"""
import logging
from datetime import datetime, timedelta, timezone
from celery import shared_task

from database.db import get_db_context
from database.models import PatreonPublishJob, PatreonCredential, ComposedVideo
from .celery_app import celery_app

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name='tasks.patreon_tasks.publish_to_patreon_task',
    queue='publishing',
    max_retries=2,
    default_retry_delay=60
)
def publish_to_patreon_task(self, job_id: int):
    """
    Publish a composed video to Patreon.

    Args:
        job_id: ID of the PatreonPublishJob to process
    """
    logger.info(f"Starting Patreon publish job {job_id}")

    with get_db_context() as db:
        # Get the job
        job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Patreon publish job {job_id} not found")
            return {"error": "Job not found"}

        # Accept both "scheduled" (publish-now override) and "pending" (dispatched).
        # Anything else means the job is mid-flight or done.
        if job.status not in ["pending", "scheduled"]:
            logger.warning(f"Job {job_id} not eligible to publish (status={job.status})")
            return {"error": f"Job is not pending: {job.status}"}

        # Get the composed video
        video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
        if not video:
            job.status = "failed"
            job.error_message = "Composed video not found"
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"error": "Composed video not found"}

        # Get credentials
        credential = db.query(PatreonCredential).filter_by(niche=job.niche).first()
        if not credential or not credential.is_configured:
            job.status = "failed"
            job.error_message = f"Patreon credentials not configured for {job.niche}"
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"error": f"Credentials not configured for {job.niche}"}

        # Extract values needed outside session (avoid detached object errors)
        niche = job.niche
        video_path = video.storage_path
        title = job.title
        description = job.description
        tags = job.tags.split(",") if job.tags else None

        # Update job status
        job.status = "uploading"
        job.started_at = datetime.utcnow()
        db.commit()

    # Do the publishing outside of the DB session to avoid long locks
    try:
        from publishers import PatreonPublisher

        publisher = PatreonPublisher(niche=niche, headless=True)

        if not publisher.connect():
            with get_db_context() as db:
                job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
                job.status = "failed"
                job.error_message = "Could not connect to Patreon. Session may have expired."
                job.completed_at = datetime.utcnow()
                db.commit()
            publisher.close()
            return {"error": "Connection failed"}

        # Publish the video
        result = publisher.publish_video(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            public=True  # Could make this configurable
        )

        publisher.close()

        # Update job with result
        with get_db_context() as db:
            job = db.query(PatreonPublishJob).filter_by(id=job_id).first()

            if result.success:
                job.status = "posted"
                job.patreon_post_id = result.post_id
                job.patreon_post_url = result.post_url
                job.posted_at = datetime.utcnow()
                job.completed_at = datetime.utcnow()
                logger.info(f"Successfully published to Patreon: {result.post_url}")
            else:
                job.status = "failed"
                job.error_message = result.error
                job.completed_at = datetime.utcnow()
                logger.error(f"Patreon publish failed: {result.error}")

            db.commit()

        return {
            "success": result.success,
            "post_url": result.post_url,
            "error": result.error
        }

    except Exception as e:
        logger.exception(f"Error publishing to Patreon: {e}")

        with get_db_context() as db:
            job = db.query(PatreonPublishJob).filter_by(id=job_id).first()
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()

        # Retry on transient errors
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise self.retry(exc=e)

        return {"error": str(e)}


@celery_app.task(name='tasks.patreon_tasks.process_pending_patreon_schedules', queue='maintenance')
def process_pending_patreon_schedules_task():
    """
    Beat task: find PatreonPublishJobs scheduled for posting within the next
    24 hours and dispatch publish_to_patreon_task for each.

    Runs every 10 minutes via Celery Beat. Mirrors postpone_tasks.process_pending_schedules.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)

    dispatched = 0
    with get_db_context() as db:
        scheduled_jobs = db.query(PatreonPublishJob).filter(
            PatreonPublishJob.status == "scheduled",
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

            task = publish_to_patreon_task.delay(job.id)
            job.celery_task_id = task.id
            db.commit()
            dispatched += 1
            logger.info(f"Dispatched Patreon job {job.id} (scheduled {post_time.isoformat()})")

    if dispatched:
        logger.info(f"Dispatched {dispatched} pending Patreon schedule jobs")
    return {"status": "ok", "dispatched": dispatched}


@shared_task(name='tasks.patreon_tasks.test_patreon_connection', queue='publishing')
def test_patreon_connection(niche: str):
    """
    Test Patreon connection for a niche.

    Args:
        niche: The niche name to test connection for
    """
    logger.info(f"Testing Patreon connection for {niche}")

    try:
        from publishers import PatreonPublisher

        publisher = PatreonPublisher(niche=niche, headless=True)
        connected = publisher.connect()
        publisher.close()

        if connected:
            logger.info(f"Patreon connection successful for {niche}")
            return {"status": "connected", "niche": niche}
        else:
            logger.warning(f"Patreon connection failed for {niche}")
            return {"status": "failed", "niche": niche}

    except Exception as e:
        logger.error(f"Error testing Patreon connection: {e}")
        return {"status": "error", "niche": niche, "error": str(e)}
