"""
Media host Celery tasks.

Publishes composed videos to the configured media host (see
``publishers/media_host.py``) and stores the resulting public URL on the
``PostponeScheduleJob`` so the Postpone scheduling task can create a Reddit
link post pointing at it.

Two tasks:
1. publish_to_media_host_task(job_id) - Publish a single job's video
2. dispatch_media_host_uploads_task() - Beat task: find jobs needing a hosted
   URL within the next 24h and dispatch them
"""
import logging
from datetime import datetime, timedelta, timezone

from .celery_app import celery_app
from database.db import get_db_context
from database.models import PostponeScheduleJob, ComposedVideo

logger = logging.getLogger(__name__)

# Maximum retries for transient errors before the job is marked permanently
# failed. Each retry waits for the next dispatcher beat (~10 min).
MAX_TRANSIENT_RETRIES = 5


def _mark_transient_failure(job, error: str) -> str:
    """
    Record a transient failure on the job.

    Soft retry: status stays 'pending' with the error recorded for visibility.
    Hard fail (after MAX_TRANSIENT_RETRIES): status='failed' so the UI shows
    the Retry button (which resets status='pending' and re-queues).

    Returns the resulting status string.
    """
    job.retry_count = (job.retry_count or 0) + 1
    job.error_message = error
    if job.retry_count < MAX_TRANSIENT_RETRIES:
        job.status = "pending"
    else:
        job.status = "failed"
    return job.status


@celery_app.task(bind=True, name='tasks.media_host_tasks.publish_to_media_host', queue='publishing')
def publish_to_media_host_task(self, job_id: int):
    """
    Publish a composed video to the media host for a PostponeScheduleJob.

    1. Load job and composed video from DB
    2. Publish via the configured MediaHost
    3. Store hosted_url / hosted_media_id on the job (and composed video)
    4. Chain straight into schedule_to_postpone_task
    """
    import redis as redis_lib
    from config.settings import settings

    logger.info(f"Publishing Postpone job {job_id} to media host")

    # Redis lock prevents two workers hosting the same job concurrently
    r = redis_lib.from_url(settings.REDIS_URL)
    lock = r.lock(f"media_host:lock:{job_id}", timeout=300, blocking=False)
    if not lock.acquire(blocking=False):
        logger.info(f"Postpone job {job_id} already being hosted (locked), skipping")
        return {"status": "skipped", "message": "Hosting already in progress"}

    try:
        return _do_publish(job_id)
    finally:
        try:
            lock.release()
        except Exception:
            pass


def _do_publish(job_id: int):
    from publishers.media_host import get_media_host, MediaHostError

    with get_db_context() as db:
        job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Postpone job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        if job.hosted_url:
            logger.info(f"Postpone job {job_id} already has hosted_url, skipping")
            return {"status": "skipped", "message": "Already has hosted_url"}

        if job.status not in ("pending", "hosting_media"):
            logger.warning(f"Postpone job {job_id} has status '{job.status}', skipping")
            return {"status": "skipped", "message": f"Job status is '{job.status}'"}

        video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
        if not video:
            job.status = "failed"
            job.error_message = f"Composed video {job.composed_video_id} not found"
            db.commit()
            return {"status": "error", "message": job.error_message}

        try:
            job.status = "hosting_media"
            db.commit()

            host = get_media_host()
            hosted = host.publish(
                video_path=video.storage_path,
                title=job.title,
                composed_video_id=video.id,
            )

            job.hosted_url = hosted.url
            job.hosted_media_id = hosted.media_id
            job.status = "pending"  # Back to pending so Postpone scheduling picks it up
            job.error_message = None
            video.hosted_url = hosted.url
            db.commit()

            logger.info(f"Postpone job {job_id} hosted via '{host.name}': {hosted.url}")

            from tasks.postpone_tasks import schedule_to_postpone_task
            schedule_to_postpone_task.delay(job_id)

            return {
                "status": "hosted",
                "job_id": job_id,
                "hosted_url": hosted.url,
                "media_host": host.name,
            }

        except MediaHostError as e:
            final = _mark_transient_failure(job, str(e))
            db.commit()
            logger.error(f"Postpone job {job_id} media host error -> status={final}: {e}")
            return {"status": final, "job_id": job_id, "message": str(e)}

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.retry_count = (job.retry_count or 0) + 1
            db.commit()
            logger.error(f"Postpone job {job_id} hosting exception: {e}")
            return {"status": "error", "job_id": job_id, "message": str(e)}


@celery_app.task(name='tasks.media_host_tasks.dispatch_media_host_uploads', queue='maintenance')
def dispatch_media_host_uploads_task():
    """
    Beat task: find pending PostponeScheduleJobs posting within 24 hours that
    still lack a hosted_url, and dispatch publish_to_media_host_task for each.

    Runs every 10 minutes via Celery Beat.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)

    with get_db_context() as db:
        candidates = db.query(PostponeScheduleJob).filter(
            PostponeScheduleJob.status == "pending",
            PostponeScheduleJob.hosted_url.is_(None),
        ).all()

        due = []
        for job in candidates:
            if not job.base_post_time:
                continue
            post_time = job.base_post_time
            if post_time.tzinfo is None:
                post_time = post_time.replace(tzinfo=timezone.utc)
            if post_time <= cutoff:
                due.append(job)

        if not due:
            logger.debug("No Postpone jobs need media hosting")
            return {"status": "ok", "dispatched": 0}

        for job in due:
            logger.info(f"Dispatching media host publish for Postpone job {job.id}")
            publish_to_media_host_task.delay(job.id)

        return {"status": "ok", "dispatched": len(due)}
