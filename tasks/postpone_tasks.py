"""
Postpone Scheduling Celery Tasks

Two tasks:
1. schedule_to_postpone_task(job_id) - Send a single job to Postpone API
2. process_pending_schedules_task() - Beat task: find due jobs and dispatch them

Flow: publish_to_media_host_task (tasks/media_host_tasks.py) stores a public
hosted_url on the PostponeScheduleJob; this task then creates a Reddit link
post via Postpone pointing at that URL.
"""
import logging
from datetime import datetime, date, timezone

from .celery_app import celery_app
from database.db import get_db_context
from database.models import PostponeScheduleJob, ComposedVideo
from config.settings import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='tasks.postpone_tasks.schedule_to_postpone', queue='publishing')
def schedule_to_postpone_task(self, job_id: int):
    """
    Send a single PostponeScheduleJob to the Postpone API as a link post.

    1. Load job from DB
    2. Verify job has a hosted_url set
    3. Call PostponePublisher.schedule_reddit_post() with the hosted link
    4. Update job status (scheduled/failed)
    """
    logger.info(f"Processing Postpone schedule job {job_id}")

    if not settings.POSTPONE_API_KEY:
        logger.error("POSTPONE_API_KEY not configured")
        return {"status": "error", "message": "POSTPONE_API_KEY not configured"}

    with get_db_context() as db:
        job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Postpone job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        if job.status not in ("pending", "scheduling"):
            logger.warning(f"Postpone job {job_id} has status '{job.status}', skipping")
            return {"status": "skipped", "message": f"Job status is '{job.status}'"}

        # Verify composed video exists
        video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
        if not video:
            job.status = "failed"
            job.error_message = f"Composed video {job.composed_video_id} not found"
            db.commit()
            return {"status": "error", "message": job.error_message}

        # Verify the video has a public URL on the media host
        if not job.hosted_url:
            job.status = "failed"
            job.error_message = "No hosted_url set on job — publish the video to the media host first"
            db.commit()
            logger.error(f"Postpone job {job_id}: no hosted_url set")
            return {"status": "error", "message": job.error_message}

        try:
            # Mark as scheduling
            job.status = "scheduling"
            job.celery_task_id = self.request.id
            job.scheduled_at = datetime.now(timezone.utc)
            db.commit()

            # Call Postpone API
            from publishers.postpone_publisher import PostponePublisher
            from config.automation_config import get_automation_config
            publisher = PostponePublisher(api_key=settings.POSTPONE_API_KEY)
            subreddit_flairs = get_automation_config().postpone.subreddit_flairs

            result = publisher.schedule_reddit_post(
                username=job.reddit_username,
                title=job.title,
                media_url=job.hosted_url,
                subreddits=job.target_subreddits or [],
                base_post_time=job.base_post_time,
                stagger_minutes=job.stagger_minutes or 10,
                subreddit_flairs=subreddit_flairs,
            )

            if result.success:
                job.status = "scheduled"
                job.postpone_post_id = result.postpone_post_id
                job.postpone_response = result.raw_response
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = None
                db.commit()

                logger.info(
                    f"Postpone job {job_id} scheduled successfully: "
                    f"post_id={result.postpone_post_id}"
                )
                return {
                    "status": "scheduled",
                    "job_id": job_id,
                    "postpone_post_id": result.postpone_post_id,
                }
            else:
                job.status = "failed"
                job.error_message = result.error or "Unknown error"
                job.postpone_response = result.raw_response
                job.retry_count = (job.retry_count or 0) + 1
                db.commit()

                logger.error(f"Postpone job {job_id} failed: {result.error}")
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": result.error,
                }

        except Exception as e:
            logger.error(f"Postpone job {job_id} exception: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.retry_count = (job.retry_count or 0) + 1
            db.commit()
            return {"status": "error", "job_id": job_id, "message": str(e)}


@celery_app.task(name='tasks.postpone_tasks.process_pending_schedules', queue='maintenance')
def process_pending_schedules_task():
    """
    Beat task: find PostponeScheduleJobs that are due within 24 hours and
    still pending, then dispatch schedule_to_postpone_task for each.

    Only dispatches jobs that have a hosted_url set. Jobs missing a
    hosted_url are sent to the media host first if within the 24-hour window.

    Runs every 10 minutes via Celery Beat.
    """
    logger.info("Checking for pending Postpone schedule jobs")

    if not settings.POSTPONE_API_KEY:
        return {"status": "skipped", "message": "POSTPONE_API_KEY not configured"}

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)

    with get_db_context() as db:
        pending_jobs = db.query(PostponeScheduleJob).filter(
            PostponeScheduleJob.status == "pending",
        ).all()

        due_jobs = []
        needs_upload = []
        for job in pending_jobs:
            if job.base_post_time:
                post_time = job.base_post_time if job.base_post_time.tzinfo else job.base_post_time.replace(tzinfo=timezone.utc)
                if post_time <= cutoff:
                    if job.hosted_url:
                        due_jobs.append(job)
                    else:
                        needs_upload.append(job)

        # Send jobs missing a hosted_url to the media host first
        uploaded = 0
        if needs_upload:
            from tasks.media_host_tasks import publish_to_media_host_task
            for job in needs_upload:
                logger.info(f"Postpone job {job.id} needs a hosted URL, dispatching to media host")
                publish_to_media_host_task.delay(job.id)
                uploaded += 1

        if not due_jobs:
            if needs_upload:
                logger.info(f"No ready Postpone jobs (dispatched {uploaded} to media host)")
            else:
                logger.debug("No pending Postpone jobs due")
            return {"status": "ok", "dispatched": 0, "uploaded": uploaded}

        dispatched = 0
        for job in due_jobs:
            logger.info(f"Dispatching Postpone job {job.id} (scheduled {job.scheduled_date})")
            schedule_to_postpone_task.delay(job.id)
            dispatched += 1

        logger.info(f"Dispatched {dispatched} Postpone schedule jobs, {uploaded} to media host")
        return {"status": "ok", "dispatched": dispatched, "uploaded": uploaded}
