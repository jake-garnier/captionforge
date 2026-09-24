"""
Extraction dispatcher - consumes video_downloaded events and triggers extraction.

Event-driven architecture for decoupling scraping (CPU/IO-bound) from
extraction (GPU-bound). Allows scrapers to run independently of GPU workers.

Components:
1. dispatch_extraction_batch - Polls queue, triggers extraction tasks
2. process_orphaned_videos - Fallback for videos missed by queue
3. requeue_stale_extractions - Handles hung extraction tasks

GPU Control Integration:
- Respects extraction:enabled Redis flag for global extraction control
- Can be disabled via /training-manager/gpu-control/extraction/disable
"""
import redis
from tasks.celery_app import celery_app
from tasks.scraping_tasks import extract_caption_task
from utils.extraction_queue import (
    pop_video_for_extraction,
    mark_extraction_complete,
    get_queue_stats,
    get_stale_processing,
    requeue_video,
    get_queue_length,
)
from database.db import get_db_context
from database.models import Video, ScrapedCaption
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Redis keys for GPU control
EXTRACTION_ENABLED_KEY = "extraction:enabled"


def is_extraction_enabled() -> bool:
    """Check if extraction is enabled via Redis flag."""
    try:
        r = redis.from_url(settings.celery_broker_url)
        value = r.get(EXTRACTION_ENABLED_KEY)
        # Default to enabled if key doesn't exist
        return value != b"0"
    except Exception as e:
        logger.warning(f"Could not check extraction flag: {e}")
        return True  # Default to enabled on error


@celery_app.task(bind=True, queue='maintenance')
def dispatch_extraction_batch(self, batch_size: int = 10):
    """
    Dispatch a batch of extraction tasks from the queue.

    Runs on maintenance worker - pops videos from queue and queues GPU tasks.
    Designed to be called frequently (every 1-2 minutes).

    Respects extraction:enabled Redis flag - when disabled (set to "0"),
    the task will skip processing and return immediately. This allows
    GPU memory to be freed for training.

    Args:
        batch_size: Max videos to process per batch (default: 10)

    Returns:
        Dictionary with dispatch results
    """
    # Check if extraction is enabled (for GPU control during training)
    if not is_extraction_enabled():
        logger.info("Extraction disabled via GPU control - skipping batch")
        stats = get_queue_stats()
        return {
            'status': 'skipped',
            'reason': 'extraction_disabled',
            'queue_length': stats.get('queue_length', 0),
            'message': 'Extraction is disabled. Enable via /training-manager/gpu-control/extraction/enable'
        }

    logger.info(f"Starting extraction batch dispatch (batch_size={batch_size})")

    dispatched = 0
    errors = 0

    try:
        for _ in range(batch_size):
            # Non-blocking pop from queue
            event = pop_video_for_extraction(timeout=0)

            if not event:
                # Queue empty
                break

            video_id = event.get("video_id")
            if not video_id:
                logger.warning(f"Invalid event in queue (no video_id): {event}")
                errors += 1
                continue

            try:
                # Verify video exists and needs extraction
                with get_db_context() as db:
                    video = db.query(Video).filter_by(id=video_id).first()

                    if not video:
                        logger.warning(f"Video {video_id} not found in database, skipping")
                        mark_extraction_complete(video_id, success=False)
                        errors += 1
                        continue

                    # Check if already has caption
                    existing_caption = db.query(ScrapedCaption).filter_by(video_id=video_id).first()
                    if existing_caption:
                        logger.info(f"Video {video_id} already has caption, skipping")
                        mark_extraction_complete(video_id, success=True)
                        continue

                    # Check video status
                    if video.processing_status == 'failed':
                        logger.info(f"Video {video_id} has failed status, skipping extraction")
                        mark_extraction_complete(video_id, success=False)
                        continue

                # Trigger extraction (async - both GPU workers can pick up tasks)
                logger.info(f"Queuing extraction for video {video_id}")
                extract_caption_task.delay(video_id)
                dispatched += 1
                # Note: mark_extraction_complete is called by the extraction task itself

            except Exception as e:
                logger.error(f"Failed to process video {video_id}: {e}")
                mark_extraction_complete(video_id, success=False)
                errors += 1

        # Get queue stats for logging
        stats = get_queue_stats()
        logger.info(f"Extraction batch complete: {dispatched} processed, {errors} errors, {stats.get('queue_length', 0)} remaining")

        return {
            'status': 'success',
            'dispatched': dispatched,
            'errors': errors,
            'queue_remaining': stats.get('queue_length', 0),
            'queue_stats': stats
        }

    except Exception as e:
        logger.error(f"Extraction dispatch failed: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(bind=True, queue='maintenance')
def process_orphaned_videos(self, max_videos: int = 50):
    """
    Fallback task to process videos that were downloaded but never extracted.

    Catches any videos that slipped through the queue (e.g., Redis restart,
    queue cleared, etc.). Runs less frequently than dispatch_extraction_batch.

    Args:
        max_videos: Max videos to queue per run

    Returns:
        Dictionary with results
    """
    logger.info(f"Checking for orphaned videos (max={max_videos})")

    try:
        with get_db_context() as db:
            # Find videos with status 'downloaded' that don't have captions
            # These are videos that were downloaded but extraction never ran
            from sqlalchemy import and_, not_, exists
            from sqlalchemy.orm import aliased

            orphaned_videos = db.query(Video).filter(
                and_(
                    Video.processing_status == 'downloaded',
                    ~exists().where(ScrapedCaption.video_id == Video.id)
                )
            ).limit(max_videos).all()

            if not orphaned_videos:
                logger.debug("No orphaned videos found")
                return {'status': 'success', 'orphaned_found': 0, 'queued': 0}

            queued = 0
            for video in orphaned_videos:
                # Requeue for extraction
                if requeue_video(video.id, {'source': 'orphan_recovery'}):
                    queued += 1

            logger.info(f"Queued {queued} orphaned videos for extraction")

            return {
                'status': 'success',
                'orphaned_found': len(orphaned_videos),
                'queued': queued
            }

    except Exception as e:
        logger.error(f"Failed to process orphaned videos: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(bind=True, queue='maintenance')
def requeue_stale_extractions(self, max_age_minutes: int = 30):
    """
    Requeue extractions that have been processing for too long (hung tasks).

    Args:
        max_age_minutes: Max processing time before requeue

    Returns:
        Dictionary with results
    """
    task_id = self.request.id
    task_name = 'requeue-stale-extractions'

    try:
        stale_ids = get_stale_processing(max_age_minutes)

        if not stale_ids:
            logger.debug("No stale extractions found")
            return {'status': 'success', 'stale_found': 0, 'requeued': 0}

        requeued = 0
        for video_id in stale_ids:
            if requeue_video(video_id, {'source': 'stale_recovery'}):
                requeued += 1
                logger.warning(f"Requeued stale extraction for video {video_id}")

        logger.info(f"Requeued {requeued}/{len(stale_ids)} stale extractions")

        return {
            'status': 'success',
            'stale_found': len(stale_ids),
            'requeued': requeued
        }

    except Exception as e:
        logger.error(f"Failed to requeue stale extractions: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(queue='maintenance')
def get_extraction_queue_status():
    """
    Get current extraction queue status.

    Returns:
        Dictionary with queue stats
    """
    return get_queue_stats()
