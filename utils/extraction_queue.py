"""
Redis-based extraction queue for event-driven decoupling.

Separates video downloading (I/O-bound, can run on CPU) from
caption extraction (GPU-bound, requires CUDA workers).

Architecture:
- download_video_task pushes video_id to extraction queue
- extraction_dispatcher pops from queue and triggers extraction
- Fallback scheduler catches any orphaned videos

Uses Redis List for reliable message delivery (vs pub/sub which loses messages).
"""
import redis
import json
import logging
from typing import Optional, List, Dict
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis keys
EXTRACTION_QUEUE_KEY = "captions:extraction_queue"
EXTRACTION_PROCESSING_KEY = "captions:extraction_processing"
EXTRACTION_STATS_KEY = "captions:extraction_stats"


def get_redis_client() -> redis.Redis:
    """Get Redis client from settings."""
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True
    )


def publish_video_downloaded(video_id: int, metadata: Optional[Dict] = None) -> bool:
    """
    Publish a video_downloaded event to the extraction queue.

    Called by download_video_task after successful download.

    Args:
        video_id: Database ID of the downloaded video
        metadata: Optional metadata (media_type, upvotes, etc.)

    Returns:
        True if successfully queued, False otherwise
    """
    try:
        client = get_redis_client()

        event = {
            "video_id": video_id,
            "event": "video_downloaded",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Push to queue (RPUSH for FIFO order)
        client.rpush(EXTRACTION_QUEUE_KEY, json.dumps(event))

        # Update stats
        client.hincrby(EXTRACTION_STATS_KEY, "total_queued", 1)
        client.hset(EXTRACTION_STATS_KEY, "last_queued_at", datetime.utcnow().isoformat())

        logger.info(f"Queued video {video_id} for extraction")
        return True

    except Exception as e:
        logger.error(f"Failed to queue video {video_id} for extraction: {e}")
        return False


def pop_video_for_extraction(timeout: int = 0) -> Optional[Dict]:
    """
    Pop a video from the extraction queue for processing.

    Uses BLPOP for blocking pop (efficient) or LPOP for non-blocking.

    Args:
        timeout: Seconds to block waiting for item (0 = non-blocking)

    Returns:
        Event dict with video_id, or None if queue empty
    """
    try:
        client = get_redis_client()

        if timeout > 0:
            # Blocking pop - waits for item
            result = client.blpop(EXTRACTION_QUEUE_KEY, timeout=timeout)
            if result:
                _, event_json = result
                event = json.loads(event_json)
            else:
                return None
        else:
            # Non-blocking pop
            event_json = client.lpop(EXTRACTION_QUEUE_KEY)
            if not event_json:
                return None
            event = json.loads(event_json)

        # Track that we're processing this video
        client.hset(
            EXTRACTION_PROCESSING_KEY,
            str(event["video_id"]),
            json.dumps({
                "started_at": datetime.utcnow().isoformat(),
                "event": event
            })
        )

        # Update stats
        client.hincrby(EXTRACTION_STATS_KEY, "total_popped", 1)

        return event

    except Exception as e:
        logger.error(f"Failed to pop video from extraction queue: {e}")
        return None


def mark_extraction_complete(video_id: int, success: bool = True) -> None:
    """
    Mark a video's extraction as complete (success or failure).

    Removes from processing set and updates stats.

    Args:
        video_id: Database ID of the video
        success: Whether extraction succeeded
    """
    try:
        client = get_redis_client()

        # Remove from processing set
        client.hdel(EXTRACTION_PROCESSING_KEY, str(video_id))

        # Update stats
        if success:
            client.hincrby(EXTRACTION_STATS_KEY, "total_completed", 1)
        else:
            client.hincrby(EXTRACTION_STATS_KEY, "total_failed", 1)

        client.hset(EXTRACTION_STATS_KEY, "last_completed_at", datetime.utcnow().isoformat())

    except Exception as e:
        logger.error(f"Failed to mark extraction complete for video {video_id}: {e}")


def requeue_video(video_id: int, metadata: Optional[Dict] = None) -> bool:
    """
    Requeue a video for extraction (e.g., after failure or for re-extraction).

    Args:
        video_id: Database ID of the video
        metadata: Optional metadata

    Returns:
        True if successfully requeued
    """
    # Remove from processing if present
    try:
        client = get_redis_client()
        client.hdel(EXTRACTION_PROCESSING_KEY, str(video_id))
    except:
        pass

    # Re-publish to queue
    return publish_video_downloaded(video_id, metadata)


def get_queue_length() -> int:
    """Get the number of videos waiting in the extraction queue."""
    try:
        client = get_redis_client()
        return client.llen(EXTRACTION_QUEUE_KEY)
    except Exception as e:
        logger.error(f"Failed to get queue length: {e}")
        return 0


def get_processing_count() -> int:
    """Get the number of videos currently being processed."""
    try:
        client = get_redis_client()
        return client.hlen(EXTRACTION_PROCESSING_KEY)
    except Exception as e:
        logger.error(f"Failed to get processing count: {e}")
        return 0


def get_queue_stats() -> Dict:
    """Get comprehensive queue statistics."""
    try:
        client = get_redis_client()

        stats = client.hgetall(EXTRACTION_STATS_KEY) or {}

        # Get current queue state
        queue_length = client.llen(EXTRACTION_QUEUE_KEY)
        processing = client.hgetall(EXTRACTION_PROCESSING_KEY) or {}

        # Peek at next items in queue (without removing)
        next_items = client.lrange(EXTRACTION_QUEUE_KEY, 0, 4)  # First 5
        next_video_ids = []
        for item in next_items:
            try:
                event = json.loads(item)
                next_video_ids.append(event.get("video_id"))
            except:
                pass

        return {
            "queue_length": queue_length,
            "processing_count": len(processing),
            "processing_video_ids": [int(vid) for vid in processing.keys()],
            "next_in_queue": next_video_ids,
            "total_queued": int(stats.get("total_queued", 0)),
            "total_popped": int(stats.get("total_popped", 0)),
            "total_completed": int(stats.get("total_completed", 0)),
            "total_failed": int(stats.get("total_failed", 0)),
            "last_queued_at": stats.get("last_queued_at"),
            "last_completed_at": stats.get("last_completed_at"),
        }

    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {"error": str(e)}


def clear_queue() -> int:
    """
    Clear the extraction queue (admin operation).

    Returns:
        Number of items cleared
    """
    try:
        client = get_redis_client()
        length = client.llen(EXTRACTION_QUEUE_KEY)
        client.delete(EXTRACTION_QUEUE_KEY)
        client.delete(EXTRACTION_PROCESSING_KEY)
        logger.warning(f"Cleared extraction queue ({length} items)")
        return length
    except Exception as e:
        logger.error(f"Failed to clear queue: {e}")
        return 0


def get_stale_processing(max_age_minutes: int = 30) -> List[int]:
    """
    Get video IDs that have been processing for too long (likely hung).

    Args:
        max_age_minutes: Max processing time before considered stale

    Returns:
        List of video IDs that are stale
    """
    try:
        client = get_redis_client()
        processing = client.hgetall(EXTRACTION_PROCESSING_KEY) or {}

        stale_ids = []
        now = datetime.utcnow()

        for video_id, data_json in processing.items():
            try:
                data = json.loads(data_json)
                started_at = datetime.fromisoformat(data["started_at"])
                age_minutes = (now - started_at).total_seconds() / 60

                if age_minutes > max_age_minutes:
                    stale_ids.append(int(video_id))

            except Exception as e:
                logger.warning(f"Failed to parse processing data for video {video_id}: {e}")
                stale_ids.append(int(video_id))

        return stale_ids

    except Exception as e:
        logger.error(f"Failed to get stale processing: {e}")
        return []
