"""
Worker heartbeat system for monitoring celery worker activity.

Tracks:
- Last activity timestamp
- Current task being processed
- Task start time (to detect hangs)

Used by watchdog to detect hung workers and trigger recovery.
"""
import redis
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Redis keys
HEARTBEAT_KEY = "celery:worker:heartbeat"
CURRENT_TASK_KEY = "celery:worker:current_task"


def get_redis_client():
    """Get Redis client using environment config."""
    host = os.getenv('REDIS_HOST', 'redis')
    port = int(os.getenv('REDIS_PORT', 6379))
    password = os.getenv('REDIS_PASSWORD', None)
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)


def heartbeat(task_name: str = None, video_id: int = None, stage: str = None):
    """
    Update worker heartbeat in Redis.

    Call this periodically during long operations to signal worker is alive.

    Args:
        task_name: Name of current task (e.g., 'extract_caption_task')
        video_id: Video being processed (if applicable)
        stage: Current stage (e.g., 'ocr_frame_10', 'downloading')
    """
    try:
        client = get_redis_client()
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'task_name': task_name,
            'video_id': video_id,
            'stage': stage,
            'pid': os.getpid()
        }
        client.set(HEARTBEAT_KEY, json.dumps(data), ex=300)  # Expires in 5 min
    except Exception as e:
        logger.warning(f"Failed to update heartbeat: {e}")


def start_task(task_name: str, task_id: str, video_id: int = None):
    """
    Mark task as started. Used to detect hung tasks.

    Args:
        task_name: Name of the task
        task_id: Celery task ID
        video_id: Video being processed (if applicable)
    """
    try:
        client = get_redis_client()
        data = {
            'task_name': task_name,
            'task_id': task_id,
            'video_id': video_id,
            'started_at': datetime.utcnow().isoformat(),
            'pid': os.getpid()
        }
        client.set(CURRENT_TASK_KEY, json.dumps(data), ex=7200)  # 2 hour max
        heartbeat(task_name, video_id, 'started')
    except Exception as e:
        logger.warning(f"Failed to mark task start: {e}")


def end_task():
    """Mark current task as completed."""
    try:
        client = get_redis_client()
        client.delete(CURRENT_TASK_KEY)
        heartbeat(stage='idle')
    except Exception as e:
        logger.warning(f"Failed to mark task end: {e}")


def get_worker_status() -> Dict:
    """
    Get current worker status for diagnostics.

    Returns:
        Dict with heartbeat info, current task, and health status
    """
    try:
        client = get_redis_client()

        # Get heartbeat
        heartbeat_raw = client.get(HEARTBEAT_KEY)
        heartbeat_data = json.loads(heartbeat_raw) if heartbeat_raw else None

        # Get current task
        task_raw = client.get(CURRENT_TASK_KEY)
        task_data = json.loads(task_raw) if task_raw else None

        # Calculate health status
        is_healthy = True
        hang_detected = False
        hang_duration_seconds = None
        worker_state = 'unknown'

        # Determine worker state based on heartbeat and task data
        if task_data:
            # Worker is processing a task
            started = datetime.fromisoformat(task_data['started_at'])
            duration = (datetime.utcnow() - started).total_seconds()
            hang_duration_seconds = duration
            worker_state = 'processing'

            # Check if task is taking too long (> 30 minutes)
            if duration > 1800:
                hang_detected = True
                is_healthy = False
                worker_state = 'potentially_hung'

            # If processing but no recent heartbeat, might be stuck
            if heartbeat_data:
                last_beat = datetime.fromisoformat(heartbeat_data['timestamp'])
                heartbeat_age = (datetime.utcnow() - last_beat).total_seconds()
                # No heartbeat in 5 minutes while task running = potential hang
                if heartbeat_age > 300:
                    is_healthy = False
                    worker_state = 'potentially_hung'
        else:
            # No current task - worker is idle
            worker_state = 'idle'
            # Idle worker is healthy (no task to hang on)
            is_healthy = True

            # Check for stale heartbeat (informational only for idle worker)
            if heartbeat_data:
                last_beat = datetime.fromisoformat(heartbeat_data['timestamp'])
                heartbeat_age = (datetime.utcnow() - last_beat).total_seconds()
                if heartbeat_age > 300:
                    worker_state = 'idle_stale_heartbeat'

        return {
            'heartbeat': heartbeat_data,
            'current_task': task_data,
            'is_healthy': is_healthy,
            'hang_detected': hang_detected,
            'hang_duration_seconds': hang_duration_seconds,
            'worker_state': worker_state
        }

    except Exception as e:
        logger.error(f"Failed to get worker status: {e}")
        return {
            'error': str(e),
            'is_healthy': False,
            'hang_detected': False,
            'worker_state': 'error'
        }


def check_for_hang(max_task_duration_seconds: int = 1800) -> bool:
    """
    Check if worker appears to be hung.

    A worker is considered hung ONLY if:
    - There's an active task AND (task duration > threshold OR no recent heartbeat)

    An idle worker (no active task) is NOT considered hung.

    Args:
        max_task_duration_seconds: Max time a task should take (default 30 min)

    Returns:
        True if hang detected, False otherwise
    """
    status = get_worker_status()

    # Check explicit hang detection from get_worker_status
    if status.get('hang_detected', False):
        return True

    # Check worker state for potentially hung
    worker_state = status.get('worker_state', '')
    if worker_state == 'potentially_hung':
        return True

    # Check if task is running too long
    if status.get('current_task') and status.get('hang_duration_seconds'):
        if status['hang_duration_seconds'] > max_task_duration_seconds:
            return True

    return False
