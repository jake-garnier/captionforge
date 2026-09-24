"""
Watchdog task for monitoring worker health and triggering recovery.

Runs on celery-beat and checks:
1. Worker heartbeat freshness
2. Task running duration
3. Whether worker responds to ping

If hang detected, logs detailed diagnostics, stores incident to database,
and can trigger recovery.
"""
from tasks.celery_app import celery_app
from celery import current_app
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

# Thresholds for hang detection
MAX_HEARTBEAT_AGE_SECONDS = 300  # 5 minutes without heartbeat
MAX_TASK_DURATION_SECONDS = 1800  # 30 minutes per task


def collect_and_store_hang_diagnostics(worker_status: dict, recovery_action: str = 'none') -> int:
    """
    Collect comprehensive diagnostics and store hang incident to database.

    Args:
        worker_status: Current worker status from get_worker_status()
        recovery_action: Action being taken ('none', 'manual', 'auto_restart')

    Returns:
        Incident ID from database
    """
    try:
        from utils.hang_diagnostics import (
            collect_hang_diagnostics,
            store_hang_incident,
            format_diagnostics_for_logging
        )

        # Collect all diagnostics
        diagnostics = collect_hang_diagnostics(worker_status)

        # Log the formatted diagnostics
        logger.warning(format_diagnostics_for_logging(diagnostics))

        # Store to database
        incident_id = store_hang_incident(
            worker_status=worker_status,
            diagnostics=diagnostics,
            recovery_action=recovery_action
        )

        return incident_id

    except Exception as e:
        logger.error(f"Failed to collect/store hang diagnostics: {e}")
        # Fall back to basic logging
        log_worker_diagnostics()
        return None


@celery_app.task
def check_worker_health():
    """
    Check worker health and log diagnostics.

    This runs on celery-beat (not celery-worker) to monitor from outside.
    Returns status dict with health info.
    """
    try:
        from utils.worker_heartbeat import get_worker_status, check_for_hang

        status = get_worker_status()
        worker_state = status.get('worker_state', 'unknown')

        # Log worker state
        logger.info(f"Worker state: {worker_state}")

        # Log heartbeat details if available
        if status.get('heartbeat'):
            hb = status['heartbeat']
            logger.info(
                f"Worker heartbeat: task={hb.get('task_name')}, "
                f"stage={hb.get('stage')}, "
                f"video_id={hb.get('video_id')}"
            )

        # Log current task if processing
        if status.get('current_task'):
            task = status['current_task']
            duration = status.get('hang_duration_seconds', 0)
            logger.info(
                f"Current task: {task.get('task_name')} "
                f"(video_id={task.get('video_id')}, "
                f"running for {duration:.0f}s)"
            )

        # Check for hang (only detects actual hangs, not idle workers)
        hang_detected = check_for_hang(MAX_TASK_DURATION_SECONDS)

        if hang_detected:
            logger.warning(
                f"HANG DETECTED! Worker state={worker_state}, "
                f"status: {status}"
            )

            # Collect comprehensive diagnostics and store to database
            incident_id = collect_and_store_hang_diagnostics(status, recovery_action='none')

            return {
                'status': 'hang_detected',
                'worker_state': worker_state,
                'details': status,
                'incident_id': incident_id,
                'action': 'manual_restart_recommended'
            }

        # Log based on worker state
        if worker_state == 'idle':
            logger.info("Worker health check: idle (healthy)")
        elif worker_state == 'processing':
            logger.info(f"Worker health check: processing (healthy)")
        else:
            logger.info(f"Worker health check: {worker_state}")

        return {
            'status': 'healthy' if status.get('is_healthy') else 'unhealthy',
            'worker_state': worker_state,
            'details': status
        }

    except Exception as e:
        logger.error(f"Error checking worker health: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task
def worker_watchdog_with_recovery():
    """
    Watchdog task that can trigger recovery if hang detected.

    CAUTION: This can restart the worker container.
    Only enable if you want automatic recovery.
    """
    try:
        from utils.worker_heartbeat import check_for_hang, get_worker_status

        if not check_for_hang(MAX_TASK_DURATION_SECONDS):
            logger.debug("Watchdog: Worker is healthy")
            return {'status': 'healthy'}

        # Hang detected - get details
        status = get_worker_status()
        logger.warning(f"Watchdog: Hang detected! Status: {status}")

        # Check if auto-recovery is enabled
        auto_recovery = os.getenv('WATCHDOG_AUTO_RECOVERY', 'false').lower() == 'true'
        recovery_action = 'auto_restart' if auto_recovery else 'none'

        # Collect diagnostics and store incident
        incident_id = collect_and_store_hang_diagnostics(status, recovery_action=recovery_action)

        if auto_recovery:
            logger.warning("Watchdog: Triggering automatic recovery...")
            success = trigger_worker_restart()

            # Update incident with recovery result
            if incident_id:
                try:
                    from database.db import get_db_context
                    from database.models import HangIncident
                    with get_db_context() as db:
                        incident = db.query(HangIncident).filter_by(id=incident_id).first()
                        if incident:
                            incident.recovery_successful = success
                            db.commit()
                except Exception as e:
                    logger.warning(f"Failed to update incident with recovery result: {e}")

            if success:
                logger.info("Watchdog: Worker restart triggered successfully")
                return {
                    'status': 'recovery_triggered',
                    'incident_id': incident_id,
                    'details': status
                }
            else:
                logger.error("Watchdog: Failed to trigger worker restart")
                return {
                    'status': 'recovery_failed',
                    'incident_id': incident_id,
                    'details': status
                }
        else:
            logger.warning(
                "Watchdog: Auto-recovery disabled. "
                "Set WATCHDOG_AUTO_RECOVERY=true to enable. "
                "Manual restart required."
            )
            return {
                'status': 'hang_detected',
                'auto_recovery': False,
                'incident_id': incident_id,
                'details': status
            }

    except Exception as e:
        logger.error(f"Watchdog error: {e}")
        return {'status': 'error', 'error': str(e)}


def log_worker_diagnostics():
    """Log detailed diagnostics about the hung worker."""
    try:
        # Try to get GPU status
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"GPU status: {result.stdout.strip()}")
        except Exception as e:
            logger.debug(f"Could not get GPU status: {e}")

        # Log memory usage
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            logger.info(f"Worker memory: RSS={mem.rss / 1024 / 1024:.0f}MB")
        except Exception as e:
            logger.debug(f"Could not get memory status: {e}")

    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")


def trigger_worker_restart():
    """
    Trigger celery-worker container restart.

    Returns True if restart command succeeded, False otherwise.
    """
    try:
        # Method 1: Use docker command (if docker socket mounted)
        result = subprocess.run(
            ['docker', 'restart', 'captions-celery-worker'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            logger.info("Docker restart command succeeded")
            return True

        logger.warning(f"Docker restart failed: {result.stderr}")

    except FileNotFoundError:
        logger.warning("Docker command not available")
    except subprocess.TimeoutExpired:
        logger.warning("Docker restart timed out")
    except Exception as e:
        logger.error(f"Error triggering restart: {e}")

    return False


@celery_app.task
def test_watchdog():
    """
    Test task to verify watchdog is working.
    Can be called manually to check monitoring.
    """
    from utils.worker_heartbeat import heartbeat, get_worker_status

    # Send test heartbeat
    heartbeat(task_name='test_watchdog', stage='testing')

    # Get status
    status = get_worker_status()

    logger.info(f"Watchdog test: {status}")

    return {
        'status': 'test_complete',
        'worker_status': status
    }
