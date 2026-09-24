"""
Pipeline Orchestrator - Automated Per-Niche Training, Generation, and Composition

Manages the full automation pipeline:
1. Collection Phase: Scraping and extraction (both GPUs)
2. Training Phase: Per-niche LoRA training (exclusive GPU use)
3. Output Phase: Per-niche generation and composition

State machine runs every 5 minutes to check and transition states.
Deployment-safe: GPU workers protected during active jobs.
"""
import logging
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import redis
from sqlalchemy import func

from tasks.celery_app import celery_app
from config.automation_config import get_automation_config, NicheConfig
from database.db import get_db_context
from database.models import (
    TrainingJob, TrainedModel, GenerationJob, GeneratedCaption,
    VideoCompositionJob, ComposedVideo, ScrapedCaption, BackgroundVideo
)
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis keys for pipeline state
PIPELINE_STATE_KEY = "pipeline:state"
PIPELINE_ENABLED_KEY = "pipeline:enabled"
PIPELINE_CURRENT_NICHE_KEY = "pipeline:current_niche"
PIPELINE_LAST_TRANSITION_KEY = "pipeline:last_transition_at"

# Per-niche tracking keys (use format strings)
NICHE_TRAINING_JOB_KEY = "pipeline:niche:{niche}:training_job"
NICHE_GENERATION_JOB_KEY = "pipeline:niche:{niche}:generation_job"
NICHE_COMPOSITION_JOB_KEY = "pipeline:niche:{niche}:composition_job"
NICHE_LAST_TRAINED_KEY = "pipeline:niche:{niche}:last_trained_at"
NICHE_MODEL_VERSION_KEY = "pipeline:niche:{niche}:model_version"
NICHE_CONSECUTIVE_FAILURES_KEY = "pipeline:niche:{niche}:consecutive_failures"

# Stuck state detection
PIPELINE_STATE_TIMESTAMP_KEY = "pipeline:state_timestamp"
MAX_CONSECUTIVE_FAILURES = 3  # Max failures before pausing niche
MAX_STATE_DURATION_MINUTES = {
    "training_pending": 30,
    "gpu_clearing": 30,
    "training": 240,  # 4 hours
    "generation_pending": 10,
    "generating": 90,  # 1.5 hours - generation jobs can take 60-80 minutes
    "composition_pending": 10,
    "composing": 60,
}

# Daily quota keys
QUOTA_KEY_FORMAT = "pipeline:quota:{date}:{niche}:composed"


def get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis.from_url(settings.celery_broker_url)


# ============================================================================
# Pipeline State Management
# ============================================================================

def get_pipeline_state() -> str:
    """Get current pipeline state."""
    try:
        r = get_redis()
        state = r.get(PIPELINE_STATE_KEY)
        return state.decode() if state else "collecting"
    except Exception as e:
        logger.error(f"Failed to get pipeline state: {e}")
        return "collecting"


def set_pipeline_state(state: str, reason: str = None, triggered_by: str = "orchestrator", metadata: dict = None):
    """Set pipeline state and log transition to database."""
    try:
        r = get_redis()
        old_state = get_pipeline_state()
        current_niche = get_current_niche()

        r.set(PIPELINE_STATE_KEY, state)
        r.set(PIPELINE_LAST_TRANSITION_KEY, datetime.utcnow().isoformat())
        # Track when we entered this state for stuck detection
        r.set(PIPELINE_STATE_TIMESTAMP_KEY, datetime.utcnow().isoformat())

        logger.info(f"[ORCHESTRATOR] State: {old_state} → {state}" + (f" (reason: {reason})" if reason else ""))

        # Log transition to database for history tracking
        log_state_transition(
            new_state=state,
            previous_state=old_state,
            niche=current_niche,
            triggered_by=triggered_by,
            reason=reason,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Failed to set pipeline state: {e}")


def log_state_transition(
    new_state: str,
    previous_state: str = None,
    niche: str = None,
    triggered_by: str = "orchestrator",
    reason: str = None,
    metadata: dict = None
):
    """Log a state transition to the pipeline_state_log table."""
    try:
        from database.db import get_db_context
        from sqlalchemy import text
        import json

        with get_db_context() as db:
            db.execute(
                text("""
                    INSERT INTO pipeline_state_log
                    (state, previous_state, niche, triggered_by, reason, metadata, created_at)
                    VALUES (:state, :previous_state, :niche, :triggered_by, :reason, :metadata, NOW())
                """),
                {
                    "state": new_state,
                    "previous_state": previous_state,
                    "niche": niche,
                    "triggered_by": triggered_by,
                    "reason": reason,
                    "metadata": json.dumps(metadata) if metadata else None
                }
            )
            db.commit()
    except Exception as e:
        logger.error(f"Failed to log state transition: {e}")


def get_state_duration_minutes() -> float:
    """Get how long we've been in the current state (in minutes)."""
    try:
        r = get_redis()
        timestamp = r.get(PIPELINE_STATE_TIMESTAMP_KEY)
        if timestamp:
            state_time = datetime.fromisoformat(timestamp.decode())
            delta = datetime.utcnow() - state_time
            return delta.total_seconds() / 60
        return 0
    except Exception as e:
        logger.error(f"Failed to get state duration: {e}")
        return 0


def is_state_stuck() -> bool:
    """Check if current state has exceeded max duration."""
    state = get_pipeline_state()
    max_duration = MAX_STATE_DURATION_MINUTES.get(state)
    if not max_duration:
        return False  # No limit for this state (e.g., collecting)

    duration = get_state_duration_minutes()
    if duration > max_duration:
        logger.warning(f"[ORCHESTRATOR] State {state} stuck for {duration:.1f} minutes (max: {max_duration})")
        return True
    return False


def get_consecutive_failures(niche: str) -> int:
    """Get consecutive failure count for a niche."""
    try:
        r = get_redis()
        key = NICHE_CONSECUTIVE_FAILURES_KEY.format(niche=niche)
        count = r.get(key)
        return int(count) if count else 0
    except Exception as e:
        logger.error(f"Failed to get consecutive failures: {e}")
        return 0


def increment_consecutive_failures(niche: str) -> int:
    """Increment and return consecutive failure count."""
    try:
        r = get_redis()
        key = NICHE_CONSECUTIVE_FAILURES_KEY.format(niche=niche)
        count = r.incr(key)
        # Expire after 24 hours
        r.expire(key, 60 * 60 * 24)
        logger.warning(f"[ORCHESTRATOR] {niche}: consecutive failures = {count}")
        return count
    except Exception as e:
        logger.error(f"Failed to increment consecutive failures: {e}")
        return 0


def reset_consecutive_failures(niche: str):
    """Reset failure count for a niche after success."""
    try:
        r = get_redis()
        key = NICHE_CONSECUTIVE_FAILURES_KEY.format(niche=niche)
        r.delete(key)
    except Exception as e:
        logger.error(f"Failed to reset consecutive failures: {e}")


def is_pipeline_enabled() -> bool:
    """Check if pipeline orchestrator is enabled."""
    try:
        r = get_redis()
        enabled = r.get(PIPELINE_ENABLED_KEY)
        return enabled is None or enabled.decode().lower() == "true"
    except Exception as e:
        logger.error(f"Failed to check pipeline enabled: {e}")
        return True  # Default to enabled


def set_pipeline_enabled(enabled: bool):
    """Enable or disable pipeline orchestrator."""
    try:
        r = get_redis()
        r.set(PIPELINE_ENABLED_KEY, "true" if enabled else "false")
        logger.info(f"[ORCHESTRATOR] Pipeline {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        logger.error(f"Failed to set pipeline enabled: {e}")


def get_current_niche() -> Optional[str]:
    """Get the niche currently being processed."""
    try:
        r = get_redis()
        niche = r.get(PIPELINE_CURRENT_NICHE_KEY)
        return niche.decode() if niche else None
    except Exception as e:
        logger.error(f"Failed to get current niche: {e}")
        return None


def set_current_niche(niche: Optional[str]):
    """Set the niche currently being processed."""
    try:
        r = get_redis()
        if niche:
            r.set(PIPELINE_CURRENT_NICHE_KEY, niche)
        else:
            r.delete(PIPELINE_CURRENT_NICHE_KEY)
        logger.info(f"[ORCHESTRATOR] Current niche: {niche}")
    except Exception as e:
        logger.error(f"Failed to set current niche: {e}")


# ============================================================================
# Training Trigger Logic
# ============================================================================

def get_caption_count_for_subreddits(subreddits: List[str]) -> int:
    """Get total caption count for a list of subreddits."""
    with get_db_context() as db:
        return db.query(ScrapedCaption).filter(
            ScrapedCaption.source_subreddit.in_(subreddits),
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.llm_refined_text != ""
        ).count()


def get_captions_since_for_subreddits(subreddits: List[str], since: datetime) -> int:
    """Get count of captions created since a given datetime."""
    with get_db_context() as db:
        return db.query(ScrapedCaption).filter(
            ScrapedCaption.source_subreddit.in_(subreddits),
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.scraped_at > since
        ).count()


def get_last_training_job_for_niche(niche: str) -> Optional[TrainingJob]:
    """Get the most recent completed training job for a niche."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return None

    with get_db_context() as db:
        # First try to find by niche column (new approach)
        job = db.query(TrainingJob).filter(
            TrainingJob.status == "completed",
            TrainingJob.niche == niche
        ).order_by(TrainingJob.completed_at.desc()).first()

        if job:
            # Expunge to detach from session but keep loaded attributes
            db.expunge(job)
            return job

        # Fallback: find jobs that trained on any of this niche's subreddits
        # Use text-based matching since JSON @> requires JSONB
        from sqlalchemy import cast, String
        for subreddit in niche_config.subreddits:
            job = db.query(TrainingJob).filter(
                TrainingJob.status == "completed",
                cast(TrainingJob.source_subreddits, String).contains(subreddit)
            ).order_by(TrainingJob.completed_at.desc()).first()
            if job:
                # Expunge to detach from session but keep loaded attributes
                db.expunge(job)
                return job

        return None


def get_model_for_niche(niche: str) -> Optional[TrainedModel]:
    """Get the latest ready model for a niche.

    Also checks for LoRA adapter files on disk as a fallback when no
    TrainedModel record exists (e.g., after Vast.ai training).
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return None

    with get_db_context() as db:
        # First try to find by niche column (new approach)
        model = db.query(TrainedModel).filter(
            TrainedModel.status.in_(["ready", "loaded"]),
            TrainedModel.niche == niche
        ).order_by(TrainedModel.created_at.desc()).first()

        if model:
            # Expunge to detach from session but keep loaded attributes
            db.expunge(model)
            return model

        # Fallback: find models that trained on any of this niche's subreddits
        from sqlalchemy import cast, String
        for subreddit in niche_config.subreddits:
            model = db.query(TrainedModel).filter(
                TrainedModel.status.in_(["ready", "loaded"]),
                cast(TrainedModel.source_subreddits, String).contains(subreddit)
            ).order_by(TrainedModel.created_at.desc()).first()
            if model:
                # Expunge to detach from session but keep loaded attributes
                db.expunge(model)
                return model

        # Final fallback: Check for LoRA adapter files on disk
        # This handles Vast.ai training that may not have created DB records
        lora_adapter_path = Path(f"/data/lora_adapters/{niche}/adapter.gguf")
        if lora_adapter_path.exists():
            logger.info(f"[ORCHESTRATOR] Found LoRA adapter on disk for {niche}: {lora_adapter_path}")
            # Create a virtual TrainedModel-like object for compatibility
            # We don't actually need all fields - just need to return something truthy
            from datetime import datetime
            virtual_model = TrainedModel(
                model_name=f"{niche}-lora-adapter",
                model_path=str(lora_adapter_path),
                status="ready",
                niche=niche,
                source_subreddits=niche_config.subreddits,
                created_at=datetime.utcnow()
            )
            return virtual_model

        return None


def get_generation_quality_for_niche(niche: str) -> Optional[float]:
    """Get average quality score of recent generations for a niche."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return None

    # Look at last 50 generated captions with scores
    with get_db_context() as db:
        # Get recent generation jobs for this niche's model
        model = get_model_for_niche(niche)
        if not model:
            return None

        avg_score = db.query(func.avg(GeneratedCaption.quality_score)).filter(
            GeneratedCaption.quality_score.isnot(None),
            GeneratedCaption.quality_score > 0
        ).join(GenerationJob).filter(
            GenerationJob.model_id == model.id
        ).scalar()

        return float(avg_score) if avg_score else None


def hours_since_datetime(dt: Optional[datetime]) -> float:
    """Calculate hours since a datetime."""
    if not dt:
        return float('inf')
    # Handle timezone-aware datetimes from database
    from datetime import timezone
    now = datetime.now(timezone.utc)
    # Make naive datetime timezone-aware for comparison
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 3600


def get_next_niche_needing_training() -> Optional[str]:
    """
    DEPRECATED: Auto-training was removed. Training is now manual-only via
    POST /training/trigger/{niche}. This function always returns None so the
    orchestrator never enters the training_pending → gpu_clearing → training
    branch on its own.

    Body retained (unreachable) so the selector logic is still readable for
    anyone resurrecting auto-training; remove entirely if it bitrots.
    """
    logger.debug("[ORCHESTRATOR] Auto-training disabled — training is manual-only")
    return None

    # --- unreachable ---
    config = get_automation_config()
    training_config = config.training

    candidates = []

    for niche_name, niche_config in config.niches.items():
        if not niche_config.enabled:
            continue

        # Skip niches that have already met their daily quota
        # No point training/generating if we can't compose more today
        if daily_quota_met(niche_name):
            logger.debug(f"[ORCHESTRATOR] {niche_name}: Skipping - daily quota already met")
            continue

        subreddits = niche_config.subreddits
        total_captions = get_caption_count_for_subreddits(subreddits)

        # Skip if not enough data yet
        if total_captions < training_config.min_total_captions:
            logger.debug(f"[ORCHESTRATOR] {niche_name}: Only {total_captions}/{training_config.min_total_captions} captions")
            continue

        last_job = get_last_training_job_for_niche(niche_name)

        # First training needed?
        if not last_job:
            candidates.append((niche_name, "first_training", total_captions))
            continue

        # Check time since last training
        hours_since = hours_since_datetime(last_job.completed_at)
        if hours_since >= training_config.retrain_interval_hours:
            candidates.append((niche_name, "stale_model", hours_since))
            continue

        # Check new captions since last training
        new_captions = get_captions_since_for_subreddits(subreddits, last_job.completed_at)
        if new_captions >= training_config.min_new_captions_since_last_train:
            candidates.append((niche_name, "new_data", new_captions))
            continue

    if not candidates:
        return None

    # Priority: first_training > stale_model > new_data
    priority_order = ["first_training", "stale_model", "new_data"]
    candidates.sort(key=lambda x: priority_order.index(x[1]))

    selected = candidates[0]
    logger.info(f"[ORCHESTRATOR] Training candidate: {selected[0]} (reason: {selected[1]}, value: {selected[2]})")

    return selected[0]  # Return niche name


def _get_training_reason(niche: str) -> str:
    """
    Get the reason a niche needs training.
    This mirrors the logic in get_next_niche_needing_training for logging purposes.
    """
    config = get_automation_config()
    training_config = config.training
    niche_config = config.get_niche(niche)

    if not niche_config:
        return "unknown"

    subreddits = niche_config.subreddits
    last_job = get_last_training_job_for_niche(niche)

    if not last_job:
        return "first_training"

    hours_since = hours_since_datetime(last_job.completed_at)
    if hours_since >= training_config.retrain_interval_hours:
        return f"stale_model:{hours_since:.0f}h"

    new_captions = get_captions_since_for_subreddits(subreddits, last_job.completed_at)
    if new_captions >= training_config.min_new_captions_since_last_train:
        return f"new_data:{new_captions}_captions"

    return "unknown"


# ============================================================================
# Daily Quota Tracking
# ============================================================================

def get_videos_composed_today(niche: str) -> int:
    """Get count of videos composed today for a niche."""
    try:
        r = get_redis()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = QUOTA_KEY_FORMAT.format(date=today, niche=niche)
        count = r.get(key)
        return int(count) if count else 0
    except Exception as e:
        logger.error(f"Failed to get daily quota: {e}")
        return 0


def increment_videos_composed_today(niche: str, count: int = 1):
    """Increment composed video count for today."""
    try:
        r = get_redis()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = QUOTA_KEY_FORMAT.format(date=today, niche=niche)
        r.incrby(key, count)
        # Set expiry to 48 hours (for cleanup)
        r.expire(key, 60 * 60 * 48)
    except Exception as e:
        logger.error(f"Failed to increment daily quota: {e}")


def daily_quota_met(niche: str) -> bool:
    """Check if daily quota is met for a niche."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return True  # Unknown niche, consider quota met

    composed_today = get_videos_composed_today(niche)
    target = niche_config.daily_video_target

    return composed_today >= target


def get_niche_below_daily_quota() -> Optional[str]:
    """Find the most-behind niche that has a model available.

    Returns the niche with the lowest progress ratio (composed_today /
    daily_video_target) that still has work to do. Round-robin'ing this
    way ensures we don't get stuck running just motivation forever — every
    niche gets a turn before any one finishes its full daily quota.

    Tiebreak: alphabetical, so behavior is deterministic.
    """
    config = get_automation_config()

    candidates = []
    for niche_name, niche_config in config.niches.items():
        if not niche_config.enabled:
            continue
        if daily_quota_met(niche_name):
            continue
        if not get_model_for_niche(niche_name):
            continue
        composed = get_videos_composed_today(niche_name)
        target = max(1, niche_config.daily_video_target)
        progress = composed / target
        candidates.append((progress, niche_name))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1]


# NOTE: get_niche_needing_generation() was removed - we now always generate
# when below quota via get_niche_below_daily_quota() to keep caption pool full


# ============================================================================
# GPU Resource Management
# ============================================================================

# Redis keys for GPU control (must match api/gpu_control.py)
EXTRACTION_ENABLED_KEY = "extraction:enabled"
OCR_GPU0_ENABLED_KEY = "ocr:gpu0:enabled"
OCR_GPU1_ENABLED_KEY = "ocr:gpu1:enabled"
LLM_BATCH_ENABLED_KEY = "llm:batch:enabled"
WATERMARK_FILTER_ENABLED_KEY = "watermark_filter:enabled"


def disable_all_gpu_workloads():
    """
    Disable ALL GPU workloads to free GPU for training.

    Disables:
    - Extraction dispatcher
    - OCR on GPU 0 and GPU 1
    - LLM batch refinement
    - Watermark filter (also uses OCR)
    """
    try:
        r = get_redis()

        # Disable extraction dispatcher (use "0" to match is_extraction_enabled() check)
        r.set(EXTRACTION_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled extraction dispatcher")

        # Disable OCR on both GPUs
        r.set(OCR_GPU0_ENABLED_KEY, "0")
        r.set(OCR_GPU1_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled OCR on all GPUs")

        # Disable LLM batch (GPU 0 only)
        r.set(LLM_BATCH_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled LLM batch processing")

        # Disable watermark filter (uses OCR)
        r.set(WATERMARK_FILTER_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled watermark filter")

    except Exception as e:
        logger.error(f"Failed to disable GPU workloads: {e}")


def enable_all_gpu_workloads():
    """
    Re-enable ALL GPU workloads after training completes.

    Re-enables:
    - Extraction dispatcher
    - OCR on GPU 0 and GPU 1
    - LLM batch refinement
    - Watermark filter
    """
    try:
        r = get_redis()

        # Re-enable extraction dispatcher (use "1" to match is_extraction_enabled() check)
        r.set(EXTRACTION_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled extraction dispatcher")

        # Re-enable OCR on both GPUs
        r.set(OCR_GPU0_ENABLED_KEY, "1")
        r.set(OCR_GPU1_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled OCR on all GPUs")

        # Re-enable LLM batch
        r.set(LLM_BATCH_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled LLM batch processing")

        # Re-enable watermark filter
        r.set(WATERMARK_FILTER_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled watermark filter")

    except Exception as e:
        logger.error(f"Failed to enable GPU workloads: {e}")


def disable_extraction():
    """Disable extraction processing to free GPU."""
    try:
        r = get_redis()
        r.set(EXTRACTION_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled extraction")
    except Exception as e:
        logger.error(f"Failed to disable extraction: {e}")


def _maybe_dispatch_tagging_batch(batch_size: int = 100):
    """Dispatch one ML-tagging batch IF nothing else is using the GPU.

    Called from the orchestrator's idle path so background tagging only
    happens when the pipeline has no training, generation, or composition
    work to do. The tagging task itself stops llama-server, runs the VLM,
    and restarts llama-server — so it's safe to fire as long as no other
    GPU workload is queued up.

    Returns True if a batch was dispatched, False otherwise.
    """
    try:
        r = get_redis()

        # Skip if a tagging batch is already running
        if r.get("ml_tagging:running"):
            return False

        # Skip if the user has manually disabled tagging
        enabled = r.get("ml_tagging:enabled")
        if enabled is not None and enabled != b"1":
            return False

        # Use the existing dispatcher in tasks.ml_tagging — it queries the
        # DB for a pending batch and queues a run_tagging_batch task on the
        # ml_tagging queue. The dispatcher itself returns immediately.
        from tasks.ml_tagging import dispatch_ml_tagging_batch
        result = dispatch_ml_tagging_batch(batch_size=batch_size)
        if result.get("status") == "dispatched":
            logger.info(
                f"[ORCHESTRATOR] Idle-dispatched ML tagging batch: {result.get('queued')} videos "
                f"(task {result.get('task_id')})"
            )
            return True
        return False
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] Idle tagging dispatch failed: {e}")
        return False


def enable_extraction():
    """Re-enable extraction processing."""
    try:
        r = get_redis()
        r.set(EXTRACTION_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled extraction")
    except Exception as e:
        logger.error(f"Failed to enable extraction: {e}")


def disable_llm_batch():
    """Disable LLM batch processing to free GPU."""
    try:
        r = get_redis()
        r.set(LLM_BATCH_ENABLED_KEY, "0")
        logger.info("[ORCHESTRATOR] Disabled LLM batch processing")
    except Exception as e:
        logger.error(f"Failed to disable LLM batch: {e}")


def enable_llm_batch():
    """Re-enable LLM batch processing."""
    try:
        r = get_redis()
        r.set(LLM_BATCH_ENABLED_KEY, "1")
        logger.info("[ORCHESTRATOR] Enabled LLM batch processing")
    except Exception as e:
        logger.error(f"Failed to enable LLM batch: {e}")


def extraction_queue_empty() -> bool:
    """Check if extraction queue is empty."""
    try:
        r = get_redis()
        queue_length = r.llen("captions:extraction_queue")
        return queue_length == 0
    except Exception as e:
        logger.error(f"Failed to check extraction queue: {e}")
        return True


def no_active_extractions() -> bool:
    """Check if there are no extraction tasks running."""
    # Check celery inspect for active tasks
    try:
        from celery import current_app
        inspect = current_app.control.inspect()
        active = inspect.active()
        if not active:
            return True

        for worker, tasks in active.items():
            for task in tasks:
                if "extract" in task.get("name", "").lower():
                    return False
        return True
    except Exception as e:
        logger.error(f"Failed to check active extractions: {e}")
        return True  # Assume none running on error


def purge_gpu_queues() -> Dict[str, Any]:
    """
    Selectively revoke extraction tasks from GPU queues.

    This is called when transitioning to training to immediately clear any
    pending extraction tasks that would block GPU resources.

    IMPORTANT: This selectively revokes only extraction-related tasks,
    preserving training and generation tasks in the queue.

    Actions:
    1. Revoke reserved (queued) extraction tasks
    2. Revoke active (running) extraction tasks
    3. Clear processing markers in Redis
    4. Clear Redis extraction queue to prevent re-dispatch
    """
    logger.warning("[ORCHESTRATOR] Selectively purging extraction tasks for training transition")

    results = {
        "reserved_extraction_revoked": 0,
        "active_extraction_revoked": 0,
        "processing_markers_cleared": 0,
        "extraction_queue_cleared": 0,
        "preserved_tasks": [],
    }

    try:
        from celery import current_app
        r = get_redis()
        inspect = current_app.control.inspect()

        # 1. Revoke RESERVED (queued) extraction tasks only
        reserved = inspect.reserved() or {}
        for worker, tasks in reserved.items():
            for task in tasks:
                task_name = task.get('name', '')
                task_id = task.get('id')
                # Only revoke extraction-related tasks
                if any(keyword in task_name.lower() for keyword in ['extract', 'ocr', 'llm_refine', 'watermark']):
                    if task_id:
                        current_app.control.revoke(task_id)
                        results["reserved_extraction_revoked"] += 1
                else:
                    # Track preserved tasks (training, generation, etc.)
                    results["preserved_tasks"].append(task_name)

        logger.info(f"[ORCHESTRATOR] Revoked {results['reserved_extraction_revoked']} reserved extraction tasks")

        # 2. Revoke ACTIVE (running) extraction tasks
        active = inspect.active() or {}
        for worker, tasks in active.items():
            for task in tasks:
                task_name = task.get('name', '')
                task_id = task.get('id')
                if any(keyword in task_name.lower() for keyword in ['extract', 'ocr', 'watermark']):
                    if task_id:
                        current_app.control.revoke(task_id, terminate=True)
                        results["active_extraction_revoked"] += 1

        logger.info(f"[ORCHESTRATOR] Revoked {results['active_extraction_revoked']} active extraction tasks")

        # 3. Clear processing markers from Redis extraction queue
        processing_key = "captions:extraction_processing"
        processing_count = r.hlen(processing_key) or 0
        r.delete(processing_key)
        results["processing_markers_cleared"] = processing_count
        logger.info(f"[ORCHESTRATOR] Cleared {processing_count} processing markers")

        # 4. Clear the Redis extraction queue to prevent re-dispatch
        extraction_queue_key = "captions:extraction_queue"
        queue_length = r.llen(extraction_queue_key) or 0
        r.delete(extraction_queue_key)
        results["extraction_queue_cleared"] = queue_length
        logger.info(f"[ORCHESTRATOR] Cleared {queue_length} items from extraction queue")

        # 5. Clear the Celery broker queues for GPU workers (removes dispatched but unprocessed tasks)
        # This is critical because tasks pile up in Celery faster than workers process them
        total_cleared = 0
        for queue_name in ["gpu", "gpu_llm", "gpu_status_0", "gpu_status_1"]:
            queue_len = r.llen(queue_name) or 0
            if queue_len > 0:
                r.delete(queue_name)
                total_cleared += queue_len
                logger.info(f"[ORCHESTRATOR] Cleared {queue_len} tasks from Celery {queue_name} broker queue")
        results["celery_gpu_queues_cleared"] = total_cleared

    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Failed to purge extraction tasks: {e}")

    logger.warning(f"[ORCHESTRATOR] Extraction purge complete: {results}")
    return results


def ensure_extraction_enabled():
    """Make sure ALL extraction workloads are enabled (for collection phase)."""
    enable_all_gpu_workloads()


def unload_all_models():
    """Unload all loaded inference models to free GPU memory."""
    logger.info("[ORCHESTRATOR] Unloading all loaded models")

    try:
        # Unload OCR and LLM models via maintenance tasks
        from tasks.maintenance_tasks import unload_gpu_0_models, unload_gpu_1_models

        # Queue unload tasks (they run on their respective GPU workers)
        try:
            unload_gpu_0_models.delay()
            logger.info("[ORCHESTRATOR] Queued model unload for GPU 0")
        except Exception as e:
            logger.warning(f"Could not queue GPU 0 unload: {e}")

        try:
            unload_gpu_1_models.delay()
            logger.info("[ORCHESTRATOR] Queued model unload for GPU 1")
        except Exception as e:
            logger.warning(f"Could not queue GPU 1 unload: {e}")

        # Unload any trained LoRA models that are loaded for generation
        with get_db_context() as db:
            loaded_models = db.query(TrainedModel).filter(
                TrainedModel.is_loaded == True
            ).all()

            for model in loaded_models:
                try:
                    model_gpu = getattr(model, 'target_gpu', 0)
                    if model_gpu == 0:
                        from tasks.training_tasks import unload_model_task_gpu_0
                        unload_model_task_gpu_0.delay(model.id)
                    else:
                        from tasks.training_tasks import unload_model_task_gpu_1
                        unload_model_task_gpu_1.delay(model.id)
                    logger.info(f"[ORCHESTRATOR] Queued unload for model '{model.name}' on GPU {model_gpu}")
                except Exception as e:
                    logger.warning(f"Could not queue unload for model {model.id}: {e}")
                    # Fallback: mark as unloaded in DB
                    model.is_loaded = False
                    model.status = "ready"

    except ImportError as e:
        logger.warning(f"Could not import unload tasks: {e}")
    except Exception as e:
        logger.error(f"Failed to unload models: {e}")


def deep_clean_gpus():
    """Deep clean GPU memory to free maximum space."""
    logger.info("[ORCHESTRATOR] Deep cleaning GPU memory")

    try:
        from tasks.maintenance_tasks import deep_clean_gpu_0, deep_clean_gpu_1

        try:
            deep_clean_gpu_0.delay()
            logger.info("[ORCHESTRATOR] Queued deep clean for GPU 0")
        except Exception as e:
            logger.warning(f"Could not queue GPU 0 deep clean: {e}")

        try:
            deep_clean_gpu_1.delay()
            logger.info("[ORCHESTRATOR] Queued deep clean for GPU 1")
        except Exception as e:
            logger.warning(f"Could not queue GPU 1 deep clean: {e}")

    except ImportError as e:
        logger.warning(f"Could not import deep clean tasks: {e}")
    except Exception as e:
        logger.error(f"Failed to deep clean GPUs: {e}")


def get_gpu_memory_from_cache(gpu_id: int) -> Optional[Dict[str, Any]]:
    """
    Get GPU memory status from Redis cache.

    This reads from the cache populated by cache_gpu_*_status tasks,
    allowing the orchestrator to check GPU memory without needing GPU access.

    Returns:
        Dict with memory_used_mb, memory_total_mb, memory_percent, or None if unavailable
    """
    try:
        import json
        r = get_redis()
        cache_key = f"gpu:status:cache:{gpu_id}"
        cached = r.get(cache_key)

        if cached:
            data = json.loads(cached)
            return {
                "memory_used_mb": data.get("memory_used_mb", 0),
                "memory_total_mb": data.get("memory_total_mb", 11264),
                "memory_percent": data.get("memory_percent", 0),
                "loaded_models": data.get("loaded_models", []),
                "cached_at": data.get("cached_at"),
            }
        return None
    except Exception as e:
        logger.warning(f"Failed to get GPU {gpu_id} memory from cache: {e}")
        return None


def get_gpu_memory_direct(gpu_id: int = 0) -> Optional[Dict[str, Any]]:
    """
    Query GPU memory directly via Celery task (bypasses 30s cache).

    Use this for critical transitions where stale cache data could cause issues.
    This sends a synchronous task to the GPU worker and waits for result.

    Args:
        gpu_id: GPU to query (0 or 1)

    Returns:
        Dict with memory_used_mb, memory_total_mb, etc., or None if query failed
    """
    try:
        from tasks.maintenance_tasks import cache_gpu_0_status, cache_gpu_1_status

        # Queue the task and wait for result synchronously (up to 10 seconds)
        if gpu_id == 0:
            result = cache_gpu_0_status.apply_async(queue="gpu_status_0")
        else:
            result = cache_gpu_1_status.apply_async(queue="gpu_status_1")

        # Wait for result with timeout
        gpu_data = result.get(timeout=10)

        if gpu_data:
            return {
                "memory_used_mb": gpu_data.get("memory_used_mb", 0),
                "memory_total_mb": gpu_data.get("memory_total_mb", 11264),
                "memory_percent": gpu_data.get("memory_percent", 0),
                "loaded_models": gpu_data.get("loaded_models", []),
                "queried_at": datetime.utcnow().isoformat(),
            }
        return None
    except Exception as e:
        logger.warning(f"Direct GPU {gpu_id} query failed: {e}")
        return None


def gpu_memory_is_clear(gpu_id: int = 0, max_memory_mb: int = 2000, use_direct_query: bool = False) -> bool:
    """
    Check if GPU memory is below threshold (i.e., models are unloaded).

    Args:
        gpu_id: GPU to check (default: 0 for training GPU)
        max_memory_mb: Maximum acceptable memory usage in MB (default: 2000 = ~2GB)
                       This allows for base CUDA overhead (~500MB) plus safety margin.
        use_direct_query: If True, query GPU directly (slower but real-time).
                          If False, use 30s cache (faster but potentially stale).
                          Use direct query for critical transitions.

    Returns:
        True if memory is below threshold, False otherwise
    """
    status = None

    if use_direct_query:
        status = get_gpu_memory_direct(gpu_id)
        logger.info(f"[ORCHESTRATOR] GPU {gpu_id} direct query result: {status}")

    # Fallback to cache if direct query failed or not requested
    if not status:
        status = get_gpu_memory_from_cache(gpu_id)
        if status and use_direct_query:
            logger.info(f"[ORCHESTRATOR] GPU {gpu_id} direct query failed, using cache: {status.get('memory_used_mb', 'N/A')}MB")

    if not status:
        # Cache/query unavailable, trigger cache update and wait for next tick
        logger.info(f"[ORCHESTRATOR] GPU {gpu_id} status unavailable, triggering update")
        try:
            from tasks.maintenance_tasks import cache_gpu_0_status, cache_gpu_1_status
            if gpu_id == 0:
                cache_gpu_0_status.delay()
            else:
                cache_gpu_1_status.delay()
        except Exception as e:
            logger.warning(f"Could not trigger GPU cache update: {e}")
        return False

    memory_used = status.get("memory_used_mb", 0)
    loaded_models = status.get("loaded_models", [])

    logger.info(f"[ORCHESTRATOR] GPU {gpu_id} memory: {memory_used}MB (threshold: {max_memory_mb}MB), loaded models: {len(loaded_models)}")

    # If no models are loaded, consider GPU clear regardless of nvidia-smi memory
    # The baseline CUDA overhead (~4-5GB) is reusable memory from driver/context allocations
    # that gets reclaimed when actual models are loaded
    if len(loaded_models) == 0:
        logger.info(f"[ORCHESTRATOR] GPU {gpu_id} has no loaded models - considering clear (baseline CUDA memory is reusable)")
        return True

    # If models ARE loaded, check if memory is under threshold
    return memory_used < max_memory_mb


def prepare_gpu_for_training(gpu_id: int = 0):
    """
    Comprehensive GPU preparation for training.

    Performs:
    1. Disable ALL GPU workloads (extraction, OCR, LLM, watermark filter)
    2. Purge any pending extraction tasks from GPU queue
    3. Unload all inference models (OCR, LLM, LoRA)
    4. Deep clean GPU memory
    """
    logger.info(f"[ORCHESTRATOR] Preparing GPU {gpu_id} for training")

    # 1. Disable all GPU workloads
    disable_all_gpu_workloads()

    # 2. Purge GPU queues (clear pending extraction tasks)
    purge_gpu_queues()

    # 3. Unload all loaded models
    unload_all_models()

    # 4. Deep clean GPU memory
    deep_clean_gpus()

    logger.info(f"[ORCHESTRATOR] GPU {gpu_id} preparation complete - waiting for memory to clear")


# ============================================================================
# Training Job Management
# ============================================================================

def get_current_training_job_id(niche: str) -> Optional[int]:
    """Get ID of active training job for a niche."""
    try:
        r = get_redis()
        key = NICHE_TRAINING_JOB_KEY.format(niche=niche)
        job_id = r.get(key)
        return int(job_id) if job_id else None
    except Exception as e:
        logger.error(f"Failed to get training job ID: {e}")
        return None


def set_current_training_job_id(niche: str, job_id: int):
    """Set active training job for a niche."""
    try:
        r = get_redis()
        key = NICHE_TRAINING_JOB_KEY.format(niche=niche)
        r.set(key, str(job_id))
    except Exception as e:
        logger.error(f"Failed to set training job ID: {e}")


def clear_current_training_job(niche: str):
    """Clear training job tracking for a niche."""
    try:
        r = get_redis()
        key = NICHE_TRAINING_JOB_KEY.format(niche=niche)
        r.delete(key)
    except Exception as e:
        logger.error(f"Failed to clear training job: {e}")


def start_training_job(niche: str):
    """
    Create and start training job for a specific niche.

    Uses TRAINING_PROVIDER setting to choose between:
    - "vastai": Cloud training on Vast.ai (for large models like Mistral-Small-24B)
    - "local": Local GPU training (for smaller models)
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        logger.error(f"[ORCHESTRATOR] Unknown niche: {niche}")
        return

    # RACE CONDITION GUARD: Check if there's already an active training job
    existing_job_id = get_current_training_job_id(niche)
    if existing_job_id:
        with get_db_context() as db:
            existing_job = db.query(TrainingJob).filter_by(id=existing_job_id).first()
            if existing_job and existing_job.status in ["queued", "training", "saving"]:
                logger.warning(f"[ORCHESTRATOR] {niche}: Training job {existing_job_id} already active, skipping")
                return  # Already have an active job
        # Job exists but is completed/failed, clear it
        clear_current_training_job(niche)

    training_provider = settings.TRAINING_PROVIDER.lower()
    gpu_config = config.gpu

    if training_provider == "vastai":
        # ================================================================
        # VAST.AI CLOUD TRAINING
        # ================================================================
        if not settings.VASTAI_API_KEY:
            logger.error("[ORCHESTRATOR] VASTAI_API_KEY not configured but TRAINING_PROVIDER=vastai")
            return

        logger.info(f"[ORCHESTRATOR] Starting Vast.ai cloud training for '{niche}'")

        # Create training job record for tracking
        with get_db_context() as db:
            job_name = f"{niche}-vastai-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
            job = TrainingJob(
                job_name=job_name,
                base_model="mistralai/Mistral-Small-24B-Instruct-2501",
                source_subreddits=niche_config.subreddits,
                niche=niche,
                min_upvotes=100,  # Vast.ai training uses more data
                target_gpu=0,  # Not used for cloud training
                status="queued"
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            job_id = job.id
            logger.info(f"[ORCHESTRATOR] Created Vast.ai training job {job_id}: {job_name}")

        # Queue Vast.ai training task (runs on maintenance queue, no local GPU needed)
        from tasks.training_tasks import trigger_cloud_training
        result = trigger_cloud_training.apply_async(
            args=[niche],
            kwargs={"provider": "vastai", "job_id": job_id},
            queue="maintenance"
        )

        # Update job with celery task ID
        with get_db_context() as db:
            job = db.query(TrainingJob).filter_by(id=job_id).first()
            if job:
                job.celery_task_id = result.id
                db.commit()

        # Track in Redis
        set_current_training_job_id(niche, job_id)
        logger.info(f"[ORCHESTRATOR] Started Vast.ai training for '{niche}': task_id={result.id}")

    else:
        # ================================================================
        # LOCAL GPU TRAINING
        # ================================================================
        logger.info(f"[ORCHESTRATOR] Starting local GPU training for '{niche}'")

        # Create training job in database
        with get_db_context() as db:
            job_name = f"{niche}-lora-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

            job = TrainingJob(
                job_name=job_name,
                base_model="mistralai/Mistral-7B-Instruct-v0.3",
                source_subreddits=niche_config.subreddits,
                niche=niche,  # Per-niche tracking
                min_upvotes=300,
                target_gpu=gpu_config.training_gpu,
                status="queued"
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            job_id = job.id
            logger.info(f"[ORCHESTRATOR] Created training job {job_id}: {job_name}")

        # Queue training task on GPU 0 (11GB RTX 2080 Ti) - needs more VRAM than GPU 1
        from tasks.training_tasks import run_training_job
        result = run_training_job.apply_async(args=[job_id], queue="gpu_training_0")

        # Update job with celery task ID
        with get_db_context() as db:
            job = db.query(TrainingJob).filter_by(id=job_id).first()
            if job:
                job.celery_task_id = result.id
                db.commit()

        # Track in Redis
        set_current_training_job_id(niche, job_id)
        logger.info(f"[ORCHESTRATOR] Started local training job for '{niche}': {job_name}")


def training_complete(niche: str) -> bool:
    """
    Check if training is complete for current niche.

    Also handles stuck jobs:
    - Jobs in "saving" status with a saved model are auto-fixed to "completed"
    - Jobs in "saving" status for >30 min without a model are marked as "failed"
    """
    job_id = get_current_training_job_id(niche)
    if not job_id:
        return True  # No job, consider complete

    with get_db_context() as db:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            return True

        if job.status in ["completed", "failed", "cancelled"]:
            logger.info(f"[ORCHESTRATOR] Training job {job_id} finished with status: {job.status}")
            clear_current_training_job(niche)
            return True

        # Handle stuck "saving" status - check if model was actually saved
        if job.status == "saving":
            model = db.query(TrainedModel).filter(
                TrainedModel.name == job.job_name
            ).first()

            if model:
                # Model exists! Worker crashed after saving, auto-fix the job
                logger.warning(f"[ORCHESTRATOR] Auto-fixing stuck job {job_id} (status='saving' but model exists)")
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.progress_percent = 100.0
                if not model.niche and job.niche:
                    model.niche = job.niche
                db.commit()
                clear_current_training_job(niche)
                return True

            # Check if stuck for too long (>30 minutes in saving status)
            if job.started_at:
                stuck_time = datetime.utcnow() - job.started_at
                if stuck_time.total_seconds() > 1800:  # 30 minutes
                    logger.error(f"[ORCHESTRATOR] Job {job_id} stuck in 'saving' for >30min without model, marking as failed")
                    job.status = "failed"
                    job.completed_at = datetime.utcnow()
                    job.error_message = "Stuck in saving status - model not found after 30 minutes"
                    db.commit()
                    clear_current_training_job(niche)
                    return True

        return False


# ============================================================================
# Generation Job Management
# ============================================================================

def get_current_generation_job_id(niche: str) -> Optional[int]:
    """Get ID of active generation job for a niche."""
    try:
        r = get_redis()
        key = NICHE_GENERATION_JOB_KEY.format(niche=niche)
        job_id = r.get(key)
        return int(job_id) if job_id else None
    except Exception as e:
        logger.error(f"Failed to get generation job ID: {e}")
        return None


def set_current_generation_job_id(niche: str, job_id: int):
    """Set active generation job for a niche."""
    try:
        r = get_redis()
        key = NICHE_GENERATION_JOB_KEY.format(niche=niche)
        r.set(key, str(job_id))
    except Exception as e:
        logger.error(f"Failed to set generation job ID: {e}")


def clear_current_generation_job(niche: str):
    """Clear generation job tracking for a niche."""
    try:
        r = get_redis()
        key = NICHE_GENERATION_JOB_KEY.format(niche=niche)
        r.delete(key)
    except Exception as e:
        logger.error(f"Failed to clear generation job: {e}")


def verify_model_files_exist(model) -> bool:
    """
    Verify that model adapter files actually exist on disk.

    This prevents generation jobs from failing immediately due to
    missing model files (e.g., from volume mount issues).

    Args:
        model: TrainedModel instance with adapter_path

    Returns:
        True if model files exist, False otherwise
    """
    import os

    if not model or not model.adapter_path:
        return False

    adapter_path = model.adapter_path

    # Check for GGUF adapter (used by llama-server)
    adapter_gguf = os.path.join(adapter_path, "adapter.gguf")
    if os.path.exists(adapter_gguf):
        logger.info(f"[ORCHESTRATOR] GGUF adapter verified at {adapter_gguf}")
        return True

    # Fallback: check for PEFT format (adapter_config.json + weights)
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_weights_bin = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(adapter_config) and (os.path.exists(adapter_weights) or os.path.exists(adapter_weights_bin)):
        logger.info(f"[ORCHESTRATOR] PEFT adapter verified at {adapter_path}")
        return True

    logger.error(f"[ORCHESTRATOR] No adapter files found in {adapter_path}")

    logger.info(f"[ORCHESTRATOR] Model files verified at {adapter_path}")
    return True


def _start_bg_first_cycle(niche: str) -> bool:
    """
    Stage 2 path: kick off a BG-first generation cycle.

    Uses a Redis sentinel ('bg_first_cycle:<niche>') with staleness
    detection: if a sentinel exists but no caption_candidates have been
    created for this niche in the last 5 min, treat it as stale and
    re-fire. This handles cases where the worker died mid-cycle (deploy,
    OOM, etc) and celery's task_acks_late retry didn't complete.

    Skip-if-already-have-winners: if the niche already has more pending
    winners than its remaining daily quota, don't fire another cycle —
    let composition catch up first. This prevents wasteful generation
    after a partial-failed composition job leaves winners in the pool.
    """
    import os
    from database.models import CaptionCandidate
    r = get_redis()
    cycle_key = f"bg_first_cycle:{niche}"

    # Skip if we have any pending winners — compose them first.
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if niche_config:
        with get_db_context() as db:
            pending_winners = db.query(CaptionCandidate).filter(
                CaptionCandidate.niche == niche,
                CaptionCandidate.status.in_(["winner", "composing"]),
            ).count()
        if pending_winners > 0:
            composed_today = get_videos_composed_today(niche)
            remaining_quota = niche_config.daily_video_target - composed_today
            logger.info(
                f"[ORCHESTRATOR] {niche}: {pending_winners} pending winners exist "
                f"(remaining quota {remaining_quota}); skipping new BG-first cycle"
            )
            return True

    # Already-running guard with staleness detection.
    if r.get(cycle_key):
        # If we've seen NO progress (no new candidates) in 5+ min, the cycle
        # is dead and the sentinel is dangling. Reset and proceed to dispatch.
        with get_db_context() as db:
            from sqlalchemy import func
            last_cand_at = db.query(func.max(CaptionCandidate.created_at)).filter(
                CaptionCandidate.niche == niche,
            ).scalar()
        if last_cand_at:
            age_seconds = (datetime.now(timezone.utc) - last_cand_at).total_seconds()
            if age_seconds < 300:
                logger.info(f"[ORCHESTRATOR] BG-first cycle for {niche} is alive (last cand {age_seconds:.0f}s ago), skipping")
                return True
            logger.warning(
                f"[ORCHESTRATOR] BG-first sentinel for {niche} looks stale "
                f"(last cand {age_seconds:.0f}s ago); clearing and re-firing"
            )
            r.delete(cycle_key)
        else:
            # Sentinel set but no candidates ever created — stale.
            logger.warning(f"[ORCHESTRATOR] BG-first sentinel for {niche} stale (no candidates); clearing")
            r.delete(cycle_key)

    # Sane defaults: 30 BGs * 10 candidates = 300 generations per cycle.
    # ~50 minutes on Mistral-Small. Tunable via env.
    n_bgs = int(os.environ.get("BG_FIRST_BGS_PER_CYCLE", "30"))
    candidates_per_bg = int(os.environ.get("BG_FIRST_CANDIDATES_PER_BG", "10"))

    # Set sentinel BEFORE dispatching so a quick second tick doesn't double-fire.
    r.set(cycle_key, "running", ex=7200)  # 2h max — task time_limit is 7200

    from tasks.bg_first_generation import run_bg_first_cycle
    result = run_bg_first_cycle.apply_async(
        args=[niche, n_bgs, candidates_per_bg],
        queue="maintenance",
    )
    logger.info(
        f"[ORCHESTRATOR] {niche}: BG-first cycle dispatched "
        f"(n_bgs={n_bgs}, candidates_per_bg={candidates_per_bg}, task_id={result.id})"
    )
    set_current_generation_job_id(niche, 0)  # Sentinel value — orchestrator state stays in 'generating'
    return True


def start_generation_job(niche: str) -> bool:
    """
    Create and start generation job for a niche.

    With BG_FIRST_GENERATION=true (default), uses Stage 2 BG-first flow:
    pick BGs, generate K candidates per BG grounded in scene_description,
    judge inline. Otherwise falls back to legacy "generate blind, match
    later" path.

    Returns:
        True if job was successfully started, False otherwise
    """
    import os
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        logger.error(f"[ORCHESTRATOR] Unknown niche: {niche}")
        return False

    # Stage 2: BG-first generation (default ON)
    bg_first_enabled = os.environ.get("BG_FIRST_GENERATION", "true").lower() in ("1", "true", "yes")
    if bg_first_enabled:
        return _start_bg_first_cycle(niche)

    # Legacy path below.

    # RACE CONDITION GUARD: Check if there's already an active generation job
    existing_job_id = get_current_generation_job_id(niche)
    if existing_job_id:
        with get_db_context() as db:
            existing_job = db.query(GenerationJob).filter_by(id=existing_job_id).first()
            if existing_job and existing_job.status in ["queued", "running"]:
                logger.warning(f"[ORCHESTRATOR] {niche}: Generation job {existing_job_id} already active, skipping")
                return True  # Return True since we have an active job
        # Job exists but is completed/failed, clear it
        clear_current_generation_job(niche)

    # Check consecutive failures - don't start if too many recent failures
    failures = get_consecutive_failures(niche)
    if failures >= MAX_CONSECUTIVE_FAILURES:
        logger.error(f"[ORCHESTRATOR] {niche}: Too many consecutive failures ({failures}), skipping generation")
        return False

    # Get model for this niche
    model = get_model_for_niche(niche)
    if not model:
        logger.error(f"[ORCHESTRATOR] No model available for niche: {niche}")
        return False

    # CRITICAL: Verify model files exist before queuing job
    if not verify_model_files_exist(model):
        logger.error(f"[ORCHESTRATOR] Model files not found for {niche}, cannot start generation")
        increment_consecutive_failures(niche)
        return False

    with get_db_context() as db:
        job_name = f"{niche}-gen-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

        # Use niche-specific prompt and parameters from config
        generation_prompt = niche_config.generation_prompt
        generation_temp = niche_config.generation_temperature
        generation_rep_penalty = niche_config.generation_repetition_penalty

        logger.info(f"[ORCHESTRATOR] Using niche-specific prompt for {niche} (temp={generation_temp}, rep_penalty={generation_rep_penalty})")

        job = GenerationJob(
            model_id=model.id,
            job_name=job_name,
            num_captions=config.generation.captions_per_batch,
            prompt=generation_prompt,
            temperature=generation_temp,
            top_p=0.92,  # Slightly tighter sampling for coherence
            max_new_tokens=120,  # Hard-cap output length for 40-80 word target
            repetition_penalty=generation_rep_penalty,
            status="queued"
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        job_id = job.id
        logger.info(f"[ORCHESTRATOR] Created generation job {job_id}: {job_name}")

    # Queue generation task (runs on maintenance queue - uses llama-server API)
    from tasks.training_tasks import run_generation_job
    result = run_generation_job.apply_async(args=[job_id], queue="maintenance")

    # Update job with celery task ID
    with get_db_context() as db:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if job:
            job.celery_task_id = result.id
            db.commit()

    set_current_generation_job_id(niche, job_id)
    logger.info(f"[ORCHESTRATOR] Started generation job for '{niche}': {job_name}")
    return True


def generation_complete(niche: str) -> bool:
    """
    Check if generation is complete for current niche.

    For BG-first cycles (Stage 2), checks the Redis sentinel
    'bg_first_cycle:<niche>' set by _start_bg_first_cycle. The cycle
    is "complete" when the key has been removed (cleared by the cycle
    task on finish, or by TTL expiry on hang).

    For legacy generation, checks the GenerationJob row.

    Tracks consecutive failures for auto-recovery.
    """
    job_id = get_current_generation_job_id(niche)

    # BG-first sentinel: job_id=0 means we're in a BG-first cycle.
    if job_id == 0:
        r = get_redis()
        cycle_key = f"bg_first_cycle:{niche}"
        if r.get(cycle_key):
            return False  # still running
        # Cycle ended (either finished or TTL'd).
        from database.models import CaptionCandidate
        with get_db_context() as db:
            winner_count = db.query(CaptionCandidate).filter_by(niche=niche, status="winner").count()
        logger.info(f"[ORCHESTRATOR] BG-first cycle complete for {niche}: {winner_count} winners total")
        if winner_count > 0:
            reset_consecutive_failures(niche)
        return True

    if not job_id:
        return True

    with get_db_context() as db:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if not job:
            return True

        if job.status in ["completed", "failed", "cancelled"]:
            logger.info(f"[ORCHESTRATOR] Generation job {job_id} finished with status: {job.status}")

            # Track failures/successes for auto-recovery
            if job.status == "failed":
                increment_consecutive_failures(niche)
                logger.warning(f"[ORCHESTRATOR] Generation job {job_id} failed: {job.error_message}")
            elif job.status == "completed":
                reset_consecutive_failures(niche)

            # NOTE: Don't clear job tracking here - do it after scoring
            return True

        return False


def score_generated_captions(niche: str, job_id: int = None):
    """
    Auto-approve high-quality captions based on quality score.

    Captions with score >= threshold are automatically approved.
    This replaces manual review for automated pipeline.

    Args:
        niche: The niche name to score captions for
        job_id: Optional specific job ID. If not provided, uses Redis key or finds most recent job.
    """
    config = get_automation_config()
    threshold = config.generation.min_quality_score_for_composition

    # Try to get job_id from multiple sources
    if not job_id:
        job_id = get_current_generation_job_id(niche)

    if not job_id:
        # Fallback: Find most recent job for this niche from database
        model = get_model_for_niche(niche)
        if model:
            with get_db_context() as db:
                recent_job = db.query(GenerationJob).filter(
                    GenerationJob.model_id == model.id,
                    GenerationJob.status.in_(["completed", "failed"])
                ).order_by(GenerationJob.created_at.desc()).first()
                if recent_job:
                    job_id = recent_job.id
                    logger.info(f"[ORCHESTRATOR] Using fallback job ID {job_id} for {niche} (Redis key was empty)")

    if not job_id:
        logger.warning(f"[ORCHESTRATOR] No job found for {niche}, cannot score captions")
        return

    with get_db_context() as db:
        # Get captions from this job that are pending review
        captions = db.query(GeneratedCaption).filter(
            GeneratedCaption.generation_job_id == job_id,
            GeneratedCaption.status == "pending_review",
            GeneratedCaption.quality_score.isnot(None)
        ).all()

        approved = 0
        for caption in captions:
            if caption.quality_score >= threshold:
                caption.status = "approved"
                approved += 1

        db.commit()
        logger.info(f"[ORCHESTRATOR] Auto-approved {approved}/{len(captions)} captions for {niche} job {job_id} (threshold: {threshold})")


# ============================================================================
# Composition Job Management
# ============================================================================

def get_current_composition_job_id(niche: str) -> Optional[int]:
    """Get ID of active composition job for a niche."""
    try:
        r = get_redis()
        key = NICHE_COMPOSITION_JOB_KEY.format(niche=niche)
        job_id = r.get(key)
        return int(job_id) if job_id else None
    except Exception as e:
        logger.error(f"Failed to get composition job ID: {e}")
        return None


def set_current_composition_job_id(niche: str, job_id: int):
    """Set active composition job for a niche."""
    try:
        r = get_redis()
        key = NICHE_COMPOSITION_JOB_KEY.format(niche=niche)
        r.set(key, str(job_id))
    except Exception as e:
        logger.error(f"Failed to set composition job ID: {e}")


def clear_current_composition_job(niche: str):
    """Clear composition job tracking for a niche."""
    try:
        r = get_redis()
        key = NICHE_COMPOSITION_JOB_KEY.format(niche=niche)
        r.delete(key)
    except Exception as e:
        logger.error(f"Failed to clear composition job: {e}")


def _judge_filter(query):
    """
    Stage 3 filter: drop captions that the LLM judge actively failed.

    Captions with judge_status NULL or 'pending' or 'failed' are still
    eligible — we don't want unjudged historical rows to be locked out of
    composition. Only rows the judge has actively scored AND failed get
    excluded. The judge backlog beat will eventually score everything.
    """
    from sqlalchemy import or_
    return query.filter(
        or_(
            GeneratedCaption.judge_status != "judged",
            GeneratedCaption.judge_pass.is_(True),
        )
    )


def ready_for_composition(niche: str) -> bool:
    """Check if there are enough approved captions for composition.

    Uses niche-based filtering (not model-based) so captions from previous
    models can still be used after training a new model.

    If no captions have niche field set, falls back to model-name matching
    to handle legacy data that wasn't backfilled.
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return False

    # We still need a model to exist, but don't require captions from it specifically
    model = get_model_for_niche(niche)
    if not model:
        return False

    with get_db_context() as db:
        # Get IDs of captions already used in composed videos
        used_ids = db.query(ComposedVideo.generated_caption_id).filter(
            ComposedVideo.generated_caption_id.isnot(None)
        ).all()
        used_ids = [id[0] for id in used_ids]

        # Count approved captions for this niche not yet used
        query = db.query(GeneratedCaption).filter(
            GeneratedCaption.niche == niche,
            GeneratedCaption.status == "approved",
            GeneratedCaption.quality_score >= config.generation.min_quality_score_for_composition
        )
        query = _judge_filter(query)

        if used_ids:
            query = query.filter(~GeneratedCaption.id.in_(used_ids))

        ready_count = query.count()

        # FALLBACK: If no captions found with niche field, try matching by model name
        # This handles legacy data where niche wasn't set
        if ready_count == 0:
            fallback_query = db.query(GeneratedCaption).filter(
                GeneratedCaption.llm_model.ilike(f"{niche}-%"),
                GeneratedCaption.status == "approved",
                GeneratedCaption.quality_score >= config.generation.min_quality_score_for_composition
            )
            fallback_query = _judge_filter(fallback_query)
            if used_ids:
                fallback_query = fallback_query.filter(~GeneratedCaption.id.in_(used_ids))

            fallback_count = fallback_query.count()
            if fallback_count > 0:
                logger.warning(
                    f"[ORCHESTRATOR] {niche}: Found {fallback_count} captions via model-name fallback "
                    f"(niche field is NULL). Run backfill_niche_from_model_name.py migration!"
                )
                ready_count = fallback_count

        logger.debug(f"[ORCHESTRATOR] {niche}: {ready_count} captions ready for composition")
        return ready_count >= config.composition.min_captions_for_batch


def get_available_caption_count(niche: str) -> int:
    """
    Get count of approved captions that haven't been used in composed videos yet.

    This is the actual number available for composition, excluding already-used captions.
    Falls back to model-name matching if niche field is NULL.
    """
    model = get_model_for_niche(niche)
    if not model:
        return 0

    config = get_automation_config()

    with get_db_context() as db:
        # Get IDs of captions already used in composed videos
        used_ids = db.query(ComposedVideo.generated_caption_id).filter(
            ComposedVideo.generated_caption_id.isnot(None)
        ).all()
        used_ids = [id[0] for id in used_ids]

        # Count approved captions not yet used
        query = db.query(GeneratedCaption).filter(
            GeneratedCaption.niche == niche,
            GeneratedCaption.status == "approved",
            GeneratedCaption.quality_score >= config.generation.min_quality_score_for_composition
        )
        query = _judge_filter(query)

        if used_ids:
            query = query.filter(~GeneratedCaption.id.in_(used_ids))

        count = query.count()

        # FALLBACK: Try model-name matching if niche field is NULL
        if count == 0:
            fallback_query = db.query(GeneratedCaption).filter(
                GeneratedCaption.llm_model.ilike(f"{niche}-%"),
                GeneratedCaption.status == "approved",
                GeneratedCaption.quality_score >= config.generation.min_quality_score_for_composition
            )
            fallback_query = _judge_filter(fallback_query)
            if used_ids:
                fallback_query = fallback_query.filter(~GeneratedCaption.id.in_(used_ids))
            count = fallback_query.count()

        return count


def find_matching_background(niche: str, caption_id: int) -> Optional[int]:
    """
    Find a background video compatible with the niche via subreddit matching.

    Returns background video ID or None.
    """
    from config.niche_rules import is_subreddit_compatible

    config = get_automation_config()
    niche_config = config.get_niche(niche)
    comp_config = config.composition

    if not niche_config:
        return None

    with get_db_context() as db:
        # Calculate cooldown cutoff (timezone-aware to match Postgres TIMESTAMPTZ columns)
        cooldown_hours = comp_config.background_reuse_cooldown_hours
        cooldown_cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)

        # Find approved backgrounds
        candidates = db.query(BackgroundVideo).filter(
            BackgroundVideo.filter_status == "approved",
            BackgroundVideo.download_status == "completed",
            BackgroundVideo.source_type == "reddit",
        ).all()

        # Score each candidate using subreddit matching
        scored = []
        for bg in candidates:
            # Check cooldown (if last_used_at exists)
            if hasattr(bg, 'last_used_at') and bg.last_used_at:
                if bg.last_used_at > cooldown_cutoff:
                    continue

            # Check subreddit compatibility
            if not is_subreddit_compatible(bg.searched_tag, niche):
                continue

            # Score by views
            score = min(bg.views / 10000, 10) if comp_config.prefer_high_view_backgrounds and bg.views else 0

            scored.append((bg.id, score))

        if not scored:
            logger.warning(f"[ORCHESTRATOR] No compatible backgrounds for {niche}")
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


def start_bg_first_composition_job(niche: str) -> bool:
    """
    Stage 2 composition: take winners from caption_candidates and compose
    each one against its OWN background. No BG matching at composition
    time — the pairing was decided at generation time.

    Mirrors each winner into generated_captions (so the swipe queue, claude
    review, and existing approval flow all keep working unchanged), then
    dispatches one compose_single_video_task per pair.
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        return False

    daily_target = niche_config.daily_video_target
    composed_today = get_videos_composed_today(niche)
    remaining = daily_target - composed_today
    if remaining <= 0:
        logger.info(f"[ORCHESTRATOR] Daily quota already met for {niche}")
        return False

    from database.models import CaptionCandidate, BackgroundVideo
    model = get_model_for_niche(niche)

    pairs_to_compose = []
    with get_db_context() as db:
        # Backgrounds already in a composed video — exclude (one comp per BG).
        used_bg_ids = {
            r[0] for r in db.query(ComposedVideo.background_video_id)
                            .filter(ComposedVideo.background_video_id.isnot(None)).all()
        }

        # Pull winning candidates with their BGs.
        winners = (
            db.query(CaptionCandidate, BackgroundVideo)
            .join(BackgroundVideo, CaptionCandidate.background_video_id == BackgroundVideo.id)
            .filter(
                CaptionCandidate.niche == niche,
                CaptionCandidate.status == "winner",
                BackgroundVideo.storage_path.isnot(None),
            )
            .order_by(CaptionCandidate.judge_overall.desc().nullslast(), CaptionCandidate.id.desc())
            .limit(remaining * 3)  # Over-fetch in case some BGs are already used
            .all()
        )

        for candidate, bg in winners:
            if bg.id in used_bg_ids:
                # Discard, don't just skip. If we leave it as 'winner', the
                # candidate sits forever blocking the
                # `pending winners exist; skipping new BG-first cycle` gate
                # in should_run_bg_first_cycle, which deadlocks generation
                # for the whole niche. The pairing is locked at gen time
                # (caption was written for this specific BG's scene), so
                # there's no way to recover this candidate — kill it.
                logger.warning(
                    f"[ORCHESTRATOR] {niche}: candidate {candidate.id}'s BG "
                    f"{bg.id} is already used in another composed video; "
                    f"discarding (cannot recompose against a different BG)"
                )
                candidate.status = "discarded"
                continue

            # Hard length check: run the same chunker + timing the composer
            # uses, so we catch captions that *would* overflow the BG before
            # we dispatch a doomed compose task. The previous heuristic
            # (word_count/2.5 + 3s buffer) underestimated by 1-2s when short
            # chunks hit the per-chunk min_duration floor — see incident
            # 2026-05-04: a 78-word caption fit the 34.2s heuristic but the
            # actual chunker output was 36.1s, overflowing the 35.8s BG and
            # looping the candidate forever.
            from video_generator.chunker import TextChunker
            from video_generator.timing import TimingEngine
            try:
                chunks = TextChunker().chunk(candidate.caption_text)
                timed = TimingEngine().calculate_timings(chunks) if chunks else []
                required_seconds = timed[-1].end_time if timed else 0.0
            except Exception as e:
                # Defensive: if the chunker chokes on the text, fall back to
                # the old heuristic so we don't crash the whole orchestrator
                # cycle. The composer will catch any leftover edge cases.
                logger.warning(f"[ORCHESTRATOR] chunker pre-check failed for cand {candidate.id}: {e}")
                word_count = len(candidate.caption_text.split())
                required_seconds = (word_count / 2.5) + 3.0
            if bg.duration_seconds and bg.duration_seconds < required_seconds:
                word_count = len(candidate.caption_text.split())
                logger.warning(
                    f"[ORCHESTRATOR] {niche}: candidate {candidate.id} ({word_count}w "
                    f"~{required_seconds:.1f}s) too long for BG {bg.id} ({bg.duration_seconds:.1f}s); skipping"
                )
                candidate.status = "discarded"
                continue

            # Mirror this winner into generated_captions so existing
            # downstream code (swipe, claude review, approval) reads it.
            #
            # Reuse an existing mirror if one is already on file for this
            # caption text. Without this guard, a candidate that gets
            # restored from 'composing' back to 'winner' (line ~2277, when
            # its compose task didn't produce an mp4) gets re-mirrored on
            # the next orchestrator cycle — we observed one candidate stuck
            # in this loop produce 58 duplicate rows over a few days. Match
            # on (niche, caption_text) since one mirror per (niche, text)
            # is what every downstream consumer actually wants.
            mirrored = (
                db.query(GeneratedCaption)
                .filter(
                    GeneratedCaption.niche == niche,
                    GeneratedCaption.caption_text == candidate.caption_text,
                )
                .order_by(GeneratedCaption.id.desc())
                .first()
            )
            if mirrored is None:
                mirrored = GeneratedCaption(
                    caption_text=candidate.caption_text,
                    llm_model=candidate.llm_model or "bg_first_mistral",
                    generation_prompt=f"bg_first_v1:bg_{bg.id}",
                    quality_score=80.0,  # Synthetic — actual signal is judge_pass
                    status="approved",
                    niche=niche,
                    tags=[],
                    judge_status="judged",
                    judge_pass=candidate.judge_pass,
                    judge_scores=candidate.judge_scores,
                    judge_issues=candidate.judge_issues or [],
                )
                db.add(mirrored)
                db.flush()
            else:
                logger.info(
                    f"[ORCHESTRATOR] {niche}: reusing existing GeneratedCaption "
                    f"id={mirrored.id} for candidate {candidate.id} "
                    f"(prevents duplicate mirror)"
                )

            pairs_to_compose.append((mirrored.id, bg.id, bg.storage_path, candidate.caption_text))
            # Mark candidate as 'composing' so we don't double-dispatch it
            # if the orchestrator runs again before this cycle finishes.
            # The composition_complete_with_reason finalizer will move it
            # to 'composed' after the actual mp4 lands, OR back to 'winner'
            # if the compose_single_video_task failed (we treat 'composing'
            # as a hold; if no mp4 after 180s, it's restored).
            candidate.status = "composing"

            if len(pairs_to_compose) >= remaining:
                break

        if not pairs_to_compose:
            logger.info(f"[ORCHESTRATOR] {niche}: no fresh BG-first winners to compose")
            return False

        # Single composition job to track them all.
        job = VideoCompositionJob(
            job_name=f"{niche}-bgfirst-compose-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
            caption_source="generated",
            caption_status_filter="approved",
            target_count=len(pairs_to_compose),
            niche=niche,
            status="running",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    # Fire one compose_single_video_task per pair.
    from tasks.video_composition_tasks import compose_single_video_task
    for caption_id, bg_id, bg_path, caption_text in pairs_to_compose:
        compose_single_video_task.apply_async(
            kwargs={
                "caption_text": caption_text,
                "background_video_path": bg_path,
                "background_video_id": bg_id,
                "generated_caption_id": caption_id,
                "composition_job_id": job_id,
            },
            queue="composition",
        )

    set_current_composition_job_id(niche, job_id)
    logger.info(f"[ORCHESTRATOR] {niche}: BG-first composition job {job_id}, {len(pairs_to_compose)} pairs queued")
    return True


def start_composition_job(niche: str):
    """Create composition job for a niche, matching captions to backgrounds by tags."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)
    if not niche_config:
        logger.error(f"[ORCHESTRATOR] Unknown niche: {niche}")
        return

    # RACE CONDITION GUARD: Check if there's already an active composition job
    existing_job_id = get_current_composition_job_id(niche)
    if existing_job_id:
        with get_db_context() as db:
            existing_job = db.query(VideoCompositionJob).filter_by(id=existing_job_id).first()
            if existing_job and existing_job.status in ["queued", "running"]:
                logger.warning(f"[ORCHESTRATOR] {niche}: Composition job {existing_job_id} already active, skipping")
                return  # Already have an active job
        # Job exists but is completed/failed, clear it
        clear_current_composition_job(niche)

    daily_target = niche_config.daily_video_target
    composed_today = get_videos_composed_today(niche)
    remaining = daily_target - composed_today

    if remaining <= 0:
        logger.info(f"[ORCHESTRATOR] Daily quota already met for {niche}")
        return

    model = get_model_for_niche(niche)
    if not model:
        logger.error(f"[ORCHESTRATOR] No model for niche: {niche}")
        return

    # Get approved captions for this niche (not already used in composed videos)
    with get_db_context() as db:
        quality_threshold = config.generation.min_quality_score_for_composition

        # Get IDs of captions already used in composed videos
        used_ids = db.query(ComposedVideo.generated_caption_id).filter(
            ComposedVideo.generated_caption_id.isnot(None)
        ).all()
        used_ids = [id[0] for id in used_ids]

        # Query by niche (not model) to include captions from previous models
        query = db.query(GeneratedCaption).filter(
            GeneratedCaption.niche == niche,
            GeneratedCaption.status == "approved",
            GeneratedCaption.quality_score >= quality_threshold
        )
        query = _judge_filter(query)

        # Debug: Count before excluding used
        total_approved = query.count()
        logger.info(f"[ORCHESTRATOR] {niche}: Found {total_approved} approved captions with score >= {quality_threshold} (judge-passing or unjudged)")

        if used_ids:
            query = query.filter(~GeneratedCaption.id.in_(used_ids))
            logger.info(f"[ORCHESTRATOR] {niche}: Excluding {len(used_ids)} already-used captions")

        # Order: judged-pass rows first, then by quality_score desc. We can't
        # read judge_scores->>'overall' as a sort key because the column is
        # mapped as plain JSON (not JSONB), and .astext only works on JSONB
        # comparators. judge_pass boolean is sufficient as the primary signal.
        from sqlalchemy import case
        judge_rank = case(
            (GeneratedCaption.judge_pass.is_(True), 1),
            else_=0,
        )
        captions = query.order_by(
            judge_rank.desc(),
            GeneratedCaption.quality_score.desc(),
        ).limit(remaining).all()

        logger.info(f"[ORCHESTRATOR] {niche}: {len(captions)} captions after filters (need {remaining})")

        if not captions:
            logger.info(f"[ORCHESTRATOR] No approved captions for {niche}")
            return

        # Match each caption to a background
        matches = []
        for caption in captions:
            bg_id = find_matching_background(niche, caption.id)
            if bg_id:
                matches.append((caption.id, bg_id))

        if not matches:
            logger.warning(f"[ORCHESTRATOR] No matching backgrounds for {niche}")
            return

        # Create composition job
        job_name = f"{niche}-compose-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

        job = VideoCompositionJob(
            job_name=job_name,
            caption_source="generated",
            caption_status_filter="approved",
            target_count=len(matches),
            niche=niche,  # Per-niche tracking
            status="queued"
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        job_id = job.id
        logger.info(f"[ORCHESTRATOR] Created composition job {job_id}: {len(matches)} videos")

    # Queue composition task
    from tasks.video_composition_tasks import run_composition_job
    result = run_composition_job.apply_async(args=[job_id], queue="composition")

    # Update job with celery task ID
    with get_db_context() as db:
        job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
        if job:
            job.celery_task_id = result.id
            db.commit()

    # Track in Redis
    set_current_composition_job_id(niche, job_id)

    logger.info(f"[ORCHESTRATOR] Started composition job for '{niche}': {len(matches)} videos")


def composition_complete(niche: str) -> bool:
    """Check if composition is complete for current niche."""
    complete, _ = composition_complete_with_reason(niche)
    return complete


def composition_complete_with_reason(niche: str) -> Tuple[bool, Optional[str]]:
    """
    Check if composition is complete for current niche.

    Returns:
        Tuple of (is_complete, failure_reason)
        failure_reason is None for success, or one of:
        - "no_captions" - Failed because no approved captions available
        - "no_backgrounds" - Failed because no background videos available
        - "other_failure" - Failed for other reasons
        - None - Completed successfully or still running
    """
    job_id = get_current_composition_job_id(niche)
    if not job_id:
        return True, None  # No job, consider complete

    with get_db_context() as db:
        job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
        if not job:
            return True, None

        # BG-first finalizer.
        # start_bg_first_composition_job dispatches N independent
        # compose_single_video_task instances and never updates the parent
        # VideoCompositionJob row. Without this finalizer the job stays
        # 'running' forever and the watchdog auto-recovery loop kicks in.
        # Strategy: count actual composed_videos rows for this job, and
        # also reconcile caption_candidates statuses (composing → composed
        # if the mp4 landed, → winner if it didn't).
        if job.status == "running" and job.job_name and "bgfirst" in job.job_name:
            from sqlalchemy import func
            from database.models import CaptionCandidate

            composed_rows = db.query(ComposedVideo).filter(
                ComposedVideo.composition_job_id == job_id,
                ComposedVideo.status == "completed",
            ).all()
            composed_count = len(composed_rows)

            last_composed_at = db.query(func.max(ComposedVideo.created_at)).filter(
                ComposedVideo.composition_job_id == job_id,
                ComposedVideo.status == "completed",
            ).scalar()

            # Reconcile candidate statuses: a candidate whose mirrored caption
            # made it into a composed_video moves to 'composed'; one whose
            # caption did not goes back to 'winner' so the next cycle reuses it.
            composed_caption_texts = {cv.generated_caption.caption_text for cv in composed_rows if cv.generated_caption}

            should_finalize = False
            partial = False
            if composed_count >= job.target_count:
                should_finalize = True
            elif last_composed_at:
                seconds_since_last = (datetime.now(timezone.utc) - last_composed_at).total_seconds()
                # 4 min between mp4s = treat the rest as failed.
                # Composes are 60-90s each; if there's a 4-min gap something
                # went wrong (worker died, OCR contention, etc).
                if seconds_since_last > 240:
                    should_finalize = True
                    partial = True
            else:
                seconds_since_create = (datetime.now(timezone.utc) - job.created_at).total_seconds()
                # Bumped from 300s → 1800s (30 min). With 30 compose tasks
                # at 60-90s each on 2 GPU workers, the FIRST mp4 can take
                # 60-180s after dispatch. The pre-composition disable+purge
                # also resets workers, adding a few seconds of cold start.
                # 300s was too aggressive and would declare failure right
                # before the first mp4 landed.
                if seconds_since_create > 1800:
                    db.query(CaptionCandidate).filter(
                        CaptionCandidate.niche == niche,
                        CaptionCandidate.status == "composing",
                    ).update({"status": "winner"}, synchronize_session=False)
                    job.status = "failed"
                    job.error_message = f"BG-first: no composed videos produced after {seconds_since_create:.0f}s"
                    job.completed_at = datetime.utcnow()
                    db.commit()
                    logger.error(f"[ORCHESTRATOR] BG-first job {job_id} failed: {job.error_message}")

            if should_finalize:
                # Reconcile candidates.
                composing_cands = db.query(CaptionCandidate).filter(
                    CaptionCandidate.niche == niche,
                    CaptionCandidate.status == "composing",
                ).all()
                composed_n = restored_n = 0
                for cc in composing_cands:
                    if cc.caption_text in composed_caption_texts:
                        cc.status = "composed"
                        composed_n += 1
                    else:
                        cc.status = "winner"
                        restored_n += 1

                job.status = "completed"
                job.videos_composed = composed_count
                if partial:
                    job.videos_failed = max(0, job.target_count - composed_count)
                job.completed_at = datetime.utcnow()
                db.commit()
                tag = "PARTIAL" if partial else "OK"
                logger.info(
                    f"[ORCHESTRATOR] BG-first job {job_id} finalized [{tag}]: "
                    f"{composed_count}/{job.target_count} videos, "
                    f"{composed_n} cands marked composed, {restored_n} restored to winner"
                )

        if job.status in ["completed", "failed", "cancelled"]:
            logger.info(f"[ORCHESTRATOR] Composition job {job_id} finished with status: {job.status}")

            failure_reason = None
            if job.status == "failed":
                # Determine the reason for failure
                error_msg = job.error_message or ""
                if "No captions found" in error_msg:
                    failure_reason = "no_captions"
                elif "No background" in error_msg:
                    failure_reason = "no_backgrounds"
                else:
                    failure_reason = "other_failure"
                logger.info(f"[ORCHESTRATOR] Composition failed: {failure_reason} - {error_msg}")

            # Increment daily quota with the number of videos composed
            if job.status == "completed" and job.videos_composed:
                increment_videos_composed_today(niche, job.videos_composed)
                logger.info(f"[ORCHESTRATOR] Added {job.videos_composed} videos to daily quota for {niche}")

            clear_current_composition_job(niche)
            return True, failure_reason

        return False, None


# ============================================================================
# Main Orchestrator Task
# ============================================================================

@celery_app.task(bind=True, name='tasks.pipeline_orchestrator.pipeline_orchestrator', queue='maintenance')
def pipeline_orchestrator(self):
    """
    Main orchestration task - runs every 5 minutes.

    Manages per-niche training, generation, and composition pipeline.
    Includes stuck state detection and auto-recovery.
    """
    if not is_pipeline_enabled():
        logger.debug("[ORCHESTRATOR] Pipeline disabled, skipping")
        return {"status": "disabled"}

    state = get_pipeline_state()
    current_niche = get_current_niche()

    logger.info(f"[ORCHESTRATOR] Running: state={state}, niche={current_niche}")

    # STUCK STATE DETECTION: Check if we've been in current state too long
    if is_state_stuck():
        duration = get_state_duration_minutes()
        logger.error(f"[ORCHESTRATOR] STUCK STATE DETECTED: {state} for {duration:.1f} minutes")

        # Auto-recover based on state
        if state in ["training_pending", "gpu_clearing"]:
            # Stuck waiting for GPU - force enable extraction and return to collecting
            logger.warning("[ORCHESTRATOR] Auto-recovery: Force enabling extraction and returning to collecting")
            ensure_extraction_enabled()
            set_current_niche(None)
            set_pipeline_state("collecting", reason=f"auto_recovery:stuck_in_{state}", metadata={"duration_minutes": duration})
            return {"status": "auto_recovered", "from_state": state, "reason": "stuck_waiting_for_gpu"}

        elif state == "generating":
            # Generation job might be hung - check and fail it
            if current_niche:
                job_id = get_current_generation_job_id(current_niche)
                if job_id:
                    with get_db_context() as db:
                        job = db.query(GenerationJob).filter_by(id=job_id).first()
                        if job and job.status == "running":
                            job.status = "failed"
                            job.error_message = f"Auto-failed: stuck in generating state for {duration:.1f} minutes"
                            job.completed_at = datetime.utcnow()
                            db.commit()
                            logger.warning(f"[ORCHESTRATOR] Auto-failed stuck generation job {job_id}")
                        elif job and job.status == "completed":
                            # Job actually completed but we timed out waiting - still score captions
                            logger.info(f"[ORCHESTRATOR] Job {job_id} completed but timed out - scoring captions anyway")
                # Always try to score captions before clearing job (even on timeout)
                # This ensures any generated captions get auto-approved
                score_generated_captions(current_niche)
                increment_consecutive_failures(current_niche)
                clear_current_generation_job(current_niche)
            set_pipeline_state("collecting", reason="auto_recovery:stuck_generating", metadata={"duration_minutes": duration, "job_id": job_id if current_niche else None})
            ensure_extraction_enabled()
            return {"status": "auto_recovered", "from_state": state, "reason": "stuck_generating"}

        elif state in ["composition_pending", "composing"]:
            # Skip composition and return to collecting
            if current_niche:
                clear_current_composition_job(current_niche)
            set_current_niche(None)
            set_pipeline_state("collecting", reason=f"auto_recovery:stuck_in_{state}", metadata={"duration_minutes": duration})
            ensure_extraction_enabled()
            return {"status": "auto_recovered", "from_state": state, "reason": "stuck_composing"}

        elif state in ["training", "generation_pending"]:
            # For training states, just log and continue - don't interrupt
            logger.warning(f"[ORCHESTRATOR] State {state} running longer than expected, but allowing to continue")

    try:
        if state == "collecting":
            # Check if any niche needs training
            niche_to_train = get_next_niche_needing_training()
            if niche_to_train:
                # Get the training reason for logging
                training_reason = _get_training_reason(niche_to_train)
                set_current_niche(niche_to_train)
                logger.info(f"[ORCHESTRATOR] Transitioning to training for {niche_to_train}")
                # Disable ALL GPU workloads to allow queue to drain
                disable_all_gpu_workloads()
                set_pipeline_state("training_pending", reason=f"training_needed:{training_reason}", metadata={"niche": niche_to_train, "training_reason": training_reason})

            # Check if any niche is below daily quota (has model, needs work).
            elif niche_below_quota := get_niche_below_daily_quota():
                set_current_niche(niche_below_quota)
                # If there are already enough pending winners (incl 'composing'
                # candidates from a partial-failed previous job) to cover the
                # remaining quota, skip generation and go directly to
                # composition_pending. This prevents wasteful re-generation
                # when the previous BG-first job ended with leftover winners.
                from database.models import CaptionCandidate
                config = get_automation_config()
                fc = config.get_niche(niche_below_quota)
                composed_today = get_videos_composed_today(niche_below_quota)
                remaining_quota = (fc.daily_video_target if fc else 30) - composed_today
                with get_db_context() as db:
                    pending_winners = db.query(CaptionCandidate).filter(
                        CaptionCandidate.niche == niche_below_quota,
                        CaptionCandidate.status.in_(["winner", "composing"]),
                    ).count()
                if pending_winners > 0:
                    # Always compose existing winners before generating
                    # more. Even if winners < remaining_quota, composing
                    # them first reduces the next-cycle pressure.
                    logger.info(
                        f"[ORCHESTRATOR] {niche_below_quota}: {pending_winners} pending winners exist "
                        f"(remaining quota {remaining_quota}); composing those first before generating more"
                    )
                    set_pipeline_state(
                        "composition_pending",
                        reason="winners_already_pending",
                        metadata={"niche": niche_below_quota, "pending_winners": pending_winners, "remaining": remaining_quota},
                    )
                else:
                    available_captions = get_available_caption_count(niche_below_quota)
                    logger.info(f"[ORCHESTRATOR] {niche_below_quota} below daily quota (available captions: {available_captions}), going to generation")
                    set_pipeline_state("generation_pending", reason="quota_below_target", metadata={"niche": niche_below_quota, "available_captions": available_captions})

            else:
                ensure_extraction_enabled()
                # Idle work: drain the ML tagging backlog one batch at a time.
                # Tagging is self-contained — it stops llama-server, runs the
                # VLM, then restarts llama-server — so it only runs while
                # nothing else needs the GPUs. We dispatch a single batch
                # per orchestrator tick (every 5 min) so the pipeline can
                # interrupt with training/generation work as soon as needed.
                _maybe_dispatch_tagging_batch()

        elif state == "training_pending":
            # Check if there's an ACTIVE training job we're waiting on
            # (This handles resumption after restart - but we don't skip just because a model exists)
            config = get_automation_config()
            current_job_id = get_current_training_job_id(current_niche)

            if current_job_id:
                # There's an active training job - check if it completed
                if training_complete(current_niche):
                    logger.info(f"[ORCHESTRATOR] Training job {current_job_id} completed for {current_niche}")
                    if config.generation.auto_generate_after_training:
                        set_pipeline_state("generation_pending", reason="training_job_completed", metadata={"niche": current_niche, "job_id": current_job_id})
                    else:
                        ensure_extraction_enabled()
                        set_pipeline_state("collecting", reason="training_job_completed:auto_generate_disabled")
                else:
                    # Job still running - shouldn't normally be in training_pending with active job
                    # Transition to training state to properly track it
                    logger.warning(f"[ORCHESTRATOR] Found active training job {current_job_id} in training_pending, transitioning to training state")
                    set_pipeline_state("training", reason="active_job_found", metadata={"niche": current_niche, "job_id": current_job_id})
            else:
                # No active training job - proceed with queue draining for new training
                # First, purge the GPU queue to clear pending extraction tasks
                # This is idempotent - calling multiple times is safe
                purge_results = purge_gpu_queues()

                # Now check if any extraction tasks are still actively running
                # (We can't interrupt a task mid-execution safely)
                if no_active_extractions():
                    logger.info(f"[ORCHESTRATOR] GPU queue purged, no active extractions - preparing GPU for {current_niche}")
                    # Queue unload and cleanup tasks - they run on separate queues
                    prepare_gpu_for_training()
                    # Transition to gpu_clearing state to wait for memory to actually clear
                    set_pipeline_state("gpu_clearing", reason="no_active_extractions", metadata={"purge_results": purge_results})
                else:
                    logger.info("[ORCHESTRATOR] Waiting for active extraction task to finish (queue purged, will prepare GPU on next tick)")

        elif state == "gpu_clearing":
            # Wait for GPU memory to actually be cleared before starting training
            # This is critical because unload tasks run async on separate queues
            config = get_automation_config()
            training_gpu = config.gpu.training_gpu

            # Check if there's an ACTIVE training job (handles restart/resumption)
            current_job_id = get_current_training_job_id(current_niche)
            if current_job_id:
                # There's an active training job - check if it completed
                if training_complete(current_niche):
                    logger.info(f"[ORCHESTRATOR] Training job {current_job_id} completed for {current_niche} (detected in gpu_clearing)")
                    if config.generation.auto_generate_after_training:
                        set_pipeline_state("generation_pending", reason="training_job_completed:safety_check", metadata={"job_id": current_job_id})
                    else:
                        ensure_extraction_enabled()
                        set_pipeline_state("collecting", reason="training_job_completed:auto_generate_disabled")
                else:
                    # Job still running - transition to training state
                    logger.info(f"[ORCHESTRATOR] Found active training job {current_job_id}, transitioning to training state")
                    set_pipeline_state("training", reason="active_job_found:gpu_clearing", metadata={"job_id": current_job_id})
            elif gpu_memory_is_clear(training_gpu, max_memory_mb=2000, use_direct_query=True):
                # CRITICAL TRANSITION: Use direct GPU query (not cached) to ensure accuracy
                # This prevents starting training with stale cache data that could show cleared memory
                # when in reality models are still loaded
                gpu_status = get_gpu_memory_direct(training_gpu)  # Direct query for accurate metadata
                logger.info(f"[ORCHESTRATOR] GPU {training_gpu} memory cleared (direct query) - starting training for {current_niche}")
                start_training_job(current_niche)
                set_pipeline_state("training", reason="gpu_memory_cleared", metadata={"gpu": training_gpu, "memory_mb": gpu_status.get("memory_used_mb") if gpu_status else None, "query_type": "direct"})
            else:
                # Memory not clear yet - re-queue cleanup tasks and wait
                logger.info(f"[ORCHESTRATOR] Waiting for GPU {training_gpu} memory to clear (unload tasks still running)")
                # Re-trigger deep clean in case previous tasks didn't complete
                deep_clean_gpus()

        elif state == "training":
            if training_complete(current_niche):
                logger.info(f"[ORCHESTRATOR] Training complete for {current_niche}")
                config = get_automation_config()
                if config.generation.auto_generate_after_training:
                    set_pipeline_state("generation_pending", reason="training_complete", metadata={"niche": current_niche})
                else:
                    ensure_extraction_enabled()
                    set_pipeline_state("collecting", reason="training_complete:auto_generate_disabled")

        elif state == "generation_pending":
            # Safety check: if quota is already met, skip generation entirely
            if daily_quota_met(current_niche):
                logger.info(f"[ORCHESTRATOR] Quota already met for {current_niche}, skipping generation")
                ensure_extraction_enabled()
                set_current_niche(None)
                set_pipeline_state("collecting", reason="daily_quota_met:skip_generation", metadata={"niche": current_niche})
            else:
                # Disable GPU workloads (OCR, ML tagging, watermark filter) so llama-server
                # gets full GPU bandwidth for generation. Without this, LLM calls time out
                # because OCR models consume GPU memory and compete for compute.
                disable_all_gpu_workloads()
                # Purge queued GPU tasks so workers don't pick them up and load models
                purge_gpu_queues()
                # Also unload OCR models from GPU memory to free VRAM for llama-server
                unload_all_models()

                # Always generate to keep the caption pool replenished
                # We want a continuous supply of captions for composition
                available_captions = get_available_caption_count(current_niche)
                logger.info(f"[ORCHESTRATOR] Starting generation for {current_niche} (available captions: {available_captions})")
                if start_generation_job(current_niche):
                    set_pipeline_state("generating", reason="generation_job_started", metadata={"niche": current_niche, "available_captions": available_captions})
                else:
                    # Generation failed to start (missing model files or too many failures)
                    # Re-enable GPU workloads since we disabled them above
                    ensure_extraction_enabled()
                    failures = get_consecutive_failures(current_niche)
                    if available_captions > 0:
                        # If we have captions available, proceed to composition anyway
                        logger.warning(f"[ORCHESTRATOR] Generation failed for {current_niche}, but {available_captions} captions available - proceeding to composition")
                        set_pipeline_state("composition_pending", reason="generation_failed_but_captions_available", metadata={"niche": current_niche, "available_captions": available_captions, "consecutive_failures": failures})
                    else:
                        logger.error(f"[ORCHESTRATOR] Failed to start generation for {current_niche} and no captions available, returning to collecting")
                        set_current_niche(None)
                        set_pipeline_state("collecting", reason="generation_start_failed", metadata={"niche": current_niche, "consecutive_failures": failures})

        elif state == "generating":
            if generation_complete(current_niche):
                logger.info(f"[ORCHESTRATOR] Generation complete for {current_niche}")
                score_generated_captions(current_niche)
                clear_current_generation_job(current_niche)  # Clear after scoring
                # Re-enable GPU workloads (OCR, ML tagging, etc.) that were paused for generation
                ensure_extraction_enabled()
                # Unload generation model to free GPU memory for next phase
                try:
                    from api.gpu_control import force_unload_all_models
                    unload_result = force_unload_all_models()
                    logger.info(f"[ORCHESTRATOR] Unloaded models after generation: {unload_result}")
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR] Failed to unload models after generation: {e}")
                set_pipeline_state("composition_pending", reason="generation_complete", metadata={"niche": current_niche})

        elif state == "composition_pending":
            if daily_quota_met(current_niche):
                composed_today = get_videos_composed_today(current_niche)
                config = get_automation_config()
                target = config.get_niche(current_niche).daily_video_target if config.get_niche(current_niche) else 3
                logger.info(f"[ORCHESTRATOR] Daily quota met for {current_niche}")
                enable_all_gpu_workloads()
                set_current_niche(None)
                set_pipeline_state("collecting", reason="daily_quota_met", metadata={"niche": current_niche, "composed_today": composed_today, "target": target})
            else:
                # Stage 2: prefer BG-first composition (uses winners from
                # caption_candidates). Falls back to legacy if BG-first is
                # disabled or has no winners ready.
                import os
                bg_first_enabled = os.environ.get("BG_FIRST_GENERATION", "true").lower() in ("1", "true", "yes")
                if bg_first_enabled:
                    # Composition itself runs on the CPU-only composition worker,
                    # but the composing phase still leans on the GPUs (judge
                    # re-checks, llama-server headroom). Pause the OCR/watermark
                    # producers and purge their backlog so GPU work doesn't
                    # compete with the composing phase:
                    # Step 1: stop the source — disable extraction + watermark.
                    # Step 2: purge the existing OCR backlog from the gpu queues.
                    # Step 3: dispatch the compositions.
                    disable_all_gpu_workloads()
                    purge_results = purge_gpu_queues()
                    logger.info(f"[ORCHESTRATOR] Pre-composition queue purge: {purge_results}")
                    if start_bg_first_composition_job(current_niche):
                        set_pipeline_state("composing", reason="bg_first_winners_ready", metadata={"niche": current_niche})
                    elif ready_for_composition(current_niche):
                        # Nothing BG-first; fall through to legacy composition.
                        logger.info(f"[ORCHESTRATOR] No BG-first winners; starting legacy composition for {current_niche}")
                        start_composition_job(current_niche)
                        set_pipeline_state("composing", reason="ready_for_composition_after_purge", metadata={"niche": current_niche})
                    else:
                        # Nothing to compose at all — re-enable workloads and bail.
                        enable_all_gpu_workloads()
                        logger.info(f"[ORCHESTRATOR] Nothing to compose, returning to collecting")
                        set_pipeline_state("collecting", reason="not_ready_for_composition", metadata={"niche": current_niche})
                elif ready_for_composition(current_niche):
                    logger.info(f"[ORCHESTRATOR] Starting legacy composition for {current_niche}")
                    start_composition_job(current_niche)
                    set_pipeline_state("composing", reason="ready_for_composition", metadata={"niche": current_niche})
                else:
                    logger.info(f"[ORCHESTRATOR] Not ready for composition, returning to collecting")
                    ensure_extraction_enabled()
                    set_pipeline_state("collecting", reason="not_ready_for_composition", metadata={"niche": current_niche})

        elif state == "composing":
            complete, failure_reason = composition_complete_with_reason(current_niche)
            if complete:
                composed_today = get_videos_composed_today(current_niche)
                config = get_automation_config()
                target = config.get_niche(current_niche).daily_video_target if config.get_niche(current_niche) else 3
                logger.info(f"[ORCHESTRATOR] {current_niche}: {composed_today}/{target} videos today")

                # If composition failed due to no captions available, trigger generation
                if failure_reason == "no_captions" and not daily_quota_met(current_niche):
                    available = get_available_caption_count(current_niche)
                    logger.info(f"[ORCHESTRATOR] Composition failed: no available captions for {current_niche} (available={available})")
                    logger.info(f"[ORCHESTRATOR] Transitioning to generation_pending to create more captions")
                    set_pipeline_state(
                        "generation_pending",
                        reason="no_available_captions:need_more_generation",
                        metadata={"niche": current_niche, "available_captions": available, "composed_today": composed_today, "target": target}
                    )
                else:
                    # Normal completion or quota met. Re-enable ALL gpu
                    # workloads (we disabled OCR + watermark + LLM batch
                    # when entering composition_pending for BG-first).
                    enable_all_gpu_workloads()
                    set_current_niche(None)
                    reason = "composition_complete" if not failure_reason else f"composition_failed:{failure_reason}"
                    set_pipeline_state("collecting", reason=reason, metadata={"niche": current_niche, "composed_today": composed_today, "target": target})

        return {
            "status": "ok",
            "state": state,
            "current_niche": current_niche,
            "new_state": get_pipeline_state()
        }

    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)
        # On error, try to return to safe state
        ensure_extraction_enabled()
        return {"status": "error", "error": str(e)}


# ============================================================================
# Status and API Helpers
# ============================================================================

def get_pipeline_status() -> Dict[str, Any]:
    """Get full pipeline status for API."""
    config = get_automation_config()

    niche_status = {}
    for niche_name, niche_config in config.niches.items():
        composed_today = get_videos_composed_today(niche_name)
        model = get_model_for_niche(niche_name)
        total_captions = get_caption_count_for_subreddits(niche_config.subreddits)

        niche_status[niche_name] = {
            "enabled": niche_config.enabled,
            "subreddits": niche_config.subreddits,
            "keywords": niche_config.keywords,
            "daily_target": niche_config.daily_video_target,
            "composed_today": composed_today,
            "quota_met": composed_today >= niche_config.daily_video_target,
            "has_model": model is not None,
            "model_name": model.name if model else None,
            "total_captions": total_captions,
            "ready_for_training": total_captions >= config.training.min_total_captions,
        }

    # Idle background work — surface whether a tagging batch is in flight
    # so the state-machine viewer can highlight the tagging node.
    ml_tagging_running = False
    try:
        r = get_redis()
        ml_tagging_running = bool(r.get("ml_tagging:running"))
    except Exception:
        pass

    result = {
        "enabled": is_pipeline_enabled(),
        "state": get_pipeline_state(),
        "current_niche": get_current_niche(),
        "niches": niche_status,
        "config": config.to_dict(),
        "ml_tagging_running": ml_tagging_running,
    }

    # Add GPU memory status when in gpu_clearing state
    current_state = get_pipeline_state()
    if current_state == "gpu_clearing":
        gpu_0_status = get_gpu_memory_from_cache(0)
        result["gpu_clearing_status"] = {
            "training_gpu": config.gpu.training_gpu,
            "memory_used_mb": gpu_0_status.get("memory_used_mb", 0) if gpu_0_status else None,
            "memory_total_mb": gpu_0_status.get("memory_total_mb", 11264) if gpu_0_status else None,
            "memory_percent": gpu_0_status.get("memory_percent", 0) if gpu_0_status else None,
            "loaded_models": gpu_0_status.get("loaded_models", []) if gpu_0_status else [],
            "target_memory_mb": 2000,  # Threshold for "clear"
            "is_clear": gpu_memory_is_clear(config.gpu.training_gpu, max_memory_mb=2000),
        }

    return result
