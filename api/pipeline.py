"""
Pipeline API - Control and monitor the automation pipeline

Provides endpoints for:
- Pipeline status and control (enable/disable)
- Per-niche status and manual triggers
- Daily quota tracking
- Configuration viewing/updating
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from tasks.pipeline_orchestrator import (
    get_pipeline_status,
    get_pipeline_state,
    set_pipeline_state,
    is_pipeline_enabled,
    set_pipeline_enabled,
    get_current_niche,
    set_current_niche,
    get_videos_composed_today,
    daily_quota_met,
    get_model_for_niche,
    get_caption_count_for_subreddits,
    get_next_niche_needing_training,
    start_training_job,
    start_generation_job,
    start_composition_job,
    ensure_extraction_enabled,
    get_current_training_job_id,
    get_current_generation_job_id,
    get_current_composition_job_id,
    get_consecutive_failures,
    reset_consecutive_failures,
    score_generated_captions,
)
from config.automation_config import get_automation_config, reload_config

router = APIRouter(prefix="/pipeline", tags=["Pipeline Automation"])


# ============================================================================
# Response Models
# ============================================================================

class PipelineStatusResponse(BaseModel):
    enabled: bool
    state: str
    current_niche: Optional[str]
    niches: Dict[str, Any]
    config: Dict[str, Any]
    ml_tagging_running: Optional[bool] = False


class NicheStatusResponse(BaseModel):
    name: str
    enabled: bool
    subreddits: List[str]
    tags: List[str]
    daily_target: int
    composed_today: int
    quota_met: bool
    has_model: bool
    model_name: Optional[str]
    total_captions: int
    ready_for_training: bool


class QuotaStatusResponse(BaseModel):
    date: str
    niches: Dict[str, Dict[str, Any]]
    total_target: int
    total_composed: int
    all_quotas_met: bool


# ============================================================================
# Pipeline Control
# ============================================================================

@router.get("/status", response_model=PipelineStatusResponse)
async def get_status():
    """
    Get full pipeline status including state, current niche, and per-niche metrics.
    """
    return get_pipeline_status()


@router.get("/state")
async def get_state():
    """Get just the current pipeline state."""
    return {
        "state": get_pipeline_state(),
        "current_niche": get_current_niche(),
        "enabled": is_pipeline_enabled()
    }


@router.post("/enable")
async def enable_pipeline():
    """
    Enable the pipeline orchestrator and restore extraction.

    This will:
    1. Enable the pipeline orchestrator
    2. Re-enable extraction processing on both GPUs
    """
    import redis
    from config.settings import settings

    set_pipeline_enabled(True)

    # Re-enable extraction
    try:
        r = redis.from_url(settings.celery_broker_url)
        r.set("extraction:enabled", "true")
        r.set("ocr:gpu0:enabled", "true")
        r.set("ocr:gpu1:enabled", "true")
        r.set("llm:batch:enabled", "true")
    except Exception as e:
        pass  # Non-critical

    return {"status": "enabled", "message": "Pipeline enabled, extraction restored"}


@router.post("/disable")
async def disable_pipeline():
    """
    Disable the pipeline orchestrator and clean up GPU resources.

    This will:
    1. Disable the pipeline orchestrator
    2. Disable extraction processing
    3. Unload all GPU models to free memory
    4. Reset pipeline state to 'collecting'
    """
    import redis
    from config.settings import settings

    set_pipeline_enabled(False)

    # Also disable extraction to prevent GPU usage
    try:
        r = redis.from_url(settings.celery_broker_url)
        r.set("extraction:enabled", "false")
        r.set("ocr:gpu0:enabled", "false")
        r.set("ocr:gpu1:enabled", "false")
        r.set("llm:batch:enabled", "false")
    except Exception as e:
        pass  # Non-critical

    # Reset pipeline state
    set_pipeline_state("collecting")
    set_current_niche(None)

    # Queue model unload tasks (async, don't wait)
    try:
        from tasks.maintenance_tasks import force_unload_gpu_models
        force_unload_gpu_models.apply_async(queue="gpu")
    except Exception as e:
        pass  # Non-critical

    return {
        "status": "disabled",
        "message": "Pipeline disabled, extraction stopped, GPU cleanup queued",
        "state": "collecting"
    }


@router.post("/pause")
async def pause_pipeline():
    """
    Pause all orchestration (alias for disable).

    Immediately stops automated transitions while allowing manual operations.
    """
    set_pipeline_enabled(False)
    return {
        "status": "paused",
        "message": "Pipeline paused. Use /pipeline/resume to continue."
    }


@router.post("/resume")
async def resume_pipeline():
    """Resume orchestration (alias for enable)."""
    set_pipeline_enabled(True)
    return {"status": "resumed", "message": "Pipeline resumed"}


@router.post("/reset")
async def reset_pipeline():
    """
    Reset pipeline to collecting state.

    Use this to recover from stuck states.
    """
    set_pipeline_state("collecting")
    set_current_niche(None)
    ensure_extraction_enabled()
    return {
        "status": "reset",
        "state": "collecting",
        "message": "Pipeline reset to collecting state"
    }


# ============================================================================
# Niche Status and Control
# ============================================================================

@router.get("/niches")
async def list_niches():
    """List all niches with their current status."""
    config = get_automation_config()

    niches = []
    for name, niche_config in config.niches.items():
        composed_today = get_videos_composed_today(name)
        model = get_model_for_niche(name)
        total_captions = get_caption_count_for_subreddits(niche_config.subreddits)

        niches.append({
            "name": name,
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
        })

    return {"niches": niches}


@router.get("/niches/{niche}")
async def get_niche_status(niche: str):
    """Get detailed status for a specific niche."""
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    composed_today = get_videos_composed_today(niche)
    model = get_model_for_niche(niche)
    total_captions = get_caption_count_for_subreddits(niche_config.subreddits)

    return {
        "name": niche,
        "enabled": niche_config.enabled,
        "subreddits": niche_config.subreddits,
        "keywords": niche_config.keywords,
        "daily_target": niche_config.daily_video_target,
        "composed_today": composed_today,
        "quota_met": composed_today >= niche_config.daily_video_target,
        "has_model": model is not None,
        "model_name": model.name if model else None,
        "model_id": model.id if model else None,
        "total_captions": total_captions,
        "ready_for_training": total_captions >= config.training.min_total_captions,
        "min_captions_required": config.training.min_total_captions,
    }


# ============================================================================
# Manual Triggers
# ============================================================================

@router.post("/trigger/train/{niche}")
async def trigger_training(niche: str):
    """
    Manually trigger training for a specific niche.

    This will:
    1. Pause extraction
    2. Queue training job
    3. Set pipeline state to training
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    if not niche_config.enabled:
        raise HTTPException(status_code=400, detail=f"Niche '{niche}' is disabled")

    # Check if we have enough data
    total_captions = get_caption_count_for_subreddits(niche_config.subreddits)
    if total_captions < config.training.min_total_captions:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough captions ({total_captions}/{config.training.min_total_captions})"
        )

    # Start training
    set_current_niche(niche)
    start_training_job(niche)
    set_pipeline_state("training")

    return {
        "status": "started",
        "niche": niche,
        "message": f"Training started for {niche}"
    }


@router.post("/trigger/generate/{niche}")
async def trigger_generation(niche: str):
    """
    Manually trigger caption generation for a specific niche.

    Requires a trained model for this niche.
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    model = get_model_for_niche(niche)
    if not model:
        raise HTTPException(
            status_code=400,
            detail=f"No trained model available for {niche}. Train first."
        )

    set_current_niche(niche)
    start_generation_job(niche)
    set_pipeline_state("generating")

    return {
        "status": "started",
        "niche": niche,
        "model": model.name,
        "message": f"Generation started for {niche}"
    }


@router.post("/trigger/compose/{niche}")
async def trigger_composition(niche: str):
    """
    Manually trigger video composition for a specific niche.

    Requires approved captions for this niche.
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    set_current_niche(niche)
    start_composition_job(niche)
    set_pipeline_state("composing")

    return {
        "status": "started",
        "niche": niche,
        "message": f"Composition started for {niche}"
    }


@router.post("/trigger/check-training")
async def check_training_needs():
    """
    Check which niche (if any) needs training next.

    Returns the niche that would be selected by the orchestrator.
    """
    next_niche = get_next_niche_needing_training()

    if next_niche:
        return {
            "needs_training": True,
            "niche": next_niche,
            "message": f"'{next_niche}' needs training"
        }
    else:
        return {
            "needs_training": False,
            "niche": None,
            "message": "No niches need training currently"
        }


@router.post("/trigger/score/{niche}")
async def trigger_caption_scoring(niche: str, job_id: Optional[int] = None):
    """
    Manually trigger caption scoring/auto-approval for a niche.

    This auto-approves pending captions with quality_score >= 80.
    Useful for recovering from pipeline timeouts where captions weren't scored.

    Args:
        niche: The niche to score captions for
        job_id: Optional specific job ID. If not provided, uses most recent job.
    """
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    # Get count before scoring
    from database.db import get_db_context
    from database.models import GeneratedCaption, GenerationJob

    model = get_model_for_niche(niche)
    if not model:
        raise HTTPException(status_code=400, detail=f"No model found for niche '{niche}'")

    with get_db_context() as db:
        # Count pending before
        if job_id:
            pending_before = db.query(GeneratedCaption).filter(
                GeneratedCaption.generation_job_id == job_id,
                GeneratedCaption.status == "pending_review",
                GeneratedCaption.quality_score >= 80
            ).count()
        else:
            # Count all pending for this model
            pending_before = db.query(GeneratedCaption).join(GenerationJob).filter(
                GenerationJob.model_id == model.id,
                GeneratedCaption.status == "pending_review",
                GeneratedCaption.quality_score >= 80
            ).count()

    # Run scoring
    score_generated_captions(niche, job_id=job_id)

    with get_db_context() as db:
        # Count pending after
        if job_id:
            pending_after = db.query(GeneratedCaption).filter(
                GeneratedCaption.generation_job_id == job_id,
                GeneratedCaption.status == "pending_review",
                GeneratedCaption.quality_score >= 80
            ).count()
        else:
            pending_after = db.query(GeneratedCaption).join(GenerationJob).filter(
                GenerationJob.model_id == model.id,
                GeneratedCaption.status == "pending_review",
                GeneratedCaption.quality_score >= 80
            ).count()

    approved_count = pending_before - pending_after

    return {
        "status": "completed",
        "niche": niche,
        "job_id": job_id,
        "approved": approved_count,
        "pending_before": pending_before,
        "pending_after": pending_after,
        "message": f"Auto-approved {approved_count} captions for {niche}"
    }


# ============================================================================
# Daily Quotas
# ============================================================================

@router.get("/quotas")
async def get_quotas():
    """Get daily quota status for all niches."""
    config = get_automation_config()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    total_target = 0
    total_composed = 0
    all_met = True

    niche_quotas = {}
    for name, niche_config in config.niches.items():
        if not niche_config.enabled:
            continue

        composed = get_videos_composed_today(name)
        target = niche_config.daily_video_target
        met = composed >= target

        niche_quotas[name] = {
            "composed": composed,
            "target": target,
            "met": met,
            "remaining": max(0, target - composed)
        }

        total_target += target
        total_composed += composed
        if not met:
            all_met = False

    return {
        "date": today,
        "niches": niche_quotas,
        "total_target": total_target,
        "total_composed": total_composed,
        "all_quotas_met": all_met
    }


@router.get("/quotas/today")
async def get_todays_quotas():
    """Simplified view of today's quota progress."""
    config = get_automation_config()

    results = []
    for name, niche_config in config.niches.items():
        if not niche_config.enabled:
            continue

        composed = get_videos_composed_today(name)
        target = niche_config.daily_video_target

        results.append({
            "niche": name,
            "composed": composed,
            "target": target,
            "progress_percent": min(100, round(composed / target * 100, 1)) if target > 0 else 100
        })

    return {"quotas": results}


@router.post("/quotas/{niche}/reset")
async def reset_niche_quota(niche: str):
    """
    Reset the daily quota count for a specific niche.

    Use this if videos were composed with wrong niche attribution
    and you need to allow recomposition.
    """
    from tasks.pipeline_orchestrator import get_redis
    from datetime import datetime

    config = get_automation_config()
    if niche not in config.niches:
        raise HTTPException(status_code=404, detail=f"Unknown niche: {niche}")

    old_count = get_videos_composed_today(niche)

    try:
        r = get_redis()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"pipeline:quota:{today}:{niche}:composed"
        r.delete(key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset quota: {e}")

    return {
        "status": "reset",
        "niche": niche,
        "previous_count": old_count,
        "new_count": 0
    }


# ============================================================================
# Configuration
# ============================================================================

@router.get("/config")
async def get_config():
    """Get current automation configuration."""
    config = get_automation_config()
    return config.to_dict()


@router.post("/config/reload")
async def reload_configuration():
    """
    Reload configuration from environment/defaults.

    Use after updating environment variables.
    """
    reload_config()
    config = get_automation_config()
    return {
        "status": "reloaded",
        "config": config.to_dict()
    }


# ============================================================================
# History
# ============================================================================

@router.get("/history")
async def get_pipeline_history(limit: int = 50):
    """
    Get pipeline state transition history.

    Returns recent state changes for debugging.
    """
    from database.db import get_db_context
    from sqlalchemy import text

    try:
        with get_db_context() as db:
            result = db.execute(text("""
                SELECT id, state, previous_state, niche, triggered_by, reason, metadata, created_at
                FROM pipeline_state_log
                ORDER BY created_at DESC
                LIMIT :limit
            """), {"limit": limit})

            history = []
            for row in result:
                history.append({
                    "id": row[0],
                    "state": row[1],
                    "previous_state": row[2],
                    "niche": row[3],
                    "triggered_by": row[4],
                    "reason": row[5],
                    "metadata": row[6],
                    "created_at": row[7].isoformat() if row[7] else None
                })

            return {"history": history}
    except Exception as e:
        return {"history": [], "error": str(e)}


# ============================================================================
# Active Jobs
# ============================================================================

@router.get("/jobs/active")
async def get_active_jobs():
    """
    Get currently active jobs across all niches.

    Returns training, generation, and composition jobs in progress.
    """
    from database.db import get_db_context
    from database.models import TrainingJob, GenerationJob, VideoCompositionJob

    config = get_automation_config()
    current_niche = get_current_niche()

    active_jobs = {
        "training": None,
        "generation": None,
        "composition": None,
    }

    with get_db_context() as db:
        # Check for active training job
        if current_niche:
            job_id = get_current_training_job_id(current_niche)
            if job_id:
                job = db.query(TrainingJob).filter_by(id=job_id).first()
                if job and job.status not in ["completed", "failed", "cancelled"]:
                    active_jobs["training"] = {
                        "job_id": job.id,
                        "job_name": job.job_name,
                        "niche": current_niche,
                        "status": job.status,
                        "progress_percent": job.progress_percent,
                        "current_epoch": job.current_epoch,
                        "total_epochs": job.num_epochs,
                        "current_loss": job.current_loss,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                    }

            # Check for active generation job
            job_id = get_current_generation_job_id(current_niche)
            if job_id:
                job = db.query(GenerationJob).filter_by(id=job_id).first()
                if job and job.status not in ["completed", "failed", "cancelled"]:
                    active_jobs["generation"] = {
                        "job_id": job.id,
                        "job_name": job.job_name,
                        "niche": current_niche,
                        "status": job.status,
                        "progress_percent": job.progress_percent,
                        "captions_generated": job.captions_generated,
                        "num_captions": job.num_captions,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                    }

            # Check for active composition job
            job_id = get_current_composition_job_id(current_niche)
            if job_id:
                job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
                if job and job.status not in ["completed", "failed", "cancelled"]:
                    active_jobs["composition"] = {
                        "job_id": job.id,
                        "job_name": job.job_name,
                        "niche": current_niche,
                        "status": job.status,
                        "progress_percent": job.progress_percent,
                        "videos_composed": job.videos_composed,
                        "target_count": job.target_count,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                    }

    return {
        "current_niche": current_niche,
        "pipeline_state": get_pipeline_state(),
        "active_jobs": active_jobs
    }


@router.get("/jobs/{niche}")
async def get_niche_jobs(niche: str, limit: int = 10):
    """
    Get recent jobs for a specific niche.

    Returns training, generation, and composition job history.
    """
    from database.db import get_db_context
    from database.models import TrainingJob, GenerationJob, VideoCompositionJob

    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    with get_db_context() as db:
        # Training jobs
        training_jobs = db.query(TrainingJob).filter(
            TrainingJob.niche == niche
        ).order_by(TrainingJob.created_at.desc()).limit(limit).all()

        # Get model for niche to find generation jobs
        model = get_model_for_niche(niche)
        generation_jobs = []
        composition_jobs = []

        if model:
            generation_jobs = db.query(GenerationJob).filter(
                GenerationJob.model_id == model.id
            ).order_by(GenerationJob.created_at.desc()).limit(limit).all()

        # Composition jobs with niche
        composition_jobs = db.query(VideoCompositionJob).filter(
            VideoCompositionJob.niche == niche
        ).order_by(VideoCompositionJob.created_at.desc()).limit(limit).all()

        return {
            "niche": niche,
            "training_jobs": [{
                "id": j.id,
                "job_name": j.job_name,
                "status": j.status,
                "progress_percent": j.progress_percent,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            } for j in training_jobs],
            "generation_jobs": [{
                "id": j.id,
                "job_name": j.job_name,
                "status": j.status,
                "captions_generated": j.captions_generated,
                "num_captions": j.num_captions,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            } for j in generation_jobs],
            "composition_jobs": [{
                "id": j.id,
                "job_name": j.job_name,
                "status": j.status,
                "videos_composed": j.videos_composed,
                "target_count": j.target_count,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            } for j in composition_jobs],
        }


# ============================================================================
# Consecutive Failures Management
# ============================================================================

@router.get("/failures/{niche}")
async def get_niche_failures(niche: str):
    """Get consecutive failure count for a specific niche."""
    config = get_automation_config()
    if not config.get_niche(niche):
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    failures = get_consecutive_failures(niche)
    return {
        "niche": niche,
        "consecutive_failures": failures,
        "max_failures": 3  # Before auto-skip
    }


@router.post("/failures/{niche}/reset")
async def reset_niche_failures(niche: str):
    """
    Reset consecutive failure count for a specific niche.

    Use this to retry a niche that was auto-skipped due to failures.
    """
    config = get_automation_config()
    if not config.get_niche(niche):
        raise HTTPException(status_code=404, detail=f"Niche '{niche}' not found")

    reset_consecutive_failures(niche)
    return {
        "status": "reset",
        "niche": niche,
        "message": f"Consecutive failures reset for '{niche}'"
    }


@router.get("/failures")
async def get_all_failures():
    """Get consecutive failure counts for all niches."""
    config = get_automation_config()

    failures = {}
    for name in config.niches.keys():
        failures[name] = {
            "consecutive_failures": get_consecutive_failures(name),
            "max_failures": 3
        }

    return {"niches": failures}


# ============================================================================
# State Machine Visualization Data
# ============================================================================

@router.get("/state-machine")
async def get_state_machine_definition():
    """
    Get the full state machine definition for visualization.

    Returns all states, transitions, and conditions for rendering
    a state machine diagram in the UI.
    """
    states = [
        {
            "id": "collecting",
            "name": "Collecting",
            "description": "Normal operation - scrapers active, extraction running",
            "color": "#27ae60"
        },
        {
            "id": "training_pending",
            "name": "Training Pending",
            "description": "Waiting for extraction queue to drain before training",
            "color": "#f39c12"
        },
        {
            "id": "gpu_clearing",
            "name": "GPU Clearing",
            "description": "Purging GPU memory and disabling workloads",
            "color": "#e74c3c"
        },
        {
            "id": "training",
            "name": "Training",
            "description": "LoRA fine-tuning in progress",
            "color": "#9b59b6"
        },
        {
            "id": "generation_pending",
            "name": "Generation Pending",
            "description": "Training complete, preparing for caption generation",
            "color": "#f39c12"
        },
        {
            "id": "generating",
            "name": "Generating",
            "description": "LLM caption generation in progress",
            "color": "#3498db"
        },
        {
            "id": "composition_pending",
            "name": "Composition Pending",
            "description": "Generation complete, checking composition readiness",
            "color": "#f39c12"
        },
        {
            "id": "composing",
            "name": "Composing",
            "description": "Video composition in progress",
            "color": "#e67e22"
        }
    ]

    transitions = [
        # From collecting - priority order: training > generation (always generate when below quota)
        {
            "from": "collecting",
            "to": "training_pending",
            "condition": "Niche needs training (first_training | stale_model | 500+ new_captions | quality_drop)",
            "trigger": "orchestrator",
            "priority": 1
        },
        {
            "from": "collecting",
            "to": "generation_pending",
            "condition": "Niche below daily quota AND has model (always generate to keep caption pool full)",
            "trigger": "orchestrator",
            "priority": 2
        },
        # Training flow
        {
            "from": "training_pending",
            "to": "gpu_clearing",
            "condition": "Extraction queue drained (queue_size == 0)",
            "trigger": "orchestrator"
        },
        {
            "from": "training_pending",
            "to": "collecting",
            "condition": "Timeout (>30 min waiting) or extraction not draining",
            "trigger": "orchestrator"
        },
        {
            "from": "gpu_clearing",
            "to": "training",
            "condition": "GPU memory < 2GB",
            "trigger": "orchestrator"
        },
        {
            "from": "gpu_clearing",
            "to": "collecting",
            "condition": "GPU clear failed after 5 attempts",
            "trigger": "orchestrator"
        },
        {
            "from": "training",
            "to": "generation_pending",
            "condition": "Training job completed successfully",
            "trigger": "job_completion"
        },
        {
            "from": "training",
            "to": "collecting",
            "condition": "Training job failed",
            "trigger": "job_failure"
        },
        # Generation flow - always generates to keep caption pool replenished
        {
            "from": "generation_pending",
            "to": "generating",
            "condition": "Model available, start generation (always runs to replenish captions)",
            "trigger": "orchestrator"
        },
        {
            "from": "generation_pending",
            "to": "composition_pending",
            "condition": "Generation failed to start BUT captions available (proceed to composition)",
            "trigger": "orchestrator"
        },
        {
            "from": "generation_pending",
            "to": "collecting",
            "condition": "No model available AND no captions available",
            "trigger": "orchestrator"
        },
        {
            "from": "generating",
            "to": "composition_pending",
            "condition": "Generation job completed",
            "trigger": "job_completion"
        },
        {
            "from": "generating",
            "to": "collecting",
            "condition": "Generation job failed",
            "trigger": "job_failure"
        },
        # Composition flow
        {
            "from": "composition_pending",
            "to": "composing",
            "condition": "Approved captions available + backgrounds available",
            "trigger": "orchestrator"
        },
        {
            "from": "composition_pending",
            "to": "collecting",
            "condition": "No approved captions or no backgrounds or quota met",
            "trigger": "orchestrator"
        },
        {
            "from": "composing",
            "to": "collecting",
            "condition": "Composition job completed or quota met",
            "trigger": "job_completion"
        },
        {
            "from": "composing",
            "to": "generation_pending",
            "condition": "Composition failed: no available captions (all approved captions already used)",
            "trigger": "job_failure"
        }
    ]

    return {
        "states": states,
        "transitions": transitions,
        "current_state": get_pipeline_state(),
        "current_niche": get_current_niche()
    }


@router.get("/history/stats")
async def get_transition_statistics(hours: int = 24):
    """
    Get statistics about state transitions over a time period.

    Useful for understanding pipeline behavior and identifying issues.
    """
    from database.db import get_db_context
    from sqlalchemy import text

    try:
        with get_db_context() as db:
            # Transition counts by state
            result = db.execute(text("""
                SELECT state, COUNT(*) as count
                FROM pipeline_state_log
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                GROUP BY state
                ORDER BY count DESC
            """.replace(":hours", str(hours))))

            state_counts = {row[0]: row[1] for row in result}

            # Transition counts by reason
            result = db.execute(text("""
                SELECT reason, COUNT(*) as count
                FROM pipeline_state_log
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                  AND reason IS NOT NULL
                GROUP BY reason
                ORDER BY count DESC
                LIMIT 20
            """.replace(":hours", str(hours))))

            reason_counts = {row[0]: row[1] for row in result}

            # Niche activity
            result = db.execute(text("""
                SELECT niche, COUNT(*) as count
                FROM pipeline_state_log
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                  AND niche IS NOT NULL
                GROUP BY niche
                ORDER BY count DESC
            """.replace(":hours", str(hours))))

            niche_counts = {row[0]: row[1] for row in result}

            # Failures (transitions back to collecting with error reasons)
            result = db.execute(text("""
                SELECT COUNT(*)
                FROM pipeline_state_log
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                  AND state = 'collecting'
                  AND reason LIKE '%failed%'
            """.replace(":hours", str(hours))))

            failure_count = result.scalar() or 0

            # Average time in each state (simplified - time until next transition)
            result = db.execute(text("""
                WITH state_durations AS (
                    SELECT
                        state,
                        created_at,
                        LEAD(created_at) OVER (ORDER BY created_at) as next_transition
                    FROM pipeline_state_log
                    WHERE created_at > NOW() - INTERVAL ':hours hours'
                )
                SELECT
                    state,
                    AVG(EXTRACT(EPOCH FROM (next_transition - created_at))) as avg_seconds
                FROM state_durations
                WHERE next_transition IS NOT NULL
                GROUP BY state
            """.replace(":hours", str(hours))))

            avg_duration = {}
            for row in result:
                if row[1]:
                    avg_duration[row[0]] = round(row[1], 1)

            return {
                "period_hours": hours,
                "state_counts": state_counts,
                "reason_counts": reason_counts,
                "niche_counts": niche_counts,
                "failure_count": failure_count,
                "avg_duration_seconds": avg_duration
            }
    except Exception as e:
        return {
            "period_hours": hours,
            "error": str(e),
            "state_counts": {},
            "reason_counts": {},
            "niche_counts": {},
            "failure_count": 0,
            "avg_duration_seconds": {}
        }


@router.get("/history/timeline")
async def get_transition_timeline(hours: int = 24, limit: int = 100):
    """
    Get a timeline of state transitions for visualization.

    Returns transitions with full details for rendering a timeline chart.
    """
    from database.db import get_db_context
    from sqlalchemy import text

    try:
        with get_db_context() as db:
            result = db.execute(text("""
                SELECT
                    id, state, previous_state, niche,
                    triggered_by, reason, metadata, created_at
                FROM pipeline_state_log
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                ORDER BY created_at DESC
                LIMIT :limit
            """.replace(":hours", str(hours))), {"limit": limit})

            timeline = []
            for row in result:
                entry = {
                    "id": row[0],
                    "state": row[1],
                    "previous_state": row[2],
                    "niche": row[3],
                    "triggered_by": row[4],
                    "reason": row[5],
                    "metadata": row[6],
                    "created_at": row[7].isoformat() if row[7] else None
                }
                timeline.append(entry)

            return {"timeline": timeline, "period_hours": hours}
    except Exception as e:
        return {"timeline": [], "period_hours": hours, "error": str(e)}


# ============================================================================
# Data Integrity Check
# ============================================================================

@router.get("/data-integrity")
async def check_data_integrity():
    """
    Check for common data integrity issues that can cause pipeline problems.

    Returns warnings for:
    - Models without niche field set
    - Training jobs without niche field set
    - Generated captions without niche field set
    - Mismatches between model names and niche assignments
    """
    from database.db import get_db_context
    from database.models import TrainedModel, TrainingJob, GeneratedCaption

    issues = []
    warnings = []

    with get_db_context() as db:
        # Check trained_models without niche
        models_no_niche = db.query(TrainedModel).filter(
            TrainedModel.niche.is_(None)
        ).count()
        if models_no_niche > 0:
            issues.append({
                "severity": "error",
                "table": "trained_models",
                "issue": f"{models_no_niche} models have NULL niche field",
                "fix": "Run: docker-compose exec api python database/migrations/backfill_niche_from_model_name.py"
            })

        # Check training_jobs without niche
        jobs_no_niche = db.query(TrainingJob).filter(
            TrainingJob.niche.is_(None)
        ).count()
        if jobs_no_niche > 0:
            issues.append({
                "severity": "warning",
                "table": "training_jobs",
                "issue": f"{jobs_no_niche} training jobs have NULL niche field",
                "fix": "Run: docker-compose exec api python database/migrations/backfill_niche_from_model_name.py"
            })

        # Check generated_captions without niche
        captions_no_niche = db.query(GeneratedCaption).filter(
            GeneratedCaption.niche.is_(None),
            GeneratedCaption.status == "approved"
        ).count()
        if captions_no_niche > 0:
            issues.append({
                "severity": "error",
                "table": "generated_captions",
                "issue": f"{captions_no_niche} approved captions have NULL niche field",
                "fix": "Run: docker-compose exec api python database/migrations/backfill_niche_from_model_name.py",
                "impact": "Pipeline cannot find captions for composition - will cycle between collecting and composition_pending"
            })

        # Check for model name / niche mismatches
        from sqlalchemy import text
        mismatch_result = db.execute(text("""
            SELECT name, niche FROM trained_models
            WHERE niche IS NOT NULL
              AND NOT (
                name ILIKE niche || '-%'
                OR name ILIKE '%-' || niche || '-%'
              )
        """))
        mismatches = mismatch_result.fetchall()
        if mismatches:
            for row in mismatches:
                warnings.append({
                    "severity": "warning",
                    "issue": f"Model '{row[0]}' has niche='{row[1]}' but name doesn't match pattern",
                    "impact": "May cause incorrect niche assignment"
                })

        # Summary stats
        total_models = db.query(TrainedModel).count()
        total_captions = db.query(GeneratedCaption).filter(GeneratedCaption.status == "approved").count()

    healthy = len(issues) == 0

    return {
        "healthy": healthy,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "total_models": total_models,
            "models_with_niche": total_models - models_no_niche,
            "total_approved_captions": total_captions,
            "captions_with_niche": total_captions - captions_no_niche
        },
        "recommendation": "Run backfill migration" if not healthy else "No action needed"
    }
