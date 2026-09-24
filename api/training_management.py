"""
Training Management API - Full control over model training and management

Endpoints:
- Training Jobs: Create, list, get status, cancel
- Trained Models: List, load, unload, delete, generate
- GPU Management: Status, memory, clear cache
- Training Data: Preview, statistics per subreddit
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from database.db import get_db
from database.models import (
    TrainedModel, TrainingJob, ScrapedCaption, Video, ScrapingProgress
)
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-manager", tags=["training-manager"])


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class TrainingJobCreate(BaseModel):
    """Request model for creating a new training job."""
    job_name: str = Field(..., description="Name for this training job")
    base_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="Base model to fine-tune"
    )
    source_subreddits: List[str] = Field(..., description="Subreddits to use for training data")
    niche: Optional[str] = Field(default=None, description="Niche category for this model (motivation, fitness, cooking, travel)")
    min_upvotes: int = Field(default=0, description="Minimum upvotes filter")
    min_caption_length: int = Field(default=50, description="Minimum caption length")
    max_caption_length: int = Field(default=2000, description="Maximum caption length")

    # GPU selection (0 or 1 for dual-GPU setup)
    target_gpu: int = Field(default=0, ge=0, le=1, description="GPU to train on (0 or 1)")

    # Hyperparameters
    lora_rank: int = Field(default=16, ge=4, le=128, description="LoRA rank (4-128)")
    lora_alpha: int = Field(default=32, ge=8, le=256, description="LoRA alpha")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    num_epochs: int = Field(default=3, ge=1, le=20, description="Number of epochs")
    batch_size: int = Field(default=1, ge=1, le=8, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    max_seq_length: int = Field(default=512, ge=128, le=2048)
    warmup_ratio: float = Field(default=0.1, ge=0, le=0.5)


class TrainingJobResponse(BaseModel):
    """Response model for training job details."""
    id: int
    job_name: str
    status: str
    progress_percent: float
    current_epoch: int
    current_step: int
    total_steps: Optional[int]
    current_loss: Optional[float]
    best_loss: Optional[float]
    total_samples: Optional[int]
    train_samples: Optional[int]
    val_samples: Optional[int]
    source_subreddits: List[str]
    base_model: str
    target_gpu: int = 0
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    training_duration_seconds: Optional[int]


class TrainedModelResponse(BaseModel):
    """Response model for trained model details."""
    id: int
    name: str
    description: Optional[str]
    base_model: str
    source_subreddits: List[str]
    training_samples: int
    status: str
    is_loaded: bool
    final_loss: Optional[float]
    validation_loss: Optional[float]
    created_at: datetime


class GenerateRequest(BaseModel):
    """Request model for caption generation."""
    prompt: str = Field(default="Generate a caption:", description="Generation prompt")
    max_new_tokens: int = Field(default=300, ge=50, le=1000)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)


class SubredditDataStats(BaseModel):
    """Statistics about training data from a subreddit."""
    subreddit: str
    total_captions: int
    with_llm_refined: int
    avg_caption_length: float
    avg_upvotes: float
    upvote_distribution: Dict[str, int]  # Bucketed upvote counts


# ============================================================================
# Training Data Endpoints
# ============================================================================

@router.get("/data/subreddits")
async def get_available_subreddits(db: Session = Depends(get_db)):
    """
    Get all subreddits with available training data.

    Returns caption counts and quality metrics per subreddit.
    """
    try:
        # Get subreddits with caption counts
        subreddit_stats = db.query(
            ScrapedCaption.source_subreddit,
            func.count(ScrapedCaption.id).label('total'),
            func.count(ScrapedCaption.llm_refined_text).label('with_llm'),
            func.avg(func.length(ScrapedCaption.llm_refined_text)).label('avg_length'),
            func.avg(ScrapedCaption.upvotes).label('avg_upvotes')
        ).group_by(ScrapedCaption.source_subreddit).all()

        subreddits = []
        for sub, total, with_llm, avg_len, avg_upvotes in subreddit_stats:
            subreddits.append({
                "subreddit": sub,
                "total_captions": total,
                "with_llm_refined": with_llm or 0,
                "llm_rate": round((with_llm or 0) / total * 100, 1) if total > 0 else 0,
                "avg_caption_length": round(avg_len or 0, 0),
                "avg_upvotes": round(avg_upvotes or 0, 0)
            })

        return {
            "subreddits": sorted(subreddits, key=lambda x: x['total_captions'], reverse=True),
            "total_subreddits": len(subreddits)
        }
    except Exception as e:
        logger.error(f"Error getting subreddit data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/subreddit/{subreddit_name}/stats")
async def get_subreddit_training_stats(
    subreddit_name: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed training data statistics for a specific subreddit.

    Includes upvote distribution and sample captions.
    """
    try:
        # Get captions for this subreddit
        captions = db.query(ScrapedCaption).filter(
            ScrapedCaption.source_subreddit == subreddit_name,
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.llm_refined_text != ""
        ).all()

        if not captions:
            return {
                "subreddit": subreddit_name,
                "total_captions": 0,
                "message": "No LLM-refined captions available"
            }

        # Calculate statistics
        lengths = [len(c.llm_refined_text) for c in captions]
        upvotes = [c.upvotes or 0 for c in captions]

        # Upvote distribution buckets
        upvote_buckets = {
            "0-100": 0,
            "100-500": 0,
            "500-1000": 0,
            "1000-5000": 0,
            "5000+": 0
        }
        for u in upvotes:
            if u < 100:
                upvote_buckets["0-100"] += 1
            elif u < 500:
                upvote_buckets["100-500"] += 1
            elif u < 1000:
                upvote_buckets["500-1000"] += 1
            elif u < 5000:
                upvote_buckets["1000-5000"] += 1
            else:
                upvote_buckets["5000+"] += 1

        # Sample captions (top 3 by upvotes)
        top_captions = sorted(captions, key=lambda x: x.upvotes or 0, reverse=True)[:3]
        samples = [
            {
                "id": c.id,
                "upvotes": c.upvotes,
                "length": len(c.llm_refined_text),
                "preview": c.llm_refined_text[:200] + "..." if len(c.llm_refined_text) > 200 else c.llm_refined_text
            }
            for c in top_captions
        ]

        return {
            "subreddit": subreddit_name,
            "total_captions": len(captions),
            "statistics": {
                "avg_length": round(sum(lengths) / len(lengths), 0),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_upvotes": round(sum(upvotes) / len(upvotes), 0),
                "max_upvotes": max(upvotes),
                "min_upvotes": min(upvotes)
            },
            "upvote_distribution": upvote_buckets,
            "sample_captions": samples
        }
    except Exception as e:
        logger.error(f"Error getting subreddit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/preview")
async def preview_training_data(
    subreddits: str,  # Comma-separated list
    min_upvotes: int = 0,
    min_length: int = 50,
    max_length: int = 2000,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Preview training data with filters applied.

    Use this to see what data will be included in training.
    """
    try:
        subreddit_list = [s.strip() for s in subreddits.split(",")]

        query = db.query(ScrapedCaption).filter(
            ScrapedCaption.source_subreddit.in_(subreddit_list),
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.llm_refined_text != "",
            ScrapedCaption.upvotes >= min_upvotes,
            func.length(ScrapedCaption.llm_refined_text) >= min_length,
            func.length(ScrapedCaption.llm_refined_text) <= max_length
        )

        total_count = query.count()
        samples = query.order_by(desc(ScrapedCaption.upvotes)).limit(limit).all()

        return {
            "filters": {
                "subreddits": subreddit_list,
                "min_upvotes": min_upvotes,
                "min_length": min_length,
                "max_length": max_length
            },
            "total_matching": total_count,
            "samples": [
                {
                    "id": c.id,
                    "subreddit": c.source_subreddit,
                    "upvotes": c.upvotes,
                    "length": len(c.llm_refined_text),
                    "text": c.llm_refined_text
                }
                for c in samples
            ]
        }
    except Exception as e:
        logger.error(f"Error previewing training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Training Job Endpoints
# ============================================================================

@router.post("/jobs", response_model=Dict[str, Any])
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db)
):
    """
    Create and start a new training job.

    This will:
    1. Validate training data availability
    2. Create a TrainingJob record
    3. Queue the training task
    """
    try:
        # Validate subreddits have data
        for subreddit in job.source_subreddits:
            count = db.query(ScrapedCaption).filter(
                ScrapedCaption.source_subreddit == subreddit,
                ScrapedCaption.llm_refined_text.isnot(None),
                ScrapedCaption.upvotes >= job.min_upvotes
            ).count()
            if count == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No training data available for r/{subreddit} with min_upvotes={job.min_upvotes}"
                )

        # Create training job record
        training_job = TrainingJob(
            job_name=job.job_name,
            base_model=job.base_model,
            source_subreddits=job.source_subreddits,
            niche=job.niche,  # Per-niche tracking for pipeline orchestrator
            min_upvotes=job.min_upvotes,
            min_caption_length=job.min_caption_length,
            max_caption_length=job.max_caption_length,
            lora_rank=job.lora_rank,
            lora_alpha=job.lora_alpha,
            learning_rate=job.learning_rate,
            num_epochs=job.num_epochs,
            batch_size=job.batch_size,
            gradient_accumulation_steps=job.gradient_accumulation_steps,
            max_seq_length=job.max_seq_length,
            warmup_ratio=job.warmup_ratio,
            target_gpu=job.target_gpu,
            status="queued"
        )

        db.add(training_job)
        db.commit()
        db.refresh(training_job)

        # Queue the training task to the appropriate GPU queue
        from tasks.training_tasks import run_training_job
        # Route to gpu_training_0 or gpu_training_1 based on target_gpu
        queue_name = f"gpu_training_{job.target_gpu}"
        task = run_training_job.apply_async(args=[training_job.id], queue=queue_name)

        # Update with celery task ID
        training_job.celery_task_id = task.id
        db.commit()

        return {
            "status": "success",
            "job_id": training_job.id,
            "celery_task_id": task.id,
            "target_gpu": job.target_gpu,
            "message": f"Training job '{job.job_name}' created and queued for GPU {job.target_gpu}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List all training jobs with optional status filter."""
    try:
        query = db.query(TrainingJob).order_by(desc(TrainingJob.created_at))

        if status:
            query = query.filter(TrainingJob.status == status)

        jobs = query.limit(limit).all()

        return {
            "jobs": [
                {
                    "id": j.id,
                    "job_name": j.job_name,
                    "status": j.status,
                    "progress_percent": j.progress_percent,
                    "current_epoch": j.current_epoch,
                    "num_epochs": j.num_epochs,
                    "current_loss": j.current_loss,
                    "source_subreddits": j.source_subreddits,
                    "base_model": j.base_model,
                    "target_gpu": getattr(j, 'target_gpu', 0),
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None
                }
                for j in jobs
            ],
            "total": len(jobs)
        }
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get detailed status of a specific training job."""
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

        return {
            "id": job.id,
            "job_name": job.job_name,
            "celery_task_id": job.celery_task_id,
            "status": job.status,
            "progress_percent": job.progress_percent,
            "current_epoch": job.current_epoch,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "current_loss": job.current_loss,
            "best_loss": job.best_loss,
            "total_samples": job.total_samples,
            "train_samples": job.train_samples,
            "val_samples": job.val_samples,
            "configuration": {
                "base_model": job.base_model,
                "source_subreddits": job.source_subreddits,
                "min_upvotes": job.min_upvotes,
                "target_gpu": getattr(job, 'target_gpu', 0),
                "lora_rank": job.lora_rank,
                "lora_alpha": job.lora_alpha,
                "learning_rate": job.learning_rate,
                "num_epochs": job.num_epochs,
                "batch_size": job.batch_size,
                "gradient_accumulation_steps": job.gradient_accumulation_steps,
                "max_seq_length": job.max_seq_length
            },
            "results": {
                "final_train_loss": job.final_train_loss,
                "final_val_loss": job.final_val_loss,
                "training_duration_seconds": job.training_duration_seconds
            },
            "error": {
                "message": job.error_message,
                "traceback": job.error_traceback
            } if job.error_message else None,
            "timestamps": {
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a running training job."""
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

        if job.status in ("completed", "failed", "cancelled"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status '{job.status}'"
            )

        # Revoke the Celery task
        if job.celery_task_id:
            from tasks.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True)

        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        db.commit()

        return {"status": "success", "message": f"Training job {job_id} cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/fix-stuck")
async def fix_stuck_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Fix a training job stuck in 'saving' or 'training' status.

    This handles cases where:
    - The worker crashed after saving the model but before updating job status
    - The job shows 100% progress but status is 'saving'

    Actions taken:
    1. If a trained model exists for this job, mark job as completed
    2. Update the trained model's niche field if missing
    3. Clear pipeline job tracking for the niche
    """
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

        if job.status == "completed":
            return {"status": "already_completed", "message": "Job is already completed"}

        actions_taken = []

        # Check if a trained model exists for this job
        model = db.query(TrainedModel).filter(
            TrainedModel.name == job.job_name
        ).first()

        if model:
            # Model was saved, mark job as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress_percent = 100.0
            actions_taken.append(f"Marked job {job_id} as completed")

            # Update model's niche if missing
            if not model.niche and job.niche:
                model.niche = job.niche
                actions_taken.append(f"Set model niche to '{job.niche}'")

            # Clear pipeline tracking for this niche
            if job.niche:
                import redis
                from config.settings import settings
                r = redis.from_url(settings.celery_broker_url)
                key = f"pipeline:niche:{job.niche}:training_job"
                r.delete(key)
                actions_taken.append(f"Cleared pipeline tracking for '{job.niche}'")

        else:
            # No model found - job truly failed
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = "Model not found - job was stuck without saving"
            actions_taken.append(f"Marked job {job_id} as failed (no model found)")

        db.commit()

        return {
            "status": "success",
            "job_id": job_id,
            "new_status": job.status,
            "model_found": model is not None,
            "actions_taken": actions_taken
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fixing stuck training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/retry")
async def retry_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Retry a training job that was interrupted (e.g., by worker restart).

    This will reset the job status and re-dispatch the Celery task.
    Only works for jobs in 'training', 'preparing_data', or 'queued' status
    that haven't actually completed.
    """
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

        # Don't retry completed or cancelled jobs
        if job.status == "completed":
            raise HTTPException(
                status_code=400,
                detail="Cannot retry completed job. Create a new job instead."
            )
        if job.status == "cancelled":
            raise HTTPException(
                status_code=400,
                detail="Cannot retry cancelled job. Create a new job instead."
            )

        # Check if job already has final results (was actually completed but status not updated)
        if job.final_train_loss is not None and job.final_train_loss > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Job already has results (final_train_loss={job.final_train_loss}). Mark as completed or create new job."
            )

        # Reset job for retry
        job.status = "queued"
        job.progress_percent = 0.0
        job.current_epoch = 0
        job.current_step = 0
        job.current_loss = None
        job.best_loss = None
        job.started_at = None
        job.error_message = None
        db.commit()

        # Re-dispatch the Celery task
        from tasks.training_tasks import run_training_job
        target_gpu = getattr(job, 'target_gpu', 0)
        queue_name = f"gpu_training_{target_gpu}"
        task = run_training_job.apply_async(args=[job.id], queue=queue_name)

        # Update with new celery task ID
        job.celery_task_id = task.id
        db.commit()

        return {
            "status": "success",
            "job_id": job.id,
            "celery_task_id": task.id,
            "target_gpu": target_gpu,
            "message": f"Training job '{job.job_name}' re-queued for GPU {target_gpu}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a training job record."""
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

        if job.status in ("queued", "preparing_data", "training"):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running job. Cancel it first."
            )

        db.delete(job)
        db.commit()

        return {"status": "success", "message": f"Training job {job_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Trained Model Endpoints
# ============================================================================

@router.get("/models")
async def list_trained_models(db: Session = Depends(get_db)):
    """List all trained models."""
    try:
        models = db.query(TrainedModel).order_by(desc(TrainedModel.created_at)).all()

        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "base_model": m.base_model,
                    "source_subreddits": m.source_subreddits,
                    "training_samples": m.training_samples,
                    "status": m.status,
                    "is_loaded": m.is_loaded,
                    "final_loss": m.final_loss,
                    "validation_loss": m.validation_loss,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ],
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"Error listing trained models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_trained_model(model_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a trained model."""
    try:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Get training job history
        jobs = db.query(TrainingJob).filter_by(model_id=model_id).order_by(
            desc(TrainingJob.created_at)
        ).all()

        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "base_model": model.base_model,
            "adapter_path": model.adapter_path,
            "source_subreddits": model.source_subreddits,
            "training_samples": model.training_samples,
            "min_upvotes_filter": model.min_upvotes_filter,
            "hyperparameters": model.hyperparameters,
            "metrics": {
                "final_loss": model.final_loss,
                "validation_loss": model.validation_loss,
                "training_duration_seconds": model.training_duration_seconds
            },
            "status": model.status,
            "is_loaded": model.is_loaded,
            "last_loaded_at": model.last_loaded_at.isoformat() if model.last_loaded_at else None,
            "training_history": [
                {
                    "job_id": j.id,
                    "job_name": j.job_name,
                    "status": j.status,
                    "created_at": j.created_at.isoformat() if j.created_at else None
                }
                for j in jobs
            ],
            "created_at": model.created_at.isoformat() if model.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trained model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/load")
async def load_model(
    model_id: int,
    target_gpu: int = Query(0, ge=0, le=1, description="GPU to load model on (0 or 1)"),
    db: Session = Depends(get_db)
):
    """
    Load a trained model into GPU memory.

    Args:
        model_id: ID of the model to load
        target_gpu: Which GPU to load on (0=RTX 2080 Ti 11GB, 1=RTX 4060 Ti 8GB)

    This will unload any currently loaded model first.
    """
    try:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        if model.is_loaded:
            return {"status": "already_loaded", "message": f"Model '{model.name}' is already loaded"}

        # Queue the load task to specific GPU
        if target_gpu == 0:
            from tasks.training_tasks import load_model_task_gpu_0
            task = load_model_task_gpu_0.delay(model_id)
        else:
            from tasks.training_tasks import load_model_task_gpu_1
            task = load_model_task_gpu_1.delay(model_id)

        return {
            "status": "loading",
            "task_id": task.id,
            "target_gpu": target_gpu,
            "message": f"Loading model '{model.name}' on GPU {target_gpu}..."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_model_loaded_gpu(model_id: int) -> int:
    """
    Get which GPU a model is loaded on from Redis tracking.

    Returns GPU index (0 or 1), or None if not tracked.
    """
    import redis
    from config.settings import settings

    try:
        r = redis.from_url(settings.celery_broker_url)
        gpu = r.get(f"model:{model_id}:loaded_on_gpu")
        if gpu:
            return int(gpu.decode('utf-8'))
    except Exception as e:
        logger.debug(f"Could not get model GPU from Redis: {e}")
    return None


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: int, force: bool = False, db: Session = Depends(get_db)):
    """
    Unload a model from GPU memory.

    Args:
        model_id: ID of the model to unload
        force: If True, immediately update database without queuing GPU task.
               Use when GPU worker is busy with extraction (trained model not in use).

    The unload task is routed to the same GPU worker that loaded the model,
    ensuring the model is actually unloaded from memory.
    """
    try:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        if not model.is_loaded:
            return {"status": "not_loaded", "message": f"Model '{model.name}' is not loaded"}

        # Check if GPU worker is busy with extraction - if so, use force mode automatically
        # since the trained model isn't actually loaded during extraction (different model)
        auto_force = False
        if not force:
            try:
                from utils.worker_heartbeat import get_worker_status
                status = get_worker_status()
                current_task = status.get('current_task') or {}
                task_name = current_task.get('task_name', '') if isinstance(current_task, dict) else str(current_task)
                if task_name and 'extract' in task_name.lower():
                    auto_force = True
                    logger.info(f"Auto-forcing unload: GPU worker busy with '{task_name}', trained model not in use")
            except Exception as e:
                logger.debug(f"Could not check worker status: {e}")
                pass  # If we can't check, proceed normally

        if force or auto_force:
            # Immediately update database - trained model not actually in GPU memory
            # during extraction (extraction uses Qwen2-VL, not the trained Mistral)
            model.is_loaded = False
            model.status = "ready"
            db.commit()

            # Clear Redis tracking
            import redis
            from config.settings import settings
            try:
                r = redis.from_url(settings.celery_broker_url)
                r.delete(f"model:{model_id}:loaded_on_gpu")
            except:
                pass

            logger.info(f"Force unloaded model {model_id} ({model.name}) - database updated")
            return {
                "status": "unloaded",
                "message": f"Model '{model.name}' unloaded (database updated)",
                "note": "Force mode: trained model not in GPU during extraction"
            }

        # Route unload to the correct GPU worker based on where the model was loaded
        target_gpu = _get_model_loaded_gpu(model_id)

        if target_gpu == 0:
            from tasks.training_tasks import unload_model_task_gpu_0
            task = unload_model_task_gpu_0.delay(model_id)
            logger.info(f"Routing unload for model {model_id} to GPU 0 worker")
        elif target_gpu == 1:
            from tasks.training_tasks import unload_model_task_gpu_1
            task = unload_model_task_gpu_1.delay(model_id)
            logger.info(f"Routing unload for model {model_id} to GPU 1 worker")
        else:
            # Fallback: send to both GPUs to ensure it gets unloaded
            logger.warning(f"Model {model_id} GPU not tracked, sending unload to both workers")
            from tasks.training_tasks import unload_model_task_gpu_0, unload_model_task_gpu_1
            unload_model_task_gpu_0.delay(model_id)
            task = unload_model_task_gpu_1.delay(model_id)

        return {
            "status": "unloading",
            "task_id": task.id,
            "target_gpu": target_gpu,
            "message": f"Unloading model '{model.name}' from GPU {target_gpu if target_gpu is not None else 'unknown'}..."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/generate")
async def generate_caption(
    model_id: int,
    request: GenerateRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a caption using a loaded model.

    The model must be loaded first with POST /models/{model_id}/load
    """
    try:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        if not model.is_loaded:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model.name}' is not loaded. Load it first with POST /models/{model_id}/load"
            )

        # Queue generation task
        from tasks.training_tasks import generate_caption_task
        task = generate_caption_task.delay(
            model_id,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            request.repetition_penalty
        )

        # Wait for result (with timeout)
        try:
            result = task.get(timeout=120)  # 2 minute timeout
            return {
                "status": "success",
                "model_name": model.name,
                "prompt": request.prompt,
                "generated_text": result.get("text", ""),
                "generation_time_seconds": result.get("time", 0)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Generation failed: {str(e)}"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    delete_files: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete a trained model.

    Args:
        delete_files: Also delete the adapter files from disk
    """
    try:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        if model.is_loaded:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete loaded model. Unload it first."
            )

        # Optionally delete files
        if delete_files and model.adapter_path:
            import shutil
            if os.path.exists(model.adapter_path):
                shutil.rmtree(model.adapter_path)
                logger.info(f"Deleted model files at {model.adapter_path}")

        db.delete(model)
        db.commit()

        return {
            "status": "success",
            "message": f"Model '{model.name}' deleted",
            "files_deleted": delete_files
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GPU Management Endpoints
# ============================================================================

@router.get("/gpu/status")
async def get_gpu_status():
    """
    Get current GPU memory usage and status.

    This endpoint runs a Celery task on the GPU worker to get accurate
    GPU status (since the API container doesn't have GPU access).
    """
    try:
        from tasks.training_tasks import get_gpu_status_task
        from utils.worker_heartbeat import get_worker_status

        # First check if worker is busy with another task
        try:
            worker_status = get_worker_status()

            if worker_status.get('current_task'):
                current_task = worker_status['current_task']
                task_name = current_task.get('task_name', 'unknown')
                video_id = current_task.get('video_id')

                return {
                    "status": "worker_busy",
                    "message": f"GPU worker busy with: {task_name}" + (f" (video {video_id})" if video_id else ""),
                    "worker_state": "processing",
                    "current_task": task_name,
                    "note": "GPU status unavailable while worker is processing. Try again later."
                }
        except Exception as e:
            logger.debug(f"Could not check worker status: {e}")

        # Run task on GPU worker and wait for result
        task = get_gpu_status_task.delay()
        try:
            result = task.get(timeout=15)  # Wait up to 15 seconds
            return result
        except Exception as e:
            logger.warning(f"GPU status task failed: {e}")
            return {
                "error": f"Failed to get GPU status from worker: {str(e)}",
                "status": "timeout",
                "note": "Worker may be busy or unresponsive. Check worker health."
            }

    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {"error": str(e)}


@router.post("/gpu/clear-cache")
async def clear_gpu_cache():
    """Clear GPU memory cache (PyTorch)."""
    try:
        from tasks.training_tasks import clear_gpu_memory_task
        task = clear_gpu_memory_task.delay()

        return {
            "status": "success",
            "task_id": task.id,
            "message": "GPU cache clear requested"
        }
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Base Model Options
# ============================================================================

@router.get("/base-models")
async def get_available_base_models():
    """Get list of supported base models for training."""
    return {
        "models": [
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.3",
                "name": "Mistral 7B Instruct v0.3",
                "description": "High quality instruction-tuned model. Good balance of quality and speed.",
                "vram_required_gb": 5,
                "recommended": True
            },
            {
                "id": "meta-llama/Llama-3-8B-Instruct",
                "name": "Llama 3 8B Instruct",
                "description": "Meta's latest instruction-tuned model. Excellent quality.",
                "vram_required_gb": 6,
                "recommended": False
            },
            {
                "id": "microsoft/Phi-3-mini-4k-instruct",
                "name": "Phi-3 Mini (3.8B)",
                "description": "Small but capable model. Fast training.",
                "vram_required_gb": 3,
                "recommended": False
            },
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen 2.5 7B Instruct",
                "description": "Alibaba's instruction-tuned model. Strong multilingual.",
                "vram_required_gb": 5,
                "recommended": False
            }
        ]
    }


# ============================================================================
# Training UI
# ============================================================================

@router.get("/", include_in_schema=False)
async def training_ui():
    """Serve the Training Management UI."""
    from fastapi.responses import HTMLResponse
    ui_path = os.path.join(os.path.dirname(__file__), "training_ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())
