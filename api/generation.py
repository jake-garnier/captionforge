"""
Generation API - Batch caption generation and management

Endpoints:
- Generation Jobs: Create, list, get status, cancel
- Generated Captions: List, view, update status, favorite, delete
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database.db import get_db
from database.models import GenerationJob, GeneratedCaption, TrainedModel, ComposedVideo
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation", tags=["generation"])


# ============================================================================
# Pydantic Models
# ============================================================================

class GenerationJobCreate(BaseModel):
    """Request model for creating a new generation job."""
    job_name: str = Field(..., description="Name for this generation job")
    model_id: int = Field(..., description="ID of the trained model to use")
    num_captions: int = Field(..., ge=1, le=100, description="Number of captions to generate (1-100)")
    prompt: str = Field(default="Generate a caption:", description="Generation prompt")
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    max_new_tokens: int = Field(default=120, ge=50, le=1000)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)


class CaptionUpdate(BaseModel):
    """Request model for updating a caption."""
    status: Optional[str] = Field(None, description="New status: pending_review, approved, rejected")
    is_favorite: Optional[bool] = Field(None, description="Mark as favorite")
    caption_text: Optional[str] = Field(None, description="Edit the caption text")


class BulkActionRequest(BaseModel):
    """Request model for bulk caption actions."""
    action: str = Field(..., description="Action: approve, reject, delete, favorite, unfavorite")
    caption_ids: List[int] = Field(..., description="List of caption IDs to act on")


# ============================================================================
# Generation Job Endpoints
# ============================================================================

@router.post("/jobs", response_model=Dict[str, Any])
async def create_generation_job(
    job: GenerationJobCreate,
    db: Session = Depends(get_db)
):
    """
    Create and start a new caption generation job.

    This will queue a Celery task to generate the specified number of captions.
    """
    try:
        # Verify model exists
        model = db.query(TrainedModel).filter_by(id=job.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {job.model_id} not found")

        if model.status == "error":
            raise HTTPException(status_code=400, detail=f"Model '{model.name}' is in error state")

        # Create generation job record
        generation_job = GenerationJob(
            job_name=job.job_name,
            model_id=job.model_id,
            num_captions=job.num_captions,
            prompt=job.prompt,
            temperature=job.temperature,
            top_p=job.top_p,
            max_new_tokens=job.max_new_tokens,
            repetition_penalty=job.repetition_penalty,
            status="queued"
        )

        db.add(generation_job)
        db.commit()
        db.refresh(generation_job)

        # Queue the generation task
        from tasks.training_tasks import run_generation_job
        task = run_generation_job.delay(generation_job.id)

        # Update with celery task ID
        generation_job.celery_task_id = task.id
        db.commit()

        return {
            "status": "success",
            "job_id": generation_job.id,
            "celery_task_id": task.id,
            "message": f"Generation job '{job.job_name}' created - will generate {job.num_captions} captions"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_generation_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List all generation jobs with optional status filter."""
    try:
        query = db.query(GenerationJob).order_by(desc(GenerationJob.created_at))

        if status:
            query = query.filter(GenerationJob.status == status)

        jobs = query.limit(limit).all()

        return {
            "jobs": [
                {
                    "id": j.id,
                    "job_name": j.job_name,
                    "model_id": j.model_id,
                    "status": j.status,
                    "progress_percent": j.progress_percent,
                    "captions_generated": j.captions_generated,
                    "num_captions": j.num_captions,
                    "prompt": j.prompt[:50] + "..." if len(j.prompt) > 50 else j.prompt,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None
                }
                for j in jobs
            ],
            "total": len(jobs)
        }
    except Exception as e:
        logger.error(f"Error listing generation jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_generation_job(job_id: int, db: Session = Depends(get_db)):
    """Get detailed status of a specific generation job."""
    try:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found")

        # Get model name
        model = db.query(TrainedModel).filter_by(id=job.model_id).first()
        model_name = model.name if model else "Unknown"

        return {
            "id": job.id,
            "job_name": job.job_name,
            "celery_task_id": job.celery_task_id,
            "model": {
                "id": job.model_id,
                "name": model_name
            },
            "configuration": {
                "num_captions": job.num_captions,
                "prompt": job.prompt,
                "temperature": job.temperature,
                "top_p": job.top_p,
                "max_new_tokens": job.max_new_tokens,
                "repetition_penalty": job.repetition_penalty
            },
            "progress": {
                "status": job.status,
                "progress_percent": job.progress_percent,
                "captions_generated": job.captions_generated
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
        logger.error(f"Error getting generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_generation_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a running generation job."""
    try:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found")

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

        return {"status": "success", "message": f"Generation job {job_id} cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_generation_job(
    job_id: int,
    delete_captions: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete a generation job record.

    Args:
        delete_captions: Also delete all captions from this job
    """
    try:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found")

        if job.status in ("queued", "running"):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running job. Cancel it first."
            )

        captions_deleted = 0
        composed_unlinked = 0
        if delete_captions:
            # Get caption IDs that will be deleted
            caption_ids = [c.id for c in db.query(GeneratedCaption.id).filter_by(generation_job_id=job_id).all()]

            if caption_ids:
                # Unlink any composed videos that reference these captions
                composed_unlinked = db.query(ComposedVideo).filter(
                    ComposedVideo.generated_caption_id.in_(caption_ids)
                ).update({ComposedVideo.generated_caption_id: None}, synchronize_session=False)

                # Now delete the captions
                captions_deleted = db.query(GeneratedCaption).filter_by(generation_job_id=job_id).delete()

        db.delete(job)
        db.commit()

        return {
            "status": "success",
            "message": f"Generation job {job_id} deleted",
            "captions_deleted": captions_deleted,
            "composed_videos_unlinked": composed_unlinked
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/cancelled/all")
async def delete_all_cancelled_jobs(
    delete_captions: bool = True,
    db: Session = Depends(get_db)
):
    """
    Delete all cancelled generation jobs and optionally their captions.

    Args:
        delete_captions: Also delete all captions from cancelled jobs (default: True)
    """
    try:
        # Find all cancelled jobs
        cancelled_jobs = db.query(GenerationJob).filter_by(status="cancelled").all()

        if not cancelled_jobs:
            return {
                "status": "success",
                "message": "No cancelled jobs to delete",
                "jobs_deleted": 0,
                "captions_deleted": 0
            }

        job_ids = [j.id for j in cancelled_jobs]
        captions_deleted = 0
        composed_unlinked = 0

        if delete_captions:
            # Get caption IDs that will be deleted
            caption_ids = [c.id for c in db.query(GeneratedCaption.id).filter(
                GeneratedCaption.generation_job_id.in_(job_ids)
            ).all()]

            if caption_ids:
                # Unlink any composed videos that reference these captions
                composed_unlinked = db.query(ComposedVideo).filter(
                    ComposedVideo.generated_caption_id.in_(caption_ids)
                ).update({ComposedVideo.generated_caption_id: None}, synchronize_session=False)

                # Now delete captions for all cancelled jobs
                captions_deleted = db.query(GeneratedCaption).filter(
                    GeneratedCaption.generation_job_id.in_(job_ids)
                ).delete(synchronize_session=False)

        # Delete the jobs
        jobs_deleted = db.query(GenerationJob).filter(
            GenerationJob.id.in_(job_ids)
        ).delete(synchronize_session=False)

        db.commit()

        return {
            "status": "success",
            "message": f"Deleted {jobs_deleted} cancelled jobs",
            "jobs_deleted": jobs_deleted,
            "captions_deleted": captions_deleted,
            "composed_videos_unlinked": composed_unlinked
        }
    except Exception as e:
        logger.error(f"Error deleting cancelled jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Generated Captions Endpoints
# ============================================================================

@router.get("/captions")
async def list_generated_captions(
    job_id: Optional[int] = None,
    niche: Optional[str] = None,
    status: Optional[str] = None,
    favorites_only: bool = False,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    sort_by: str = "date",  # "date", "score_high", "score_low"
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List generated captions with optional filters.

    Args:
        job_id: Filter by generation job ID
        niche: Filter by niche category (motivation, fitness, travel, cooking)
        status: Filter by status (pending_review, approved, rejected)
        favorites_only: Only show favorited captions
        min_score: Minimum quality score (1-100)
        max_score: Maximum quality score (1-100)
        sort_by: Sort order - "date" (default), "score_high", "score_low"
        limit: Max results to return
        offset: Pagination offset
    """
    try:
        query = db.query(GeneratedCaption)

        if job_id is not None:
            query = query.filter(GeneratedCaption.generation_job_id == job_id)
        if niche:
            query = query.filter(GeneratedCaption.niche == niche)
        if status:
            query = query.filter(GeneratedCaption.status == status)
        if favorites_only:
            query = query.filter(GeneratedCaption.is_favorite == True)
        if min_score is not None:
            query = query.filter(GeneratedCaption.quality_score >= min_score)
        if max_score is not None:
            query = query.filter(GeneratedCaption.quality_score <= max_score)

        # Apply sorting
        if sort_by == "score_high":
            query = query.order_by(desc(GeneratedCaption.quality_score))
        elif sort_by == "score_low":
            query = query.order_by(GeneratedCaption.quality_score)
        else:  # date (default)
            query = query.order_by(desc(GeneratedCaption.generated_at))

        total = query.count()
        captions = query.offset(offset).limit(limit).all()

        return {
            "captions": [
                {
                    "id": c.id,
                    "caption_text": c.caption_text,
                    "status": c.status,
                    "is_favorite": c.is_favorite,
                    "quality_score": c.quality_score,  # LLM quality score (1-100)
                    "llm_model": c.llm_model,
                    "generation_job_id": c.generation_job_id,
                    "temperature": c.temperature,
                    "tags": c.tags or [],  # Activity tags for background matching
                    "niche": c.niche,  # Niche category (motivation, fitness, etc.)
                    "generated_at": c.generated_at.isoformat() if c.generated_at else None,
                    # Stage 3 LLM judge fields
                    "judge_status": c.judge_status,
                    "judge_pass": c.judge_pass,
                    "judge_scores": c.judge_scores,
                    "judge_issues": c.judge_issues or [],
                }
                for c in captions
            ],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing captions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tag management endpoints - must come BEFORE /captions/{caption_id} to avoid route conflicts
@router.get("/captions/tag-stats")
async def get_caption_tag_stats(db: Session = Depends(get_db)):
    """Get statistics about caption tags."""
    try:
        from sqlalchemy import cast, String

        total = db.query(GeneratedCaption).count()

        # Count captions without tags (null or empty array)
        without_tags = db.query(GeneratedCaption).filter(
            (GeneratedCaption.tags == None) |
            (cast(GeneratedCaption.tags, String) == '[]')
        ).count()
        with_tags = total - without_tags

        # Get most common tags
        all_captions = db.query(GeneratedCaption).filter(
            GeneratedCaption.tags != None,
            cast(GeneratedCaption.tags, String) != '[]'
        ).all()

        tag_counts = {}
        for c in all_captions:
            for tag in (c.tags or []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_captions": total,
            "with_tags": with_tags,
            "without_tags": without_tags,
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags]
        }
    except Exception as e:
        logger.error(f"Error getting tag stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/backfill-tags")
async def backfill_caption_tags(
    limit: int = 100,
    use_llm: bool = False,
    db: Session = Depends(get_db)
):
    """
    Backfill tags for existing captions that don't have them.

    This extracts action/position tags from caption text and updates the database.

    Args:
        limit: Maximum number of captions to process
        use_llm: If True, triggers LLM-based extraction via Celery task (async, more accurate).
                 If False, uses fast regex-based extraction (synchronous).
    """
    try:
        from sqlalchemy import cast, String

        if use_llm:
            # Queue LLM-based extraction as Celery task (runs on GPU worker)
            from tasks.training_tasks import extract_tags_llm_batch
            task = extract_tags_llm_batch.delay(limit=limit)
            return {
                "status": "queued",
                "task_id": task.id,
                "message": f"Queued LLM-based tag extraction for up to {limit} captions",
                "check_status": f"/generation/captions/backfill-tags/status/{task.id}"
            }

        # Synchronous regex-based extraction
        from utils.generation_postprocessor import extract_tags

        # Find captions with empty or null tags using text cast for comparison
        captions = db.query(GeneratedCaption).filter(
            (GeneratedCaption.tags == None) |
            (cast(GeneratedCaption.tags, String) == '[]')
        ).limit(limit).all()

        updated = 0
        for caption in captions:
            if caption.caption_text:
                tags = extract_tags(caption.caption_text)
                caption.tags = tags
                updated += 1

        db.commit()

        # Count remaining
        remaining = db.query(GeneratedCaption).filter(
            (GeneratedCaption.tags == None) |
            (cast(GeneratedCaption.tags, String) == '[]')
        ).count()

        return {
            "status": "success",
            "method": "regex",
            "updated": updated,
            "remaining": remaining,
            "message": f"Extracted tags for {updated} captions using regex"
        }
    except Exception as e:
        logger.error(f"Error backfilling tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/captions/backfill-tags/status/{task_id}")
async def get_backfill_tags_status(task_id: str):
    """Check status of LLM-based tag extraction task."""
    from celery.result import AsyncResult
    from tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    elif result.status == "PROGRESS":
        response["progress"] = result.info

    return response


# Quality scoring endpoints
@router.get("/captions/score-stats")
async def get_caption_score_stats(db: Session = Depends(get_db)):
    """Get statistics about caption quality scores."""
    try:
        total = db.query(GeneratedCaption).count()

        # Count captions with scores
        with_scores = db.query(GeneratedCaption).filter(
            GeneratedCaption.quality_score != None,
            GeneratedCaption.quality_score > 0
        ).count()
        without_scores = total - with_scores

        # Get score distribution
        if with_scores > 0:
            scored_captions = db.query(GeneratedCaption).filter(
                GeneratedCaption.quality_score != None,
                GeneratedCaption.quality_score > 0
            ).all()

            scores = [c.quality_score for c in scored_captions]
            avg_score = sum(scores) / len(scores)
            min_score_val = min(scores)
            max_score_val = max(scores)

            # Score buckets
            excellent = len([s for s in scores if s >= 80])  # 80-100
            good = len([s for s in scores if 60 <= s < 80])  # 60-79
            fair = len([s for s in scores if 40 <= s < 60])  # 40-59
            poor = len([s for s in scores if s < 40])  # 0-39
        else:
            avg_score = None
            min_score_val = None
            max_score_val = None
            excellent = good = fair = poor = 0

        return {
            "total_captions": total,
            "with_scores": with_scores,
            "without_scores": without_scores,
            "average_score": round(avg_score, 1) if avg_score else None,
            "min_score": min_score_val,
            "max_score": max_score_val,
            "distribution": {
                "excellent_80_100": excellent,
                "good_60_79": good,
                "fair_40_59": fair,
                "poor_0_39": poor
            }
        }
    except Exception as e:
        logger.error(f"Error getting score stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/score-batch")
async def score_captions_batch(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Score captions that don't have quality scores yet.

    This triggers an LLM-based quality evaluation that scores captions on:
    - Grammar & Writing Quality (0-30 points)
    - Sensuality & Appeal (0-40 points)
    - Story Clarity (0-30 points)

    Args:
        limit: Maximum number of captions to process
    """
    try:
        # Queue LLM-based scoring as Celery task (runs on GPU worker)
        from tasks.training_tasks import score_captions_llm_batch
        task = score_captions_llm_batch.delay(limit=limit)

        # Get count of captions needing scores
        without_scores = db.query(GeneratedCaption).filter(
            (GeneratedCaption.quality_score == None) |
            (GeneratedCaption.quality_score == 0)
        ).count()

        return {
            "status": "queued",
            "task_id": task.id,
            "message": f"Queued LLM scoring for up to {limit} captions ({without_scores} need scoring)",
            "check_status": f"/generation/captions/score-batch/status/{task.id}"
        }
    except Exception as e:
        logger.error(f"Error queuing score batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/captions/score-batch/status/{task_id}")
async def get_score_batch_status(task_id: str):
    """Check status of LLM-based scoring task."""
    from celery.result import AsyncResult
    from tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    elif result.status == "PROGRESS":
        response["progress"] = result.info

    return response


@router.post("/captions/rescore-all")
async def rescore_all_captions(
    limit: int = 100,
    min_current_score: Optional[float] = None,
    max_current_score: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Rescore existing captions using the new strict scoring system.

    This applies:
    1. Pre-rejection filter for obviously low-quality captions
    2. Stricter LLM scoring prompt with explicit penalties
    3. Rule-based deductions for specific issues (ALL CAPS, missing punctuation, etc.)

    Args:
        limit: Maximum number of captions to rescore
        min_current_score: Only rescore captions with current score >= this value
        max_current_score: Only rescore captions with current score <= this value

    Returns task_id to track progress.
    """
    try:
        # Queue rescoring as Celery task (runs on GPU worker)
        from tasks.training_tasks import rescore_captions_strict_batch
        task = rescore_captions_strict_batch.delay(
            limit=limit,
            min_current_score=min_current_score,
            max_current_score=max_current_score
        )

        # Count captions to be rescored
        query = db.query(GeneratedCaption)
        if min_current_score is not None:
            query = query.filter(GeneratedCaption.quality_score >= min_current_score)
        if max_current_score is not None:
            query = query.filter(GeneratedCaption.quality_score <= max_current_score)
        eligible_count = query.count()

        return {
            "status": "queued",
            "task_id": task.id,
            "message": f"Queued strict rescoring for up to {limit} captions ({eligible_count} eligible)",
            "check_status": f"/generation/captions/rescore-all/status/{task.id}",
            "scoring_changes": [
                "Pre-rejection filter for gibberish, excessive ALL CAPS, frame delimiters",
                "Stricter LLM prompt with explicit penalties for issues",
                "Rule-based deductions: -5 per missing ?, -3 per ALL CAPS phrase, -10 incomplete ending"
            ]
        }
    except Exception as e:
        logger.error(f"Error queuing rescore batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/captions/rescore-all/status/{task_id}")
async def get_rescore_status(task_id: str):
    """Check status of strict rescoring task."""
    from celery.result import AsyncResult
    from tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    elif result.status == "PROGRESS":
        response["progress"] = result.info

    return response


@router.get("/captions/{caption_id}")
async def get_caption(caption_id: int, db: Session = Depends(get_db)):
    """Get a specific generated caption."""
    try:
        caption = db.query(GeneratedCaption).filter_by(id=caption_id).first()
        if not caption:
            raise HTTPException(status_code=404, detail=f"Caption {caption_id} not found")

        return {
            "id": caption.id,
            "caption_text": caption.caption_text,
            "status": caption.status,
            "is_favorite": caption.is_favorite,
            "llm_model": caption.llm_model,
            "generation_prompt": caption.generation_prompt,
            "generation_job_id": caption.generation_job_id,
            "video_id": caption.video_id,
            "quality_score": caption.quality_score,
            "temperature": caption.temperature,
            "top_p": caption.top_p,
            "max_tokens": caption.max_tokens,
            "tags": caption.tags or [],  # Activity tags for background matching
            "generated_at": caption.generated_at.isoformat() if caption.generated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/captions/{caption_id}")
async def update_caption(
    caption_id: int,
    update: CaptionUpdate,
    db: Session = Depends(get_db)
):
    """Update a caption's status, favorite flag, or text."""
    try:
        caption = db.query(GeneratedCaption).filter_by(id=caption_id).first()
        if not caption:
            raise HTTPException(status_code=404, detail=f"Caption {caption_id} not found")

        if update.status is not None:
            if update.status not in ("pending_review", "approved", "rejected", "published"):
                raise HTTPException(status_code=400, detail=f"Invalid status: {update.status}")
            caption.status = update.status

        if update.is_favorite is not None:
            caption.is_favorite = update.is_favorite

        if update.caption_text is not None:
            caption.caption_text = update.caption_text

        db.commit()

        return {
            "status": "success",
            "caption": {
                "id": caption.id,
                "status": caption.status,
                "is_favorite": caption.is_favorite,
                "caption_text": caption.caption_text[:100] + "..." if len(caption.caption_text) > 100 else caption.caption_text
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/captions/{caption_id}")
async def delete_caption(caption_id: int, db: Session = Depends(get_db)):
    """Delete a generated caption."""
    try:
        caption = db.query(GeneratedCaption).filter_by(id=caption_id).first()
        if not caption:
            raise HTTPException(status_code=404, detail=f"Caption {caption_id} not found")

        db.delete(caption)
        db.commit()

        return {"status": "success", "message": f"Caption {caption_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/{caption_id}/favorite")
async def toggle_favorite(caption_id: int, db: Session = Depends(get_db)):
    """Toggle the favorite status of a caption."""
    try:
        caption = db.query(GeneratedCaption).filter_by(id=caption_id).first()
        if not caption:
            raise HTTPException(status_code=404, detail=f"Caption {caption_id} not found")

        caption.is_favorite = not caption.is_favorite
        db.commit()

        return {
            "status": "success",
            "caption_id": caption_id,
            "is_favorite": caption.is_favorite
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/bulk-action")
async def bulk_caption_action(
    request: BulkActionRequest,
    db: Session = Depends(get_db)
):
    """
    Perform bulk actions on captions.

    Actions: approve, reject, delete, favorite, unfavorite
    """
    try:
        action = request.action
        caption_ids = request.caption_ids

        if action not in ("approve", "reject", "delete", "favorite", "unfavorite"):
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

        affected = 0

        if action == "delete":
            affected = db.query(GeneratedCaption).filter(
                GeneratedCaption.id.in_(caption_ids)
            ).delete(synchronize_session=False)
        else:
            captions = db.query(GeneratedCaption).filter(
                GeneratedCaption.id.in_(caption_ids)
            ).all()

            for caption in captions:
                if action == "approve":
                    caption.status = "approved"
                elif action == "reject":
                    caption.status = "rejected"
                elif action == "favorite":
                    caption.is_favorite = True
                elif action == "unfavorite":
                    caption.is_favorite = False
                affected += 1

        db.commit()

        return {
            "status": "success",
            "action": action,
            "affected": affected
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics
# ============================================================================

@router.get("/stats")
async def get_generation_stats(db: Session = Depends(get_db)):
    """Get overall generation statistics."""
    try:
        # Job stats
        total_jobs = db.query(GenerationJob).count()
        completed_jobs = db.query(GenerationJob).filter_by(status="completed").count()
        running_jobs = db.query(GenerationJob).filter_by(status="running").count()

        # Caption stats
        total_captions = db.query(GeneratedCaption).count()
        approved_captions = db.query(GeneratedCaption).filter_by(status="approved").count()
        pending_captions = db.query(GeneratedCaption).filter_by(status="pending_review").count()
        favorite_captions = db.query(GeneratedCaption).filter_by(is_favorite=True).count()

        # Recent activity
        recent_jobs = db.query(GenerationJob).order_by(
            desc(GenerationJob.created_at)
        ).limit(5).all()

        return {
            "jobs": {
                "total": total_jobs,
                "completed": completed_jobs,
                "running": running_jobs
            },
            "captions": {
                "total": total_captions,
                "approved": approved_captions,
                "pending_review": pending_captions,
                "favorites": favorite_captions
            },
            "recent_jobs": [
                {
                    "id": j.id,
                    "job_name": j.job_name,
                    "status": j.status,
                    "captions_generated": j.captions_generated,
                    "created_at": j.created_at.isoformat() if j.created_at else None
                }
                for j in recent_jobs
            ]
        }
    except Exception as e:
        logger.error(f"Error getting generation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
