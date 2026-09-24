"""
Publishing API Endpoints

Handles publishing composed videos: media host (self-hosted URL) -> Reddit profile post -> crossposts.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from database.db import get_db
from database.models import (
    VideoPublishJob, RedditCrosspost, ComposedVideo, GeneratedCaption
)
from config.automation_config import get_automation_config

router = APIRouter(prefix="/publishing", tags=["publishing"])


# === Request/Response Models ===

class PublishRequest(BaseModel):
    """Request to publish a composed video."""
    composed_video_id: int
    title: Optional[str] = None  # Auto-populated from generated caption if not provided
    tags: Optional[str] = None  # Comma-separated tags
    crosspost_delay_minutes: int = 30


class PublishResponse(BaseModel):
    """Response from publish request."""
    job_id: int
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Publish job status response."""
    id: int
    composed_video_id: int
    title: str
    niche: Optional[str]
    status: str
    hosted_url: Optional[str]
    profile_post_url: Optional[str]
    crosspost_scheduled_at: Optional[str]
    crossposts: List[dict]
    error_message: Optional[str]
    created_at: str


# === API Endpoints ===

@router.post("/publish", response_model=PublishResponse)
def publish_video(request: PublishRequest, db: Session = Depends(get_db)):
    """
    Start the publish workflow for a composed video.

    Steps:
    1. Publish to the media host (self-hosted URL by default)
    2. Post to Reddit profile
    3. Wait 30 minutes
    4. Crosspost to all subreddits for the video's niche
    """
    # Get the composed video
    video = db.query(ComposedVideo).filter_by(id=request.composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    if video.is_published:
        raise HTTPException(status_code=400, detail="Video is already published")

    # Check for existing pending/running job
    existing = db.query(VideoPublishJob).filter(
        VideoPublishJob.composed_video_id == request.composed_video_id,
        VideoPublishJob.status.in_(["pending", "hosting_media", "posting_profile", "waiting_crosspost", "crossposting"])
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Video already has a pending publish job (id={existing.id}, status={existing.status})"
        )

    # Determine niche and title from the generated caption
    niche = video.niche
    title = request.title
    caption = None

    if video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption:
            if not niche:
                niche = caption.niche
            # Use generated title if no title provided
            if not title and caption.generated_title:
                title = caption.generated_title

    # Require a title (either provided or generated)
    if not title:
        raise HTTPException(
            status_code=400,
            detail="No title provided and no generated title available. Please provide a title."
        )

    # Build default tags from niche config
    tags = request.tags
    if not tags and niche:
        config = get_automation_config()
        niche_config = config.get_niche(niche)
        if niche_config:
            tags = ",".join(niche_config.keywords)

    # Create publish job
    job = VideoPublishJob(
        composed_video_id=request.composed_video_id,
        title=title,
        niche=niche,
        tags=tags,
        crosspost_delay_minutes=request.crosspost_delay_minutes,
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Trigger the publish task
    from tasks.publishing_tasks import publish_video_task
    task = publish_video_task.delay(job.id)

    job.celery_task_id = task.id
    db.commit()

    return PublishResponse(
        job_id=job.id,
        status="started",
        message=f"Publishing started. Crosspost scheduled in {request.crosspost_delay_minutes} minutes after profile post."
    )


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_publish_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List publish jobs with optional status filter."""
    query = db.query(VideoPublishJob).order_by(VideoPublishJob.created_at.desc())

    if status:
        query = query.filter(VideoPublishJob.status == status)

    jobs = query.limit(limit).all()

    return [_format_job(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_publish_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific publish job."""
    job = db.query(VideoPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _format_job(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_publish_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a pending or in-progress publish job."""
    job = db.query(VideoPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'"
        )

    # Revoke Celery task if running
    if job.celery_task_id:
        from tasks.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = "cancelled"
    job.completed_at = datetime.utcnow()
    db.commit()

    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry-crossposts")
def retry_failed_crossposts(job_id: int, db: Session = Depends(get_db)):
    """Retry failed crossposts for a job."""
    job = db.query(VideoPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.profile_post_id:
        raise HTTPException(status_code=400, detail="No profile post to crosspost from")

    # Count failed crossposts that can be retried
    failed = db.query(RedditCrosspost).filter(
        RedditCrosspost.publish_job_id == job_id,
        RedditCrosspost.status == "failed",
        RedditCrosspost.retry_count < 3
    ).count()

    if failed == 0:
        raise HTTPException(status_code=400, detail="No failed crossposts to retry")

    # Trigger retry task
    from tasks.publishing_tasks import retry_failed_crossposts_task
    task = retry_failed_crossposts_task.delay(job_id)

    return {"status": "started", "task_id": task.id, "crossposts_to_retry": failed}


@router.delete("/jobs/{job_id}")
def delete_publish_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a publish job (only if completed, failed, or cancelled)."""
    job = db.query(VideoPublishJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job with status '{job.status}'. Cancel it first."
        )

    db.delete(job)
    db.commit()

    return {"status": "deleted", "job_id": job_id}


@router.get("/videos/{video_id}/status")
def get_video_publish_status(video_id: int, db: Session = Depends(get_db)):
    """Get publishing status for a composed video."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")

    # Get latest publish job
    job = db.query(VideoPublishJob).filter_by(
        composed_video_id=video_id
    ).order_by(VideoPublishJob.created_at.desc()).first()

    return {
        "video_id": video_id,
        "is_published": video.is_published,
        "latest_job": _format_job(job) if job else None
    }


@router.get("/config/niches")
def get_publishing_niches(db: Session = Depends(get_db)):
    """Get niche configurations for publishing."""
    config = get_automation_config()
    niches = config.get_enabled_niches()

    return {
        "niches": [
            {
                "name": f.name,
                "subreddits": f.subreddits,
                "tags": f.tags
            }
            for f in niches
        ]
    }


@router.get("/stats")
def get_publishing_stats(db: Session = Depends(get_db)):
    """Get publishing statistics."""
    from sqlalchemy import func

    total_jobs = db.query(VideoPublishJob).count()
    by_status = db.query(
        VideoPublishJob.status,
        func.count(VideoPublishJob.id)
    ).group_by(VideoPublishJob.status).all()

    total_crossposts = db.query(RedditCrosspost).count()
    crossposts_by_status = db.query(
        RedditCrosspost.status,
        func.count(RedditCrosspost.id)
    ).group_by(RedditCrosspost.status).all()

    return {
        "jobs": {
            "total": total_jobs,
            "by_status": {status: count for status, count in by_status}
        },
        "crossposts": {
            "total": total_crossposts,
            "by_status": {status: count for status, count in crossposts_by_status}
        }
    }


def _format_job(job: VideoPublishJob) -> dict:
    """Format a publish job for API response."""
    crossposts = []
    for cp in job.crossposts:
        crossposts.append({
            "id": cp.id,
            "subreddit": cp.subreddit,
            "status": cp.status,
            "post_url": cp.post_url,
            "error_message": cp.error_message,
            "retry_count": cp.retry_count
        })

    return {
        "id": job.id,
        "composed_video_id": job.composed_video_id,
        "title": job.title,
        "niche": job.niche,
        "status": job.status,
        "hosted_url": job.hosted_url,
        "profile_post_url": job.profile_post_url,
        "crosspost_scheduled_at": job.crosspost_scheduled_at.isoformat() if job.crosspost_scheduled_at else None,
        "crossposts": crossposts,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None
    }
