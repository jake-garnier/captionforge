"""
Postpone Scheduling API

Manage composed video approval and Postpone-scheduled Reddit posting.

Endpoints:
- POST /postpone/approve - Approve composed videos for scheduling
- POST /postpone/reject - Reject composed videos
- POST /postpone/schedule-batch - Auto-schedule approved videos 1/day
- GET /postpone/jobs - List Postpone schedule jobs
- GET /postpone/jobs/{id} - Get single job detail
- POST /postpone/jobs/{id}/cancel - Cancel a pending job
- POST /postpone/jobs/{id}/retry - Retry a failed job
- GET /postpone/calendar - Calendar view of scheduled posts
- GET /postpone/health - Check Postpone API connectivity
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, cast, Date
from database.db import get_db
from database.models import ComposedVideo, PostponeScheduleJob, GeneratedCaption, BackgroundVideo
from config.settings import settings
from config.automation_config import get_automation_config
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/postpone", tags=["postpone"])


# ============================================================================
# Pydantic Models
# ============================================================================

class ApproveRequest(BaseModel):
    video_ids: List[int]


class RejectRequest(BaseModel):
    video_ids: List[int]


class ScheduleBatchRequest(BaseModel):
    niche: str
    start_date: Optional[str] = None  # ISO date string, defaults to tomorrow
    posting_hour_utc: Optional[int] = None  # Override default hour
    posting_minute: Optional[int] = None  # Override default minute
    stagger_minutes: Optional[int] = None  # Override default stagger
    max_videos: Optional[int] = None  # Cap number of videos to schedule


class ScheduleBatchResponse(BaseModel):
    scheduled_count: int
    start_date: str
    end_date: str
    jobs: List[Dict[str, Any]]


class ScheduleSingleRequest(BaseModel):
    composed_video_id: int
    scheduled_date: Optional[str] = None  # YYYY-MM-DD; defaults to next free day
    posting_hour_utc: Optional[int] = None
    posting_minute: Optional[int] = None
    stagger_minutes: Optional[int] = None


class SetHostedUrlRequest(BaseModel):
    hosted_url: str


class SetHostedUrlBatchRequest(BaseModel):
    """Set hosted video URLs for multiple jobs at once."""
    jobs: List[Dict[str, Any]]  # [{"job_id": 1, "hosted_url": "https://..."}, ...]


class PostponeJobResponse(BaseModel):
    id: int
    composed_video_id: int
    title: str
    niche: Optional[str]
    reddit_username: str
    scheduled_date: str
    base_post_time: str
    stagger_minutes: int
    target_subreddits: List[str]
    postpone_post_id: Optional[str]
    status: str
    hosted_url: Optional[str]
    error_message: Optional[str]
    retry_count: int
    created_at: datetime
    scheduled_at: Optional[datetime]
    completed_at: Optional[datetime]
    # Background video source (joined from composed_videos.background_video_id)
    bg_source_url: Optional[str] = None
    bg_source_type: Optional[str] = None
    bg_subreddit: Optional[str] = None


class ApprovalStatsResponse(BaseModel):
    pending: int
    approved: int
    scheduled: int
    rejected: int
    by_niche: Dict[str, Dict[str, int]]


# ============================================================================
# Approval Endpoints
# ============================================================================

@router.post("/approve")
async def approve_videos(request: ApproveRequest, db: Session = Depends(get_db)):
    """Approve composed videos for scheduling."""
    now = datetime.now(timezone.utc)
    updated = 0
    for vid_id in request.video_ids:
        video = db.query(ComposedVideo).filter_by(id=vid_id, status="completed").first()
        if video and video.approval_status in ("pending", "rejected"):
            video.approval_status = "approved"
            video.approved_at = now
            updated += 1

    db.commit()
    return {"approved": updated, "total_requested": len(request.video_ids)}


@router.post("/reject")
async def reject_videos(request: RejectRequest, db: Session = Depends(get_db)):
    """Reject composed videos."""
    updated = 0
    for vid_id in request.video_ids:
        video = db.query(ComposedVideo).filter_by(id=vid_id, status="completed").first()
        if video and video.approval_status in ("pending", "approved"):
            video.approval_status = "rejected"
            updated += 1

    db.commit()
    return {"rejected": updated, "total_requested": len(request.video_ids)}


@router.get("/approval-stats", response_model=ApprovalStatsResponse)
async def get_approval_stats(db: Session = Depends(get_db)):
    """Get approval status counts for composed videos."""
    videos = db.query(
        ComposedVideo.approval_status,
        ComposedVideo.niche,
        func.count(ComposedVideo.id)
    ).filter(
        ComposedVideo.status == "completed"
    ).group_by(
        ComposedVideo.approval_status,
        ComposedVideo.niche
    ).all()

    totals = {"pending": 0, "approved": 0, "scheduled": 0, "rejected": 0}
    by_niche: Dict[str, Dict[str, int]] = {}

    for approval_status, niche, count in videos:
        status_key = approval_status or "pending"
        if status_key in totals:
            totals[status_key] += count

        niche_key = niche or "unknown"
        if niche_key not in by_niche:
            by_niche[niche_key] = {"pending": 0, "approved": 0, "scheduled": 0, "rejected": 0}
        if status_key in by_niche[niche_key]:
            by_niche[niche_key][status_key] += count

    return ApprovalStatsResponse(**totals, by_niche=by_niche)


# ============================================================================
# Scheduling Endpoints
# ============================================================================

@router.post("/schedule-batch", response_model=ScheduleBatchResponse)
async def schedule_batch(request: ScheduleBatchRequest, db: Session = Depends(get_db)):
    """
    Auto-schedule approved videos 1/day across consecutive days.

    Takes all approved + unscheduled composed videos for the given niche
    and creates PostponeScheduleJob records on consecutive days.
    """
    if not settings.POSTPONE_API_KEY:
        raise HTTPException(status_code=400, detail="POSTPONE_API_KEY not configured")

    config = get_automation_config()
    niche_config = config.get_niche(request.niche)
    if not niche_config:
        raise HTTPException(status_code=400, detail=f"Unknown niche: {request.niche}")

    # Determine scheduling parameters
    postpone_config = config.postpone
    posting_hour = request.posting_hour_utc if request.posting_hour_utc is not None else postpone_config.default_posting_hour_utc
    posting_minute = request.posting_minute if request.posting_minute is not None else postpone_config.default_posting_minute
    stagger = request.stagger_minutes if request.stagger_minutes is not None else postpone_config.default_stagger_minutes
    stagger = max(10, stagger)

    # Determine Reddit username
    reddit_username = niche_config.postpone_reddit_username or postpone_config.reddit_username

    # Parse start date
    if request.start_date:
        try:
            start = date.fromisoformat(request.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    else:
        start = date.today() + timedelta(days=1)  # Default: tomorrow

    # Get approved, unscheduled videos for this niche
    query = db.query(ComposedVideo).filter(
        ComposedVideo.status == "completed",
        ComposedVideo.approval_status == "approved",
        ComposedVideo.niche == request.niche,
    ).order_by(ComposedVideo.created_at.asc())

    if request.max_videos:
        query = query.limit(request.max_videos)

    videos = query.all()

    if not videos:
        raise HTTPException(status_code=404, detail=f"No approved videos found for niche '{request.niche}'")

    # Check for existing scheduled jobs on these dates to avoid conflicts
    existing_dates = set()
    existing_jobs = db.query(PostponeScheduleJob).filter(
        PostponeScheduleJob.niche == request.niche,
        PostponeScheduleJob.status.in_(["pending", "scheduling", "scheduled"]),
    ).all()
    for job in existing_jobs:
        if job.scheduled_date:
            existing_dates.add(job.scheduled_date.date() if hasattr(job.scheduled_date, 'date') else job.scheduled_date)

    # Create jobs on consecutive days, skipping dates that already have a job
    jobs_created = []
    current_date = start
    for video in videos:
        # Skip dates that already have scheduled posts
        while current_date in existing_dates:
            current_date += timedelta(days=1)

        post_time = datetime(
            current_date.year, current_date.month, current_date.day,
            posting_hour, posting_minute,
            tzinfo=timezone.utc
        )

        job = PostponeScheduleJob(
            composed_video_id=video.id,
            title=_get_video_title(video, db),
            niche=request.niche,
            reddit_username=reddit_username,
            scheduled_date=post_time,
            base_post_time=post_time,
            stagger_minutes=stagger,
            target_subreddits=niche_config.subreddits,
            status="pending",
        )
        db.add(job)
        video.approval_status = "scheduled"

        jobs_created.append({
            "composed_video_id": video.id,
            "scheduled_date": current_date.isoformat(),
            "post_time": post_time.isoformat(),
            "subreddits": niche_config.subreddits,
        })
        existing_dates.add(current_date)
        current_date += timedelta(days=1)

    db.commit()

    end_date = (current_date - timedelta(days=1)).isoformat()
    return ScheduleBatchResponse(
        scheduled_count=len(jobs_created),
        start_date=start.isoformat(),
        end_date=end_date,
        jobs=jobs_created,
    )


@router.post("/schedule-single")
async def schedule_single(request: ScheduleSingleRequest, db: Session = Depends(get_db)):
    """
    Schedule a single approved composed video to Postpone.

    If scheduled_date is not provided, picks the next day (starting tomorrow) that
    has no existing pending/scheduling/scheduled job for the same niche.
    """
    if not settings.POSTPONE_API_KEY:
        raise HTTPException(status_code=400, detail="POSTPONE_API_KEY not configured")

    video = db.query(ComposedVideo).filter_by(id=request.composed_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Composed video not found")
    if video.status != "completed":
        raise HTTPException(status_code=400, detail="Video is not completed")
    if video.approval_status != "approved":
        raise HTTPException(status_code=400, detail=f"Video must be approved (current: {video.approval_status})")
    if not video.niche:
        raise HTTPException(status_code=400, detail="Video has no niche set")

    config = get_automation_config()
    niche_config = config.get_niche(video.niche)
    if not niche_config:
        raise HTTPException(status_code=400, detail=f"Unknown niche: {video.niche}")

    postpone_config = config.postpone
    posting_hour = request.posting_hour_utc if request.posting_hour_utc is not None else postpone_config.default_posting_hour_utc
    posting_minute = request.posting_minute if request.posting_minute is not None else postpone_config.default_posting_minute
    stagger = request.stagger_minutes if request.stagger_minutes is not None else postpone_config.default_stagger_minutes
    stagger = max(10, stagger)

    reddit_username = niche_config.postpone_reddit_username or postpone_config.reddit_username

    # Determine the scheduled date
    if request.scheduled_date:
        try:
            target_date = date.fromisoformat(request.scheduled_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid scheduled_date format. Use YYYY-MM-DD")
    else:
        # Find next free day for this niche starting tomorrow
        existing_dates = set()
        existing_jobs = db.query(PostponeScheduleJob).filter(
            PostponeScheduleJob.niche == video.niche,
            PostponeScheduleJob.status.in_(["pending", "scheduling", "scheduled"]),
        ).all()
        for job in existing_jobs:
            if job.scheduled_date:
                existing_dates.add(job.scheduled_date.date() if hasattr(job.scheduled_date, 'date') else job.scheduled_date)

        target_date = date.today() + timedelta(days=1)
        while target_date in existing_dates:
            target_date += timedelta(days=1)

    post_time = datetime(
        target_date.year, target_date.month, target_date.day,
        posting_hour, posting_minute,
        tzinfo=timezone.utc,
    )

    job = PostponeScheduleJob(
        composed_video_id=video.id,
        title=_get_video_title(video, db),
        niche=video.niche,
        reddit_username=reddit_username,
        scheduled_date=post_time,
        base_post_time=post_time,
        stagger_minutes=stagger,
        target_subreddits=niche_config.subreddits,
        status="pending",
    )
    db.add(job)
    video.approval_status = "scheduled"
    db.commit()
    db.refresh(job)

    return {
        "job_id": job.id,
        "composed_video_id": video.id,
        "scheduled_date": target_date.isoformat(),
        "post_time": post_time.isoformat(),
        "subreddits": niche_config.subreddits,
        "niche": video.niche,
    }


# ============================================================================
# Job Management Endpoints
# ============================================================================

@router.get("/jobs", response_model=List[PostponeJobResponse])
async def list_jobs(
    status: Optional[str] = None,
    niche: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List Postpone schedule jobs."""
    query = db.query(PostponeScheduleJob)
    if status:
        query = query.filter(PostponeScheduleJob.status == status)
    if niche:
        query = query.filter(PostponeScheduleJob.niche == niche)
    jobs = query.order_by(PostponeScheduleJob.scheduled_date.asc()).offset(offset).limit(limit).all()

    bg_lookup = _fetch_bg_info_for_jobs(db, jobs)
    return [_job_to_response(j, bg_lookup.get(j.composed_video_id)) for j in jobs]


@router.get("/jobs/{job_id}", response_model=PostponeJobResponse)
async def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get single Postpone schedule job."""
    job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    bg_lookup = _fetch_bg_info_for_jobs(db, [job])
    return _job_to_response(job, bg_lookup.get(job.composed_video_id))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a pending schedule job."""
    job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending",):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job.status}'")

    job.status = "cancelled"
    # Revert video approval status
    video = db.query(ComposedVideo).filter_by(id=job.composed_video_id).first()
    if video and video.approval_status == "scheduled":
        video.approval_status = "approved"

    db.commit()
    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: int, db: Session = Depends(get_db)):
    """Retry a failed schedule job."""
    job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "failed":
        raise HTTPException(status_code=400, detail=f"Can only retry failed jobs, current status: '{job.status}'")

    job.status = "pending"
    job.error_message = None
    db.commit()
    return {"status": "pending", "job_id": job_id, "retry_count": job.retry_count}


@router.post("/jobs/{job_id}/publish-now")
async def publish_now(job_id: int, db: Session = Depends(get_db)):
    """
    Immediately execute a schedule job, skipping the wait for the beat task.

    If the job has a hosted_url, dispatches scheduling directly.
    If not, publishes to the media host first (which chains to scheduling on completion).
    """
    job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot publish job with status '{job.status}'")

    # Reset failed jobs to pending
    if job.status == "failed":
        job.status = "pending"
        job.error_message = None
        db.commit()

    if job.hosted_url:
        from tasks.postpone_tasks import schedule_to_postpone_task
        schedule_to_postpone_task.delay(job.id)
        return {"status": "dispatched", "job_id": job_id, "action": "scheduling"}
    else:
        from tasks.media_host_tasks import publish_to_media_host_task
        publish_to_media_host_task.delay(job.id)
        return {"status": "dispatched", "job_id": job_id, "action": "hosting"}


def _validate_hosted_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(("https://", "http://"))


@router.post("/jobs/{job_id}/set-hosted-url")
async def set_hosted_url(job_id: int, request: SetHostedUrlRequest, db: Session = Depends(get_db)):
    """
    Manually set the hosted video URL for a pending job.

    Normally the media host task fills this in automatically; use this to
    point a job at a video you hosted elsewhere.
    """
    job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot update job with status '{job.status}'")

    if not _validate_hosted_url(request.hosted_url):
        raise HTTPException(status_code=400, detail="hosted_url must be an absolute http(s) URL")

    job.hosted_url = request.hosted_url
    if job.status == "failed":
        job.status = "pending"
        job.error_message = None
    db.commit()
    return {"status": "ok", "job_id": job_id, "hosted_url": job.hosted_url}


@router.post("/jobs/set-hosted-urls")
async def set_hosted_urls_batch(request: SetHostedUrlBatchRequest, db: Session = Depends(get_db)):
    """
    Set hosted video URLs for multiple jobs at once.

    Body: {"jobs": [{"job_id": 1, "hosted_url": "https://..."}, ...]}
    """
    updated = 0
    errors = []
    for item in request.jobs:
        job_id = item.get("job_id")
        url = item.get("hosted_url", "")

        if not job_id:
            errors.append({"job_id": None, "error": "Missing job_id"})
            continue

        if not _validate_hosted_url(url):
            errors.append({"job_id": job_id, "error": "Invalid hosted_url (must be absolute http(s) URL)"})
            continue

        job = db.query(PostponeScheduleJob).filter_by(id=job_id).first()
        if not job:
            errors.append({"job_id": job_id, "error": "Job not found"})
            continue
        if job.status not in ("pending", "failed"):
            errors.append({"job_id": job_id, "error": f"Cannot update job with status '{job.status}'"})
            continue

        job.hosted_url = url
        if job.status == "failed":
            job.status = "pending"
            job.error_message = None
        updated += 1

    db.commit()
    return {"updated": updated, "errors": errors}


# ============================================================================
# Calendar & Status Endpoints
# ============================================================================

@router.get("/calendar")
async def get_calendar(
    niche: Optional[str] = None,
    month: Optional[str] = None,  # YYYY-MM format
    db: Session = Depends(get_db)
):
    """Get calendar view of scheduled posts."""
    query = db.query(PostponeScheduleJob)
    if niche:
        query = query.filter(PostponeScheduleJob.niche == niche)
    if month:
        try:
            year, mo = month.split("-")
            start_date = date(int(year), int(mo), 1)
            if int(mo) == 12:
                end_date = date(int(year) + 1, 1, 1)
            else:
                end_date = date(int(year), int(mo) + 1, 1)
            query = query.filter(
                PostponeScheduleJob.scheduled_date >= datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc),
                PostponeScheduleJob.scheduled_date < datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc),
            )
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")

    jobs = query.order_by(PostponeScheduleJob.scheduled_date.asc()).all()

    # Group by date
    calendar: Dict[str, List[Dict]] = {}
    for job in jobs:
        day_key = job.scheduled_date.strftime("%Y-%m-%d") if job.scheduled_date else "unknown"
        if day_key not in calendar:
            calendar[day_key] = []
        calendar[day_key].append({
            "job_id": job.id,
            "composed_video_id": job.composed_video_id,
            "title": job.title,
            "niche": job.niche,
            "status": job.status,
            "subreddits": job.target_subreddits,
            "post_time": job.base_post_time.isoformat() if job.base_post_time else None,
        })

    return calendar


@router.get("/health")
async def health_check():
    """Check Postpone API connectivity."""
    if not settings.POSTPONE_API_KEY:
        return {"status": "not_configured", "message": "POSTPONE_API_KEY not set"}

    from publishers.postpone_publisher import PostponePublisher
    publisher = PostponePublisher(api_key=settings.POSTPONE_API_KEY)
    healthy = publisher.health_check()

    return {
        "status": "healthy" if healthy else "error",
        "api_key_configured": True,
    }


# ============================================================================
# Helpers
# ============================================================================

def _get_video_title(video: ComposedVideo, db: Session) -> str:
    """Get a title for the composed video from its generated caption.

    Preference order:
      1. generated_title (written by Stage 4 Claude review)
      2. first ~100 chars of the caption text (cuts at last sentence/word
         boundary). Reddit allows 300 chars; ~100 reads better.
      3. "Caption video #N" placeholder (last resort).
    """
    caption_text = video.edited_caption_text
    if video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption:
            if caption.generated_title:
                from utils.generation_postprocessor import clean_generated_title
                title = clean_generated_title(caption.generated_title)
                if title:
                    return title[:300]
            if not caption_text:
                caption_text = caption.caption_text

    if caption_text:
        text = caption_text.strip()
        if len(text) <= 100:
            return text
        # Prefer cutting at a sentence boundary near 100 chars; fall back to
        # a word boundary; final fallback is a hard cut.
        head = text[:120]
        for stop in (".", "!", "?"):
            idx = head.rfind(stop)
            if 40 <= idx <= 100:
                return head[: idx + 1]
        space_idx = text.rfind(" ", 60, 100)
        if space_idx >= 60:
            return text[:space_idx] + "..."
        return text[:100] + "..."

    return f"Caption video #{video.id}"


def _job_to_response(job: PostponeScheduleJob, bg_info: Optional[Dict[str, Any]] = None) -> PostponeJobResponse:
    """Convert job model to response. bg_info has keys source_url/source_type/subreddit."""
    bg_info = bg_info or {}
    return PostponeJobResponse(
        id=job.id,
        composed_video_id=job.composed_video_id,
        title=job.title,
        niche=job.niche,
        reddit_username=job.reddit_username,
        scheduled_date=job.scheduled_date.strftime("%Y-%m-%d") if job.scheduled_date else "",
        base_post_time=job.base_post_time.isoformat() if job.base_post_time else "",
        stagger_minutes=job.stagger_minutes or 10,
        target_subreddits=job.target_subreddits or [],
        postpone_post_id=job.postpone_post_id,
        status=job.status,
        hosted_url=job.hosted_url,
        error_message=job.error_message,
        retry_count=job.retry_count or 0,
        created_at=job.created_at,
        scheduled_at=job.scheduled_at,
        completed_at=job.completed_at,
        bg_source_url=bg_info.get("source_url"),
        bg_source_type=bg_info.get("source_type"),
        bg_subreddit=bg_info.get("subreddit"),
    )


def _fetch_bg_info_for_jobs(db: Session, jobs) -> Dict[int, Dict[str, Any]]:
    """One-shot join: composed_video_id -> {source_url, source_type, subreddit}.

    Avoids N+1 round-trips when listing many jobs.
    """
    composed_ids = [j.composed_video_id for j in jobs if j.composed_video_id]
    if not composed_ids:
        return {}

    rows = (
        db.query(
            ComposedVideo.id,
            BackgroundVideo.source_url,
            BackgroundVideo.source_type,
            BackgroundVideo.reddit_subreddit,
        )
        .join(BackgroundVideo, ComposedVideo.background_video_id == BackgroundVideo.id)
        .filter(ComposedVideo.id.in_(composed_ids))
        .all()
    )
    return {
        composed_id: {
            "source_url": source_url,
            "source_type": source_type,
            "subreddit": subreddit,
        }
        for composed_id, source_url, source_type, subreddit in rows
    }
