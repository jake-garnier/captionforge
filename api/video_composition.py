"""
Video Composition API - Create caption videos from backgrounds + text

Endpoints for:
- Creating composition jobs (batch video generation)
- Listing and managing composed videos
- Preview timing without rendering
- Single video composition
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database.db import get_db
from database.models import (
    VideoCompositionJob, ComposedVideo, GeneratedCaption,
    ScrapedCaption, BackgroundVideo
)
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import os
from utils.generation_postprocessor import clean_generated_title
from config.niche_rules import is_subreddit_compatible
from video_generator import TextChunker, TimingEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/composition", tags=["video-composition"])


# ============================================================================
# Pydantic Models
# ============================================================================

class CompositionJobCreate(BaseModel):
    """Request model for creating a composition job."""
    job_name: str = Field(..., description="Name for this composition job")
    caption_source: str = Field(default="generated", description="'generated' or 'scraped'")
    caption_status_filter: Optional[str] = Field(default="approved", description="Status filter for generated captions")
    min_upvotes: int = Field(default=0, description="Min upvotes for scraped captions")
    background_tag_filter: Optional[str] = Field(default=None, description="Filter backgrounds by source tag (subreddit)")
    target_count: int = Field(default=10, ge=1, le=100, description="Number of videos to compose")
    niche: Optional[str] = Field(default=None, description="Niche for compatibility filtering (e.g., 'motivation', 'cooking')")


class CompositionJobResponse(BaseModel):
    """Response model for composition job details."""
    id: int
    job_name: str
    caption_source: str
    caption_status_filter: Optional[str]
    min_upvotes: int
    background_tag_filter: Optional[str]
    target_count: int
    niche: Optional[str]
    status: str
    progress_percent: float
    videos_composed: int
    videos_failed: int
    videos_skipped: int
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class ComposedVideoResponse(BaseModel):
    """Response model for composed video details."""
    id: int
    composition_job_id: Optional[int]
    generated_caption_id: Optional[int]
    scraped_caption_id: Optional[int]
    background_video_id: int
    storage_path: str
    file_size_bytes: Optional[int]
    duration_seconds: Optional[float]
    resolution: Optional[str]
    caption_chunks: Optional[int]
    status: str
    is_favorite: bool
    is_published: bool
    created_at: datetime
    # Include caption text for display
    caption_text: Optional[str] = None
    # Edited caption text (if user modified it)
    edited_caption_text: Optional[str] = None
    # Niche and tag metadata
    niche: Optional[str] = None
    caption_tags: Optional[List[str]] = None
    background_tags: Optional[List[str]] = None  # Source tags of the background video
    quality_score: Optional[float] = None
    # LLM-generated title for publishing
    generated_title: Optional[str] = None
    # Background video source
    background_source_url: Optional[str] = None
    background_subreddit: Optional[str] = None
    # Reddit posting fields
    hosted_url: Optional[str] = None  # Public media-host URL (set once published)
    reddit_post_url: Optional[str] = None
    reddit_posted_at: Optional[datetime] = None
    # Postpone approval workflow
    approval_status: Optional[str] = None
    approved_at: Optional[datetime] = None
    # Stage 4 Claude Code visual review
    claude_review_status: Optional[str] = None
    claude_review_verdict: Optional[str] = None
    claude_review_scores: Optional[dict] = None
    claude_review_issues: Optional[list] = None
    # Stage 3 LLM judge — pulled from the linked generated_caption row
    judge_pass: Optional[bool] = None
    judge_scores: Optional[dict] = None
    judge_issues: Optional[list] = None


class SingleComposeRequest(BaseModel):
    """Request for composing a single video."""
    caption_text: str = Field(..., description="Caption text to overlay")
    background_video_id: int = Field(..., description="ID of background video to use")


class TimingPreviewRequest(BaseModel):
    """Request for previewing timing without rendering."""
    caption_text: str = Field(..., description="Caption text to analyze")


class UpdateCaptionRequest(BaseModel):
    """Request for updating caption text on a composed video."""
    caption_text: str = Field(..., description="New caption text to use")


class RecomposeRequest(BaseModel):
    """Request for recomposing a video with a new background."""
    background_video_id: Optional[int] = Field(default=None, description="Specific background video ID to use")
    background_tag_filter: Optional[str] = Field(default=None, description="Filter for new background video tag")
    use_same_background: bool = Field(default=False, description="Recompose with same background video")


class TimingPreviewResponse(BaseModel):
    """Response with timing information."""
    chunks: List[dict]
    total_duration: float
    chunk_count: int


# ============================================================================
# Composition Jobs
# ============================================================================

@router.post("/jobs", response_model=CompositionJobResponse)
async def create_composition_job(
    request: CompositionJobCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new video composition job.

    This will queue a Celery task to compose multiple videos by
    matching captions with compatible background videos.
    """
    try:
        # Validate caption source
        if request.caption_source not in ("generated", "scraped"):
            raise HTTPException(
                status_code=400,
                detail="caption_source must be 'generated' or 'scraped'"
            )

        # Create job record
        job = VideoCompositionJob(
            job_name=request.job_name,
            caption_source=request.caption_source,
            caption_status_filter=request.caption_status_filter,
            min_upvotes=request.min_upvotes,
            background_tag_filter=request.background_tag_filter,
            target_count=request.target_count,
            niche=request.niche,
            status="queued"
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Queue Celery task
        from tasks.video_composition_tasks import run_composition_job
        task = run_composition_job.delay(job.id)

        job.celery_task_id = task.id
        db.commit()

        logger.info(f"Created composition job {job.id}: {job.job_name}")

        return _job_to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating composition job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[CompositionJobResponse])
async def list_composition_jobs(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List composition jobs, optionally filtered by status."""
    try:
        query = db.query(VideoCompositionJob)

        if status:
            query = query.filter(VideoCompositionJob.status == status)

        jobs = query.order_by(desc(VideoCompositionJob.created_at)).limit(limit).all()

        return [_job_to_response(job) for job in jobs]

    except Exception as e:
        logger.error(f"Error listing composition jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=CompositionJobResponse)
async def get_composition_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific composition job."""
    job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_to_response(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_composition_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a running composition job."""
    job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ("completed", "cancelled", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job.status}")

    job.status = "cancelled"
    job.completed_at = datetime.utcnow()
    db.commit()

    # Revoke Celery task if running
    if job.celery_task_id:
        from tasks.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    return {"status": "cancelled", "job_id": job_id}


@router.delete("/jobs/{job_id}")
async def delete_composition_job(
    job_id: int,
    delete_files: bool = Query(True, description="Also delete video files from disk"),
    db: Session = Depends(get_db)
):
    """
    Delete a composition job and all its composed videos.

    Args:
        job_id: The job ID to delete
        delete_files: If True, also delete video files from disk
    """
    job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get all composed videos for this job
    videos = db.query(ComposedVideo).filter_by(composition_job_id=job_id).all()
    deleted_files = 0
    failed_deletes = 0

    # Delete video files if requested
    if delete_files:
        for video in videos:
            if video.storage_path and os.path.exists(video.storage_path):
                try:
                    os.remove(video.storage_path)
                    deleted_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete file {video.storage_path}: {e}")
                    failed_deletes += 1

    # Delete composed video records
    videos_deleted = len(videos)
    for video in videos:
        db.delete(video)

    # Delete the job
    db.delete(job)
    db.commit()

    logger.info(f"Deleted composition job {job_id} with {videos_deleted} videos")

    return {
        "status": "deleted",
        "job_id": job_id,
        "videos_deleted": videos_deleted,
        "files_deleted": deleted_files,
        "files_failed": failed_deletes
    }


# ============================================================================
# Composed Videos
# ============================================================================

@router.get("/videos", response_model=List[ComposedVideoResponse])
async def list_composed_videos(
    job_id: Optional[int] = None,
    niche: Optional[str] = None,
    is_favorite: Optional[bool] = None,
    is_published: Optional[bool] = None,
    approval_status: Optional[str] = None,
    claude_review_verdict: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List composed videos with optional filters."""
    try:
        query = db.query(ComposedVideo)

        if job_id is not None:
            query = query.filter(ComposedVideo.composition_job_id == job_id)
        if niche is not None:
            query = query.filter(ComposedVideo.niche == niche)
        if is_favorite is not None:
            query = query.filter(ComposedVideo.is_favorite == is_favorite)
        if is_published is not None:
            query = query.filter(ComposedVideo.is_published == is_published)
        if approval_status is not None:
            query = query.filter(ComposedVideo.approval_status == approval_status)
        if claude_review_verdict is not None:
            query = query.filter(ComposedVideo.claude_review_verdict == claude_review_verdict)

        query = query.filter(ComposedVideo.status == "completed")
        videos = query.order_by(desc(ComposedVideo.created_at)).offset(offset).limit(limit).all()

        return [_video_to_response(v, db) for v in videos]

    except Exception as e:
        logger.error(f"Error listing composed videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}", response_model=ComposedVideoResponse)
async def get_composed_video(video_id: int, db: Session = Depends(get_db)):
    """Get details of a specific composed video."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return _video_to_response(video, db)


@router.get("/videos/{video_id}/stream")
async def stream_composed_video(video_id: int, db: Session = Depends(get_db)):
    """Stream a composed video file."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.storage_path or not os.path.exists(video.storage_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video.storage_path,
        media_type="video/mp4",
        filename=f"composed_{video_id}.mp4"
    )


@router.get("/swipe-queue")
async def swipe_queue(
    niche: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    """
    Stage 5 swipe queue.

    Returns a batch of pending composed videos to review one at a time.
    Gated on Stage 4: only videos with claude_review_verdict in
    ('pass', 'maybe') are eligible. 'fail' and unreviewed items are
    hidden until Claude review runs.

    Order within the eligible set: pass before maybe, then Stage 3
    judge_pass, then quality_score desc.
    """
    from sqlalchemy import case
    from database.models import GeneratedCaption

    query = db.query(ComposedVideo).filter(
        ComposedVideo.status == "completed",
        # Only pending — once swiped (approved/scheduled/rejected) we don't show it again.
        (ComposedVideo.approval_status == "pending") | (ComposedVideo.approval_status.is_(None)),
        # Stage 4 gate: only surface videos Claude has reviewed and not failed.
        ComposedVideo.claude_review_verdict.in_(["pass", "maybe"]),
    )
    if niche:
        query = query.filter(ComposedVideo.niche == niche)

    # Tiered ordering:
    # 1) Claude verdict: pass > maybe > unreviewed > fail
    # 2) Stage 3 judge_pass
    # 3) quality_score (NULLs last)
    # We don't use ->>'overall' from JSON because the column is mapped as
    # generic JSON, not JSONB; .astext breaks on that. judge_pass alone
    # is a good enough secondary signal.
    claude_rank = case(
        (ComposedVideo.claude_review_verdict == "pass", 3),
        (ComposedVideo.claude_review_verdict == "maybe", 2),
        (ComposedVideo.claude_review_verdict.is_(None), 1),
        (ComposedVideo.claude_review_verdict == "fail", 0),
        else_=1,
    )

    judge_rank = case(
        (GeneratedCaption.judge_pass.is_(True), 1),
        else_=0,
    )

    query = query.outerjoin(
        GeneratedCaption,
        ComposedVideo.generated_caption_id == GeneratedCaption.id,
    ).order_by(
        claude_rank.desc(),
        judge_rank.desc(),
        GeneratedCaption.quality_score.desc().nullslast(),
        ComposedVideo.id.desc(),
    )

    rows = query.limit(limit).all()
    return [_video_to_response(v, db) for v in rows]


@router.get("/stage4-pending-count")
async def stage4_pending_count(
    niche: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Count of composed videos eligible for Stage 4 (Claude visual review).

    Mirrors the filter in scripts/claude_review_videos.py export_pending so
    the swipe-tab counter matches what the export command would actually
    grab on the next run: completed videos that have not yet been reviewed
    by Claude AND haven't been hard-rejected upstream.
    """
    query = db.query(func.count(ComposedVideo.id)).filter(
        ComposedVideo.status == "completed",
        ComposedVideo.claude_review_status.is_(None),
        ComposedVideo.approval_status.in_(["pending", "approved"]),
    )
    if niche:
        query = query.filter(ComposedVideo.niche == niche)
    return {"niche": niche, "count": query.scalar() or 0}


@router.get("/debug-frames")
async def list_debug_frames():
    """List available debug frames."""
    # Check both locations - primary and copied
    debug_dirs = ["/data/debug_frames", "/data/composed_videos/debug_frames"]
    for debug_dir in debug_dirs:
        if os.path.exists(debug_dir):
            frames = sorted([f for f in os.listdir(debug_dir) if f.endswith('.png')])
            return {"frames": frames, "count": len(frames), "location": debug_dir}

    return {"frames": [], "message": "No debug frames directory"}


@router.get("/debug-frames/{frame_name}")
async def get_debug_frame(frame_name: str):
    """Get a specific debug frame image."""
    # Check both locations
    debug_dirs = ["/data/debug_frames", "/data/composed_videos/debug_frames"]
    frame_path = None
    for debug_dir in debug_dirs:
        path = os.path.join(debug_dir, frame_name)
        if os.path.exists(path):
            frame_path = path
            break

    if not frame_path:
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(
        frame_path,
        media_type="image/png",
        filename=frame_name
    )


@router.patch("/videos/{video_id}")
async def update_composed_video(
    video_id: int,
    is_favorite: Optional[bool] = None,
    is_published: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Update composed video properties (favorite, published status)."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if is_favorite is not None:
        video.is_favorite = is_favorite
    if is_published is not None:
        video.is_published = is_published

    db.commit()

    return {"status": "updated", "video_id": video_id}


@router.put("/videos/{video_id}/caption")
async def update_video_caption(
    video_id: int,
    request: UpdateCaptionRequest,
    db: Session = Depends(get_db)
):
    """
    Update the caption text for a composed video.

    The edited text is stored separately and will be used when recomposing.
    The original caption source (generated/scraped) is preserved.
    """
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video.edited_caption_text = request.caption_text
    db.commit()

    logger.info(f"Updated caption for composed video {video_id}")

    return {
        "status": "updated",
        "video_id": video_id,
        "edited_caption_text": request.caption_text
    }


def _ml_tag_similarity(prev_tags: dict, candidate_tags: dict) -> float:
    """
    Score a candidate BG's ml_tags against a reference (previous) BG's ml_tags.
    Higher = more similar. Used by the recompose 'New BG' path so the picked
    BG resembles what the user just rejected — same flavor, different clip —
    not a random niche-compatible BG.

    Schema (tasks/ml_tagging.py output): {subjects:[{type, description}],
    activities:[...], setting, mood, camera, text_on_screen}.
    """
    if not isinstance(prev_tags, dict) or not isinstance(candidate_tags, dict):
        return 0.0
    score = 0.0

    # Activities overlap (+5/match) — strongest signal: e.g. running vs cooking vs hiking.
    prev_actions = set(prev_tags.get('activities') or [])
    cand_actions = set(candidate_tags.get('activities') or [])
    if prev_actions and cand_actions:
        score += 5 * len(prev_actions & cand_actions)

    # Subject types overlap (+3/match) — keeps e.g. person/food/landscape footage together.
    def _subject_types(tags: dict) -> set:
        return {
            s.get('type') for s in (tags.get('subjects') or [])
            if isinstance(s, dict) and s.get('type')
        }
    prev_types = _subject_types(prev_tags)
    cand_types = _subject_types(candidate_tags)
    if prev_types and cand_types:
        score += 3 * len(prev_types & cand_types)

    # Same mood (+2) — an energetic clip is replaced by another energetic clip.
    if prev_tags.get('mood') and prev_tags.get('mood') == candidate_tags.get('mood'):
        score += 2

    # Same setting (+1) — kitchen, gym, outdoors, etc. Tiebreaker.
    if prev_tags.get('setting') and prev_tags.get('setting') == candidate_tags.get('setting'):
        score += 1

    # Same camera style (+1) — drone stays drone, handheld stays handheld.
    if prev_tags.get('camera') and prev_tags.get('camera') == candidate_tags.get('camera'):
        score += 1

    return score


@router.post("/videos/{video_id}/recompose")
async def recompose_video(
    video_id: int,
    request: RecomposeRequest = None,
    db: Session = Depends(get_db)
):
    """
    Recompose a video with a new background, keeping the same caption.

    Uses edited_caption_text if available, otherwise falls back to original caption.
    Selects a new unused background video (optionally filtered by tag).
    The old composed video file is deleted and replaced.
    """
    try:
        # Get the existing composed video
        video = db.query(ComposedVideo).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get the caption text to use (edited or original)
        caption_text = video.edited_caption_text
        if not caption_text:
            # Fall back to original caption
            if video.generated_caption_id:
                caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
                if caption:
                    caption_text = caption.caption_text
            elif video.scraped_caption_id:
                caption = db.query(ScrapedCaption).filter_by(id=video.scraped_caption_id).first()
                if caption:
                    caption_text = caption.llm_refined_text or caption.caption_text

        if not caption_text:
            raise HTTPException(status_code=400, detail="No caption text found for recomposition")

        # Calculate required duration for caption
        chunker = TextChunker()
        timing = TimingEngine()
        chunks = chunker.chunk(caption_text)
        required_duration = timing.get_total_duration(chunks) if chunks else 0
        logger.info(f"Recompose video {video_id}: caption requires {required_duration:.1f}s")

        # Check background selection mode
        specific_bg_id = request.background_video_id if request else None
        use_same_background = request.use_same_background if request else False

        if specific_bg_id:
            # Use a specific background video by ID
            new_bg = db.query(BackgroundVideo).filter_by(id=specific_bg_id).first()
            if not new_bg:
                raise HTTPException(status_code=404, detail=f"Background video {specific_bg_id} not found")
            if not new_bg.storage_path or not os.path.exists(new_bg.storage_path):
                raise HTTPException(status_code=404, detail="Background video file not found")
            if new_bg.duration_seconds and new_bg.duration_seconds < required_duration:
                raise HTTPException(
                    status_code=400,
                    detail=f"Background video ({new_bg.duration_seconds:.1f}s) is too short for caption ({required_duration:.1f}s)."
                )
        elif use_same_background and video.background_video_id:
            # Use the same background video
            new_bg = db.query(BackgroundVideo).filter_by(id=video.background_video_id).first()
            if not new_bg:
                raise HTTPException(status_code=404, detail="Original background video not found")
            if not new_bg.storage_path or not os.path.exists(new_bg.storage_path):
                raise HTTPException(status_code=404, detail="Background video file not found")
            # Check if background is still long enough (caption may have been edited longer)
            if new_bg.duration_seconds and new_bg.duration_seconds < required_duration:
                raise HTTPException(
                    status_code=400,
                    detail=f"Background video ({new_bg.duration_seconds:.1f}s) is too short for edited caption ({required_duration:.1f}s). Use 'New BG' to find a longer one."
                )
        else:
            # Find a new unused background video with niche compatibility enforcement
            # Get IDs of all backgrounds already used in composed videos
            used_bg_ids = db.query(ComposedVideo.background_video_id).filter(
                ComposedVideo.background_video_id.isnot(None),
                ComposedVideo.id != video_id  # Exclude current video (freeing its background)
            ).distinct().all()
            used_bg_ids = [r[0] for r in used_bg_ids]

            # Also exclude the CURRENT video's background to ensure we get a DIFFERENT one
            current_bg_id = video.background_video_id
            if current_bg_id and current_bg_id not in used_bg_ids:
                used_bg_ids.append(current_bg_id)

            # Query for available backgrounds with sufficient duration
            bg_query = db.query(BackgroundVideo).filter(
                BackgroundVideo.filter_status == 'approved',
                BackgroundVideo.storage_path.isnot(None),
                BackgroundVideo.source_type == 'reddit',  # Only use Reddit background videos
                BackgroundVideo.duration_seconds >= required_duration  # Must be long enough for caption
            )

            if used_bg_ids:
                bg_query = bg_query.filter(~BackgroundVideo.id.in_(used_bg_ids))

            # Apply tag filter if specified
            tag_filter = request.background_tag_filter if request else None
            if tag_filter:
                bg_query = bg_query.filter(BackgroundVideo.tags.contains([tag_filter]))

            # Get all matching backgrounds for niche compatibility scoring
            candidate_backgrounds = bg_query.all()

            if not candidate_backgrounds:
                # Check if it's a duration issue
                any_long_enough = db.query(BackgroundVideo).filter(
                    BackgroundVideo.filter_status == 'approved',
                    BackgroundVideo.source_type == 'reddit',
                    BackgroundVideo.duration_seconds >= required_duration
                ).first()
                if not any_long_enough:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No background videos long enough. Caption needs {required_duration:.1f}s."
                    )
                raise HTTPException(
                    status_code=404,
                    detail="No available background videos found. All approved backgrounds are in use."
                )

            # Pull the previous BG's ml_tags so we can score candidates by
            # similarity. When the user hits "New BG" they're saying "this
            # caption-BG combo was right in spirit but the specific clip was
            # off" — so the replacement should resemble it (same action,
            # participants, position) rather than be a random pick.
            prev_bg_tags = None
            if video.background_video_id:
                prev_bg = db.query(BackgroundVideo).filter_by(id=video.background_video_id).first()
                if prev_bg and isinstance(prev_bg.ml_tags, dict):
                    prev_bg_tags = prev_bg.ml_tags

            # Filter by niche compatibility if video has a niche
            niche = video.niche
            if niche:
                compatible_backgrounds = []
                for bg in candidate_backgrounds:
                    if not bg.storage_path or not os.path.exists(bg.storage_path):
                        continue
                    if is_subreddit_compatible(bg.searched_tag, niche):
                        score = min(bg.views / 10000, 10) if bg.views else 0
                        if prev_bg_tags and isinstance(bg.ml_tags, dict):
                            score += _ml_tag_similarity(prev_bg_tags, bg.ml_tags)
                        compatible_backgrounds.append((bg, score))

                if not compatible_backgrounds:
                    logger.warning(
                        f"Recompose video {video_id}: No compatible backgrounds for niche '{niche}'"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"No available backgrounds compatible with {niche}. "
                               f"All {len(candidate_backgrounds)} candidates were from incompatible subreddits."
                    )

                # Sort by combined score (niche-compat + ml-tag similarity + views).
                # Pick from a tighter top-N when we have a similarity anchor so
                # the "same kind of clip" intent isn't diluted.
                compatible_backgrounds.sort(key=lambda x: x[1], reverse=True)
                top_n = 5 if prev_bg_tags else 10
                top_backgrounds = compatible_backgrounds[:top_n]
                import random
                new_bg = random.choice(top_backgrounds)[0]
                anchor_note = f"anchored to prev bg {video.background_video_id} ml_tags" if prev_bg_tags else "no ml_tag anchor"
                logger.info(
                    f"Recompose video {video_id}: Selected bg {new_bg.id} ({anchor_note}, "
                    f"top_n={top_n}) from {len(compatible_backgrounds)} compatible options"
                )
            else:
                # No niche — still anchor to prev BG ml_tags if we have them.
                import random
                valid_candidates = [bg for bg in candidate_backgrounds
                                   if bg.storage_path and os.path.exists(bg.storage_path)]
                if not valid_candidates:
                    raise HTTPException(status_code=404, detail="No valid background video files found")
                if prev_bg_tags:
                    scored = []
                    for bg in valid_candidates:
                        sim = _ml_tag_similarity(prev_bg_tags, bg.ml_tags) if isinstance(bg.ml_tags, dict) else 0.0
                        views_score = min(bg.views / 10000, 10) if bg.views else 0
                        scored.append((bg, sim + views_score))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    new_bg = random.choice(scored[:5])[0]
                else:
                    new_bg = random.choice(valid_candidates)

            if not new_bg.storage_path or not os.path.exists(new_bg.storage_path):
                raise HTTPException(status_code=404, detail="Background video file not found")

        # NOTE: We do NOT delete the old file here. If the queued task fails
        # or never runs, deleting up-front orphans the DB row (path points
        # to a missing file). The recompose task itself deletes the old
        # file only after the new render succeeds — see
        # recompose_video_task in tasks/video_composition_tasks.py.

        # Queue recomposition task
        from tasks.video_composition_tasks import recompose_video_task
        task = recompose_video_task.delay(
            composed_video_id=video_id,
            caption_text=caption_text,
            background_video_id=new_bg.id,
            background_video_path=new_bg.storage_path
        )

        return {
            "status": "queued",
            "task_id": task.id,
            "video_id": video_id,
            "new_background_id": new_bg.id,
            "same_background": use_same_background,
            "message": "Recomposition started with same background" if use_same_background else "Recomposition started with new background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting recomposition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos/{video_id}/append-outro")
async def append_outro_endpoint(video_id: int, db: Session = Depends(get_db)):
    """
    Queue a CTA/watermark outro append to a composed video.

    Used by the Reddit publish flow: before approving for Reddit, the swipe
    UI calls this to bake the per-niche CTA onto the end of the video.
    The new mp4 replaces storage_path. Patreon/Telegram-only publishes do
    NOT call this (the CTA promotes Patreon — would be self-referential).

    No-op success when the video has no niche or its niche has no
    reddit_outro_text configured. The caller can poll /task/{task_id}.
    """
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.storage_path or not os.path.exists(video.storage_path):
        raise HTTPException(status_code=404, detail="Composed video file not found")

    from tasks.video_composition_tasks import append_outro_task
    task = append_outro_task.delay(composed_video_id=video_id)

    return {
        "status": "queued",
        "task_id": task.id,
        "video_id": video_id,
    }


@router.post("/videos/{video_id}/restyle")
async def restyle_video(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Restyle a video with a new random text style.

    Keeps the same caption and background, but re-renders with a different
    random style preset (font, color, outline, glow, etc.).
    """
    try:
        # Get the existing composed video
        video = db.query(ComposedVideo).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Verify background exists
        bg = db.query(BackgroundVideo).filter_by(id=video.background_video_id).first()
        if not bg or not bg.storage_path or not os.path.exists(bg.storage_path):
            raise HTTPException(status_code=404, detail="Background video not found")

        # Queue restyle task
        from tasks.video_composition_tasks import restyle_video_task
        task = restyle_video_task.delay(composed_video_id=video_id)

        return {
            "status": "queued",
            "task_id": task.id,
            "video_id": video_id,
            "message": "Restyle started - video will be re-rendered with a new random text style"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting restyle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/videos/{video_id}")
async def delete_composed_video(
    video_id: int,
    delete_file: bool = Query(True, description="Also delete the video file"),
    db: Session = Depends(get_db)
):
    """Delete a composed video."""
    video = db.query(ComposedVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete file if requested
    if delete_file and video.storage_path and os.path.exists(video.storage_path):
        try:
            os.remove(video.storage_path)
        except Exception as e:
            logger.warning(f"Could not delete file {video.storage_path}: {e}")

    db.delete(video)
    db.commit()

    return {"status": "deleted", "video_id": video_id}


# ============================================================================
# Single Video Composition
# ============================================================================

@router.post("/single")
async def compose_single_video(
    request: SingleComposeRequest,
    db: Session = Depends(get_db)
):
    """
    Compose a single video from caption text and background video.

    This is a synchronous operation for testing/preview purposes.
    For batch processing, use the jobs API.
    """
    try:
        # Get background video
        bg_video = db.query(BackgroundVideo).filter_by(id=request.background_video_id).first()
        if not bg_video:
            raise HTTPException(status_code=404, detail="Background video not found")

        if not bg_video.storage_path or not os.path.exists(bg_video.storage_path):
            raise HTTPException(status_code=404, detail="Background video file not found")

        # Queue composition task
        from tasks.video_composition_tasks import compose_single_video_task
        task = compose_single_video_task.delay(
            caption_text=request.caption_text,
            background_video_path=bg_video.storage_path,
            background_video_id=bg_video.id
        )

        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Video composition started"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting single composition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Timing Preview
# ============================================================================

@router.post("/preview-timing", response_model=TimingPreviewResponse)
async def preview_timing(request: TimingPreviewRequest):
    """
    Preview how a caption will be chunked and timed without rendering.

    Useful for checking if a caption will fit in a given background video duration.
    """
    try:
        from video_generator import TextChunker, TimingEngine

        chunker = TextChunker()
        timing = TimingEngine()

        chunks = chunker.chunk(request.caption_text)
        timed_chunks = timing.calculate_timings(chunks)

        return TimingPreviewResponse(
            chunks=[
                {
                    "text": tc.text,
                    "start": round(tc.start_time, 2),
                    "end": round(tc.end_time, 2),
                    "duration": round(tc.duration, 2)
                }
                for tc in timed_chunks
            ],
            total_duration=round(timed_chunks[-1].end_time, 2) if timed_chunks else 0,
            chunk_count=len(chunks)
        )

    except Exception as e:
        logger.error(f"Error previewing timing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_composition_stats(db: Session = Depends(get_db)):
    """Get composition statistics including background availability."""
    try:
        total_jobs = db.query(func.count(VideoCompositionJob.id)).scalar() or 0
        completed_jobs = db.query(func.count(VideoCompositionJob.id)).filter(
            VideoCompositionJob.status == "completed"
        ).scalar() or 0

        total_videos = db.query(func.count(ComposedVideo.id)).scalar() or 0
        favorite_videos = db.query(func.count(ComposedVideo.id)).filter(
            ComposedVideo.is_favorite == True
        ).scalar() or 0
        published_videos = db.query(func.count(ComposedVideo.id)).filter(
            ComposedVideo.is_published == True
        ).scalar() or 0

        # Background availability stats (Reddit only)
        total_approved_backgrounds = db.query(func.count(BackgroundVideo.id)).filter(
            BackgroundVideo.filter_status == 'approved',
            BackgroundVideo.source_type == 'reddit'
        ).scalar() or 0

        # Get unique backgrounds used in composed videos
        used_background_ids = db.query(ComposedVideo.background_video_id).filter(
            ComposedVideo.background_video_id.isnot(None)
        ).distinct().all()
        backgrounds_used = len(used_background_ids)
        backgrounds_available = total_approved_backgrounds - backgrounds_used

        return {
            "jobs": {
                "total": total_jobs,
                "completed": completed_jobs
            },
            "videos": {
                "total": total_videos,
                "favorites": favorite_videos,
                "published": published_videos
            },
            "backgrounds": {
                "total_approved": total_approved_backgrounds,
                "used": backgrounds_used,
                "available": backgrounds_available
            }
        }

    except Exception as e:
        logger.error(f"Error getting composition stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def _job_to_response(job: VideoCompositionJob) -> CompositionJobResponse:
    """Convert job model to response."""
    return CompositionJobResponse(
        id=job.id,
        job_name=job.job_name,
        caption_source=job.caption_source,
        caption_status_filter=job.caption_status_filter,
        min_upvotes=job.min_upvotes,
        background_tag_filter=job.background_tag_filter,
        target_count=job.target_count,
        niche=job.niche,
        status=job.status,
        progress_percent=job.progress_percent or 0.0,
        videos_composed=job.videos_composed or 0,
        videos_failed=job.videos_failed or 0,
        videos_skipped=job.videos_skipped or 0,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


def _video_to_response(video: ComposedVideo, db: Session) -> ComposedVideoResponse:
    """Convert video model to response with caption text and metadata."""
    caption_text = None
    caption_tags = None
    quality_score = None
    generated_title = None
    judge_pass = None
    judge_scores = None
    judge_issues = None

    if video.generated_caption_id:
        caption = db.query(GeneratedCaption).filter_by(id=video.generated_caption_id).first()
        if caption:
            caption_text = caption.caption_text
            caption_tags = caption.tags or []
            quality_score = caption.quality_score
            # Clean the title to remove OCR artifacts and promotional spam
            generated_title = clean_generated_title(caption.generated_title) if caption.generated_title else None
            # Stage 3 LLM judge fields, surfaced for the swipe UI badge.
            judge_pass = caption.judge_pass
            judge_scores = caption.judge_scores
            judge_issues = caption.judge_issues or []
    elif video.scraped_caption_id:
        caption = db.query(ScrapedCaption).filter_by(id=video.scraped_caption_id).first()
        if caption:
            caption_text = caption.llm_refined_text or caption.caption_text

    # Get background video tags, source URL, and subreddit
    background_tags = None
    background_source_url = None
    background_subreddit = None
    if video.background_video_id:
        bg_video = db.query(BackgroundVideo).filter_by(id=video.background_video_id).first()
        if bg_video:
            # Show ml_tags activities (VLM scene tags) instead of raw source tags
            if bg_video.ml_tags and bg_video.ml_tags.get('activities'):
                background_tags = bg_video.ml_tags['activities']
            elif bg_video.tags:
                background_tags = bg_video.tags if isinstance(bg_video.tags, list) else []
            background_source_url = bg_video.source_url
            background_subreddit = bg_video.searched_tag

    return ComposedVideoResponse(
        id=video.id,
        composition_job_id=video.composition_job_id,
        generated_caption_id=video.generated_caption_id,
        scraped_caption_id=video.scraped_caption_id,
        background_video_id=video.background_video_id,
        storage_path=video.storage_path,
        file_size_bytes=video.file_size_bytes,
        duration_seconds=video.duration_seconds,
        resolution=video.resolution,
        caption_chunks=video.caption_chunks,
        status=video.status,
        is_favorite=video.is_favorite or False,
        is_published=video.is_published or False,
        created_at=video.created_at,
        caption_text=caption_text,
        edited_caption_text=video.edited_caption_text,
        niche=video.niche,
        caption_tags=caption_tags,
        background_tags=background_tags,
        background_source_url=background_source_url,
        background_subreddit=background_subreddit,
        quality_score=quality_score,
        generated_title=generated_title,
        hosted_url=video.hosted_url,
        reddit_post_url=video.reddit_post_url,
        reddit_posted_at=video.reddit_posted_at,
        approval_status=video.approval_status or "pending",
        approved_at=video.approved_at,
        claude_review_status=video.claude_review_status,
        claude_review_verdict=video.claude_review_verdict,
        claude_review_scores=video.claude_review_scores,
        claude_review_issues=video.claude_review_issues,
        judge_pass=judge_pass,
        judge_scores=judge_scores,
        judge_issues=judge_issues,
    )
