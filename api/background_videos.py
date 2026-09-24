"""
API endpoints for background video management.

Provides:
- Video listing, streaming, and deletion
- Watermark filter control and stats
- Statistics

Background videos are scraped from Reddit; the per-subreddit scraper config
and its enable/disable toggles live in api/reddit_background_videos.py
(`/background-videos/reddit/*`).
"""
import logging
import redis
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from database.db import get_db
from database.models import BackgroundVideo, RedditBackgroundSubreddit
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/background-videos", tags=["background-videos"])

# Redis keys for scraper and filter state
# (must match tasks/watermark_filter.py and tasks/reddit_background_scraping.py)
REDDIT_BG_ENABLED_KEY = "reddit_bg:enabled"
WATERMARK_FILTER_ENABLED_KEY = "watermark_filter:enabled"
WATERMARK_FILTER_GPU_KEY = "watermark_filter:gpu"  # "0", "1", or "both"


# === Pydantic Models ===

class BackgroundVideoResponse(BaseModel):
    id: int
    source_type: str = "reddit"  # video source (reddit, or a future custom importer)
    reddit_post_id: Optional[str] = None
    reddit_subreddit: Optional[str] = None
    reddit_score: Optional[int] = None
    source_url: str
    thumbnail_url: Optional[str]
    duration_seconds: Optional[int]
    width: Optional[int]
    height: Optional[int]
    file_size_mb: Optional[float]
    views: int
    likes: int
    tags: Optional[List[str]]
    searched_tag: Optional[str] = None  # The tag/subreddit we used to find this video
    username: Optional[str]
    download_status: str
    filter_status: Optional[str] = None  # pending, approved, rejected, error
    filter_text: Optional[str] = None  # OCR detected text or error message
    ml_tags: Optional[Dict[str, Any]] = None  # VLM tags: subjects/activities/setting/mood/camera/text_on_screen
    class Config:
        from_attributes = True


# === Helper Functions ===

def get_redis_client():
    return redis.from_url(settings.celery_broker_url)


def _video_to_response(video: BackgroundVideo) -> BackgroundVideoResponse:
    return BackgroundVideoResponse(
        id=video.id,
        source_type=video.source_type or "reddit",
        reddit_post_id=video.reddit_post_id,
        reddit_subreddit=video.reddit_subreddit,
        reddit_score=video.reddit_score,
        source_url=video.source_url,
        thumbnail_url=f"/background-videos/{video.id}/thumbnail" if video.thumbnail_path else None,
        duration_seconds=video.duration_seconds,
        width=video.width,
        height=video.height,
        file_size_mb=round(video.file_size_bytes / 1024 / 1024, 1) if video.file_size_bytes else None,
        views=video.views or 0,
        likes=video.likes or 0,
        tags=video.tags,
        searched_tag=video.searched_tag,
        username=video.username,
        download_status=video.download_status,
        filter_status=video.filter_status,
        filter_text=video.filter_text,
        ml_tags=video.ml_tags if isinstance(video.ml_tags, dict) else None
    )


# === Stats (must be before /{video_id} routes) ===

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get detailed statistics."""
    # Video stats
    total = db.query(func.count(BackgroundVideo.id)).scalar() or 0
    completed = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    total_size = db.query(func.sum(BackgroundVideo.file_size_bytes)).filter(
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    avg_duration = db.query(func.avg(BackgroundVideo.duration_seconds)).filter(
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    avg_views = db.query(func.avg(BackgroundVideo.views)).filter(
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    # Filter stats (watermark detection)
    pending_filter = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'pending'
    ).scalar() or 0
    approved_filter = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'approved'
    ).scalar() or 0
    rejected_filter = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'rejected'
    ).scalar() or 0
    error_filter = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'error'
    ).scalar() or 0

    filter_stats = {
        "pending": pending_filter,
        "approved": approved_filter,
        "rejected": rejected_filter,
        "error": error_filter
    }

    # Disk usage estimate
    disk_used_gb = round(total_size / 1024 / 1024 / 1024, 2) if total_size else 0

    # Per-source breakdown
    source_breakdown = {}
    source_counts = db.query(
        BackgroundVideo.source_type,
        func.count(BackgroundVideo.id),
        func.coalesce(func.sum(BackgroundVideo.file_size_bytes), 0)
    ).filter(
        BackgroundVideo.download_status == 'completed'
    ).group_by(BackgroundVideo.source_type).all()

    for source, count, size in source_counts:
        source_name = source or "reddit"
        source_breakdown[source_name] = {
            "count": count,
            "size_gb": round((size or 0) / 1024 / 1024 / 1024, 2)
        }

    return {
        "total_videos": total,
        "completed_videos": completed,
        "total_size_bytes": total_size or 0,
        "total_size_gb": disk_used_gb,
        "avg_duration_seconds": round(avg_duration, 1) if avg_duration else 0,
        "avg_views": int(avg_views) if avg_views else 0,
        "filter_stats": filter_stats,
        "source_breakdown": source_breakdown
    }


@router.get("/count")
def get_video_count(
    tag: Optional[str] = None,
    min_views: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get total count of background videos (for pagination)."""
    query = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.download_status == 'completed'
    )

    if tag:
        query = query.filter(BackgroundVideo.tags.contains([tag]))
    if min_views:
        query = query.filter(BackgroundVideo.views >= min_views)

    count = query.scalar() or 0
    return {"count": count}


# === Video Endpoints ===

@router.get("/")
def list_background_videos(
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    tag: Optional[str] = None,
    min_views: Optional[int] = None,
    source_type: Optional[str] = Query(None, pattern="^(reddit|all)$"),
    filter_status: Optional[str] = Query(None, pattern="^(pending|approved|rejected|error|all)$"),
    sort_by: str = Query("views", pattern="^(views|likes|duration|created_at)$"),
    sort_desc: bool = True,
    db: Session = Depends(get_db)
):
    """List background videos with filtering and pagination.

    Default behavior depends on filter status:
    - If filter is enabled: shows only approved videos (passed watermark filter)
    - If filter is disabled: shows all videos (pending, approved, etc.)
    Use filter_status=all to explicitly see all videos.
    """
    query = db.query(BackgroundVideo).filter(
        BackgroundVideo.download_status == 'completed'
    )

    # Apply source type filter
    if source_type and source_type != 'all':
        query = query.filter(BackgroundVideo.source_type == source_type)

    # Determine effective filter_status
    # If not specified, check if filter is enabled to determine default
    effective_filter = filter_status
    if effective_filter is None:
        try:
            r = get_redis_client()
            filter_enabled = r.get(WATERMARK_FILTER_ENABLED_KEY) == b"1"
            effective_filter = "approved" if filter_enabled else "all"
        except Exception:
            effective_filter = "all"  # Default to all if Redis unavailable

    # Apply filter status
    if effective_filter and effective_filter != 'all':
        query = query.filter(BackgroundVideo.filter_status == effective_filter)

    if tag:
        # Filter by tag (JSONB contains)
        query = query.filter(BackgroundVideo.tags.contains([tag]))

    if min_views:
        query = query.filter(BackgroundVideo.views >= min_views)

    # Get total count
    total = query.count()

    # Sorting
    sort_map = {
        "views": BackgroundVideo.views,
        "likes": BackgroundVideo.likes,
        "duration": BackgroundVideo.duration_seconds,
        "created_at": BackgroundVideo.created_at
    }
    sort_col = sort_map.get(sort_by, BackgroundVideo.views)
    if sort_desc:
        query = query.order_by(desc(sort_col))
    else:
        query = query.order_by(sort_col)

    videos = query.offset(skip).limit(limit).all()
    return {
        "videos": [_video_to_response(v) for v in videos],
        "total": total,
        "limit": limit,
        "skip": skip
    }


# === Bulk Operations (must be before /{video_id} routes) ===

@router.delete("/orphans")
def delete_orphaned_videos(db: Session = Depends(get_db)):
    """Delete videos where the file no longer exists on disk."""
    videos = db.query(BackgroundVideo).all()
    deleted_count = 0

    for video in videos:
        file_missing = not video.storage_path or not os.path.exists(video.storage_path)
        if file_missing:
            # Delete thumbnail if exists
            if video.thumbnail_path and os.path.exists(video.thumbnail_path):
                try:
                    os.remove(video.thumbnail_path)
                except:
                    pass
            db.delete(video)
            deleted_count += 1

    db.commit()
    logger.info(f"Deleted {deleted_count} orphaned video records")
    return {"deleted": deleted_count, "remaining": len(videos) - deleted_count}


@router.delete("/all")
def delete_all_background_videos(
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
    db: Session = Depends(get_db)
):
    """Delete ALL background videos - database records and files.

    WARNING: This is destructive and cannot be undone.
    Pass confirm=true to proceed.
    """
    if not confirm:
        return {
            "error": "Must pass confirm=true to delete all videos",
            "total_videos": db.query(func.count(BackgroundVideo.id)).scalar() or 0
        }

    # Get all videos
    videos = db.query(BackgroundVideo).all()
    deleted_files = 0
    deleted_thumbnails = 0

    for video in videos:
        # Delete video file
        if video.storage_path and os.path.exists(video.storage_path):
            try:
                os.remove(video.storage_path)
                deleted_files += 1
            except Exception as e:
                logger.warning(f"Could not delete file {video.storage_path}: {e}")

        # Delete thumbnail
        if video.thumbnail_path and os.path.exists(video.thumbnail_path):
            try:
                os.remove(video.thumbnail_path)
                deleted_thumbnails += 1
            except Exception as e:
                logger.warning(f"Could not delete thumbnail {video.thumbnail_path}: {e}")

    # Delete all database records
    total_deleted = db.query(BackgroundVideo).delete()
    db.commit()

    # Reset per-subreddit scrape counters so progress reflects the empty library
    subreddits = db.query(RedditBackgroundSubreddit).all()
    for sub in subreddits:
        sub.videos_downloaded = 0
        sub.videos_failed = 0
        sub.posts_scraped = 0
    db.commit()

    logger.warning(f"Deleted all background videos: {total_deleted} records, {deleted_files} files, {deleted_thumbnails} thumbnails")

    return {
        "status": "deleted",
        "records_deleted": total_deleted,
        "files_deleted": deleted_files,
        "thumbnails_deleted": deleted_thumbnails,
        "subreddits_reset": len(subreddits)
    }


# === Individual Video Endpoints ===

@router.get("/{video_id}", response_model=BackgroundVideoResponse)
def get_background_video(video_id: int, db: Session = Depends(get_db)):
    """Get single background video details."""
    video = db.query(BackgroundVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_to_response(video)


@router.get("/{video_id}/stream")
def stream_background_video(
    video_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Stream video file with proper HTTP Range support for video seeking."""

    video = db.query(BackgroundVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not os.path.exists(video.storage_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = os.path.getsize(video.storage_path)
    file_path = video.storage_path

    # Parse Range header for seeking support
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        # Clamp to valid range
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        content_length = end - start + 1

        def iterfile_range():
            with open(file_path, mode="rb") as file:
                file.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    chunk = file.read(read_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iterfile_range(),
            status_code=206,  # Partial Content
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Content-Disposition": f'inline; filename="bg_{video_id}.mp4"'
            }
        )
    else:
        # No range requested - return full file
        def iterfile():
            with open(file_path, mode="rb") as file:
                chunk_size = 64 * 1024
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="bg_{video_id}.mp4"'
            }
        )


@router.get("/{video_id}/thumbnail")
def get_thumbnail(video_id: int, db: Session = Depends(get_db)):
    """Get thumbnail image."""
    video = db.query(BackgroundVideo).filter_by(id=video_id).first()
    if not video or not video.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    if not os.path.exists(video.thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found on disk")
    return FileResponse(video.thumbnail_path, media_type="image/jpeg")


@router.delete("/{video_id}")
def delete_background_video(video_id: int, db: Session = Depends(get_db)):
    """Delete a background video."""
    video = db.query(BackgroundVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete files
    if video.storage_path and os.path.exists(video.storage_path):
        os.remove(video.storage_path)
    if video.thumbnail_path and os.path.exists(video.thumbnail_path):
        os.remove(video.thumbnail_path)

    db.delete(video)
    db.commit()

    return {"message": f"Deleted video {video_id}"}


# === Watermark Filter ===

@router.get("/filter/stats")
def get_filter_stats(db: Session = Depends(get_db)):
    """Get watermark filter statistics."""
    total = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    pending = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'pending',
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    approved = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'approved'
    ).scalar() or 0

    rejected = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'rejected'
    ).scalar() or 0

    errors = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'error'
    ).scalar() or 0

    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "errors": errors,
        "rejection_rate": f"{(rejected / (approved + rejected) * 100):.1f}%" if (approved + rejected) > 0 else "N/A"
    }


@router.post("/filter/process-pending")
def process_pending_filters(
    batch_size: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Queue pending videos for watermark filtering."""
    pending_videos = db.query(BackgroundVideo).filter(
        BackgroundVideo.filter_status == 'pending',
        BackgroundVideo.download_status == 'completed'
    ).limit(batch_size).all()

    if not pending_videos:
        return {"status": "no_pending", "queued": 0}

    from tasks.watermark_filter import filter_background_video
    tasks = []
    for video in pending_videos:
        task = filter_background_video.delay(video.id)
        tasks.append({"video_id": video.id, "task_id": task.id})

    return {
        "status": "queued",
        "queued": len(tasks),
        "tasks": tasks
    }


@router.post("/filter/retry-errors")
def retry_filter_errors(db: Session = Depends(get_db)):
    """Retry videos that failed watermark filtering."""
    error_videos = db.query(BackgroundVideo).filter(
        BackgroundVideo.filter_status == 'error',
        BackgroundVideo.download_status == 'completed'
    ).all()

    if not error_videos:
        return {"status": "no_errors", "queued": 0}

    # Reset to pending
    for video in error_videos:
        video.filter_status = 'pending'
    db.commit()

    from tasks.watermark_filter import filter_background_video
    tasks = []
    for video in error_videos:
        task = filter_background_video.delay(video.id)
        tasks.append({"video_id": video.id, "task_id": task.id})

    return {
        "status": "queued",
        "queued": len(tasks),
        "tasks": tasks
    }


@router.post("/filter/reset-rejected")
def reset_rejected_videos(
    requeue: bool = Query(True, description="Immediately queue for re-filtering"),
    db: Session = Depends(get_db)
):
    """Reset all rejected videos back to pending status for re-filtering.

    Use this after updating filter logic to re-evaluate previously rejected videos.
    """
    rejected_videos = db.query(BackgroundVideo).filter(
        BackgroundVideo.filter_status == 'rejected',
        BackgroundVideo.download_status == 'completed'
    ).all()

    if not rejected_videos:
        return {"status": "no_rejected", "reset": 0}

    # Reset to pending and clear filter text
    for video in rejected_videos:
        video.filter_status = 'pending'
        video.filter_text = None
        video.filter_checked_at = None
    db.commit()

    result = {
        "status": "reset",
        "reset": len(rejected_videos),
        "message": f"Reset {len(rejected_videos)} rejected videos to pending"
    }

    if requeue:
        from tasks.watermark_filter import filter_background_video
        tasks = []
        for video in rejected_videos[:50]:  # Limit to 50 to avoid queue flood
            task = filter_background_video.delay(video.id)
            tasks.append({"video_id": video.id, "task_id": task.id})

        result["queued"] = len(tasks)
        result["tasks"] = tasks
        if len(rejected_videos) > 50:
            result["note"] = f"Queued first 50 of {len(rejected_videos)}. Rest will be picked up by dispatcher."

    return result


@router.post("/filter/{video_id}")
def filter_single_video(video_id: int, db: Session = Depends(get_db)):
    """Manually trigger watermark filter for a specific video."""
    video = db.query(BackgroundVideo).filter_by(id=video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Reset status to pending
    video.filter_status = 'pending'
    db.commit()

    from tasks.watermark_filter import filter_background_video
    task = filter_background_video.delay(video_id)

    return {
        "video_id": video_id,
        "task_id": task.id,
        "message": f"Watermark filter queued for video {video_id}"
    }


# === Filter Control (Decoupled from Scraping) ===

class FilterControlStatus(BaseModel):
    filter_enabled: bool
    filter_gpu: str  # "0", "1", or "both"
    scraper_enabled: bool
    pending_count: int
    message: str


@router.get("/filter/control/status", response_model=FilterControlStatus)
def get_filter_control_status(db: Session = Depends(get_db)):
    """Get current filter control status (separate from scraper status)."""
    r = get_redis_client()

    # Get filter state
    filter_enabled = r.get(WATERMARK_FILTER_ENABLED_KEY) == b"1"
    filter_gpu = (r.get(WATERMARK_FILTER_GPU_KEY) or b"0").decode()
    scraper_enabled = r.get(REDDIT_BG_ENABLED_KEY) == b"1"

    # Count pending
    pending_count = db.query(func.count(BackgroundVideo.id)).filter(
        BackgroundVideo.filter_status == 'pending',
        BackgroundVideo.download_status == 'completed'
    ).scalar() or 0

    gpu_description = {
        "0": "GPU 0 only (RTX 2080 Ti)",
        "1": "GPU 1 only (RTX 4060 Ti)",
        "both": "Both GPUs"
    }.get(filter_gpu, f"GPU {filter_gpu}")

    return FilterControlStatus(
        filter_enabled=filter_enabled,
        filter_gpu=filter_gpu,
        scraper_enabled=scraper_enabled,
        pending_count=pending_count,
        message=f"Filter: {'enabled' if filter_enabled else 'disabled'}, GPU: {gpu_description}, {pending_count} pending"
    )


@router.post("/filter/control/enable")
def enable_filtering():
    """Enable watermark filtering (can run independently of scraping)."""
    r = get_redis_client()
    r.set(WATERMARK_FILTER_ENABLED_KEY, "1")

    # Set default GPU if not set
    if not r.get(WATERMARK_FILTER_GPU_KEY):
        r.set(WATERMARK_FILTER_GPU_KEY, "0")

    logger.info("Watermark filtering enabled")
    return {
        "filter_enabled": True,
        "filter_gpu": (r.get(WATERMARK_FILTER_GPU_KEY) or b"0").decode(),
        "message": "Watermark filtering enabled. Will process pending videos on next dispatch cycle."
    }


@router.post("/filter/control/disable")
def disable_filtering(unload_models: bool = Query(False, description="Unload OCR models from GPU(s)")):
    """Disable watermark filtering."""
    r = get_redis_client()
    r.set(WATERMARK_FILTER_ENABLED_KEY, "0")

    logger.info("Watermark filtering disabled")

    unload_results = {}
    if unload_models:
        try:
            from tasks.maintenance_tasks import unload_gpu_0_models, unload_gpu_1_models

            filter_gpu = (r.get(WATERMARK_FILTER_GPU_KEY) or b"0").decode()

            if filter_gpu == "0" or filter_gpu == "both":
                task0 = unload_gpu_0_models.delay()
                try:
                    unload_results['gpu_0'] = task0.get(timeout=10)
                except Exception as e:
                    unload_results['gpu_0'] = {'status': 'pending', 'note': str(e)}

            if filter_gpu == "1" or filter_gpu == "both":
                task1 = unload_gpu_1_models.delay()
                try:
                    unload_results['gpu_1'] = task1.get(timeout=10)
                except Exception as e:
                    unload_results['gpu_1'] = {'status': 'pending', 'note': str(e)}

        except Exception as e:
            logger.warning(f"Could not trigger model unload: {e}")
            unload_results['error'] = str(e)

    return {
        "filter_enabled": False,
        "message": "Watermark filtering disabled.",
        "models_unloaded": unload_results if unload_models else "skipped"
    }


@router.post("/filter/control/set-gpu")
def set_filter_gpu(gpu: str = Query(..., pattern="^(0|1|both)$", description="GPU to use: '0', '1', or 'both'")):
    """Set which GPU(s) to use for watermark filtering.

    - '0': GPU 0 only (RTX 2080 Ti, 11GB) - Recommended, more VRAM
    - '1': GPU 1 only (RTX 4060 Ti, 8GB)
    - 'both': Both GPUs (parallel processing, but loads model on both)
    """
    r = get_redis_client()
    r.set(WATERMARK_FILTER_GPU_KEY, gpu)

    gpu_descriptions = {
        "0": "GPU 0 (RTX 2080 Ti, 11GB VRAM) - Recommended",
        "1": "GPU 1 (RTX 4060 Ti, 8GB VRAM)",
        "both": "Both GPUs - Faster but loads model on both (uses more total VRAM)"
    }

    logger.info(f"Watermark filter GPU set to: {gpu}")
    return {
        "filter_gpu": gpu,
        "description": gpu_descriptions.get(gpu, f"GPU {gpu}"),
        "message": f"Filter will use {gpu_descriptions.get(gpu, gpu)}. Changes take effect on next filter dispatch."
    }
