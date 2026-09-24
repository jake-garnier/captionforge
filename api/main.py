"""
FastAPI application - Main API server
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database.db import get_db, init_db
from database.models import Video, ScrapedCaption, ScrapingProgress
from tasks.scraping_tasks import scrape_subreddit, scrape_all_subreddits, get_scraping_stats, extract_caption_task
from tasks.incremental_scraping import incremental_scrape_subreddit, get_scraping_progress, reset_scraping_progress
from tasks.upvote_updater import update_video_upvotes, update_stale_upvotes, update_popular_videos_upvotes
from tasks.celery_app import celery_app
from config.settings import settings
from pydantic import BaseModel
from typing import Optional, List
import logging

# Import video gallery router
from api.video_gallery import router as gallery_router
from api.dashboard import router as dashboard_router
from api.training import router as training_router
from api.scraper_control import router as scraper_control_router
from api.subreddit_management import router as subreddit_router
from api.training_management import router as training_manager_router
from api.generation import router as generation_router
from api.background_videos import router as background_videos_router
from api.gpu_control import router as gpu_control_router
from api.video_composition import router as composition_router
from api.pipeline import router as pipeline_router
from api.publishing import router as publishing_router
from api.patreon import router as patreon_router
from api.telegram import router as telegram_router
from api.telegram_scraping import router as telegram_scraping_router
from api.reddit import router as reddit_router
from api.workflows import router as workflows_router
from api.llama_server import router as llama_server_router
from api.reddit_background_videos import router as reddit_bg_router
from api.postpone import router as postpone_router
from api.membership_sync import router as membership_sync_router
from api.analytics import router as analytics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Captioned-video content pipeline: Reddit scraping, OCR, caption generation, composition and publishing"
)

# Include routers
app.include_router(gallery_router)
app.include_router(dashboard_router)
app.include_router(training_router)
app.include_router(scraper_control_router)
app.include_router(subreddit_router)
app.include_router(training_manager_router)
app.include_router(generation_router)
app.include_router(background_videos_router)
app.include_router(gpu_control_router)
app.include_router(composition_router)
app.include_router(pipeline_router)
app.include_router(publishing_router)
app.include_router(patreon_router)
app.include_router(telegram_router)
app.include_router(telegram_scraping_router)
app.include_router(reddit_router)
app.include_router(workflows_router)
app.include_router(llama_server_router)
app.include_router(reddit_bg_router)
app.include_router(postpone_router)
app.include_router(membership_sync_router)
app.include_router(analytics_router)


# Pydantic models for API
class ScrapeRequest(BaseModel):
    subreddit: str
    limit: int = 100
    min_score: int = 0


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check Redis connection
        redis_status = celery_app.backend.client.ping()

        # Check database connection
        from database.db import engine
        with engine.connect() as conn:
            db_status = True

        return {
            "status": "healthy",
            "redis": "connected" if redis_status else "disconnected",
            "database": "connected" if db_status else "disconnected",
            "scraping_enabled": settings.scraping_enabled
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Scraping endpoints
@app.post("/scrape/{subreddit}", response_model=dict)
async def trigger_scrape(subreddit: str, limit: int = 100, min_score: int = 0, sort: str = "top", time_filter: Optional[str] = None):
    """
    Trigger scraping of a specific subreddit

    Args:
        subreddit: Name of subreddit to scrape
        limit: Maximum posts to fetch
        min_score: Minimum upvote score
        sort: Sort method (top, hot, new, rising)
        time_filter: Time filter for 'top' sort (hour, day, week, month, year, all)

    Returns:
        Task ID and status
    """
    try:
        task = scrape_subreddit.delay(subreddit, limit, min_score, sort, time_filter)

        return {
            "task_id": task.id,
            "status": "queued",
            "subreddit": subreddit,
            "message": f"Scraping r/{subreddit} in background"
        }
    except Exception as e:
        logger.error(f"Error triggering scrape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape/all", response_model=dict)
async def trigger_scrape_all():
    """
    Trigger scraping of all configured subreddits

    Returns:
        Task ID and status
    """
    try:
        task = scrape_all_subreddits.delay()

        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Scraping all configured subreddits"
        }
    except Exception as e:
        logger.error(f"Error triggering scrape all: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a Celery task

    Args:
        task_id: Task ID returned from scraping endpoint

    Returns:
        Task status and result
    """
    try:
        task = celery_app.AsyncResult(task_id)

        response = {
            "task_id": task_id,
            "status": task.state,
        }

        if task.state == 'PENDING':
            response["result"] = {"message": "Task is waiting in queue"}
        elif task.state == 'STARTED':
            response["result"] = {"message": "Task is being processed"}
        elif task.state == 'SUCCESS':
            response["result"] = task.result
        elif task.state == 'FAILURE':
            response["error"] = str(task.info)
        else:
            response["result"] = {"message": f"Task state: {task.state}"}

        return response

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reset-database")
def reset_database(db: Session = Depends(get_db)):
    """
    Reset database by deleting all videos and captions
    USE WITH CAUTION - This deletes all data!
    """
    try:
        # Delete all captions first (foreign key constraint)
        deleted_captions = db.query(ScrapedCaption).delete()
        # Delete all videos
        deleted_videos = db.query(Video).delete()
        db.commit()

        logger.info(f"Database reset: Deleted {deleted_captions} captions and {deleted_videos} videos")

        return {
            "status": "success",
            "deleted_captions": deleted_captions,
            "deleted_videos": deleted_videos,
            "message": "Database reset successfully"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video management endpoints
@app.get("/videos", response_model=dict)
async def list_videos(
    skip: int = 0,
    limit: int = 50,
    subreddit: Optional[str] = None,
    status: Optional[str] = None,
    include_all: bool = False,
    db: Session = Depends(get_db)
):
    """
    List videos in database

    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        subreddit: Filter by subreddit
        status: Filter by processing status (use __all__ to see all videos)
        include_all: If True, include videos not yet in gallery
        db: Database session

    Returns:
        List of videos with metadata
    """
    try:
        # Start with base query
        query = db.query(Video)

        # When filtering by specific status or __all__, show all videos (not just gallery)
        # This allows viewing failed/pending videos that aren't in the gallery
        if status == "__all__":
            # Show all videos regardless of status or gallery
            pass
        elif status:
            # Specific status filter - show all videos with that status
            query = query.filter(Video.processing_status == status)
        elif include_all:
            # Show all videos
            pass
        else:
            # Default: Only show videos that have been added to gallery
            query = query.filter(Video.gallery_added_at.isnot(None))

        if subreddit:
            query = query.filter(Video.source_subreddit == subreddit)

        total = query.count()
        videos = query.offset(skip).limit(limit).all()

        # Get captions for each video (all versions for comparison)
        video_data = []
        for v in videos:
            # Get ALL captions for this video, ordered by date (oldest first)
            all_captions = db.query(ScrapedCaption).filter(
                ScrapedCaption.video_id == v.id
            ).order_by(ScrapedCaption.scraped_at.asc()).all()

            # Build caption comparison data
            caption_versions = []
            for cap in all_captions:
                caption_versions.append({
                    "id": cap.id,
                    "scraped_at": cap.scraped_at.isoformat() if cap.scraped_at else None,
                    "caption_text": cap.caption_text,
                    "raw_ocr_text": cap.raw_ocr_text,
                    "rule_based_text": cap.rule_based_text,
                    "llm_refined_text": cap.llm_refined_text,
                    "caption_length": len(cap.caption_text) if cap.caption_text else 0,
                    "raw_ocr_length": len(cap.raw_ocr_text) if cap.raw_ocr_text else 0,
                })

            # For backwards compatibility, also include first/latest caption
            first_caption = all_captions[0] if all_captions else None
            latest_caption = all_captions[-1] if all_captions else None

            video_data.append({
                "id": v.id,
                "post_id": v.source_post_id,
                "subreddit": v.source_subreddit,
                "upvotes": v.upvotes,
                "last_upvote_check": v.last_upvote_check.isoformat() if v.last_upvote_check else None,
                "duration": v.duration_seconds,
                "resolution": v.resolution,
                "file_size_mb": v.file_size_bytes / 1024 / 1024 if v.file_size_bytes else None,
                "status": v.processing_status,
                "media_type": v.media_type or "video",
                "storage_path": v.storage_path,
                "source_url": v.source_url,
                "download_date": v.download_date.isoformat() if v.download_date else None,
                # Backwards compatible fields (latest caption)
                "caption": latest_caption.caption_text if latest_caption else None,
                "raw_ocr": latest_caption.raw_ocr_text if latest_caption else None,
                "rule_based": latest_caption.rule_based_text if latest_caption else None,
                "llm_refined": latest_caption.llm_refined_text if latest_caption else None,
                # New comparison fields
                "caption_count": len(all_captions),
                "caption_versions": caption_versions,
                # Quick access to old vs new
                "old_caption": first_caption.caption_text if first_caption and len(all_captions) > 1 else None,
                "new_caption": latest_caption.caption_text if latest_caption and len(all_captions) > 1 else None,
                "old_caption_date": first_caption.scraped_at.strftime("%Y-%m-%d") if first_caption and len(all_captions) > 1 else None,
                "new_caption_date": latest_caption.scraped_at.strftime("%Y-%m-%d") if latest_caption and len(all_captions) > 1 else None,
            })

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "videos": video_data
        }

    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos/{video_id}", response_model=dict)
async def get_video(video_id: int, db: Session = Depends(get_db)):
    """Get details of a specific video"""
    video = db.query(Video).filter(Video.id == video_id).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": video.id,
        "post_id": video.source_post_id,
        "subreddit": video.source_subreddit,
        "source_url": video.source_url,
        "storage_path": video.storage_path,
        "file_hash": video.file_hash,
        "duration": video.duration_seconds,
        "resolution": video.resolution,
        "file_size_bytes": video.file_size_bytes,
        "status": video.processing_status,
        "download_date": video.download_date.isoformat() if video.download_date else None,
        "created_at": video.created_at.isoformat() if video.created_at else None
    }


@app.delete("/videos/{video_id}")
async def delete_video(video_id: int, db: Session = Depends(get_db)):
    """Delete a video from database and disk"""
    video = db.query(Video).filter(Video.id == video_id).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Delete file from disk
        from scrapers.video_downloader import VideoDownloader
        from database.models import VideoAnalysis, GeneratedCaption, PublishQueue
        downloader = VideoDownloader()
        downloader.delete_video(video.storage_path)

        # Delete related records first (cascade delete)
        db.query(ScrapedCaption).filter(ScrapedCaption.video_id == video_id).delete()
        db.query(VideoAnalysis).filter(VideoAnalysis.video_id == video_id).delete()
        db.query(PublishQueue).filter(PublishQueue.video_id == video_id).delete()
        db.query(GeneratedCaption).filter(GeneratedCaption.video_id == video_id).delete()

        # Delete video from database
        db.delete(video)
        db.commit()

        return {"message": f"Video {video_id} deleted successfully"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoints
@app.get("/stats")
async def get_stats():
    """Get scraping statistics"""
    try:
        task = get_scraping_stats.delay()
        result = task.get(timeout=5)  # Wait up to 5 seconds
        return result
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Incremental scraping endpoints
@app.post("/scrape/incremental/{subreddit}", response_model=dict)
async def trigger_incremental_scrape(
    subreddit: str,
    batch_size: int = 25,
    target_min_score: int = 300
):
    """
    Trigger incremental scraping of a subreddit
    Resumes from last position, processes batch_size posts

    Args:
        subreddit: Name of subreddit to scrape
        batch_size: Number of posts to process in this batch (default: 25)
        target_min_score: Stop scraping when posts drop below this score (default: 300)

    Returns:
        Task ID and status
    """
    try:
        task = incremental_scrape_subreddit.delay(subreddit, batch_size, target_min_score)

        return {
            "task_id": task.id,
            "status": "queued",
            "subreddit": subreddit,
            "batch_size": batch_size,
            "target_min_score": target_min_score,
            "message": f"Incremental scraping of r/{subreddit} started"
        }
    except Exception as e:
        logger.error(f"Error triggering incremental scrape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/progress/{subreddit}", response_model=dict)
async def get_progress(subreddit: str):
    """
    Get scraping progress for a specific subreddit

    Args:
        subreddit: Name of subreddit

    Returns:
        Scraping progress details
    """
    try:
        task = get_scraping_progress.delay(subreddit)
        result = task.get(timeout=5)
        return result
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/progress", response_model=dict)
async def get_all_progress():
    """
    Get scraping progress for all subreddits

    Returns:
        Progress for all subreddits
    """
    try:
        task = get_scraping_progress.delay()
        result = task.get(timeout=5)
        return result
    except Exception as e:
        logger.error(f"Error getting all progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape/progress/{subreddit}/reset", response_model=dict)
async def reset_progress(subreddit: str):
    """
    Reset scraping progress for a subreddit
    Use this to restart scraping from the beginning

    Args:
        subreddit: Name of subreddit to reset

    Returns:
        Status message
    """
    try:
        task = reset_scraping_progress.delay(subreddit)
        result = task.get(timeout=5)
        return result
    except Exception as e:
        logger.error(f"Error resetting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Upvote tracking endpoints
@app.post("/upvotes/update/{video_id}", response_model=dict)
async def trigger_upvote_update(video_id: int):
    """
    Trigger upvote update for a specific video

    Args:
        video_id: Database ID of video to update

    Returns:
        Task ID and status
    """
    try:
        task = update_video_upvotes.delay(video_id)

        return {
            "task_id": task.id,
            "status": "queued",
            "video_id": video_id,
            "message": f"Upvote update queued for video {video_id}"
        }
    except Exception as e:
        logger.error(f"Error triggering upvote update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upvotes/update/stale", response_model=dict)
async def trigger_stale_upvotes_update(max_videos: int = 50, stale_hours: int = 24):
    """
    Update upvotes for videos not checked recently

    Args:
        max_videos: Maximum number of videos to update (default: 50)
        stale_hours: Consider upvotes stale after this many hours (default: 24)

    Returns:
        Task ID and update summary
    """
    try:
        task = update_stale_upvotes.delay(max_videos, stale_hours)
        result = task.get(timeout=5)
        return result
    except Exception as e:
        logger.error(f"Error triggering stale upvotes update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upvotes/update/popular", response_model=dict)
async def trigger_popular_upvotes_update(min_upvotes: int = 1000, max_videos: int = 100):
    """
    Update upvotes for popular videos (high upvote count)

    Args:
        min_upvotes: Only update videos with at least this many upvotes (default: 1000)
        max_videos: Maximum number of videos to update (default: 100)

    Returns:
        Task ID and update summary
    """
    try:
        task = update_popular_videos_upvotes.delay(min_upvotes, max_videos)
        result = task.get(timeout=5)
        return result
    except Exception as e:
        logger.error(f"Error triggering popular upvotes update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch re-extraction endpoint
@app.post("/captions/reextract/all", response_model=dict)
async def reextract_all_captions(
    media_type: Optional[str] = None,
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """
    Queue caption re-extraction for all videos in database.
    Uses Qwen2-VL-2B for high-quality OCR extraction.

    Args:
        media_type: Optional filter by media type (video, image, gif, gallery)
        batch_size: Number of tasks to queue at once (default: 50, use 0 for all)

    Returns:
        Number of tasks queued and task IDs
    """
    try:
        # Get all videos with storage paths
        query = db.query(Video).filter(Video.storage_path.isnot(None))

        if media_type:
            query = query.filter(Video.media_type == media_type)

        # Order by ID for consistent processing
        videos = query.order_by(Video.id).all()

        if batch_size > 0:
            videos = videos[:batch_size]

        # Queue extraction tasks
        task_ids = []
        for video in videos:
            task = extract_caption_task.delay(video.id)
            task_ids.append({"video_id": video.id, "task_id": task.id})

        logger.info(f"Queued {len(task_ids)} caption re-extraction tasks")

        return {
            "status": "queued",
            "total_queued": len(task_ids),
            "media_type_filter": media_type,
            "message": f"Queued {len(task_ids)} videos for Qwen2-VL-2B caption re-extraction",
            "tasks": task_ids[:20] if len(task_ids) > 20 else task_ids,  # First 20 task IDs
            "note": "Processing ~2 min per video. Monitor via /task/{task_id} or Flower dashboard"
        }
    except Exception as e:
        logger.error(f"Error queuing re-extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/reextract/status", response_model=dict)
async def get_reextract_status(db: Session = Depends(get_db)):
    """
    Get status of caption extraction across all videos.
    Shows how many have been processed, pending, etc.
    """
    try:
        # Count videos by caption status
        total_videos = db.query(Video).filter(Video.storage_path.isnot(None)).count()

        # Videos with captions
        videos_with_captions = db.query(Video).join(
            ScrapedCaption, Video.id == ScrapedCaption.video_id
        ).filter(Video.storage_path.isnot(None)).distinct().count()

        # Videos without captions
        videos_without_captions = total_videos - videos_with_captions

        # By media type
        by_media_type = db.query(
            Video.media_type,
            db.query(Video).filter(Video.media_type == Video.media_type).count()
        ).group_by(Video.media_type).all()

        return {
            "total_videos": total_videos,
            "with_captions": videos_with_captions,
            "without_captions": videos_without_captions,
            "by_media_type": {mt: c for mt, c in by_media_type} if by_media_type else {}
        }
    except Exception as e:
        logger.error(f"Error getting extraction status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/captions/llm-refine", response_model=dict)
async def refine_captions_with_llm(
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """
    Queue LLM refinement for captions where it was skipped or failed.
    Dispatches to GPU worker since LLM requires CUDA.

    Finds captions where llm_refined_text == rule_based_text (LLM passthrough)
    and re-processes them with the working LLM.

    Much faster than full re-extraction since OCR is not re-run.
    """
    try:
        from tasks.scraping_tasks import llm_refine_batch_task

        # Check how many need refinement (NULL = never processed by LLM)
        needs_refinement = db.query(ScrapedCaption).filter(
            ScrapedCaption.rule_based_text.isnot(None),
            ScrapedCaption.llm_refined_text.is_(None)
        ).count()

        if needs_refinement == 0:
            return {
                "status": "complete",
                "message": "All captions already have LLM refinement",
                "remaining": 0
            }

        # Dispatch to GPU worker
        task = llm_refine_batch_task.delay(batch_size=batch_size)

        return {
            "status": "queued",
            "task_id": task.id,
            "batch_size": batch_size,
            "remaining": needs_refinement,
            "message": f"Queued LLM refinement for up to {batch_size} captions. {needs_refinement} total need refinement.",
            "note": "Task runs on GPU worker. Check status via /task/{task_id} or Flower dashboard."
        }

    except Exception as e:
        logger.error(f"Error queuing LLM refinement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/llm-refine/status", response_model=dict)
async def get_llm_refine_status(db: Session = Depends(get_db)):
    """
    Check how many captions need LLM refinement.
    """
    try:
        # Count captions where LLM didn't process (passthrough)
        needs_refinement = db.query(ScrapedCaption).filter(
            ScrapedCaption.rule_based_text.isnot(None),
            ScrapedCaption.llm_refined_text == ScrapedCaption.rule_based_text
        ).count()

        # Count captions with actual LLM refinement
        has_refinement = db.query(ScrapedCaption).filter(
            ScrapedCaption.rule_based_text.isnot(None),
            ScrapedCaption.llm_refined_text.isnot(None),
            ScrapedCaption.llm_refined_text != ScrapedCaption.rule_based_text
        ).count()

        # Total captions
        total = db.query(ScrapedCaption).count()

        return {
            "total_captions": total,
            "needs_llm_refinement": needs_refinement,
            "has_llm_refinement": has_refinement,
            "no_rule_based": total - needs_refinement - has_refinement
        }

    except Exception as e:
        logger.error(f"Error getting LLM refine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/captions/reprocess-rule-based", response_model=dict)
async def reprocess_rule_based(
    batch_size: int = 200,
    db: Session = Depends(get_db)
):
    """
    Re-run rule-based postprocessing on all existing captions.

    Use after updating spam patterns in CaptionPostProcessor. Re-processes
    raw_ocr_text through the updated rules without re-running OCR.

    Captions that change will have llm_refined_text nulled so the
    LLM batch refinement task automatically re-processes them.

    Auto-queues successive batches until all captions are processed.
    """
    try:
        from tasks.scraping_tasks import reprocess_rule_based_batch_task

        total = db.query(ScrapedCaption).filter(
            ScrapedCaption.raw_ocr_text.isnot(None),
            ScrapedCaption.raw_ocr_text != ""
        ).count()

        if total == 0:
            return {
                "status": "complete",
                "message": "No captions with raw OCR text to reprocess",
                "total": 0
            }

        task = reprocess_rule_based_batch_task.delay(
            batch_size=batch_size, offset=0
        )

        return {
            "status": "queued",
            "task_id": task.id,
            "batch_size": batch_size,
            "total_captions": total,
            "message": f"Queued rule-based reprocessing for {total} captions in batches of {batch_size}. "
                       f"Changed captions will be auto-queued for LLM refinement.",
            "note": "Runs on maintenance worker (CPU). Monitor via Flower or /task/{task_id}."
        }

    except Exception as e:
        logger.error(f"Error queuing rule-based reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/reprocess-rule-based/status", response_model=dict)
async def get_reprocess_status():
    """
    Get progress of rule-based reprocessing job.

    Returns live progress from Redis including:
    - total/processed/changed/errors counts
    - needs_llm: how many captions now need LLM refinement
    - status: running/complete/idle
    """
    import json
    import redis as redis_lib

    try:
        r = redis_lib.from_url(settings.REDIS_URL)
        status_raw = r.get('reprocess:status')

        if not status_raw:
            return {
                "status": "idle",
                "message": "No reprocessing job has been run"
            }

        status = json.loads(status_raw)

        # Add LLM refinement queue depth for context
        from database.db import get_db_context
        with get_db_context() as db:
            needs_llm = db.query(ScrapedCaption).filter(
                ScrapedCaption.rule_based_text.isnot(None),
                ScrapedCaption.llm_refined_text.is_(None)
            ).count()
            status['needs_llm_current'] = needs_llm

        return status

    except Exception as e:
        logger.error(f"Error getting reprocess status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/captions/reextract/clean-and-run", response_model=dict)
async def clean_and_reextract_captions(
    media_type: Optional[str] = None,
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """
    Delete all existing captions and queue videos for fresh re-extraction.
    This is the preferred method when iterating on the extraction model.

    Steps:
    1. Delete all existing captions from database
    2. Reset video processing status to 'downloaded'
    3. Queue videos for extraction

    Args:
        media_type: Optional filter by media type (video, image, gif, gallery)
        batch_size: Number of tasks to queue at once (default: 50, use 0 for all)

    Returns:
        Number of captions deleted and tasks queued
    """
    try:
        # Step 1: Count and delete existing captions
        caption_query = db.query(ScrapedCaption)
        if media_type:
            caption_query = caption_query.join(Video).filter(Video.media_type == media_type)

        deleted_count = caption_query.count()
        caption_query.delete(synchronize_session='fetch')

        # Step 2: Reset video processing status to 'downloaded' so they get re-processed
        video_query = db.query(Video).filter(Video.storage_path.isnot(None))
        if media_type:
            video_query = video_query.filter(Video.media_type == media_type)

        video_query.update(
            {'processing_status': 'downloaded'},
            synchronize_session='fetch'
        )

        db.commit()
        logger.info(f"Deleted {deleted_count} captions, reset video statuses")

        # Step 3: Get videos to queue for extraction
        query = db.query(Video).filter(Video.storage_path.isnot(None))
        if media_type:
            query = query.filter(Video.media_type == media_type)

        videos = query.order_by(Video.id).all()
        total_videos = len(videos)

        if batch_size > 0:
            videos = videos[:batch_size]

        # Queue extraction tasks
        task_ids = []
        for video in videos:
            task = extract_caption_task.delay(video.id)
            task_ids.append({"video_id": video.id, "task_id": task.id})

        logger.info(f"Queued {len(task_ids)} caption extraction tasks")

        return {
            "status": "started",
            "deleted_captions": deleted_count,
            "total_videos": total_videos,
            "queued_count": len(task_ids),
            "batch_size": batch_size if batch_size > 0 else "all",
            "media_type_filter": media_type,
            "message": f"Deleted {deleted_count} captions and queued {len(task_ids)} videos for fresh extraction",
            "tasks": task_ids[:20] if len(task_ids) > 20 else task_ids,
            "note": "Processing ~2 min per video. Monitor progress at /dashboard/extraction/progress"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error in clean-and-reextract: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/captions/reextract/failed", response_model=dict)
async def retry_failed_captions(
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """
    Retry extraction for videos that failed or had no caption found.

    This targets videos with processing_status='failed' or 'no_caption_found'
    and requeues them for extraction.

    Args:
        batch_size: Number of tasks to queue at once (default: 50, use 0 for all)

    Returns:
        Number of tasks queued and summary
    """
    try:
        from utils.extraction_queue import publish_video_downloaded

        # Find videos with failed or no_caption_found status
        failed_videos = db.query(Video).filter(
            Video.storage_path.isnot(None),
            Video.processing_status.in_(['failed', 'no_caption_found'])
        ).order_by(Video.id).all()

        total_failed = len(failed_videos)

        if total_failed == 0:
            return {
                "status": "complete",
                "message": "No failed or skipped videos to retry",
                "queued": 0
            }

        if batch_size > 0:
            failed_videos = failed_videos[:batch_size]

        # Reset status and queue for extraction
        queued_count = 0
        for video in failed_videos:
            # Reset status to downloaded so extraction dispatcher picks it up
            video.processing_status = 'downloaded'

            # Publish to extraction queue
            publish_video_downloaded(
                video_id=video.id,
                metadata={
                    'media_type': video.media_type or 'video',
                    'upvotes': video.upvotes or 0,
                    'subreddit': video.source_subreddit,
                    'post_id': video.source_post_id,
                    'retry': True
                }
            )
            queued_count += 1

        db.commit()

        logger.info(f"Queued {queued_count} failed/skipped videos for re-extraction")

        return {
            "status": "queued",
            "total_failed": total_failed,
            "queued": queued_count,
            "batch_size": batch_size if batch_size > 0 else "all",
            "message": f"Queued {queued_count} failed/skipped videos for re-extraction",
            "note": "Videos will be processed by the extraction dispatcher"
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error retrying failed captions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/reextract/failed/status", response_model=dict)
async def get_failed_captions_status(db: Session = Depends(get_db)):
    """
    Get count of videos with failed or no_caption_found status.
    """
    try:
        failed_count = db.query(Video).filter(
            Video.storage_path.isnot(None),
            Video.processing_status == 'failed'
        ).count()

        no_caption_count = db.query(Video).filter(
            Video.storage_path.isnot(None),
            Video.processing_status == 'no_caption_found'
        ).count()

        return {
            "failed": failed_count,
            "no_caption_found": no_caption_count,
            "total_retriable": failed_count + no_caption_count
        }

    except Exception as e:
        logger.error(f"Error getting failed caption status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
