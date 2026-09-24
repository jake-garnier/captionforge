"""
Celery tasks for web scraping
"""
from celery import group, chain
from tasks.celery_app import celery_app
from scrapers.reddit_scraper import RedditScraper
from scrapers.video_downloader import VideoDownloader
from scrapers.image_downloader import ImageDownloader
from scrapers.exceptions import PermanentDownloadError, TemporaryDownloadError
from database.db import get_db_context
from database.models import Video, ScrapedCaption, ScrapingProgress
from config.settings import settings
from sqlalchemy.sql import func
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def scrape_subreddit(self, subreddit_name: str, limit: int = 100, min_score: int = 0, sort: str = "top", time_filter: str = None):
    """
    Scrape a subreddit for video posts

    Args:
        subreddit_name: Name of subreddit to scrape
        limit: Maximum posts to fetch
        min_score: Minimum upvote score
        sort: Sort method (top, hot, new, rising) - defaults to 'top' for highest voted
        time_filter: Time filter for 'top' sort (hour, day, week, month, year, all)

    Returns:
        Dictionary with results
    """
    try:
        logger.info(f"Starting scrape of r/{subreddit_name} ({sort}{f' t={time_filter}' if time_filter else ''})")

        with RedditScraper() as scraper:
            posts = scraper.get_video_posts(
                subreddit_name=subreddit_name,
                limit=limit,
                min_score=min_score,
                sort=sort,
                time_filter=time_filter
            )

        if not posts:
            logger.warning(f"No video posts found in r/{subreddit_name}")
            return {
                'status': 'success',
                'subreddit': subreddit_name,
                'posts_found': 0,
                'downloads_queued': 0
            }

        # Check which posts are already in database
        with get_db_context() as db:
            existing_post_ids = {
                v.source_post_id
                for v in db.query(Video.source_post_id).all()
            }

        # Filter out already processed posts
        new_posts = [p for p in posts if p['id'] not in existing_post_ids]

        logger.info(
            f"Found {len(posts)} video posts in r/{subreddit_name}, "
            f"{len(new_posts)} are new"
        )

        if not new_posts:
            return {
                'status': 'success',
                'subreddit': subreddit_name,
                'posts_found': len(posts),
                'downloads_queued': 0,
                'message': 'All posts already processed'
            }

        # Queue download tasks for new posts (parallel execution)
        # Pass upvote count along with download task
        download_tasks = group(
            download_video_task.s(post['video_url'], post['id'], post['subreddit'], post.get('score', 0))
            for post in new_posts
        )

        result = download_tasks.apply_async()

        return {
            'status': 'success',
            'subreddit': subreddit_name,
            'posts_found': len(posts),
            'downloads_queued': len(new_posts),
            'group_task_id': result.id
        }

    except Exception as exc:
        logger.error(f"Error scraping r/{subreddit_name}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=2)
def download_video_task(
    self,
    video_url: str,
    post_id: str,
    subreddit: str,
    upvotes: int = 0,
    media_type: str = 'video',
    gallery_urls: Optional[List[str]] = None,
    gallery_count: Optional[int] = None
):
    """
    Download media (video, image, gallery, gif) and store metadata in database

    Args:
        video_url: URL of media to download
        post_id: Reddit post ID
        subreddit: Source subreddit
        upvotes: Current upvote count from Reddit
        media_type: Type of media ('video', 'image', 'gallery', 'gif')
        gallery_urls: List of image URLs for gallery posts
        gallery_count: Number of images in gallery

    Returns:
        Dictionary with download result
    """
    try:
        logger.info(f"Downloading {media_type} for post {post_id} ({upvotes} upvotes)")

        # Use appropriate downloader based on media type
        if media_type in ['image', 'gif']:
            downloader = ImageDownloader()
            result = downloader.download(video_url, post_id)
        elif media_type == 'gallery' and gallery_urls:
            downloader = ImageDownloader()
            result = downloader.download_gallery(gallery_urls, post_id)
        else:
            # Default to video downloader for videos and unknown types
            downloader = VideoDownloader()
            result = downloader.download(video_url, post_id)

        # Store in database
        with get_db_context() as db:
            # Check if already exists
            existing = db.query(Video).filter_by(source_post_id=post_id).first()
            if existing:
                logger.warning(f"Media for post {post_id} already exists")
                return {'status': 'skipped', 'reason': 'already_exists'}

            video = Video(
                source_url=video_url,
                storage_path=result['file_path'],
                file_hash=result['file_hash'],
                duration_seconds=result.get('duration'),
                resolution=result.get('resolution'),
                file_size_bytes=result.get('file_size'),
                source_subreddit=subreddit,
                source_post_id=post_id,
                media_type=media_type,
                gallery_image_count=result.get('gallery_count') or gallery_count,
                upvotes=upvotes,
                last_upvote_check=func.now(),
                processing_status='downloaded'
            )

            db.add(video)
            db.commit()
            db.refresh(video)

            logger.info(f"{media_type.capitalize()} saved: {post_id} (DB ID: {video.id})")

            # Queue for caption extraction (event-driven decoupling)
            # Extraction runs on GPU worker independently of download
            from utils.extraction_queue import publish_video_downloaded
            publish_video_downloaded(
                video_id=video.id,
                metadata={
                    'media_type': media_type,
                    'upvotes': upvotes,
                    'subreddit': subreddit,
                    'post_id': post_id
                }
            )

            return {
                'status': 'success',
                'video_id': video.id,
                'post_id': post_id,
                'media_type': media_type,
                'file_path': result['file_path'],
                'file_size_mb': result.get('file_size', 0) / 1024 / 1024
            }

    except PermanentDownloadError as exc:
        # Permanent failures (404, 410, dead domains) - do NOT retry
        logger.error(f"Permanent download failure for post {post_id}: {exc}")
        # Add failed video to database to prevent re-queueing
        with get_db_context() as db:
            # Check if already exists
            existing = db.query(Video).filter_by(source_post_id=post_id).first()
            if not existing:
                # Create stub record with failed status
                failed_video = Video(
                    source_url=video_url,
                    storage_path="",  # No file downloaded
                    file_hash=f"failed_{post_id}",  # Unique hash for failed downloads
                    source_subreddit=subreddit,
                    source_post_id=post_id,
                    upvotes=upvotes,
                    processing_status='failed'
                )
                db.add(failed_video)

            # Update failure counter
            progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit).first()
            if progress:
                progress.videos_failed = (progress.videos_failed or 0) + 1
            db.commit()
        return {
            'status': 'failed',
            'reason': 'permanent_error',
            'error': str(exc),
            'post_id': post_id
        }

    except TemporaryDownloadError as exc:
        # Temporary failures (timeouts, rate limits) - retry with backoff
        logger.warning(f"Temporary download failure for post {post_id} (attempt {self.request.retries + 1}/{self.max_retries}): {exc}")
        raise self.retry(exc=exc, countdown=180 * (2 ** self.request.retries))

    except Exception as exc:
        # Check for duplicate file_hash (IntegrityError) - permanent error, don't retry
        if 'duplicate key value violates unique constraint "ix_videos_file_hash"' in str(exc):
            logger.warning(f"Duplicate video content for post {post_id} (same file already exists) - skipping")
            with get_db_context() as db:
                # Check if already exists
                existing = db.query(Video).filter_by(source_post_id=post_id).first()
                if not existing:
                    # Create stub record with failed status
                    failed_video = Video(
                        source_url=video_url,
                        storage_path="",
                        file_hash=f"duplicate_{post_id}",  # Unique hash for duplicates
                        source_subreddit=subreddit,
                        source_post_id=post_id,
                        upvotes=upvotes,
                        processing_status='failed'
                    )
                    db.add(failed_video)

                # Update failure counter
                progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit).first()
                if progress:
                    progress.videos_failed = (progress.videos_failed or 0) + 1
                db.commit()
            return {
                'status': 'failed',
                'reason': 'duplicate_content',
                'error': 'Video content already exists with different post_id',
                'post_id': post_id
            }

        # All other unknown errors - treat as temporary and retry
        logger.error(f"Unexpected download error for post {post_id}: {exc}")
        raise self.retry(exc=exc, countdown=180 * (2 ** self.request.retries))


def _delete_video_file(video):
    """Delete video/image file from disk to free space. DB record is kept to prevent re-scraping."""
    if not video or not video.storage_path:
        return
    try:
        if os.path.exists(video.storage_path):
            os.remove(video.storage_path)
            logger.info(f"Deleted video file: {video.storage_path} (video_id={video.id})")

            # For galleries, also delete associated gallery images
            if video.media_type == 'gallery' and video.source_post_id:
                import glob
                base_path = video.storage_path.rsplit('.', 1)[0]
                gallery_pattern = base_path.replace(video.source_post_id, f"{video.source_post_id}_gallery_*")
                for gallery_file in glob.glob(gallery_pattern + ".*"):
                    os.remove(gallery_file)
                    logger.info(f"Deleted gallery file: {gallery_file}")
    except Exception as e:
        logger.warning(f"Failed to delete video file {video.storage_path}: {e}")


@celery_app.task(bind=True, max_retries=2)
def extract_caption_task(self, video_id: int):
    """
    Extract text captions from downloaded video using OCR

    Args:
        video_id: Database ID of video

    Returns:
        Dictionary with extraction result
    """
    # Import heartbeat for activity tracking
    try:
        from utils.worker_heartbeat import start_task, end_task, heartbeat
    except ImportError:
        start_task = end_task = heartbeat = lambda *args, **kwargs: None

    # Import extraction queue completion tracker
    try:
        from utils.extraction_queue import mark_extraction_complete
    except ImportError:
        mark_extraction_complete = lambda *args, **kwargs: None

    try:
        logger.info(f"Extracting caption from video {video_id}")
        start_task('extract_caption_task', self.request.id, video_id)

        # Get video from database
        with get_db_context() as db:
            video = db.query(Video).filter_by(id=video_id).first()

            if not video:
                raise ValueError(f"Video {video_id} not found in database")

            if not video.storage_path or not os.path.exists(video.storage_path):
                logger.warning(f"Media file not found for video {video_id}: {video.storage_path} - marking as failed")
                video.processing_status = 'failed'
                db.commit()
                end_task()
                mark_extraction_complete(video_id, success=False)
                return {
                    'status': 'failed',
                    'video_id': video_id,
                    'error': 'file_not_found'
                }

            # Extract captions using Qwen2-VL-2B OCR
            from scrapers.caption_postprocessor import CaptionPostProcessor
            from scrapers.caption_extractor_qwen2vl import Qwen2VLCaptionExtractor

            heartbeat(task_name='extract_caption_task', video_id=video_id, stage='initializing_ocr')

            extractor = Qwen2VLCaptionExtractor(gpu_id=0)
            logger.info(f"Using Qwen2-VL-2B extractor (single GPU, ~5GB VRAM)")

            # Choose extraction method based on media type
            media_type = video.media_type or 'video'

            if media_type == 'image':
                # Single image extraction
                logger.info(f"Extracting caption from image: {video.storage_path}")
                result = extractor.extract_from_image(
                    video.storage_path,
                    video_id=video_id
                )
            elif media_type == 'gif':
                # Animated GIF extraction - sample multiple frames
                logger.info(f"Extracting caption from animated GIF: {video.storage_path}")
                result = extractor.extract_from_gif(
                    video.storage_path,
                    video_id=video_id,
                    max_frames=30  # Sample up to 30 frames from the GIF
                )
            elif media_type == 'gallery' and video.gallery_image_count:
                # Gallery extraction - find all gallery images
                # Gallery images are stored as {post_id}_gallery_0.jpg, _gallery_1.jpg, etc.
                import glob
                base_path = video.storage_path.rsplit('.', 1)[0]  # Remove extension
                # Handle both patterns: post_id_gallery_0.ext and post_id.ext (primary)
                gallery_pattern = base_path.replace(video.source_post_id, f"{video.source_post_id}_gallery_*")
                gallery_paths = sorted(glob.glob(gallery_pattern + ".*"))

                if gallery_paths:
                    logger.info(f"Extracting captions from gallery: {len(gallery_paths)} images")
                    result = extractor.extract_from_gallery(gallery_paths, video_id=video_id)
                else:
                    # Fall back to single image if gallery files not found
                    logger.warning(f"Gallery files not found, using primary image: {video.storage_path}")
                    result = extractor.extract_from_image(video.storage_path, video_id=video_id)
            else:
                # Default: video extraction
                result = extractor.extract_from_video(
                    video.storage_path,
                    sample_rate=30,  # Sample every 30 frames (~1 fps for 30fps video)
                    video_id=video_id  # Pass video_id for heartbeat tracking
                )

            # Get raw OCR output
            raw_ocr = result.get('caption_text', '').strip()

            # Post-process to get all 3 versions (raw OCR, rule-based, LLM-refined)
            processor = CaptionPostProcessor()
            versions = processor.process(raw_ocr, aggressive=True, return_all_versions=True) if raw_ocr else None

            if versions and versions.get('final'):
                # Get quality metrics from post-processor
                quality_metrics = versions.get('quality_metrics', {})

                # Build extraction metadata from OCR result
                extraction_metadata = {
                    'frames_sampled': result.get('frames_sampled'),
                    'frames_processed': result.get('num_frames_processed'),
                    'frames_skipped_dedup': result.get('frames_skipped_dedup'),
                    'frames_skipped_text_dedup': result.get('frames_skipped_text_dedup'),
                    'visual_dedup_reduction_percent': result.get('visual_dedup_reduction_percent'),
                    'text_dedup_reduction_percent': result.get('text_dedup_reduction_percent'),
                    'unique_text_segments': result.get('unique_text_segments'),
                    'extractor': result.get('extractor', 'qwen2vl'),
                }

                # Save caption with all 3 versions + quality metrics to database
                scraped_caption = ScrapedCaption(
                    video_id=video.id,
                    caption_text=versions['final'],  # Final processed version
                    raw_ocr_text=versions['raw_ocr'],  # Original OCR output
                    rule_based_text=versions['rule_based'],  # After rule-based processing
                    llm_refined_text=versions['llm_refined'],  # After LLM (None if not applied)
                    source_subreddit=video.source_subreddit,
                    upvotes=video.upvotes or 0,  # Copy upvotes from the video
                    # Quality metrics
                    compression_ratio=quality_metrics.get('compression_ratio'),
                    slide_count_raw=quality_metrics.get('slide_count_raw'),
                    slide_count_final=quality_metrics.get('slide_count_final'),
                    unique_word_ratio=quality_metrics.get('unique_word_ratio'),
                    extraction_metadata=extraction_metadata
                )
                db.add(scraped_caption)

                # Update video status and add to gallery
                video.processing_status = 'caption_extracted'
                from sqlalchemy.sql import func
                video.gallery_added_at = func.now()

                # Increment videos_downloaded counter for this subreddit
                # This tracks successful unique video downloads toward the target
                progress = db.query(ScrapingProgress).filter_by(
                    subreddit=video.source_subreddit
                ).first()
                if progress:
                    progress.videos_downloaded += 1
                    logger.info(
                        f"Incremented videos_downloaded for r/{video.source_subreddit}: "
                        f"{progress.videos_downloaded}/{progress.target_videos}"
                    )

                db.commit()

                logger.info(
                    f"Caption extracted from video {video_id}: "
                    f"{len(versions['final'])} chars (final), "
                    f"OCR: {len(versions['raw_ocr'])}, "
                    f"Rule-based: {len(versions['rule_based'])}, "
                    f"LLM: {len(versions['llm_refined']) if versions['llm_refined'] else 'N/A'}, "
                    f"{result.get('num_frames_processed', 0)} frames processed"
                )

                end_task()  # Mark task complete
                mark_extraction_complete(video_id, success=True)  # Update extraction queue

                # Delete video file to free disk space (caption is saved in DB)
                _delete_video_file(video)

                return {
                    'status': 'success',
                    'video_id': video_id,
                    'caption_length': len(versions['final']),
                    'frames_processed': result.get('num_frames_processed', 0),
                    'caption_preview': versions['final'][:100]
                }
            else:
                logger.warning(f"No caption text found in video {video_id}")
                video.processing_status = 'no_caption_found'
                db.commit()

                end_task()  # Mark task complete
                mark_extraction_complete(video_id, success=True)  # Still a success (just no caption)

                # Delete video file to free disk space (no caption to extract)
                _delete_video_file(video)

                return {
                    'status': 'no_caption',
                    'video_id': video_id,
                    'frames_processed': result.get('num_frames_processed', 0)
                }

    except Exception as exc:
        logger.error(f"Caption extraction failed for video {video_id}: {exc}")
        end_task()  # Mark task complete even on failure
        mark_extraction_complete(video_id, success=False)  # Mark failed in queue
        # Retry with backoff, but on final failure mark as failed and delete file
        try:
            raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for video {video_id}, marking as failed")
            with get_db_context() as db:
                video = db.query(Video).filter_by(id=video_id).first()
                if video:
                    video.processing_status = 'failed'
                    db.commit()
                    # Delete video file to free disk space (post record stays to prevent re-scraping)
                    _delete_video_file(video)
            return {
                'status': 'failed',
                'video_id': video_id,
                'error': str(exc)
            }


@celery_app.task
def scrape_all_subreddits():
    """
    Scrape all enabled subreddits from database.

    This task is scheduled to run periodically via Celery Beat.
    Reads subreddit configuration from scraping_progress table.
    """
    try:
        with get_db_context() as db:
            # Get enabled subreddits from database
            enabled_subreddits = db.query(ScrapingProgress).filter(
                ScrapingProgress.scraping_active == True
            ).all()

            if not enabled_subreddits:
                logger.warning("No enabled subreddits found in database")
                return {'status': 'no_subreddits', 'scraped': 0}

            logger.info(f"Scraping {len(enabled_subreddits)} subreddits")

            # Create scraping tasks for each subreddit
            scrape_tasks = group(
                scrape_subreddit.s(
                    subreddit_name=sub.subreddit,
                    limit=getattr(sub, 'batch_size', None) or 25,
                    min_score=sub.target_min_score or 0,
                    sort='top'
                )
                for sub in enabled_subreddits
            )

            result = scrape_tasks.apply_async()

            return {
                'status': 'queued',
                'subreddits_count': len(enabled_subreddits),
                'subreddits': [sub.subreddit for sub in enabled_subreddits],
                'group_task_id': result.id
            }

    except Exception as e:
        logger.error(f"Error in scrape_all_subreddits: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def get_scraping_stats():
    """Get statistics about scraped videos"""
    try:
        with get_db_context() as db:
            total_videos = db.query(Video).count()
            by_subreddit = db.query(
                Video.source_subreddit,
                func.count(Video.id)
            ).group_by(Video.source_subreddit).all()

            by_status = db.query(
                Video.processing_status,
                func.count(Video.id)
            ).group_by(Video.processing_status).all()

            return {
                'total_videos': total_videos,
                'by_subreddit': {sub: count for sub, count in by_subreddit},
                'by_status': {status: count for status, count in by_status}
            }

    except Exception as e:
        logger.error(f"Error getting scraping stats: {e}")
        return {'error': str(e)}


@celery_app.task(bind=True, queue='gpu')
def llm_refine_batch_task(self, batch_size: int = 50):
    """
    Re-run LLM refinement on captions where it was skipped or failed.
    Finds captions where llm_refined_text == rule_based_text (LLM passthrough)
    and re-processes them with the working LLM.

    Runs on GPU worker since LLM requires CUDA.

    Args:
        batch_size: Number of captions to process per batch

    Returns:
        Dictionary with processing results
    """
    try:
        from scrapers.caption_postprocessor import CaptionPostProcessor

        with get_db_context() as db:
            # Find captions where LLM refinement is needed (never processed)
            # Note: We always set llm_refined_text after LLM runs, even if unchanged
            captions_needing_llm = db.query(ScrapedCaption).filter(
                ScrapedCaption.rule_based_text.isnot(None),
                ScrapedCaption.llm_refined_text.is_(None)
            ).limit(batch_size).all()

            if not captions_needing_llm:
                return {
                    "status": "complete",
                    "message": "All captions already have LLM refinement",
                    "processed": 0,
                    "remaining": 0
                }

            processor = CaptionPostProcessor()
            processed = 0
            errors = []

            for caption in captions_needing_llm:
                try:
                    # Re-run LLM refinement on the rule_based_text
                    # force=True because this batch task MUST use LLM even when inline is disabled
                    result = processor._llm_refine_caption(caption.rule_based_text, force=True)

                    # Always mark as processed (even if LLM didn't change text)
                    # This prevents re-querying the same captions forever
                    caption.llm_refined_text = result

                    # Only update caption_text if LLM actually improved it
                    if result != caption.rule_based_text:
                        caption.caption_text = result
                        processed += 1
                        logger.info(f"LLM refined caption {caption.id}: {len(caption.rule_based_text)} -> {len(result)} chars")
                    else:
                        logger.debug(f"LLM kept caption {caption.id} unchanged ({len(result)} chars)")

                except Exception as e:
                    errors.append({"caption_id": caption.id, "error": str(e)})
                    logger.error(f"Error refining caption {caption.id}: {e}")

            db.commit()

            # Count remaining (same logic as selection query)
            remaining = db.query(ScrapedCaption).filter(
                ScrapedCaption.rule_based_text.isnot(None),
                ScrapedCaption.llm_refined_text.is_(None)
            ).count()

            logger.info(f"LLM refinement batch complete: {processed} processed, {len(errors)} errors, {remaining} remaining")

            return {
                "status": "success",
                "processed": processed,
                "errors": len(errors),
                "remaining": remaining,
                "error_details": errors[:10] if errors else [],
                "message": f"LLM refined {processed} captions. {remaining} remaining."
            }

    except Exception as e:
        logger.error(f"Error in LLM refinement task: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(bind=True, queue='maintenance')
def reprocess_rule_based_batch_task(self, batch_size: int = 200, offset: int = 0):
    """
    Re-run rule-based postprocessing on existing raw_ocr_text.

    Use after updating spam patterns in CaptionPostProcessor to apply
    new rules to all existing captions without re-running OCR.

    After updating rule_based_text, nulls llm_refined_text so the
    existing llm_refine_batch_task picks them up for LLM re-processing.

    Tracks progress in Redis key 'reprocess:status' for UI monitoring.
    Disables pipeline during reprocessing to prevent stale data usage.

    Runs on maintenance queue (CPU-only, no GPU needed).
    """
    import json
    import redis as redis_lib
    import time

    r = redis_lib.from_url(settings.REDIS_URL)
    status_key = 'reprocess:status'

    try:
        from scrapers.caption_postprocessor import CaptionPostProcessor

        with get_db_context() as db:
            # Get total count on first batch
            total = db.query(ScrapedCaption).filter(
                ScrapedCaption.raw_ocr_text.isnot(None),
                ScrapedCaption.raw_ocr_text != ""
            ).count()

            # Initialize Redis status on first batch
            if offset == 0:
                # Disable pipeline to prevent stale data usage
                pipeline_was_enabled = r.get('pipeline:enabled')
                r.set('reprocess:pipeline_was_enabled',
                       pipeline_was_enabled.decode() if pipeline_was_enabled else 'false')
                r.set('pipeline:enabled', 'false')
                logger.info("Disabled pipeline for reprocessing")

                r.set(status_key, json.dumps({
                    'status': 'running',
                    'total': total,
                    'processed': 0,
                    'changed': 0,
                    'errors': 0,
                    'needs_llm': 0,
                    'started_at': time.time(),
                    'updated_at': time.time(),
                    'current_batch': 0,
                }))

            captions = db.query(ScrapedCaption).filter(
                ScrapedCaption.raw_ocr_text.isnot(None),
                ScrapedCaption.raw_ocr_text != ""
            ).order_by(ScrapedCaption.id).offset(offset).limit(batch_size).all()

            if not captions:
                # Done - update final status
                status = json.loads(r.get(status_key) or '{}')
                status['status'] = 'complete'
                status['updated_at'] = time.time()
                # Count how many now need LLM refinement
                needs_llm = db.query(ScrapedCaption).filter(
                    ScrapedCaption.rule_based_text.isnot(None),
                    ScrapedCaption.llm_refined_text.is_(None)
                ).count()
                status['needs_llm'] = needs_llm

                # Re-enable pipeline if it was enabled before
                prev = r.get('reprocess:pipeline_was_enabled')
                if prev and prev.decode() == 'true':
                    r.set('pipeline:enabled', 'true')
                    logger.info("Re-enabled pipeline after reprocessing")

                r.set(status_key, json.dumps(status))
                return status

            # Disable LLM for this task - we only want rule-based processing
            # LLM refinement is handled separately by llm_refine_batch_task
            import scrapers.caption_postprocessor as _cpp
            _cpp._LLM_ENABLED = False

            processor = CaptionPostProcessor()
            batch_processed = 0
            batch_changed = 0
            errors = []

            for caption in captions:
                try:
                    versions = processor.process(
                        caption.raw_ocr_text,
                        aggressive=True,
                        return_all_versions=True
                    )

                    new_rule_based = versions['rule_based']
                    old_rule_based = caption.rule_based_text or ""

                    if new_rule_based != old_rule_based:
                        caption.rule_based_text = new_rule_based
                        caption.caption_text = new_rule_based
                        # Null out LLM so batch refinement re-processes it
                        caption.llm_refined_text = None
                        batch_changed += 1

                    batch_processed += 1

                except Exception as e:
                    errors.append({"caption_id": caption.id, "error": str(e)})
                    logger.error(f"Error reprocessing caption {caption.id}: {e}")

            db.commit()

            # Update Redis progress
            status = json.loads(r.get(status_key) or '{}')
            status['processed'] = status.get('processed', 0) + batch_processed
            status['changed'] = status.get('changed', 0) + batch_changed
            status['errors'] = status.get('errors', 0) + len(errors)
            status['updated_at'] = time.time()
            status['current_batch'] = offset // batch_size + 1
            status['total'] = total

            next_offset = offset + batch_size
            remaining = max(0, total - next_offset)

            r.set(status_key, json.dumps(status))

            # Auto-queue next batch if there are more
            if remaining > 0:
                reprocess_rule_based_batch_task.delay(
                    batch_size=batch_size, offset=next_offset
                )
            else:
                # Final batch done
                needs_llm = db.query(ScrapedCaption).filter(
                    ScrapedCaption.rule_based_text.isnot(None),
                    ScrapedCaption.llm_refined_text.is_(None)
                ).count()
                status['status'] = 'complete'
                status['needs_llm'] = needs_llm

                # Re-enable pipeline if it was enabled before
                prev = r.get('reprocess:pipeline_was_enabled')
                if prev and prev.decode() == 'true':
                    r.set('pipeline:enabled', 'true')
                    logger.info("Re-enabled pipeline after reprocessing")

                r.set(status_key, json.dumps(status))

            logger.info(
                f"Rule-based reprocess batch: {batch_processed} processed, {batch_changed} changed, "
                f"{len(errors)} errors, {remaining} remaining"
            )

            return {
                "status": "success" if remaining > 0 else "complete",
                "processed": batch_processed,
                "changed": batch_changed,
                "errors": len(errors),
                "offset": offset,
                "remaining": remaining,
                "total": total,
            }

    except Exception as e:
        logger.error(f"Error in rule-based reprocess task: {e}")
        return {"status": "error", "error": str(e)}
