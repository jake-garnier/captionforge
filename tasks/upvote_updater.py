"""
Upvote updater tasks - periodically refresh upvote counts from Reddit

Uses proxy pool to route requests through the same proxy assigned to each subreddit,
ensuring consistent IP usage and proper block detection per-proxy.
"""
from tasks.celery_app import celery_app
from scrapers.reddit_json_scraper import RedditJsonScraper
from database.db import get_db_context
from database.models import Video
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_proxy_for_subreddit(subreddit: str):
    """Get the proxy assigned to a subreddit from the proxy pool."""
    try:
        from utils.proxy_pool import get_proxy_for_subreddit as pool_get_proxy
        return pool_get_proxy(subreddit, auto_assign=False)  # Don't auto-assign, just use if exists
    except Exception as e:
        logger.debug(f"Could not get proxy for r/{subreddit}: {e}")
        return None


@celery_app.task(bind=True, max_retries=3)
def update_video_upvotes(self, video_id: int):
    """
    Update upvote count for a specific video by scraping Reddit.

    Uses the proxy assigned to the video's subreddit from the proxy pool
    to ensure consistent IP usage and proper block detection.

    Args:
        video_id: Database ID of video to update

    Returns:
        Dictionary with update result
    """
    try:
        with get_db_context() as db:
            video = db.query(Video).filter_by(id=video_id).first()

            if not video:
                logger.error(f"Video {video_id} not found")
                return {'status': 'error', 'error': 'Video not found'}

            if not video.source_post_id:
                logger.warning(f"Video {video_id} has no source_post_id")
                return {'status': 'error', 'error': 'No source post ID'}

            # Get subreddit for proxy lookup
            subreddit = video.source_subreddit

            # Get proxy assigned to this subreddit (if any)
            proxy_url = get_proxy_for_subreddit(subreddit) if subreddit else None
            if proxy_url:
                proxy_host = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
                logger.info(f"Updating upvotes for video {video_id} via proxy {proxy_host}")
            else:
                logger.info(f"Updating upvotes for video {video_id} (direct, no proxy)")

            # Fetch post from Reddit using JSON API with proxy support
            # check_blocks=True ensures block events are recorded with proxy info
            with RedditJsonScraper(proxy_url=proxy_url, check_blocks=True) as scraper:
                post = scraper.get_post_by_id(video.source_post_id, subreddit=subreddit)

            if not post:
                logger.warning(f"Could not fetch post {video.source_post_id} from Reddit")
                return {'status': 'error', 'error': 'Post not found on Reddit'}

            # Update upvote count
            new_upvotes = post.get('score', 0)
            old_upvotes = video.upvotes
            video.upvotes = new_upvotes
            video.last_upvote_check = func.now()
            db.commit()

            logger.info(
                f"Updated video {video_id} upvotes: {old_upvotes} → {new_upvotes} "
                f"(change: {new_upvotes - old_upvotes:+d})"
            )

            return {
                'status': 'success',
                'video_id': video_id,
                'post_id': video.source_post_id,
                'subreddit': subreddit,
                'old_upvotes': old_upvotes,
                'new_upvotes': new_upvotes,
                'change': new_upvotes - old_upvotes,
                'proxy_used': proxy_url is not None
            }

    except Exception as exc:
        logger.error(f"Error updating upvotes for video {video_id}: {exc}")
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task
def update_stale_upvotes(max_videos: int = 50, stale_hours: int = 24):
    """
    Update upvotes for videos that haven't been checked recently

    Args:
        max_videos: Maximum number of videos to update per run (default: 50)
        stale_hours: Consider upvotes stale after this many hours (default: 24)

    Returns:
        Dictionary with update summary
    """
    try:
        logger.info(f"Updating stale upvotes (stale after {stale_hours}h, max {max_videos} videos)")

        with get_db_context() as db:
            stale_threshold = datetime.utcnow() - timedelta(hours=stale_hours)

            # Get videos with stale upvote data
            # Priority: videos with no check yet, then oldest checks first
            stale_videos = db.query(Video).filter(
                (Video.last_upvote_check.is_(None)) |
                (Video.last_upvote_check < stale_threshold)
            ).order_by(
                Video.last_upvote_check.asc().nullsfirst()
            ).limit(max_videos).all()

            if not stale_videos:
                logger.info("No stale videos found")
                return {
                    'status': 'success',
                    'videos_found': 0,
                    'updates_queued': 0,
                    'message': 'All upvotes are up to date'
                }

            logger.info(f"Found {len(stale_videos)} videos with stale upvotes")

            # Queue update tasks for each video
            from celery import group
            update_tasks = group(
                update_video_upvotes.s(video.id)
                for video in stale_videos
            )

            result = update_tasks.apply_async()

            return {
                'status': 'success',
                'videos_found': len(stale_videos),
                'updates_queued': len(stale_videos),
                'stale_hours': stale_hours,
                'group_task_id': result.id
            }

    except Exception as e:
        logger.error(f"Error in update_stale_upvotes: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def update_popular_videos_upvotes(min_upvotes: int = 1000, max_videos: int = 100):
    """
    Update upvotes for popular videos (high upvote count)
    These are more likely to gain upvotes over time

    Args:
        min_upvotes: Only update videos with at least this many upvotes (default: 1000)
        max_videos: Maximum number of videos to update (default: 100)

    Returns:
        Dictionary with update summary
    """
    try:
        logger.info(f"Updating popular videos (min {min_upvotes} upvotes, max {max_videos} videos)")

        with get_db_context() as db:
            # Get popular videos sorted by upvotes
            popular_videos = db.query(Video).filter(
                Video.upvotes >= min_upvotes
            ).order_by(
                Video.upvotes.desc()
            ).limit(max_videos).all()

            if not popular_videos:
                logger.info(f"No videos with {min_upvotes}+ upvotes found")
                return {
                    'status': 'success',
                    'videos_found': 0,
                    'updates_queued': 0,
                    'message': f'No videos with {min_upvotes}+ upvotes'
                }

            logger.info(f"Found {len(popular_videos)} popular videos to update")

            # Queue update tasks
            from celery import group
            update_tasks = group(
                update_video_upvotes.s(video.id)
                for video in popular_videos
            )

            result = update_tasks.apply_async()

            return {
                'status': 'success',
                'videos_found': len(popular_videos),
                'updates_queued': len(popular_videos),
                'min_upvotes': min_upvotes,
                'group_task_id': result.id
            }

    except Exception as e:
        logger.error(f"Error in update_popular_videos_upvotes: {e}")
        return {'status': 'error', 'error': str(e)}
