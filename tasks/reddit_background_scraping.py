"""
Reddit background video scraping tasks.

Scrapes the per-niche background subreddits for background videos (not captions).
Downloads videos and feeds them into the background pipeline:
BackgroundVideo table → watermark filter → ML tagging → composition.

Uses the same RedditJsonScraper and VideoDownloader as caption scraping,
but stores in BackgroundVideo instead of Video, and does NOT queue OCR extraction.
"""
import os
import hashlib
from tasks.celery_app import celery_app
from scrapers.reddit_json_scraper import RedditJsonScraper
from scrapers.video_downloader import VideoDownloader
from scrapers.exceptions import PermanentDownloadError, TemporaryDownloadError
from database.db import get_db_context
from database.models import BackgroundVideo, RedditBackgroundSubreddit, Video
from sqlalchemy.sql import func
from config.settings import settings
import redis
import logging

logger = logging.getLogger(__name__)

# Storage path (background_videos Docker volume)
BG_VIDEO_STORAGE = "/data/background_videos"

# Redis control key
REDDIT_BG_ENABLED_KEY = "reddit_bg:enabled"

# Reuse stage config from caption scraping
SCRAPE_STAGES = ['top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new']

STAGE_CONFIG = {
    'top_all': {'sort': 'top', 'time_filter': 'all'},
    'top_year': {'sort': 'top', 'time_filter': 'year'},
    'top_month': {'sort': 'top', 'time_filter': 'month'},
    'top_week': {'sort': 'top', 'time_filter': 'week'},
    'top_day': {'sort': 'top', 'time_filter': 'day'},
    'new': {'sort': 'new', 'time_filter': None},
}


def get_redis_client():
    return redis.from_url(settings.celery_broker_url)


def get_next_stage(current_stage: str) -> str:
    """Get the next stage in progression, or stay at 'new' if already there."""
    try:
        idx = SCRAPE_STAGES.index(current_stage)
        if idx < len(SCRAPE_STAGES) - 1:
            return SCRAPE_STAGES[idx + 1]
    except ValueError:
        pass
    return 'new'


def _calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


@celery_app.task(bind=True, max_retries=3, queue='scraping')
def scrape_reddit_background_subreddit(self, subreddit_name: str, batch_size: int = 10):
    """
    Scrape a Reddit subreddit for background videos.

    Uses progressive depth strategy (top_all → new), downloads video-only posts,
    and stores in BackgroundVideo table for watermark filter → ML tagging pipeline.

    Args:
        subreddit_name: Subreddit to scrape (without r/)
        batch_size: Max videos to download per run
    """
    try:
        logger.info(f"Starting Reddit background scrape of r/{subreddit_name}")

        # Load subreddit config
        with get_db_context() as db:
            sub_config = db.query(RedditBackgroundSubreddit).filter_by(
                subreddit=subreddit_name
            ).first()

            if not sub_config:
                return {'status': 'error', 'message': f'Subreddit not configured: {subreddit_name}'}
            if not sub_config.enabled:
                return {'status': 'skipped', 'message': f'Subreddit disabled: {subreddit_name}'}

            current_stage = sub_config.scrape_stage or 'top_all'
            last_pagination_url = sub_config.last_pagination_url
            posts_scraped = sub_config.posts_scraped or 0
            min_score = sub_config.min_score or 100
            min_duration = sub_config.min_duration or 10
            max_duration = sub_config.max_duration or 60
            configured_batch = sub_config.batch_size or batch_size

        # Stage config
        stage_config = STAGE_CONFIG.get(current_stage, STAGE_CONFIG['new'])
        sort_method = stage_config['sort']
        time_filter = stage_config['time_filter']

        logger.info(
            f"r/{subreddit_name} - Stage: {current_stage} "
            f"(sort={sort_method}, time={time_filter}, min_score={min_score})"
        )

        # Get existing post IDs for deduplication
        with get_db_context() as db:
            existing_bg_post_ids = {
                v.reddit_post_id for v in
                db.query(BackgroundVideo.reddit_post_id).filter(
                    BackgroundVideo.reddit_post_id.isnot(None)
                ).all()
            }
            existing_caption_post_ids = {
                v.source_post_id for v in
                db.query(Video.source_post_id).all()
            }

        all_existing_ids = existing_bg_post_ids | existing_caption_post_ids

        # Get proxy
        proxy_url = None
        try:
            from utils.proxy_pool import get_proxy_for_subreddit
            proxy_url = get_proxy_for_subreddit(f"bg_{subreddit_name}", auto_assign=True)
            if proxy_url:
                proxy_host = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
                logger.info(f"Using proxy {proxy_host} for r/{subreddit_name}")
        except Exception as e:
            logger.warning(f"Proxy pool not available: {e}")

        os.makedirs(BG_VIDEO_STORAGE, exist_ok=True)

        videos_downloaded = 0
        videos_skipped = 0
        videos_failed = 0
        should_advance_stage = False
        total_pages_fetched = 0
        max_pages_per_run = 10
        current_after_token = last_pagination_url
        current_count = posts_scraped

        with RedditJsonScraper(proxy_url=proxy_url) as scraper:
            while total_pages_fetched < max_pages_per_run and videos_downloaded < configured_batch:
                total_pages_fetched += 1

                posts, next_after_token, updated_count = scraper.get_video_posts(
                    subreddit_name=subreddit_name,
                    limit=min(100, configured_batch * 3),
                    sort=sort_method,
                    min_score=min_score,
                    time_filter=time_filter,
                    after=current_after_token,
                    count=current_count,
                )

                if not posts and not next_after_token:
                    # Pagination exhausted — advance stage
                    should_advance_stage = True
                    logger.info(f"r/{subreddit_name}: stage '{current_stage}' exhausted, advancing")
                    break

                current_after_token = next_after_token
                current_count = updated_count

                # Filter for video-only posts not already downloaded
                new_video_posts = []
                for post in posts:
                    if post.get('media_type') != 'video':
                        continue
                    if post['id'] in all_existing_ids:
                        videos_skipped += 1
                        continue
                    new_video_posts.append(post)
                    all_existing_ids.add(post['id'])

                if not new_video_posts:
                    if not next_after_token:
                        should_advance_stage = True
                        break
                    continue

                # Download each video
                for post in new_video_posts:
                    if videos_downloaded >= configured_batch:
                        break

                    post_id = post['id']
                    video_url = post.get('video_url')
                    if not video_url:
                        continue

                    try:
                        # Check duration before full download using yt-dlp info extraction
                        downloader = VideoDownloader(
                            output_path=BG_VIDEO_STORAGE, use_proxy=False
                        )
                        info = downloader.get_info(video_url)
                        if info:
                            duration = info.get('duration') or 0
                            if duration < min_duration or duration > max_duration:
                                logger.debug(
                                    f"Skipping {post_id}: duration {duration}s "
                                    f"(range: {min_duration}-{max_duration}s)"
                                )
                                continue

                        # Download the video
                        result = downloader.download(video_url, f"bg_{post_id}")

                        file_path = result['file_path']
                        file_hash = result['file_hash']
                        duration = result.get('duration', 0)
                        file_size = result.get('file_size', 0)

                        # Post-download duration check (in case info extraction was incomplete)
                        if duration and (duration < min_duration or duration > max_duration):
                            logger.debug(f"Removing {post_id}: duration {duration}s out of range")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            continue

                        # Check hash dedup
                        with get_db_context() as db:
                            existing_hash = db.query(BackgroundVideo).filter_by(
                                file_hash=file_hash
                            ).first()
                            if existing_hash:
                                logger.info(f"Skipping {post_id}: duplicate content (hash match)")
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                continue

                            # Parse resolution
                            width = None
                            height = None
                            res = result.get('resolution')
                            if res and 'x' in str(res):
                                try:
                                    parts = str(res).split('x')
                                    width = int(float(parts[0]))
                                    height = int(float(parts[1]))
                                except (ValueError, IndexError):
                                    pass

                            # Save to database
                            bg_video = BackgroundVideo(
                                source_type='reddit',
                                reddit_post_id=post_id,
                                reddit_subreddit=subreddit_name,
                                reddit_score=post.get('score', 0),
                                source_url=f"https://www.reddit.com/r/{subreddit_name}/comments/{post_id}",
                                storage_path=file_path,
                                file_hash=file_hash,
                                duration_seconds=duration,
                                width=width,
                                height=height,
                                file_size_bytes=file_size,
                                views=post.get('score', 0),  # Use score as views proxy
                                tags=[subreddit_name],
                                searched_tag=subreddit_name,
                                download_status='completed',
                                filter_status='pending',
                            )
                            db.add(bg_video)
                            db.commit()

                            # Queue watermark filter if enabled
                            video_db_id = bg_video.id
                            r = get_redis_client()
                            filter_key = "watermark_filter:enabled"
                            if r.get(filter_key) == b"1":
                                try:
                                    from tasks.watermark_filter import filter_background_video
                                    filter_background_video.delay(video_db_id)
                                except Exception as filter_err:
                                    logger.warning(f"Could not queue filter: {filter_err}")

                        videos_downloaded += 1
                        logger.info(
                            f"Downloaded r/{subreddit_name}/{post_id}: "
                            f"{file_size // 1024 // 1024}MB, {duration}s, "
                            f"{post.get('score', 0)} upvotes"
                        )

                    except PermanentDownloadError as e:
                        logger.warning(f"Permanent error for {post_id}: {e}")
                        videos_failed += 1
                        continue
                    except (TemporaryDownloadError, Exception) as e:
                        logger.error(f"Failed to download {post_id}: {e}")
                        videos_failed += 1
                        continue

                if not next_after_token:
                    should_advance_stage = True
                    break

        # Update subreddit progress
        with get_db_context() as db:
            sub_config = db.query(RedditBackgroundSubreddit).filter_by(
                subreddit=subreddit_name
            ).first()
            if sub_config:
                if should_advance_stage:
                    next_stage = get_next_stage(current_stage)
                    sub_config.scrape_stage = next_stage
                    sub_config.last_pagination_url = None
                    sub_config.posts_scraped = 0
                    logger.info(f"r/{subreddit_name}: advanced from '{current_stage}' to '{next_stage}'")
                else:
                    sub_config.last_pagination_url = current_after_token
                    sub_config.posts_scraped = current_count

                sub_config.videos_downloaded = (sub_config.videos_downloaded or 0) + videos_downloaded
                sub_config.videos_failed = (sub_config.videos_failed or 0) + videos_failed
                sub_config.last_scrape_at = func.now()
                db.commit()

                final_stage = sub_config.scrape_stage

        return {
            'status': 'success',
            'subreddit': subreddit_name,
            'scrape_stage': final_stage,
            'videos_downloaded': videos_downloaded,
            'videos_skipped': videos_skipped,
            'videos_failed': videos_failed,
            'pages_fetched': total_pages_fetched,
            'stage_advanced': should_advance_stage,
        }

    except Exception as exc:
        logger.error(f"Error scraping r/{subreddit_name} for backgrounds: {exc}")
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(
    name='tasks.reddit_background_scraping.dispatch_reddit_background_scraper',
    queue='maintenance'
)
def dispatch_reddit_background_scraper():
    """
    Dispatcher - runs every 10 min via Celery Beat.
    Checks Redis flag, block status, and dispatches per-subreddit scraping.
    """
    try:
        r = get_redis_client()

        # Check if enabled
        if r.get(REDDIT_BG_ENABLED_KEY) != b"1":
            logger.debug("Reddit background scraping disabled - skipping")
            return {'status': 'disabled', 'dispatched': 0}

        # Check Reddit block status (reuses caption scraping block detector)
        try:
            from utils.reddit_block_detector import should_pause_scraping
            should_pause, pause_reason = should_pause_scraping()
            if should_pause:
                logger.warning(f"Reddit background scraping paused: {pause_reason}")
                return {'status': 'blocked', 'reason': pause_reason, 'dispatched': 0}
        except ImportError:
            pass  # Block detector not available, continue anyway

        # Get enabled subreddits
        with get_db_context() as db:
            enabled_subs = db.query(RedditBackgroundSubreddit).filter_by(enabled=True).all()

            if not enabled_subs:
                return {'status': 'no_subreddits', 'dispatched': 0}

            # Sort by least recently scraped first
            sorted_subs = sorted(
                enabled_subs,
                key=lambda s: s.last_scrape_at or s.created_at
            )

            # Dispatch up to 5 per cycle
            dispatched = []
            for sub in sorted_subs[:5]:
                logger.info(f"Dispatching Reddit BG scrape: r/{sub.subreddit}")
                scrape_reddit_background_subreddit.delay(
                    sub.subreddit, batch_size=sub.batch_size or 10
                )
                dispatched.append(sub.subreddit)

        logger.info(f"Dispatched Reddit BG scrape for {len(dispatched)} subreddits: {dispatched}")
        return {'status': 'ok', 'dispatched': len(dispatched), 'subreddits': dispatched}

    except Exception as e:
        logger.error(f"Reddit BG dispatcher error: {e}")
        return {'error': str(e), 'status': 'error'}
