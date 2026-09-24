"""
Incremental scraping tasks for persistent, resumable scraping
Scrapes subreddits in batches, tracking progress to resume later

Progressive Depth Scraping Strategy:
1. Start at 'top_all' (top posts of all time)
2. Progress through time filters when pagination ends
3. Eventually settle on 'new' for ongoing scraping

Stage progression: top_all -> top_year -> top_month -> top_week -> top_day -> new
"""
from celery import group
from tasks.celery_app import celery_app
from scrapers.reddit_json_scraper import RedditJsonScraper
from database.db import get_db_context
from database.models import Video, ScrapingProgress
from tasks.scraping_tasks import download_video_task
from sqlalchemy.sql import func
import logging

logger = logging.getLogger(__name__)

# Stage progression order for progressive depth scraping
SCRAPE_STAGES = ['top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new']

# Map stages to Reddit API parameters
STAGE_CONFIG = {
    'top_all': {'sort': 'top', 'time_filter': 'all', 'min_score': 300},
    'top_year': {'sort': 'top', 'time_filter': 'year', 'min_score': 300},
    'top_month': {'sort': 'top', 'time_filter': 'month', 'min_score': 300},
    'top_week': {'sort': 'top', 'time_filter': 'week', 'min_score': 300},
    'top_day': {'sort': 'top', 'time_filter': 'day', 'min_score': 300},
    'new': {'sort': 'new', 'time_filter': None, 'min_score': 25},
}


def get_next_stage(current_stage: str) -> str:
    """Get the next stage in progression, or stay at 'new' if already there"""
    try:
        current_index = SCRAPE_STAGES.index(current_stage)
        if current_index < len(SCRAPE_STAGES) - 1:
            return SCRAPE_STAGES[current_index + 1]
    except ValueError:
        pass
    return 'new'  # Default to 'new' if current stage is invalid or already at end


@celery_app.task(bind=True, max_retries=3)
def incremental_scrape_subreddit(
    self,
    subreddit_name: str,
    batch_size: int = 25,
    target_min_score: int = 300
):
    """
    Incrementally scrape a subreddit using progressive depth strategy.

    PROGRESSIVE DEPTH STRATEGY:
    1. Start at 'top_all' (top posts of all time) - best content first
    2. When pagination ends, advance to next time filter
    3. Progression: top_all -> top_year -> top_month -> top_week -> top_day -> new
    4. Stay on 'new' for continuous ongoing scraping

    Stage transition occurs when Reddit returns no 'after' pagination token.

    Args:
        subreddit_name: Name of subreddit to scrape
        batch_size: Number of posts to fetch per run (default: 25)
        target_min_score: Base min score, but uses stage-specific thresholds

    Returns:
        Dictionary with scraping results
    """
    try:
        logger.info(f"Starting incremental scrape of r/{subreddit_name}")

        # Get or create scraping progress tracker
        with get_db_context() as db:
            progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

            if not progress:
                # First time scraping this subreddit - start at top_all
                progress = ScrapingProgress(
                    subreddit=subreddit_name,
                    target_min_score=target_min_score,
                    scrape_stage='top_all',
                    scraping_active=True
                )
                db.add(progress)
                db.commit()
                db.refresh(progress)
                logger.info(f"Created new scraping progress tracker for r/{subreddit_name} (starting at top_all)")

            # Check if scraping is still active (manual disable only)
            if not progress.scraping_active:
                videos_downloaded = progress.videos_downloaded
                current_stage = progress.scrape_stage or 'top_all'
                logger.info(f"Scraping is disabled for r/{subreddit_name} (manually disabled)")
                return {
                    'status': 'completed',
                    'subreddit': subreddit_name,
                    'message': f'Scraping disabled - collected {videos_downloaded} videos so far',
                    'videos_downloaded': videos_downloaded,
                    'scrape_stage': current_stage
                }

            # Get current scraping state
            current_stage = progress.scrape_stage or 'top_all'
            last_post_id = progress.last_post_id
            last_post_score = progress.last_post_score
            last_pagination_url = progress.last_pagination_url
            posts_scraped = progress.posts_scraped
            configured_batch_size = progress.batch_size or batch_size

        # Get stage-specific configuration
        stage_config = STAGE_CONFIG.get(current_stage, STAGE_CONFIG['new'])
        sort_method = stage_config['sort']
        time_filter = stage_config['time_filter']
        min_score_for_stage = stage_config['min_score']

        logger.info(
            f"Scraping r/{subreddit_name} - Stage: {current_stage} "
            f"(sort={sort_method}, time={time_filter}, min_score={min_score_for_stage})"
        )

        if last_pagination_url:
            logger.info(f"Resuming from saved pagination position (after token: {last_pagination_url})")

        # Get all post IDs we've already processed from database
        # This provides persistent deduplication across all stages
        with get_db_context() as db:
            existing_post_ids = {
                v.source_post_id
                for v in db.query(Video.source_post_id).all()
            }

        # Track seen post IDs within this session for faster duplicate skipping
        seen_post_ids = set()

        # Get dedicated proxy for this subreddit from pool
        proxy_url = None
        try:
            from utils.proxy_pool import get_proxy_for_subreddit
            proxy_url = get_proxy_for_subreddit(subreddit_name, auto_assign=True)
            if proxy_url:
                proxy_host = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
                logger.info(f"Using dedicated proxy {proxy_host} for r/{subreddit_name}")
        except Exception as e:
            logger.warning(f"Proxy pool not available: {e}, scraping without proxy")

        # Continue paginating until we find new posts or hit Reddit's limit
        # This ensures we exhaust each stage before advancing
        posts_to_process = []
        should_advance_stage = False
        total_duplicates_skipped = 0
        total_pages_fetched = 0
        max_pages_per_run = 20  # Safety limit to prevent infinite loops
        current_after_token = last_pagination_url
        current_count = posts_scraped

        with RedditJsonScraper(proxy_url=proxy_url) as scraper:
            while total_pages_fetched < max_pages_per_run:
                total_pages_fetched += 1

                # Fetch posts using JSON API with stage-specific parameters
                posts, next_after_token, updated_count = scraper.get_video_posts(
                    subreddit_name=subreddit_name,
                    limit=configured_batch_size,
                    min_score=min_score_for_stage,
                    sort=sort_method,
                    time_filter=time_filter,
                    after=current_after_token,
                    count=current_count
                )

                # Check if pagination is exhausted (should advance stage)
                if not next_after_token and current_stage != 'new':
                    should_advance_stage = True
                    next_stage = get_next_stage(current_stage)
                    logger.info(
                        f"Pagination ended for r/{subreddit_name} at stage '{current_stage}' - "
                        f"will advance to '{next_stage}'"
                    )

                if not posts:
                    logger.info(f"No posts returned for r/{subreddit_name} at stage {current_stage} (page {total_pages_fetched})")
                    break

                # Filter out posts we've already processed (database) or seen this session
                new_posts = [
                    p for p in posts
                    if p['id'] not in existing_post_ids and p['id'] not in seen_post_ids
                ]

                # Track these as seen for this session
                seen_post_ids.update(p['id'] for p in posts)
                duplicates_this_page = len(posts) - len(new_posts)
                total_duplicates_skipped += duplicates_this_page

                logger.info(
                    f"Page {total_pages_fetched}: Found {len(new_posts)} new posts out of {len(posts)} fetched "
                    f"(stage: {current_stage}, duplicates skipped: {duplicates_this_page})"
                )

                if new_posts:
                    # Found new posts - add them and stop paginating for this run
                    posts_to_process.extend(new_posts)
                    current_after_token = next_after_token
                    current_count = updated_count
                    break

                # All posts were duplicates - continue to next page if available
                if not next_after_token:
                    logger.info(f"No more pages available for r/{subreddit_name} at stage {current_stage}")
                    break

                # Continue to next page
                current_after_token = next_after_token
                current_count = updated_count
                logger.info(f"All {len(posts)} posts were duplicates, fetching next page...")

        # Sort by score descending to process highest upvotes first
        if posts_to_process:
            posts_to_process.sort(key=lambda x: x.get('score', 0), reverse=True)

        if not posts_to_process:
            logger.info(f"No new posts to process for r/{subreddit_name} after {total_pages_fetched} pages")

            # Update state even if no posts to process
            with get_db_context() as db:
                progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()
                if progress:
                    if should_advance_stage:
                        # Advance to next stage and reset pagination
                        next_stage = get_next_stage(current_stage)
                        progress.scrape_stage = next_stage
                        progress.last_pagination_url = None
                        progress.posts_scraped = 0  # Reset count for new stage
                        logger.info(f"Advanced r/{subreddit_name} from '{current_stage}' to '{next_stage}'")
                    elif current_after_token:
                        # Save pagination for next run
                        progress.last_pagination_url = current_after_token
                        progress.posts_scraped = current_count

                    progress.last_scrape_at = func.now()
                    db.commit()

                    current_stage = progress.scrape_stage

            return {
                'status': 'no_new_posts',
                'subreddit': subreddit_name,
                'scrape_stage': current_stage,
                'message': f'No new posts at stage {current_stage} after {total_pages_fetched} pages',
                'stage_advanced': should_advance_stage,
                'duplicates_skipped': total_duplicates_skipped,
                'pages_fetched': total_pages_fetched
            }

        logger.info(f"Processing {len(posts_to_process)} new posts from r/{subreddit_name} (after {total_pages_fetched} pages)")

        # Queue download tasks for new posts
        download_tasks = group(
            download_video_task.s(
                post['video_url'],
                post['id'],
                post['subreddit'],
                post.get('score', 0),
                post.get('media_type', 'video'),
                post.get('gallery_urls'),
                post.get('gallery_count')
            )
            for post in posts_to_process
        )

        result = download_tasks.apply_async()

        # Update scraping progress
        last_processed_post = posts_to_process[-1]
        with get_db_context() as db:
            progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()
            progress.last_post_id = last_processed_post['id']
            progress.last_post_score = last_processed_post.get('score', 0)
            progress.last_scrape_at = func.now()

            if should_advance_stage:
                # Advance to next stage and reset pagination
                next_stage = get_next_stage(current_stage)
                progress.scrape_stage = next_stage
                progress.last_pagination_url = None
                progress.posts_scraped = 0
                logger.info(f"Advanced r/{subreddit_name} from '{current_stage}' to '{next_stage}'")
            else:
                # Update pagination for current stage
                progress.last_pagination_url = current_after_token
                progress.posts_scraped = current_count

            db.commit()

            # Store values while session is still active
            total_posts_scraped = progress.posts_scraped
            videos_downloaded = progress.videos_downloaded
            scraping_active = progress.scraping_active
            final_stage = progress.scrape_stage

        logger.info(
            f"Scrape completed - processed {len(posts_to_process)} posts, "
            f"queued downloads (stage: {final_stage})"
        )

        return {
            'status': 'success',
            'subreddit': subreddit_name,
            'scrape_stage': final_stage,
            'posts_in_batch': len(posts_to_process),
            'downloads_queued': len(posts_to_process),
            'last_post_score': last_processed_post.get('score', 0),
            'total_posts_scraped': total_posts_scraped,
            'videos_downloaded': videos_downloaded,
            'scraping_active': scraping_active,
            'stage_advanced': should_advance_stage,
            'duplicates_skipped': total_duplicates_skipped,
            'pages_fetched': total_pages_fetched,
            'group_task_id': result.id
        }

    except Exception as exc:
        logger.error(f"Error in incremental scrape of r/{subreddit_name}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task
def get_scraping_progress(subreddit_name: str = None):
    """
    Get scraping progress for a subreddit or all subreddits

    Args:
        subreddit_name: Optional subreddit to check (defaults to all)

    Returns:
        Dictionary with progress information
    """
    try:
        with get_db_context() as db:
            if subreddit_name:
                progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()
                if not progress:
                    return {
                        'error': f'No scraping progress found for r/{subreddit_name}'
                    }

                current_stage = progress.scrape_stage or 'top_all'
                stage_config = STAGE_CONFIG.get(current_stage, STAGE_CONFIG['new'])

                return {
                    'subreddit': progress.subreddit,
                    'scrape_stage': current_stage,
                    'stage_config': stage_config,
                    'posts_scraped': progress.posts_scraped,
                    'videos_downloaded': progress.videos_downloaded,
                    'target_videos': progress.target_videos,
                    'videos_failed': progress.videos_failed,
                    'last_post_id': progress.last_post_id,
                    'last_post_score': progress.last_post_score,
                    'target_min_score': progress.target_min_score,
                    'scraping_active': progress.scraping_active,
                    'last_scrape_at': progress.last_scrape_at.isoformat() if progress.last_scrape_at else None,
                    'created_at': progress.created_at.isoformat() if progress.created_at else None
                }
            else:
                # Get all progress trackers
                all_progress = db.query(ScrapingProgress).all()
                return {
                    'total_subreddits': len(all_progress),
                    'stage_progression': SCRAPE_STAGES,
                    'subreddits': [
                        {
                            'subreddit': p.subreddit,
                            'scrape_stage': p.scrape_stage or 'top_all',
                            'posts_scraped': p.posts_scraped,
                            'videos_downloaded': p.videos_downloaded,
                            'target_videos': p.target_videos,
                            'videos_failed': p.videos_failed,
                            'last_post_score': p.last_post_score,
                            'scraping_active': p.scraping_active,
                            'last_scrape_at': p.last_scrape_at.isoformat() if p.last_scrape_at else None
                        }
                        for p in all_progress
                    ]
                }

    except Exception as e:
        logger.error(f"Error getting scraping progress: {e}")
        return {'error': str(e)}


@celery_app.task
def reset_scraping_progress(subreddit_name: str, reset_to_stage: str = 'top_all'):
    """
    Reset scraping progress for a subreddit.
    Use this to restart scraping from the beginning or a specific stage.

    Args:
        subreddit_name: Subreddit to reset
        reset_to_stage: Stage to reset to (default: 'top_all')
                       Valid: top_all, top_year, top_month, top_week, top_day, new

    Returns:
        Status message
    """
    try:
        # Validate stage
        if reset_to_stage not in SCRAPE_STAGES:
            return {
                'error': f'Invalid stage: {reset_to_stage}. Valid stages: {SCRAPE_STAGES}'
            }

        with get_db_context() as db:
            progress = db.query(ScrapingProgress).filter_by(subreddit=subreddit_name).first()

            if not progress:
                return {
                    'error': f'No scraping progress found for r/{subreddit_name}'
                }

            old_stage = progress.scrape_stage or 'top_all'

            # Reset progress
            progress.scrape_stage = reset_to_stage
            progress.last_post_id = None
            progress.last_post_score = None
            progress.last_pagination_url = None
            progress.posts_scraped = 0
            # Note: videos_downloaded is a cumulative count, don't reset
            # progress.videos_downloaded = 0
            progress.videos_failed = 0
            progress.scraping_active = True
            db.commit()

            logger.info(
                f"Reset scraping progress for r/{subreddit_name} "
                f"from '{old_stage}' to '{reset_to_stage}'"
            )

            return {
                'status': 'success',
                'subreddit': subreddit_name,
                'previous_stage': old_stage,
                'new_stage': reset_to_stage,
                'message': f'Scraping progress reset for r/{subreddit_name} (stage: {reset_to_stage})'
            }

    except Exception as e:
        logger.error(f"Error resetting scraping progress: {e}")
        return {'error': str(e)}
