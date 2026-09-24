"""
Dispatcher task that checks Redis flags and dispatches scraper tasks.

This allows enabling/disabling scrapers via API without deployment.
Includes Reddit access health checking and auto-disable on block detection.
"""
import redis
import logging
from tasks.celery_app import celery_app
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis keys (must match api/scraper_control.py)
SCRAPER_ENABLED_KEY = "scraper:enabled"
UPVOTES_ENABLED_KEY = "scraper:upvotes_enabled"
AUTO_DISABLE_ON_BLOCK_KEY = "scraper:auto_disable_on_block"


def get_redis_client():
    """Get Redis client from Celery broker URL."""
    return redis.from_url(settings.celery_broker_url)


@celery_app.task(name='tasks.scraper_dispatcher.dispatch_scraper_tasks')
def dispatch_scraper_tasks():
    """
    Check Redis flags and dispatch scraper tasks if enabled.

    This task runs every 10 minutes via beat schedule and:
    - Checks if Reddit is blocked -> skips if blocked and auto-disable is on
    - Checks if scraper is enabled -> dispatches incremental_scrape_subreddit
    """
    try:
        r = get_redis_client()

        # Check if we should auto-disable on block
        auto_disable = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"  # Default to True

        # Check for blocks before dispatching
        if auto_disable:
            from utils.reddit_block_detector import should_pause_scraping
            should_pause, pause_reason = should_pause_scraping()
            if should_pause:
                logger.warning(f"Scraper paused due to block: {pause_reason}")
                return {
                    'scraper_dispatched': False,
                    'status': 'blocked',
                    'reason': pause_reason
                }

        # Check scraper flag
        scraper_enabled = r.get(SCRAPER_ENABLED_KEY) == b"1"
        if scraper_enabled:
            logger.info("Scraper enabled - dispatching incremental scrape tasks for all enabled subreddits")
            # Import here to avoid circular imports
            from tasks.incremental_scraping import incremental_scrape_subreddit
            from database.db import get_db_context
            from database.models import ScrapingProgress

            # Get all enabled subreddits from database
            dispatched_subreddits = []
            with get_db_context() as db:
                enabled_subreddits = db.query(ScrapingProgress).filter_by(scraping_active=True).all()
                for progress in enabled_subreddits:
                    subreddit_name = progress.subreddit
                    batch_size = progress.batch_size or 25
                    min_score = progress.target_min_score or 300
                    logger.info(f"Dispatching scrape for r/{subreddit_name} (batch={batch_size}, min_score={min_score})")
                    incremental_scrape_subreddit.delay(subreddit_name, batch_size, min_score)
                    dispatched_subreddits.append(subreddit_name)

            logger.info(f"Dispatched scrape tasks for {len(dispatched_subreddits)} subreddits: {dispatched_subreddits}")
        else:
            logger.debug("Scraper disabled - skipping incremental scrape")
            dispatched_subreddits = []

        return {
            'scraper_dispatched': scraper_enabled,
            'subreddits_dispatched': dispatched_subreddits,
            'status': 'ok'
        }

    except Exception as e:
        logger.error(f"Dispatcher error: {e}")
        return {'error': str(e), 'status': 'error'}


@celery_app.task(name='tasks.scraper_dispatcher.check_reddit_health')
def check_reddit_health():
    """
    Periodic health check that tests Reddit access through assigned proxies.

    Runs hourly to detect blocks on proxies actively used by scrapers.
    Only checks proxies currently assigned to subreddits - if no proxies
    assigned, performs a single direct check.

    Records per-proxy status in Redis for monitoring.
    """
    try:
        from utils.reddit_block_detector import (
            check_reddit_access,
            record_block_event,
            get_block_stats
        )
        from utils.proxy_pool import get_proxy_pool

        logger.info("Running periodic Reddit health check (proxy-aware)")

        # Get proxy pool and find assigned proxies
        pool = get_proxy_pool()
        assignments = pool._get_assignments()

        results = []

        if not assignments:
            # No proxies assigned - do a single check without proxy
            logger.info("No proxy assignments found, checking direct access")
            status = check_reddit_access(test_subreddit="pics", timeout=15)
            record_block_event(status)
            results.append({
                'proxy': 'direct',
                'blocked': status.blocked,
                'reason': status.reason
            })
        else:
            # Check each assigned proxy
            checked_proxies = set()
            for subreddit, proxy_url in assignments.items():
                # Extract host:port for deduplication
                if '@' in proxy_url:
                    proxy_host = proxy_url.split('@')[-1]
                else:
                    proxy_host = proxy_url

                # Skip if we already checked this proxy
                if proxy_host in checked_proxies:
                    continue
                checked_proxies.add(proxy_host)

                # Format full proxy URL
                full_proxy_url = f"http://{proxy_url}" if not proxy_url.startswith('http') else proxy_url

                logger.info(f"Health checking proxy {proxy_host} (assigned to r/{subreddit})")

                status = check_reddit_access(
                    test_subreddit="pics",
                    timeout=15,
                    proxy_url=full_proxy_url
                )
                record_block_event(status)

                results.append({
                    'proxy': proxy_host,
                    'subreddit': subreddit,
                    'blocked': status.blocked,
                    'block_type': status.block_type.value,
                    'reason': status.reason,
                    'response_time_ms': status.response_time_ms
                })

                if status.blocked:
                    logger.warning(
                        f"Proxy {proxy_host} BLOCKED: {status.block_type.value} - {status.reason}"
                    )
                else:
                    logger.info(
                        f"Proxy {proxy_host} OK: {status.reason} "
                        f"({status.response_time_ms:.0f}ms)"
                    )

        # Count blocked vs ok
        blocked_count = sum(1 for r in results if r.get('blocked'))
        ok_count = len(results) - blocked_count

        # Get overall stats for return
        stats = get_block_stats()

        return {
            'status': 'some_blocked' if blocked_count > 0 else 'all_ok',
            'proxies_checked': len(results),
            'proxies_blocked': blocked_count,
            'proxies_ok': ok_count,
            'results': results,
            'block_rate_percent': stats.get('block_rate_percent', 0)
        }

    except Exception as e:
        logger.error(f"Reddit health check error: {e}")
        return {'error': str(e), 'status': 'error'}


@celery_app.task(name='tasks.scraper_dispatcher.dispatch_upvote_tasks')
def dispatch_upvote_tasks():
    """
    Check Redis flag and dispatch upvote update task if enabled.

    This task runs hourly via beat schedule.
    """
    try:
        r = get_redis_client()

        # Check upvotes flag
        upvotes_enabled = r.get(UPVOTES_ENABLED_KEY) == b"1"
        if upvotes_enabled:
            logger.info("Upvotes enabled - dispatching upvote update task")
            from tasks.upvote_updater import update_stale_upvotes
            update_stale_upvotes.delay(50, 24)
        else:
            logger.debug("Upvotes disabled - skipping upvote update")

        return {
            'upvotes_dispatched': upvotes_enabled,
            'status': 'ok'
        }

    except Exception as e:
        logger.error(f"Upvote dispatcher error: {e}")
        return {'error': str(e), 'status': 'error'}
