"""
API endpoints for controlling scraper tasks without deployment.

Uses Redis to store scraper state, checked by a periodic Celery task.
"""
import redis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config.settings import settings

router = APIRouter(prefix="/scraper/control", tags=["scraper-control"])

# Redis key for scraper state (must match tasks/scraper_dispatcher.py)
SCRAPER_ENABLED_KEY = "scraper:enabled"
UPVOTES_ENABLED_KEY = "scraper:upvotes_enabled"
AUTO_DISABLE_ON_BLOCK_KEY = "scraper:auto_disable_on_block"

def get_redis_client():
    """Get Redis client from Celery broker URL."""
    # Parse redis URL from celery broker
    redis_url = settings.celery_broker_url
    return redis.from_url(redis_url)


class ScraperStatus(BaseModel):
    scraper_enabled: bool
    upvotes_enabled: bool
    auto_disable_on_block: bool
    message: str


class ScraperConfig(BaseModel):
    scraper_enabled: Optional[bool] = None
    upvotes_enabled: Optional[bool] = None
    auto_disable_on_block: Optional[bool] = None


@router.get("/status", response_model=ScraperStatus)
def get_scraper_status():
    """Get current scraper and upvote task status."""
    try:
        r = get_redis_client()
        scraper_enabled = r.get(SCRAPER_ENABLED_KEY)
        upvotes_enabled = r.get(UPVOTES_ENABLED_KEY)
        auto_disable = r.get(AUTO_DISABLE_ON_BLOCK_KEY)

        # Default to disabled if not set (except auto_disable defaults to True)
        scraper_on = scraper_enabled == b"1" if scraper_enabled else False
        upvotes_on = upvotes_enabled == b"1" if upvotes_enabled else False
        auto_disable_on = auto_disable != b"0"  # Default to True

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=auto_disable_on,
            message=f"Scraper: {'enabled' if scraper_on else 'disabled'}, Upvotes: {'enabled' if upvotes_on else 'disabled'}, Auto-disable on block: {'enabled' if auto_disable_on else 'disabled'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/enable", response_model=ScraperStatus)
def enable_all():
    """Enable both scraper and upvote tasks."""
    try:
        r = get_redis_client()
        r.set(SCRAPER_ENABLED_KEY, "1")
        r.set(UPVOTES_ENABLED_KEY, "1")

        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=True,
            upvotes_enabled=True,
            auto_disable_on_block=auto_disable_on,
            message="All scraper tasks enabled. Will start on next check cycle (every 10 minutes)."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable: {str(e)}")


@router.post("/disable", response_model=ScraperStatus)
def disable_all():
    """Disable both scraper and upvote tasks."""
    try:
        r = get_redis_client()
        r.set(SCRAPER_ENABLED_KEY, "0")
        r.set(UPVOTES_ENABLED_KEY, "0")

        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=False,
            upvotes_enabled=False,
            auto_disable_on_block=auto_disable_on,
            message="All scraper tasks disabled. Running tasks will complete but no new ones will start."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable: {str(e)}")


@router.post("/configure", response_model=ScraperStatus)
def configure_scrapers(config: ScraperConfig):
    """Configure individual scraper tasks."""
    try:
        r = get_redis_client()

        if config.scraper_enabled is not None:
            r.set(SCRAPER_ENABLED_KEY, "1" if config.scraper_enabled else "0")

        if config.upvotes_enabled is not None:
            r.set(UPVOTES_ENABLED_KEY, "1" if config.upvotes_enabled else "0")

        if config.auto_disable_on_block is not None:
            r.set(AUTO_DISABLE_ON_BLOCK_KEY, "1" if config.auto_disable_on_block else "0")

        # Get current state
        scraper_on = r.get(SCRAPER_ENABLED_KEY) == b"1"
        upvotes_on = r.get(UPVOTES_ENABLED_KEY) == b"1"
        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=auto_disable_on,
            message="Configuration updated."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure: {str(e)}")


@router.post("/scraper/enable", response_model=ScraperStatus)
def enable_scraper_only():
    """Enable only the scraper task (not upvotes)."""
    try:
        r = get_redis_client()
        r.set(SCRAPER_ENABLED_KEY, "1")

        upvotes_on = r.get(UPVOTES_ENABLED_KEY) == b"1"
        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=True,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=auto_disable_on,
            message="Scraper enabled."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable scraper: {str(e)}")


@router.post("/scraper/disable", response_model=ScraperStatus)
def disable_scraper_only():
    """Disable only the scraper task."""
    try:
        r = get_redis_client()
        r.set(SCRAPER_ENABLED_KEY, "0")

        upvotes_on = r.get(UPVOTES_ENABLED_KEY) == b"1"
        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=False,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=auto_disable_on,
            message="Scraper disabled."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable scraper: {str(e)}")


@router.post("/upvotes/enable", response_model=ScraperStatus)
def enable_upvotes_only():
    """Enable only the upvote update task."""
    try:
        r = get_redis_client()
        r.set(UPVOTES_ENABLED_KEY, "1")

        scraper_on = r.get(SCRAPER_ENABLED_KEY) == b"1"
        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=True,
            auto_disable_on_block=auto_disable_on,
            message="Upvote updates enabled."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable upvotes: {str(e)}")


@router.post("/upvotes/disable", response_model=ScraperStatus)
def disable_upvotes_only():
    """Disable only the upvote update task."""
    try:
        r = get_redis_client()
        r.set(UPVOTES_ENABLED_KEY, "0")

        scraper_on = r.get(SCRAPER_ENABLED_KEY) == b"1"
        auto_disable_on = r.get(AUTO_DISABLE_ON_BLOCK_KEY) != b"0"

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=False,
            auto_disable_on_block=auto_disable_on,
            message="Upvote updates disabled."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable upvotes: {str(e)}")


@router.post("/auto-disable/enable", response_model=ScraperStatus)
def enable_auto_disable_on_block():
    """Enable auto-disable when Reddit block is detected."""
    try:
        r = get_redis_client()
        r.set(AUTO_DISABLE_ON_BLOCK_KEY, "1")

        scraper_on = r.get(SCRAPER_ENABLED_KEY) == b"1"
        upvotes_on = r.get(UPVOTES_ENABLED_KEY) == b"1"

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=True,
            message="Auto-disable on block enabled. Scrapers will pause when Reddit blocks are detected."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable auto-disable: {str(e)}")


@router.post("/auto-disable/disable", response_model=ScraperStatus)
def disable_auto_disable_on_block():
    """Disable auto-disable when Reddit block is detected (scraping will continue even if blocked)."""
    try:
        r = get_redis_client()
        r.set(AUTO_DISABLE_ON_BLOCK_KEY, "0")

        scraper_on = r.get(SCRAPER_ENABLED_KEY) == b"1"
        upvotes_on = r.get(UPVOTES_ENABLED_KEY) == b"1"

        return ScraperStatus(
            scraper_enabled=scraper_on,
            upvotes_enabled=upvotes_on,
            auto_disable_on_block=False,
            message="Auto-disable on block disabled. Scrapers will continue even when blocks are detected (not recommended)."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable auto-disable: {str(e)}")
