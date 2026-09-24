"""
Reddit IP block detection utility.

Detects various forms of blocking:
- HTTP 429: Rate limited (temporary)
- HTTP 403: Forbidden (blocked)
- HTTP 503: Service unavailable (often blocking)
- Empty responses: Shadow ban (returns OK but no data)
- CAPTCHA/login redirects: Flagged as bot

Stores block events in Redis for monitoring and auto-disables scraper.
"""
import requests
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis keys for block tracking
BLOCK_STATUS_KEY = "captions:reddit_block_status"
BLOCK_HISTORY_KEY = "captions:reddit_block_history"
BLOCK_INCIDENT_COUNT_KEY = "captions:reddit_block_count"


class BlockType(Enum):
    """Types of blocking detected."""
    NONE = "none"
    RATE_LIMITED = "rate_limited"  # 429 - temporary, back off
    FORBIDDEN = "forbidden"  # 403 - IP blocked
    SERVICE_UNAVAILABLE = "service_unavailable"  # 503 - often blocking
    SHADOW_BLOCKED = "shadow_blocked"  # 200 but empty data
    CAPTCHA = "captcha"  # Requires human verification
    LOGIN_REQUIRED = "login_required"  # Redirected to login
    CONNECTION_ERROR = "connection_error"  # Network issues
    INVALID_RESPONSE = "invalid_response"  # Unexpected response format


@dataclass
class BlockStatus:
    """Result of a block detection check."""
    blocked: bool
    block_type: BlockType
    reason: str
    status_code: Optional[int]
    response_time_ms: Optional[float]
    timestamp: str
    retry_after: Optional[int] = None  # Seconds to wait (from 429 header)
    proxy_host: Optional[str] = None  # Which proxy was used (host:port only, no credentials)

    def to_dict(self) -> Dict:
        return {
            "blocked": str(self.blocked).lower(),  # Redis requires string, not bool
            "block_type": self.block_type.value,
            "reason": self.reason,
            "status_code": self.status_code if self.status_code is not None else "",
            "response_time_ms": self.response_time_ms if self.response_time_ms is not None else "",
            "timestamp": self.timestamp,
            "retry_after": self.retry_after if self.retry_after is not None else "",
            "proxy_host": self.proxy_host if self.proxy_host else "direct"
        }


def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True
    )


def check_reddit_access(
    test_subreddit: str = "pics",
    timeout: int = 10,
    proxy_url: Optional[str] = None
) -> BlockStatus:
    """
    Check if Reddit is accessible and detect blocking.

    Args:
        test_subreddit: Subreddit to test (use popular one for reliability)
        timeout: Request timeout in seconds
        proxy_url: Optional proxy URL (http://user:pass@host:port)

    Returns:
        BlockStatus with detection results
    """
    import time
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    # Extract proxy host for logging (strip credentials)
    proxy_host = None
    if proxy_url:
        # Extract host:port from http://user:pass@host:port
        if '@' in proxy_url:
            proxy_host = proxy_url.split('@')[-1]
        else:
            proxy_host = proxy_url.replace('http://', '').replace('https://', '')

    # Browser-like headers (must match scrapers/reddit_json_scraper.py to avoid bot detection)
    # Important: Don't use "Accept: application/json" - triggers 403 blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    }

    url = f"https://www.reddit.com/r/{test_subreddit}/new.json?limit=5"

    # Configure proxy if provided
    proxies = None
    if proxy_url:
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }

    try:
        # Note: allow_redirects=True (default) works better with proxies
        response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        response_time_ms = (time.time() - start_time) * 1000

        # Check for login/captcha pages in final URL after redirects
        if 'login' in response.url.lower():
            return BlockStatus(
                blocked=True,
                block_type=BlockType.LOGIN_REQUIRED,
                reason=f"Redirected to login: {response.url}",
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                proxy_host=proxy_host
            )
        if 'captcha' in response.url.lower():
            return BlockStatus(
                blocked=True,
                block_type=BlockType.CAPTCHA,
                reason=f"CAPTCHA required: {response.url}",
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                proxy_host=proxy_host
            )

        # Rate limited
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 60
            return BlockStatus(
                blocked=True,
                block_type=BlockType.RATE_LIMITED,
                reason=f"Rate limited (429). Retry after {retry_seconds}s",
                status_code=429,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                retry_after=retry_seconds,
                proxy_host=proxy_host
            )

        # Forbidden
        if response.status_code == 403:
            return BlockStatus(
                blocked=True,
                block_type=BlockType.FORBIDDEN,
                reason="Forbidden (403) - IP may be blocked",
                status_code=403,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                proxy_host=proxy_host
            )

        # Service unavailable
        if response.status_code == 503:
            return BlockStatus(
                blocked=True,
                block_type=BlockType.SERVICE_UNAVAILABLE,
                reason="Service unavailable (503) - possible blocking",
                status_code=503,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                proxy_host=proxy_host
            )

        # Other error codes
        if response.status_code >= 400:
            return BlockStatus(
                blocked=True,
                block_type=BlockType.INVALID_RESPONSE,
                reason=f"HTTP error: {response.status_code}",
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                proxy_host=proxy_host
            )

        # Check response content for 200 OK
        if response.status_code == 200:
            try:
                data = response.json()

                # Check for error in response
                if "error" in data:
                    error_code = data.get("error")
                    message = data.get("message", "Unknown error")
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.FORBIDDEN,
                        reason=f"API error {error_code}: {message}",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )

                # Check for empty children (shadow ban indicator)
                children = data.get("data", {}).get("children", [])
                if not children:
                    # Could be legitimate empty subreddit, but suspicious for r/pics
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.SHADOW_BLOCKED,
                        reason="Empty response from populated subreddit - possible shadow block",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )

                # Check for expected structure
                if "kind" not in data or data.get("kind") != "Listing":
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.INVALID_RESPONSE,
                        reason="Unexpected response structure",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )

                # All good!
                return BlockStatus(
                    blocked=False,
                    block_type=BlockType.NONE,
                    reason=f"OK - received {len(children)} posts",
                    status_code=200,
                    response_time_ms=response_time_ms,
                    timestamp=timestamp,
                    proxy_host=proxy_host
                )

            except json.JSONDecodeError:
                # Check if it's an HTML page (CAPTCHA, error page)
                content = response.text[:500].lower()
                if "captcha" in content:
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.CAPTCHA,
                        reason="CAPTCHA page returned instead of JSON",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )
                elif "blocked" in content or "denied" in content:
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.FORBIDDEN,
                        reason="Block page returned instead of JSON",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )
                else:
                    return BlockStatus(
                        blocked=True,
                        block_type=BlockType.INVALID_RESPONSE,
                        reason="Invalid JSON response",
                        status_code=200,
                        response_time_ms=response_time_ms,
                        timestamp=timestamp,
                        proxy_host=proxy_host
                    )

        # Unexpected status code
        return BlockStatus(
            blocked=False,
            block_type=BlockType.NONE,
            reason=f"Unexpected status: {response.status_code}",
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            timestamp=timestamp,
            proxy_host=proxy_host
        )

    except requests.exceptions.Timeout:
        return BlockStatus(
            blocked=True,
            block_type=BlockType.CONNECTION_ERROR,
            reason=f"Request timeout after {timeout}s",
            status_code=None,
            response_time_ms=timeout * 1000,
            timestamp=timestamp,
            proxy_host=proxy_host
        )
    except requests.exceptions.ConnectionError as e:
        return BlockStatus(
            blocked=True,
            block_type=BlockType.CONNECTION_ERROR,
            reason=f"Connection error: {str(e)[:100]}",
            status_code=None,
            response_time_ms=None,
            timestamp=timestamp,
            proxy_host=proxy_host
        )
    except Exception as e:
        return BlockStatus(
            blocked=True,
            block_type=BlockType.INVALID_RESPONSE,
            reason=f"Unexpected error: {str(e)[:100]}",
            status_code=None,
            response_time_ms=None,
            timestamp=timestamp,
            proxy_host=proxy_host
        )


def record_block_event(status: BlockStatus) -> None:
    """
    Record a block event in Redis for monitoring.

    Args:
        status: BlockStatus from check
    """
    try:
        client = get_redis_client()

        # Update current status
        client.hset(BLOCK_STATUS_KEY, mapping=status.to_dict())

        # Add to history (keep last 100 events)
        event = {
            **status.to_dict(),
            "recorded_at": datetime.utcnow().isoformat()
        }
        client.lpush(BLOCK_HISTORY_KEY, json.dumps(event))
        client.ltrim(BLOCK_HISTORY_KEY, 0, 99)

        # Increment block count if blocked
        if status.blocked:
            client.incr(BLOCK_INCIDENT_COUNT_KEY)
            logger.warning(f"Reddit block detected: {status.block_type.value} - {status.reason}")

    except Exception as e:
        logger.error(f"Failed to record block event: {e}")


def get_current_status() -> Dict:
    """Get the current block status from Redis."""
    try:
        client = get_redis_client()
        status = client.hgetall(BLOCK_STATUS_KEY)

        if not status:
            return {"status": "unknown", "message": "No status recorded yet"}

        # Parse boolean
        status["blocked"] = status.get("blocked", "False").lower() == "true"

        return status

    except Exception as e:
        logger.error(f"Failed to get current status: {e}")
        return {"error": str(e)}


def get_block_history(limit: int = 20) -> list:
    """Get recent block check history."""
    try:
        client = get_redis_client()
        history = client.lrange(BLOCK_HISTORY_KEY, 0, limit - 1)
        return [json.loads(event) for event in history]
    except Exception as e:
        logger.error(f"Failed to get block history: {e}")
        return []


def get_block_stats() -> Dict:
    """Get block statistics."""
    try:
        client = get_redis_client()

        total_incidents = int(client.get(BLOCK_INCIDENT_COUNT_KEY) or 0)
        history = get_block_history(100)

        # Count by type
        by_type = {}
        blocked_count = 0
        ok_count = 0

        for event in history:
            block_type = event.get("block_type", "unknown")
            # Check if actually blocked (handle both string and bool values from Redis)
            is_blocked = event.get("blocked") in (True, "true", "True")
            if is_blocked:
                by_type[block_type] = by_type.get(block_type, 0) + 1
                blocked_count += 1
            else:
                ok_count += 1

        # Calculate block rate
        total_checks = blocked_count + ok_count
        block_rate = (blocked_count / total_checks * 100) if total_checks > 0 else 0

        return {
            "total_incidents_all_time": total_incidents,
            "recent_checks": total_checks,
            "recent_blocked": blocked_count,
            "recent_ok": ok_count,
            "block_rate_percent": round(block_rate, 1),
            "by_type": by_type,
            "current_status": get_current_status()
        }

    except Exception as e:
        logger.error(f"Failed to get block stats: {e}")
        return {"error": str(e)}


def should_pause_scraping() -> Tuple[bool, str]:
    """
    Determine if scraping should be paused based on block status.

    Returns:
        (should_pause, reason)
    """
    try:
        status = get_current_status()

        if status.get("error"):
            return False, "Unable to check status"

        if not status.get("blocked"):
            return False, "Not blocked"

        block_type = status.get("block_type", "")

        # Always pause for these
        if block_type in ("forbidden", "captcha", "shadow_blocked"):
            return True, f"Blocked: {status.get('reason', block_type)}"

        # Pause temporarily for rate limiting
        if block_type == "rate_limited":
            retry_after = status.get("retry_after")
            if retry_after:
                return True, f"Rate limited - retry after {retry_after}s"
            return True, "Rate limited"

        # Service unavailable - might be temporary
        if block_type == "service_unavailable":
            return True, "Service unavailable - possible blocking"

        return False, "Block type not severe enough to pause"

    except Exception as e:
        logger.error(f"Error checking pause status: {e}")
        return False, f"Error: {e}"


def clear_block_status() -> None:
    """Clear block status and history (admin function, use after IP change)."""
    try:
        client = get_redis_client()
        client.delete(BLOCK_STATUS_KEY)
        client.delete(BLOCK_INCIDENT_COUNT_KEY)
        client.delete(BLOCK_HISTORY_KEY)
        logger.info("Block status and history cleared")
    except Exception as e:
        logger.error(f"Failed to clear block status: {e}")
