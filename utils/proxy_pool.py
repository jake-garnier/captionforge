"""
Proxy Pool Management for Concurrent Scraping

Manages a pool of proxy IPs, assigning dedicated proxies to each subreddit
for concurrent scraping without IP conflicts.

Proxy format: user:pass@host:port
Stored in: PROXY_POOL env var (comma-separated) or proxy_list.txt file
"""
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import redis

logger = logging.getLogger(__name__)

# Redis keys for proxy pool state
REDIS_PROXY_ASSIGNMENTS_KEY = "proxy_pool:assignments"  # Hash: subreddit -> proxy_url
REDIS_PROXY_STATS_KEY = "proxy_pool:stats"  # Hash: proxy_url -> JSON stats


@dataclass
class ProxyInfo:
    """Information about a proxy"""
    url: str  # Full URL: http://user:pass@host:port
    host: str
    port: int
    username: Optional[str] = None
    assigned_to: Optional[str] = None  # Subreddit name if assigned

    @property
    def display_url(self) -> str:
        """URL with credentials hidden"""
        return f"{self.host}:{self.port}"

    @property
    def full_url(self) -> str:
        """Full URL with http:// prefix"""
        if self.url.startswith('http'):
            return self.url
        return f"http://{self.url}"


class ProxyPool:
    """
    Manages a pool of proxies for concurrent scraping.

    Each subreddit gets assigned a dedicated proxy to avoid IP conflicts.
    Assignments are persisted in Redis for consistency across restarts.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize proxy pool.

        Args:
            redis_client: Redis client for persistence (auto-connects if None)
        """
        self._proxies: List[ProxyInfo] = []
        self._redis: Optional[redis.Redis] = redis_client
        self._load_proxies()

    def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        return self._redis

    def _load_proxies(self):
        """Load proxies from environment or file"""
        proxies = []

        # Try environment variable first (comma-separated)
        env_proxies = os.environ.get('PROXY_POOL', '')
        if env_proxies:
            proxies = [p.strip() for p in env_proxies.split(',') if p.strip()]
            logger.info(f"Loaded {len(proxies)} proxies from PROXY_POOL env var")

        # Fall back to file
        if not proxies:
            proxy_file = os.environ.get('PROXY_POOL_FILE', '/app/proxy_list.txt')
            # Also check local path for development
            if not os.path.exists(proxy_file):
                proxy_file = 'proxy_list.txt'
            if not os.path.exists(proxy_file):
                proxy_file = 'proxies_full.txt'

            if os.path.exists(proxy_file):
                with open(proxy_file, 'r') as f:
                    proxies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"Loaded {len(proxies)} proxies from {proxy_file}")

        # Parse proxies into ProxyInfo objects
        for proxy_str in proxies:
            try:
                proxy_info = self._parse_proxy(proxy_str)
                if proxy_info:
                    self._proxies.append(proxy_info)
            except Exception as e:
                logger.warning(f"Failed to parse proxy '{proxy_str}': {e}")

        logger.info(f"Proxy pool initialized with {len(self._proxies)} proxies")

    def _parse_proxy(self, proxy_str: str) -> Optional[ProxyInfo]:
        """Parse proxy string into ProxyInfo"""
        # Remove protocol prefix if present
        proxy_str = proxy_str.replace('http://', '').replace('https://', '').replace('socks5://', '')

        username = None
        if '@' in proxy_str:
            auth, host_port = proxy_str.rsplit('@', 1)
            if ':' in auth:
                username = auth.split(':')[0]
        else:
            host_port = proxy_str

        if ':' in host_port:
            host, port_str = host_port.rsplit(':', 1)
            port = int(port_str)
        else:
            host = host_port
            port = 80

        return ProxyInfo(
            url=proxy_str,
            host=host,
            port=port,
            username=username
        )

    @property
    def size(self) -> int:
        """Number of proxies in pool"""
        return len(self._proxies)

    @property
    def proxies(self) -> List[ProxyInfo]:
        """Get all proxies with assignment info"""
        assignments = self._get_assignments()
        for proxy in self._proxies:
            # Check if this proxy is assigned
            proxy.assigned_to = None
            for subreddit, assigned_url in assignments.items():
                if proxy.url in assigned_url or assigned_url in proxy.url:
                    proxy.assigned_to = subreddit
                    break
        return self._proxies

    def _get_assignments(self) -> Dict[str, str]:
        """Get current proxy assignments from Redis"""
        try:
            r = self._get_redis()
            assignments = r.hgetall(REDIS_PROXY_ASSIGNMENTS_KEY)
            return assignments or {}
        except Exception as e:
            logger.warning(f"Failed to get proxy assignments from Redis: {e}")
            return {}

    def get_proxy_for_subreddit(self, subreddit: str) -> Optional[str]:
        """
        Get the proxy URL assigned to a subreddit.

        Returns None if no proxy assigned.
        Note: Lookups work even if proxy file isn't loaded (uses Redis).
        """
        assignments = self._get_assignments()

        # Return existing assignment from Redis
        if subreddit in assignments:
            return f"http://{assignments[subreddit]}"

        return None

    def assign_proxy_to_subreddit(self, subreddit: str, proxy_index: Optional[int] = None) -> Optional[str]:
        """
        Assign a proxy to a subreddit.

        Args:
            subreddit: Subreddit name
            proxy_index: Specific proxy index to assign (auto-selects if None)

        Returns:
            Assigned proxy URL or None if pool is empty
        """
        if not self._proxies:
            logger.warning("No proxies available in pool")
            return None

        assignments = self._get_assignments()

        # If already assigned, return existing
        if subreddit in assignments:
            return f"http://{assignments[subreddit]}"

        # Find an unassigned proxy
        assigned_urls = set(assignments.values())

        if proxy_index is not None:
            # Use specific index
            if 0 <= proxy_index < len(self._proxies):
                proxy = self._proxies[proxy_index]
            else:
                logger.error(f"Invalid proxy index {proxy_index}")
                return None
        else:
            # Auto-select first unassigned proxy
            proxy = None
            for p in self._proxies:
                if p.url not in assigned_urls:
                    proxy = p
                    break

            if not proxy:
                # All proxies assigned - reuse least used or first
                logger.warning("All proxies assigned, reusing first proxy")
                proxy = self._proxies[0]

        # Store assignment
        try:
            r = self._get_redis()
            r.hset(REDIS_PROXY_ASSIGNMENTS_KEY, subreddit, proxy.url)
            logger.info(f"Assigned proxy {proxy.display_url} to r/{subreddit}")
        except Exception as e:
            logger.error(f"Failed to store proxy assignment: {e}")

        return proxy.full_url

    def unassign_proxy(self, subreddit: str) -> bool:
        """Remove proxy assignment for a subreddit"""
        try:
            r = self._get_redis()
            result = r.hdel(REDIS_PROXY_ASSIGNMENTS_KEY, subreddit)
            if result:
                logger.info(f"Unassigned proxy from r/{subreddit}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to unassign proxy: {e}")
            return False

    def get_available_proxies(self) -> List[ProxyInfo]:
        """Get list of proxies not currently assigned"""
        assignments = self._get_assignments()
        assigned_urls = set(assignments.values())
        return [p for p in self._proxies if p.url not in assigned_urls]

    def get_pool_status(self) -> Dict:
        """Get full status of proxy pool"""
        assignments = self._get_assignments()
        assigned_urls = set(assignments.values())

        available = [p for p in self._proxies if p.url not in assigned_urls]
        assigned = [p for p in self._proxies if p.url in assigned_urls]

        # Build assignment map
        assignment_details = []
        for subreddit, proxy_url in assignments.items():
            assignment_details.append({
                "subreddit": subreddit,
                "proxy_host": proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
            })

        return {
            "total_proxies": len(self._proxies),
            "available": len(available),
            "assigned": len(assigned),
            "assignments": assignment_details,
            "available_proxies": [p.display_url for p in available],
            "all_proxies": [
                {
                    "host": p.display_url,
                    "assigned_to": p.assigned_to
                }
                for p in self.proxies
            ]
        }

    def clear_all_assignments(self):
        """Clear all proxy assignments (admin operation)"""
        try:
            r = self._get_redis()
            r.delete(REDIS_PROXY_ASSIGNMENTS_KEY)
            logger.info("Cleared all proxy assignments")
        except Exception as e:
            logger.error(f"Failed to clear assignments: {e}")


# Global proxy pool instance
_proxy_pool: Optional[ProxyPool] = None


def get_proxy_pool() -> ProxyPool:
    """Get or create the global proxy pool instance"""
    global _proxy_pool
    if _proxy_pool is None:
        _proxy_pool = ProxyPool()
    return _proxy_pool


def get_proxy_for_subreddit(subreddit: str, auto_assign: bool = True) -> Optional[str]:
    """
    Convenience function to get proxy for a subreddit.

    Args:
        subreddit: Subreddit name
        auto_assign: If True, automatically assign a proxy if not already assigned

    Returns:
        Full proxy URL (http://user:pass@host:port) or None
    """
    pool = get_proxy_pool()

    proxy_url = pool.get_proxy_for_subreddit(subreddit)
    if proxy_url:
        return proxy_url

    if auto_assign:
        return pool.assign_proxy_to_subreddit(subreddit)

    return None
