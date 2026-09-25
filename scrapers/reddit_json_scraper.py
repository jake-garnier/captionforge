"""
Reddit scraper using JSON API (no browser required)
Much faster and more reliable than Playwright-based scraping

Includes IP block detection and automatic status recording.
"""
import requests
import time
import random
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

# Block detection status codes
BLOCK_STATUS_CODES = {429, 403, 503}


class RedditJsonScraper:
    """Reddit scraper using JSON API endpoints (unofficial but reliable)"""

    def __init__(self, check_blocks: bool = True, use_proxy: bool = True, proxy_url: str = None):
        """
        Initialize Reddit JSON scraper

        Args:
            check_blocks: Whether to check and record IP block status
            use_proxy: Whether to use proxy if configured in settings
            proxy_url: Specific proxy URL to use (overrides settings.proxy_url)
        """
        # Don't use requests.Session() - it adds headers like 'Accept-Encoding: gzip, deflate'
        # and 'Connection: keep-alive' which trigger Reddit's bot detection
        self.session = None  # We'll use direct requests.get() instead

        # Browser-like headers (no gzip/keep-alive that Sessions add)
        self.headers = {
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
        self.check_blocks = check_blocks
        self.last_block_status = None
        self.proxy_enabled = False
        self.proxy_url = None
        self.proxies = None

        # Determine which proxy to use (explicit > settings > none)
        effective_proxy = proxy_url
        if not effective_proxy and use_proxy and settings.proxy_enabled and settings.proxy_url:
            effective_proxy = settings.proxy_url

        # Configure proxy if available
        if effective_proxy:
            self.proxies = {
                'http': effective_proxy,
                'https': effective_proxy,
            }
            self.proxy_enabled = True
            self.proxy_url = effective_proxy
            # Log proxy host only (not credentials)
            proxy_host = effective_proxy.split('@')[-1] if '@' in effective_proxy else effective_proxy
            logger.info(f"Reddit JSON scraper initialized with proxy: {proxy_host}")
        else:
            logger.info("Reddit JSON scraper initialized (no proxy)")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session"""
        self.close()

    def _random_delay(self, min_seconds: float = 2.0, max_seconds: float = 4.0):
        """Random delay to avoid rate limiting"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def _check_response_for_block(self, response: requests.Response, response_time_ms: float, check_body: bool = True) -> Dict:
        """
        Check response for signs of IP blocking.

        Args:
            response: The HTTP response
            response_time_ms: Response time in milliseconds
            check_body: If True, read and check response body (set False when using stream=True)

        Returns:
            Dictionary with block status info
        """
        from datetime import datetime

        result = {
            'blocked': False,
            'block_type': 'none',
            'reason': 'OK',
            'status_code': response.status_code,
            'response_time_ms': response_time_ms,
            'retry_after': None
        }

        # Check HTTP status codes
        if response.status_code == 429:
            result['blocked'] = True
            result['block_type'] = 'rate_limited'
            retry_after = response.headers.get('Retry-After')
            result['retry_after'] = int(retry_after) if retry_after and retry_after.isdigit() else 60
            result['reason'] = f"Rate limited (429). Retry after {result['retry_after']}s"
            return result

        if response.status_code == 403:
            result['blocked'] = True
            result['block_type'] = 'forbidden'
            result['reason'] = "Forbidden (403) - IP may be blocked"
            return result

        if response.status_code == 503:
            result['blocked'] = True
            result['block_type'] = 'service_unavailable'
            result['reason'] = "Service unavailable (503) - possible blocking"
            return result

        # Check for redirects (we disabled follow_redirects)
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get('Location', '').lower()
            if 'login' in location:
                result['blocked'] = True
                result['block_type'] = 'login_required'
                result['reason'] = "Redirected to login"
                return result
            if 'captcha' in location:
                result['blocked'] = True
                result['block_type'] = 'captcha'
                result['reason'] = "CAPTCHA required"
                return result

        # Check response content for 200 OK (skip if using stream=True)
        if response.status_code == 200 and check_body:
            try:
                data = response.json()

                # Check for API error
                if 'error' in data:
                    result['blocked'] = True
                    result['block_type'] = 'forbidden'
                    result['reason'] = f"API error: {data.get('error')} - {data.get('message', '')}"
                    return result

                # Check for empty response (shadow ban indicator)
                children = data.get('data', {}).get('children', [])
                if not children and data.get('kind') == 'Listing':
                    # Could be empty subreddit, but flag it
                    result['block_type'] = 'possibly_shadow_blocked'
                    result['reason'] = "Empty response - might be normal or shadow block"
                    # Don't mark as blocked - might be legitimate

            except Exception:
                # If we can't parse JSON, check for HTML block pages
                content = response.text[:500].lower()
                if 'captcha' in content:
                    result['blocked'] = True
                    result['block_type'] = 'captcha'
                    result['reason'] = "CAPTCHA page returned"
                elif 'blocked' in content or 'denied' in content:
                    result['blocked'] = True
                    result['block_type'] = 'forbidden'
                    result['reason'] = "Block page returned"

        return result

    def _record_block_status(self, block_info: Dict) -> None:
        """Record block status to Redis for monitoring."""
        if not self.check_blocks:
            return

        try:
            from utils.reddit_block_detector import (
                BlockStatus, BlockType, record_block_event
            )

            # Map string type to enum
            type_map = {
                'none': BlockType.NONE,
                'rate_limited': BlockType.RATE_LIMITED,
                'forbidden': BlockType.FORBIDDEN,
                'service_unavailable': BlockType.SERVICE_UNAVAILABLE,
                'shadow_blocked': BlockType.SHADOW_BLOCKED,
                'possibly_shadow_blocked': BlockType.SHADOW_BLOCKED,
                'captcha': BlockType.CAPTCHA,
                'login_required': BlockType.LOGIN_REQUIRED,
                'connection_error': BlockType.CONNECTION_ERROR,
            }

            block_type = type_map.get(block_info.get('block_type', 'none'), BlockType.NONE)

            # Extract proxy host for logging (strip credentials)
            proxy_host = None
            if self.proxy_url:
                if '@' in self.proxy_url:
                    proxy_host = self.proxy_url.split('@')[-1]
                else:
                    proxy_host = self.proxy_url.replace('http://', '').replace('https://', '')

            status = BlockStatus(
                blocked=block_info.get('blocked', False),
                block_type=block_type,
                reason=block_info.get('reason', ''),
                status_code=block_info.get('status_code'),
                response_time_ms=block_info.get('response_time_ms'),
                timestamp=datetime.utcnow().isoformat(),
                retry_after=block_info.get('retry_after'),
                proxy_host=proxy_host
            )

            record_block_event(status)
            self.last_block_status = block_info

        except ImportError:
            logger.debug("Block detector not available")
        except Exception as e:
            logger.warning(f"Failed to record block status: {e}")

    def get_video_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort: str = "hot",
        min_score: int = 0,
        time_filter: str = None,
        after: str = None,
        count: int = 0
    ) -> Tuple[List[Dict], Optional[str], int]:
        """
        Get video posts from a subreddit using JSON API

        Args:
            subreddit_name: Name of subreddit (without r/)
            limit: Maximum number of posts to fetch
            sort: Sort method (hot, new, top, rising)
            min_score: Minimum upvote score
            time_filter: Time filter for 'top' sort (hour, day, week, month, year, all)
            after: Cursor for pagination (fullname of last post)
            count: Number of posts already seen (for proper pagination)

        Returns:
            Tuple of (posts_list, next_after_token, total_count)
            - posts: List of post dictionaries with metadata
            - next_after_token: Token for next page (None if no more pages)
            - total_count: Updated count of posts seen
        """
        posts = []
        current_after = after
        posts_fetched = 0
        current_count = count

        logger.info(f"Scraping r/{subreddit_name} ({sort}{f' t={time_filter}' if time_filter else ''}) - target: {limit} posts, starting after: {current_after}, count: {current_count}")

        while posts_fetched < limit:
            # Build URL
            url = f"https://www.reddit.com/r/{subreddit_name}/{sort}.json"

            # Build query parameters
            params = {
                'limit': min(100, limit - posts_fetched),  # Reddit max is 100 per request
                'raw_json': 1,  # Get unescaped JSON
            }

            if time_filter and sort == 'top':
                params['t'] = time_filter

            if current_after:
                params['after'] = current_after
                params['count'] = current_count

            try:
                # Small delay before each request to be more human-like
                self._random_delay(2, 5)

                logger.info(f"Fetching: {url} with params: {params}")
                start_time = time.time()
                # Use direct requests.get() instead of Session to avoid bot-like headers
                # (Session adds Accept-Encoding: gzip, deflate and Connection: keep-alive)
                response = requests.get(
                    url,
                    params=params,
                    headers=self.headers,
                    proxies=self.proxies,
                    timeout=30,
                    stream=True
                )
                response_time_ms = (time.time() - start_time) * 1000

                # Check for blocking before processing (status code only, skip body since we use stream=True)
                block_info = self._check_response_for_block(response, response_time_ms, check_body=False)
                self._record_block_status(block_info)

                if block_info['blocked']:
                    logger.error(f"Block detected: {block_info['reason']}")
                    response.close()
                    # Return what we have so far
                    break

                response.raise_for_status()

                # Read content in chunks to handle unstable proxy connections
                try:
                    content = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                    data = json.loads(content.decode('utf-8'))
                except Exception as e:
                    logger.error(f"Failed to read/parse response: {e}")
                    response.close()
                    break
                finally:
                    response.close()

                # Check if we got valid data
                if 'data' not in data or 'children' not in data['data']:
                    logger.error(f"Invalid JSON response structure: {data}")
                    break

                listing = data['data']
                children = listing['children']

                if not children:
                    logger.info("No more posts available")
                    break

                logger.info(f"Fetched {len(children)} posts from API")

                # Process posts
                for child in children:
                    post_data_raw = child.get('data', {})

                    # Extract and validate post
                    post_data = self._extract_post_data(post_data_raw, subreddit_name)

                    if not post_data:
                        current_count += 1  # Still count it for pagination
                        continue

                    # Filter by score
                    if post_data.get('score', 0) < min_score:
                        current_count += 1
                        continue

                    # Check if post has video
                    if not post_data.get('video_url'):
                        current_count += 1
                        continue

                    posts.append(post_data)
                    posts_fetched += 1
                    current_count += 1

                    if posts_fetched >= limit:
                        break

                # Get next page token
                current_after = listing.get('after')

                if not current_after:
                    logger.info("No 'after' token - reached end of available posts")
                    break

                # Human-like rate limiting between pagination requests
                if posts_fetched < limit:
                    self._random_delay(8, 15)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                break
            except ValueError as e:
                logger.error(f"JSON decode error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        logger.info(f"Scraped {len(posts)} media posts from r/{subreddit_name} (total count: {current_count})")
        return posts, current_after, current_count

    def _extract_post_data(self, post_raw: Dict, subreddit_name: str) -> Optional[Dict]:
        """
        Extract post data from Reddit JSON API response

        Args:
            post_raw: Raw post data from Reddit API
            subreddit_name: Subreddit name

        Returns:
            Dictionary with post metadata or None if not a video post
        """
        try:
            # Get post ID (without t3_ prefix)
            post_id = post_raw.get('id', '')

            # Get title
            title = post_raw.get('title', '')

            # Get URL
            url = post_raw.get('url', '')

            # Get score
            score = post_raw.get('score', 0)

            # Get author
            author = post_raw.get('author', '[deleted]')

            # Get permalink
            permalink = post_raw.get('permalink', '')
            full_permalink = f"https://www.reddit.com{permalink}" if permalink.startswith('/') else permalink

            # Get number of comments
            num_comments = post_raw.get('num_comments', 0)

            # Get domain
            domain = post_raw.get('domain', '')

            # Get upvote ratio
            upvote_ratio = post_raw.get('upvote_ratio', None)

            # Get created timestamp
            created_utc = post_raw.get('created_utc', None)
            if created_utc:
                created_utc = datetime.utcfromtimestamp(created_utc)
            else:
                created_utc = datetime.utcnow()

            # Get selftext (for self-posts)
            selftext = post_raw.get('selftext', None)
            is_self = post_raw.get('is_self', False)

            # Extract media info (video, image, gallery, gif)
            media_info = self._extract_media_info(url, domain, post_raw)

            if not media_info:
                return None

            return {
                'id': post_id,
                'title': title,
                'author': author,
                'score': score,
                'upvote_ratio': upvote_ratio,
                'num_comments': num_comments,
                'created_utc': created_utc,
                'permalink': full_permalink,
                'url': url,
                'video_url': media_info.get('url'),  # Keep for backward compatibility
                'media_url': media_info.get('url'),
                'media_type': media_info.get('media_type', 'video'),
                'gallery_urls': media_info.get('gallery_urls'),
                'gallery_count': media_info.get('gallery_count'),
                'subreddit': subreddit_name,
                'domain': domain,
                'flair': post_raw.get('link_flair_text'),
                'is_self': is_self,
                'selftext': selftext
            }

        except Exception as e:
            logger.debug(f"Error extracting post data: {e}")
            return None

    def _extract_media_info(self, url: str, domain: str, post_raw: Dict) -> Optional[Dict]:
        """
        Extract media URL and type from post data.
        Supports videos, images, galleries, and GIFs.

        Args:
            url: Post URL
            domain: Domain name
            post_raw: Raw post data (may contain additional media info)

        Returns:
            Dictionary with 'url', 'media_type', and optionally 'gallery_urls'/'gallery_count'
            or None if no supported media found
        """
        if not url:
            return None

        url_lower = url.lower()
        domain_lower = domain.lower()

        # Check if post is video type
        is_video = post_raw.get('is_video', False)
        is_gallery = post_raw.get('is_gallery', False)

        # === REDDIT GALLERIES ===
        if is_gallery:
            gallery_data = post_raw.get('gallery_data', {})
            media_metadata = post_raw.get('media_metadata', {})

            if gallery_data and media_metadata:
                gallery_urls = []
                items = gallery_data.get('items', [])

                for item in items:
                    media_id = item.get('media_id')
                    if media_id and media_id in media_metadata:
                        media = media_metadata[media_id]
                        # Get the source image (highest quality)
                        source = media.get('s', {})
                        img_url = source.get('u') or source.get('gif')
                        if img_url:
                            # Unescape URL (Reddit escapes ampersands)
                            img_url = img_url.replace('&amp;', '&')
                            gallery_urls.append(img_url)

                if gallery_urls:
                    return {
                        'url': gallery_urls[0],  # Primary URL for download
                        'media_type': 'gallery',
                        'gallery_urls': gallery_urls,
                        'gallery_count': len(gallery_urls)
                    }

        # === REDDIT HOSTED VIDEO (v.redd.it) ===
        if is_video or 'v.redd.it' in url_lower or 'v.redd.it' in domain_lower:
            media = post_raw.get('media', {})
            if media:
                reddit_video = media.get('reddit_video', {})
                if reddit_video:
                    fallback_url = reddit_video.get('fallback_url')
                    if fallback_url:
                        return {'url': fallback_url, 'media_type': 'video'}
            return {'url': url, 'media_type': 'video'}

        # === REDDIT IMAGES (i.redd.it) ===
        if 'i.redd.it' in url_lower or 'i.redd.it' in domain_lower:
            if any(ext in url_lower for ext in ['.gif']):
                return {'url': url, 'media_type': 'gif'}
            return {'url': url, 'media_type': 'image'}

        # === IMGUR ===
        if 'imgur.com' in url_lower or 'imgur.com' in domain_lower:
            # Video formats
            if any(ext in url_lower for ext in ['.mp4', '.webm']):
                return {'url': url, 'media_type': 'video'}
            # GIF formats (gifv is video, gif is image)
            if '.gifv' in url_lower:
                return {'url': url, 'media_type': 'video'}
            if '.gif' in url_lower:
                return {'url': url, 'media_type': 'gif'}
            # Album/gallery
            if '/a/' in url_lower or '/gallery/' in url_lower:
                return {'url': url, 'media_type': 'gallery', 'gallery_count': None}
            # Single image
            if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                return {'url': url, 'media_type': 'image'}
            # Imgur link without extension (could be image)
            if 'i.imgur.com' in url_lower:
                return {'url': url, 'media_type': 'image'}

        # === GFYCAT (service shut down in 2023, but keep for legacy) ===
        if 'gfycat.com' in url_lower or 'gfycat.com' in domain_lower:
            return {'url': url, 'media_type': 'video'}

        # === STREAMABLE ===
        if 'streamable.com' in url_lower:
            return {'url': url, 'media_type': 'video'}

        # === DIRECT VIDEO LINKS ===
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v']
        if any(ext in url_lower for ext in video_extensions):
            return {'url': url, 'media_type': 'video'}

        # === DIRECT IMAGE LINKS ===
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if any(ext in url_lower for ext in image_extensions):
            return {'url': url, 'media_type': 'image'}

        # === DIRECT GIF LINKS ===
        if '.gif' in url_lower and '.gifv' not in url_lower:
            return {'url': url, 'media_type': 'gif'}

        # === REDDIT PREVIEW IMAGES ===
        # Sometimes posts have preview images we can use
        preview = post_raw.get('preview', {})
        if preview:
            images = preview.get('images', [])
            if images:
                source = images[0].get('source', {})
                preview_url = source.get('url')
                if preview_url:
                    preview_url = preview_url.replace('&amp;', '&')
                    return {'url': preview_url, 'media_type': 'image'}

        # Skip self-posts and unsupported URLs
        if url.startswith('/r/') or 'reddit.com/r/' in url:
            return None

        return None

    # Keep backward compatibility
    def _extract_video_url(self, url: str, domain: str, post_raw: Dict) -> Optional[str]:
        """Backward compatibility wrapper for _extract_media_info"""
        media_info = self._extract_media_info(url, domain, post_raw)
        if media_info:
            return media_info.get('url')
        return None

    def get_post_by_id(self, post_id: str, subreddit: str = None) -> Optional[Dict]:
        """
        Get a specific post by ID using JSON API

        Args:
            post_id: Reddit post ID (without t3_ prefix)
            subreddit: Optional subreddit name (helps with rate limiting)

        Returns:
            Post dictionary or None
        """
        try:
            # If we know the subreddit, use it for more efficient lookup
            if subreddit:
                url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            else:
                # Generic lookup (works but might be slower)
                url = f"https://www.reddit.com/comments/{post_id}.json"

            params = {'raw_json': 1}

            logger.info(f"Fetching post {post_id} from {url}")
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                proxies=self.proxies,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # JSON returns array with [post, comments]
            if not data or len(data) < 1:
                return None

            post_listing = data[0]['data']['children']
            if not post_listing:
                return None

            post_raw = post_listing[0]['data']
            subreddit_name = post_raw.get('subreddit', subreddit or 'unknown')

            return self._extract_post_data(post_raw, subreddit_name)

        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # User analytics endpoints
    # Used by tasks/reddit_analytics.py to track per-account karma + post
    # stats. Go through the same proxy / block-detection path as the
    # subreddit scraping methods above.
    # ------------------------------------------------------------------

    def get_user_about(self, username: str) -> Optional[Dict]:
        """Fetch /user/<username>/about.json.

        Returns a flat dict with the karma fields we care about, plus an
        is_suspended hint. Returns None on network/parse error. A user
        that does not exist OR has been suspended will set is_suspended
        on the response rather than returning None — callers want to
        record the snapshot either way so the dashboard can render the
        suspended-account badge.
        """
        url = f"https://www.reddit.com/user/{username}/about.json"
        params = {"raw_json": 1}
        try:
            t0 = time.time()
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                proxies=self.proxies,
                timeout=30,
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            block_info = self._check_response_for_block(response, elapsed_ms, check_body=False)
            if self.check_blocks:
                self._record_block_status(block_info)

            if response.status_code == 404:
                # Account deleted/banned — Reddit returns 404 + JSON body
                logger.warning(f"User {username} returned 404 (likely banned or deleted)")
                return {
                    "username": username,
                    "is_suspended": True,
                    "raw": None,
                }

            response.raise_for_status()
            data = response.json()
            inner = (data or {}).get("data") or {}

            return {
                "username": username,
                "total_karma": inner.get("total_karma"),
                "link_karma": inner.get("link_karma"),
                "comment_karma": inner.get("comment_karma"),
                "awardee_karma": inner.get("awardee_karma"),
                "awarder_karma": inner.get("awarder_karma"),
                "created_utc": inner.get("created_utc"),
                "verified_email": inner.get("has_verified_email"),
                "is_suspended": bool(inner.get("is_suspended", False)),
                "raw": inner,
            }
        except Exception as e:
            logger.error(f"Error fetching /user/{username}/about.json: {e}")
            return None

    def get_user_submitted(
        self,
        username: str,
        after: Optional[str] = None,
        limit: int = 100,
        sort: str = "new",
    ) -> Tuple[List[Dict], Optional[str]]:
        """Paginate /user/<username>/submitted.json.

        Returns (posts, next_after). next_after is None when Reddit
        signals the end of the feed (or when an error occurred). On
        error returns ([], None) so callers can treat both cases the
        same way.

        sort: "new" (default), "hot", "top", "controversial".
        """
        url = f"https://www.reddit.com/user/{username}/submitted.json"
        params = {"raw_json": 1, "limit": limit, "sort": sort}
        if after:
            params["after"] = after

        try:
            t0 = time.time()
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                proxies=self.proxies,
                timeout=30,
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            block_info = self._check_response_for_block(response, elapsed_ms, check_body=False)
            if self.check_blocks:
                self._record_block_status(block_info)

            if response.status_code == 404:
                # Suspended / deleted users return 404 here too; treat as
                # empty rather than raising. Caller decides what to do.
                logger.warning(f"User {username} submitted feed returned 404")
                return [], None

            response.raise_for_status()
            data = response.json()
            inner = (data or {}).get("data") or {}
            children = inner.get("children") or []

            posts = []
            for child in children:
                if child.get("kind") != "t3":
                    continue
                raw = child.get("data") or {}
                extracted = self._extract_user_post_data(raw)
                if extracted is not None:
                    posts.append(extracted)

            return posts, inner.get("after")
        except Exception as e:
            logger.error(f"Error fetching /user/{username}/submitted.json: {e}")
            return [], None

    def _extract_user_post_data(self, post_raw: Dict) -> Optional[Dict]:
        """Media-agnostic extractor used by get_user_submitted.

        Different from _extract_post_data (above) which only returns
        video posts. For analytics we want every submission the user
        has made, including selftext / images / archived posts.
        """
        try:
            post_id = post_raw.get("id")
            if not post_id:
                return None

            permalink = post_raw.get("permalink") or ""
            full_permalink = (
                f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink
            )

            created_utc = post_raw.get("created_utc")
            if created_utc:
                try:
                    created_utc = datetime.utcfromtimestamp(created_utc)
                except (ValueError, TypeError):
                    created_utc = None

            return {
                "id": post_id,
                "title": post_raw.get("title") or "",
                "subreddit": post_raw.get("subreddit") or "",
                "url": post_raw.get("url") or "",
                "permalink": full_permalink,
                "score": post_raw.get("score"),
                "upvote_ratio": post_raw.get("upvote_ratio"),
                "num_comments": post_raw.get("num_comments"),
                "created_utc": created_utc,
                "is_video": bool(post_raw.get("is_video", False)),
                "domain": post_raw.get("domain"),
                # Reddit's removed-by-category: 'moderator', 'reddit',
                # 'deleted', 'copyright_takedown', 'automod_filtered',
                # or null when present. Useful for analytics drill-down.
                "removed_by_category": post_raw.get("removed_by_category"),
            }
        except Exception as e:
            logger.debug(f"Error extracting user post data: {e}")
            return None

    def close(self):
        """Cleanup - no session to close (using direct requests)"""
        logger.debug("Scraper cleanup complete (no session to close)")


# Convenience function for one-off scraping
def scrape_subreddit(subreddit_name: str, limit: int = 100, **kwargs) -> Tuple[List[Dict], Optional[str], int]:
    """
    Convenience function to scrape a subreddit using JSON API

    Usage:
        posts, next_after, count = scrape_subreddit('GetMotivated', limit=50)
    """
    with RedditJsonScraper() as scraper:
        return scraper.get_video_posts(subreddit_name, limit=limit, **kwargs)
