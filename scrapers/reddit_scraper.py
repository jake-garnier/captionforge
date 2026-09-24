"""
Reddit scraper using Playwright (no API required)
Monitors target subreddits and extracts video posts via web scraping
"""
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from typing import List, Dict, Optional
from config.settings import settings
import logging
from datetime import datetime
import time
import random

logger = logging.getLogger(__name__)


class RedditScraper:
    """Reddit scraper using Playwright for browser automation"""

    def __init__(self, headless: bool = True):
        """
        Initialize Reddit scraper

        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        logger.info("Reddit scraper initialized (Playwright mode)")

    def __enter__(self):
        """Context manager entry"""
        self._start_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup browser"""
        self._stop_browser()

    def _start_browser(self):
        """Start Playwright browser"""
        if self.browser:
            return

        logger.info("Starting Playwright browser...")
        self.playwright = sync_playwright().start()

        # Launch Firefox (better for headless scraping than Chrome)
        self.browser = self.playwright.firefox.launch(
            headless=self.headless,
            args=['--no-sandbox']
        )

        # Create browser context with realistic user agent
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US'
        )

        logger.info("Browser started successfully")

    def _stop_browser(self):
        """Stop Playwright browser"""
        if self.context:
            self.context.close()
            self.context = None

        if self.browser:
            self.browser.close()
            self.browser = None

        if self.playwright:
            self.playwright.stop()
            self.playwright = None

        logger.info("Browser stopped")

    def _random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Random delay to avoid detection"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def get_video_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort: str = "hot",
        min_score: int = 0,
        time_filter: str = None,
        after: str = None,
        start_url: str = None
    ) -> tuple[List[Dict], str]:
        """
        Get video posts from a subreddit via web scraping

        Args:
            subreddit_name: Name of subreddit (without r/)
            limit: Maximum number of posts to fetch
            sort: Sort method (hot, new, top, rising)
            min_score: Minimum upvote score
            time_filter: Time filter for 'top' sort (hour, day, week, month, year, all)
            after: Cursor for pagination (post ID to start after) - DEPRECATED, use start_url
            start_url: Full URL to resume pagination from (from previous scrape's next_url)

        Returns:
            Tuple of (posts list, next_pagination_url)
            - posts: List of post dictionaries with metadata
            - next_pagination_url: URL to continue pagination (None if no more pages)
        """
        if not self.browser:
            self._start_browser()

        try:
            page = self.context.new_page()
            posts = []
            next_pagination_url = None

            # Use start_url if provided (resume from previous scrape)
            # Otherwise build URL from scratch
            if start_url:
                url = start_url
                logger.info(f"Resuming scrape from saved position: {url[:100]}...")
            else:
                # Build URL
                url = f"https://old.reddit.com/r/{subreddit_name}/{sort}/"

                # Add query parameters
                params = []
                if time_filter and sort == 'top':
                    params.append(f"t={time_filter}")
                if after:
                    # Format after parameter (Reddit expects t3_ prefix)
                    after_id = after if after.startswith('t3_') else f"t3_{after}"
                    params.append(f"after={after_id}")

                if params:
                    url += "?" + "&".join(params)

            logger.info(f"Scraping r/{subreddit_name} ({sort}{f' t={time_filter}' if time_filter else ''}) - target: {limit} posts")

            # Navigate to subreddit (use domcontentloaded for faster loading)
            page.goto(url, wait_until='domcontentloaded', timeout=45000)
            self._random_delay(2, 4)  # Reasonable delay on initial load

            posts_scraped = 0
            pages_loaded = 0
            # Calculate max pages needed (Reddit shows ~25 posts per page)
            max_pages = min((limit // 25) + 5, 50)  # Cap at 50 pages to avoid excessive scraping

            while posts_scraped < limit and pages_loaded < max_pages:
                # Get all post elements on current page
                post_elements = page.locator('div.thing[data-type="link"]').all()

                logger.info(f"Found {len(post_elements)} posts on page {pages_loaded + 1}")

                for element in post_elements:
                    if posts_scraped >= limit:
                        break

                    try:
                        post_data = self._extract_post_data(element, subreddit_name)

                        if not post_data:
                            continue

                        # Filter by score
                        if post_data.get('score', 0) < min_score:
                            continue

                        # Check if post has video
                        if not post_data.get('video_url'):
                            continue

                        posts.append(post_data)
                        posts_scraped += 1

                    except Exception as e:
                        logger.warning(f"Error extracting post: {e}")
                        continue

                # Try to load more posts (click "next" or scroll)
                logger.info(f"Pagination check: posts_scraped={posts_scraped}, limit={limit}, pages_loaded={pages_loaded}, max_pages={max_pages}")
                if posts_scraped < limit and pages_loaded < max_pages - 1:
                    logger.info(f"Entering pagination block (condition TRUE: {posts_scraped} < {limit} and {pages_loaded} < {max_pages - 1})")
                    try:
                        # Look for "next" button
                        next_button = page.locator('span.next-button a[rel="nofollow next"]')
                        button_count = next_button.count()
                        logger.info(f"Next button locator found {button_count} elements")

                        is_visible = next_button.is_visible(timeout=2000)
                        logger.info(f"Next button is_visible() returned: {is_visible}")

                        if is_visible:
                            next_url = next_button.get_attribute('href')
                            if next_url:
                                # Store the next URL before navigating
                                next_pagination_url = next_url
                                logger.info(f"Loading next page... ({posts_scraped}/{limit} posts)")
                                # Optimized delay between pages
                                self._random_delay(3, 5)  # 3-5 seconds between pages
                                page.goto(next_url, wait_until='domcontentloaded', timeout=45000)
                                pages_loaded += 1
                            else:
                                break
                        else:
                            logger.info("Next button not visible - attempting fallback URL capture before breaking")
                            # Button not visible - try to capture URL anyway before breaking
                            try:
                                next_button_locator = page.locator('span.next-button a[rel="nofollow next"]')
                                fallback_count = next_button_locator.count()
                                logger.info(f"Fallback locator found {fallback_count} elements")
                                if fallback_count > 0:
                                    next_url = next_button_locator.first.get_attribute('href')
                                    logger.info(f"Fallback extracted href: {next_url[:80] if next_url else 'None'}...")
                                    if next_url:
                                        next_pagination_url = next_url
                                        logger.info(f"✓ Captured pagination URL (button not visible): {next_url[:80]}...")
                                    else:
                                        logger.warning("Fallback found element but href attribute is None")
                                else:
                                    logger.warning("Fallback found 0 elements - next button does NOT exist on page")
                                    # Capture pagination area HTML to understand what's actually there
                                    try:
                                        nav_buttons = page.locator('div.nav-buttons').inner_html()
                                        logger.info(f"Pagination area HTML: {nav_buttons[:500]}")
                                    except Exception as html_err:
                                        logger.warning(f"Could not capture pagination HTML: {html_err}")
                            except Exception as e:
                                logger.warning(f"Could not capture hidden pagination URL: {e}")
                            logger.info("Breaking out of while loop - stopping pagination")
                            break
                    except Exception as e:
                        logger.warning(f"Pagination failed: {e}")
                        break
                else:
                    logger.info(f"Entering ELSE block (condition FALSE: NOT ({posts_scraped} < {limit} and {pages_loaded} < {max_pages - 1}))")
                    # Store the last available next URL even if we didn't click it
                    try:
                        next_button_locator = page.locator('span.next-button a[rel="nofollow next"]')
                        # Check if element exists
                        if next_button_locator.count() > 0:
                            next_url = next_button_locator.first.get_attribute('href')
                            if next_url:
                                next_pagination_url = next_url
                                logger.info(f"Captured next pagination URL without clicking: {next_url[:80]}...")
                            else:
                                logger.warning("Next button found but has no href attribute")
                        else:
                            logger.warning("No next button found on page - might be last page")
                    except Exception as e:
                        logger.warning(f"Could not capture pagination URL: {e}")
                    break

            page.close()
            logger.info(f"Scraped {len(posts)} video posts from r/{subreddit_name}")
            logger.info(f"Next pagination URL: {next_pagination_url[:100] if next_pagination_url else 'None'}...")
            return posts, next_pagination_url

        except PlaywrightTimeout as e:
            logger.error(f"Timeout scraping r/{subreddit_name}: {e}")
            return [], None
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
            return [], None

    def _extract_post_data(self, element, subreddit_name: str) -> Optional[Dict]:
        """
        Extract post data from a post element

        Args:
            element: Playwright element locator
            subreddit_name: Subreddit name

        Returns:
            Dictionary with post metadata or None if not a video post
        """
        try:
            # Get post ID
            post_id = element.get_attribute('data-fullname')
            if post_id:
                post_id = post_id.replace('t3_', '')  # Remove prefix

            # Get title
            title_elem = element.locator('a.title')
            title = title_elem.inner_text() if title_elem.count() > 0 else ''

            # Get URL
            url_elem = element.locator('a.title')
            url = url_elem.get_attribute('href') if url_elem.count() > 0 else ''

            # Get score
            score_elem = element.locator('div.score.unvoted')
            score_text = score_elem.inner_text() if score_elem.count() > 0 else '0'
            try:
                score = int(score_text) if score_text.isdigit() else 0
            except:
                score = 0

            # Get author
            author_elem = element.locator('a.author')
            author = author_elem.inner_text() if author_elem.count() > 0 else '[deleted]'

            # Get permalink
            permalink_elem = element.locator('a.comments')
            permalink = permalink_elem.get_attribute('href') if permalink_elem.count() > 0 else ''

            # Get number of comments
            comments_elem = element.locator('a.comments')
            comments_text = comments_elem.inner_text() if comments_elem.count() > 0 else '0 comments'
            try:
                num_comments = int(comments_text.split()[0]) if comments_text.split()[0].isdigit() else 0
            except:
                num_comments = 0

            # Get domain (to identify video hosts)
            domain_elem = element.locator('span.domain a')
            domain = domain_elem.inner_text() if domain_elem.count() > 0 else ''

            # Determine if this is a video post
            video_url = self._extract_video_url(url, domain, permalink)

            if not video_url:
                return None

            # Get timestamp (relative time)
            time_elem = element.locator('time')
            time_str = time_elem.get_attribute('datetime') if time_elem.count() > 0 else None

            if time_str:
                try:
                    created_utc = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except:
                    created_utc = datetime.utcnow()
            else:
                created_utc = datetime.utcnow()

            return {
                'id': post_id,
                'title': title,
                'author': author,
                'score': score,
                'upvote_ratio': None,  # Not available via scraping
                'num_comments': num_comments,
                'created_utc': created_utc,
                'permalink': f"https://reddit.com{permalink}" if permalink.startswith('/') else permalink,
                'url': url,
                'video_url': video_url,
                'subreddit': subreddit_name,
                'domain': domain,
                'flair': None,  # Could extract if needed
                'is_self': False,
                'selftext': None
            }

        except Exception as e:
            logger.debug(f"Error extracting post data: {e}")
            return None

    def _extract_video_url(self, url: str, domain: str, permalink: str) -> Optional[str]:
        """
        Determine if URL is a video and extract video URL

        Args:
            url: Post URL
            domain: Domain name
            permalink: Reddit permalink

        Returns:
            Video URL if this is a video post, None otherwise
        """
        if not url:
            return None

        url_lower = url.lower()
        domain_lower = domain.lower()

        # Reddit hosted video (v.redd.it)
        if 'v.redd.it' in url_lower or 'v.redd.it' in domain_lower:
            return url

        # Imgur (videos and gifs)
        if 'imgur.com' in url_lower or 'imgur.com' in domain_lower:
            # Only if it looks like a video/gif
            if any(ext in url_lower for ext in ['.mp4', '.webm', '.gif', '.gifv', '/a/']):
                return url

        # Gfycat
        if 'gfycat.com' in url_lower or 'gfycat.com' in domain_lower:
            return url

        # Streamable
        if 'streamable.com' in url_lower:
            return url

        # Direct video links
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v']
        if any(ext in url_lower for ext in video_extensions):
            return url

        # If it's a Reddit self-post, skip
        if url.startswith('/r/') or 'reddit.com/r/' in url:
            return None

        return None

    def get_post_by_id(self, post_id: str) -> Optional[Dict]:
        """
        Get a specific post by ID

        Args:
            post_id: Reddit post ID

        Returns:
            Post dictionary or None
        """
        if not self.browser:
            self._start_browser()

        try:
            # Reddit post URLs: /comments/POST_ID/
            page = self.context.new_page()
            url = f"https://old.reddit.com/comments/{post_id}/"

            page.goto(url, wait_until='networkidle', timeout=30000)
            self._random_delay(1, 2)

            # Get the main post element
            post_element = page.locator('div.thing[data-type="link"]').first

            if post_element.count() > 0:
                # Extract subreddit from URL
                subreddit_elem = page.locator('a.subreddit')
                subreddit = subreddit_elem.inner_text().replace('r/', '') if subreddit_elem.count() > 0 else 'unknown'

                post_data = self._extract_post_data(post_element, subreddit)
                page.close()
                return post_data

            page.close()
            return None

        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {e}")
            return None

    def close(self):
        """Cleanup - close browser"""
        self._stop_browser()


# Convenience function for one-off scraping
def scrape_subreddit(subreddit_name: str, limit: int = 100, **kwargs):
    """
    Convenience function to scrape a subreddit

    Usage:
        posts = scrape_subreddit('GetMotivated', limit=50)
    """
    with RedditScraper() as scraper:
        return scraper.get_video_posts(subreddit_name, limit=limit, **kwargs)
