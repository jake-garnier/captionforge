"""
Reddit posting integration using Playwright (browser automation).

Alternative to PRAW when API credentials are not available.
Uses old.reddit.com for simpler HTML structure.

Required environment variables:
- REDDIT_USERNAME: Reddit account username
- REDDIT_PASSWORD: Reddit account password
"""
import os
import json
import logging
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)

# Reuse dataclasses from PRAW version for compatibility
@dataclass
class PostResult:
    """Result from Reddit post/crosspost."""
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CrosspostResults:
    """Results from batch crossposting."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)


class RedditPosterPlaywright:
    """
    Post and crosspost content to Reddit using Playwright browser automation.

    Drop-in replacement for RedditPoster when API credentials aren't available.
    Uses old.reddit.com for simpler, more stable HTML structure.

    Session cookies are persisted to avoid repeated logins.

    Usage:
        poster = RedditPosterPlaywright()
        if poster.connect():
            result = poster.post_to_profile(
                title="Check out this video",
                url="https://captions.example.com/composition/videos/123/stream"
            )
            if result.success:
                crossposts = poster.crosspost_to_subreddits(
                    source_post_id=result.post_id,
                    subreddits=["GetMotivated"]
                )
        poster.close()
    """

    # Cookie storage path
    COOKIES_FILE = "/data/reddit_session_cookies.json"

    def __init__(self, headless: bool = True):
        """
        Initialize Reddit poster.

        Args:
            headless: Run browser in headless mode (default True for server)
        """
        self.headless = headless
        self.username = os.environ.get("REDDIT_USERNAME", "")
        self.password = os.environ.get("REDDIT_PASSWORD", "")

        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._connected = False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Random delay to appear more human-like."""
        time.sleep(random.uniform(min_sec, max_sec))

    def _start_browser(self):
        """Start Playwright browser with persistent session."""
        if self.browser:
            return

        logger.info("Starting Playwright browser for Reddit posting...")
        self.playwright = sync_playwright().start()

        # Use Firefox for better compatibility
        self.browser = self.playwright.firefox.launch(
            headless=self.headless,
            args=['--no-sandbox']
        )

        # Create context with realistic settings
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US'
        )

        # Load saved cookies if they exist
        self._load_cookies()

        self.page = self.context.new_page()
        logger.info("Browser started")

    def _stop_browser(self):
        """Stop browser and save session."""
        # Save cookies before closing
        if self.context and self._connected:
            self._save_cookies()

        if self.page:
            self.page.close()
            self.page = None

        if self.context:
            self.context.close()
            self.context = None

        if self.browser:
            self.browser.close()
            self.browser = None

        if self.playwright:
            self.playwright.stop()
            self.playwright = None

        self._connected = False
        logger.info("Browser stopped")

    def _save_cookies(self):
        """Save session cookies to file."""
        if not self.context:
            return

        try:
            cookies = self.context.cookies()
            # Filter to Reddit cookies only
            reddit_cookies = [c for c in cookies if 'reddit.com' in c.get('domain', '')]

            # Ensure directory exists
            Path(self.COOKIES_FILE).parent.mkdir(parents=True, exist_ok=True)

            with open(self.COOKIES_FILE, 'w') as f:
                json.dump(reddit_cookies, f)
            logger.debug(f"Saved {len(reddit_cookies)} Reddit cookies")
        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")

    def _load_cookies(self):
        """Load saved session cookies."""
        if not self.context:
            return

        try:
            if Path(self.COOKIES_FILE).exists():
                with open(self.COOKIES_FILE, 'r') as f:
                    cookies = json.load(f)
                if cookies:
                    self.context.add_cookies(cookies)
                    logger.debug(f"Loaded {len(cookies)} saved cookies")
        except Exception as e:
            logger.warning(f"Failed to load cookies: {e}")

    def _is_logged_in(self) -> bool:
        """Check if we're currently logged in."""
        if not self.page:
            return False

        try:
            # Go to old.reddit.com and check for login state
            self.page.goto('https://old.reddit.com', wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Check for username in top bar (logged in indicator)
            user_elem = self.page.locator(f'span.user a[href*="/user/{self.username}"]')
            if user_elem.count() > 0:
                logger.info(f"Already logged in as u/{self.username}")
                return True

            # Alternative: check for logout link
            logout_link = self.page.locator('a[href*="logout"]')
            if logout_link.count() > 0:
                return True

            return False
        except Exception as e:
            logger.debug(f"Login check failed: {e}")
            return False

    def _do_login(self) -> bool:
        """Perform login to Reddit."""
        if not self.page:
            return False

        logger.info(f"Logging in to Reddit as u/{self.username}...")

        try:
            # Go to login page
            self.page.goto('https://old.reddit.com/login', wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Fill login form
            self.page.fill('input[name="user"]', self.username)
            self._random_delay(0.3, 0.7)
            self.page.fill('input[name="passwd"]', self.password)
            self._random_delay(0.3, 0.7)

            # Check "remember me"
            remember_checkbox = self.page.locator('input[name="rem"]')
            if remember_checkbox.count() > 0 and not remember_checkbox.is_checked():
                remember_checkbox.check()

            # Submit login
            self.page.click('button[type="submit"], input[type="submit"]')

            # Wait for navigation
            self.page.wait_for_load_state('domcontentloaded', timeout=30000)
            self._random_delay(2, 3)

            # Check if login succeeded
            if self._check_login_success():
                logger.info(f"Successfully logged in as u/{self.username}")
                self._save_cookies()
                return True
            else:
                logger.error("Login failed - check credentials")
                return False

        except PlaywrightTimeout:
            logger.error("Login timed out")
            return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _check_login_success(self) -> bool:
        """Verify login was successful."""
        try:
            # Check for error messages
            error_elem = self.page.locator('.error, .status.error')
            if error_elem.count() > 0 and error_elem.is_visible():
                error_text = error_elem.text_content()
                logger.error(f"Login error: {error_text}")
                return False

            # Check for user link in header
            user_link = self.page.locator(f'a[href*="/user/{self.username}"]')
            return user_link.count() > 0

        except Exception:
            return False

    def connect(self) -> bool:
        """
        Connect to Reddit (start browser and login if needed).

        Returns:
            True if connected and logged in successfully
        """
        if not self.username or not self.password:
            logger.error("Missing REDDIT_USERNAME or REDDIT_PASSWORD")
            return False

        self._start_browser()

        # Check if already logged in (via saved cookies)
        if self._is_logged_in():
            self._connected = True
            return True

        # Need to login
        if self._do_login():
            self._connected = True
            return True

        return False

    def close(self):
        """Close browser and cleanup."""
        self._stop_browser()

    def post_to_profile(
        self,
        title: str,
        url: str,
        flair_text: Optional[str] = None
    ) -> PostResult:
        """
        Post a link to user's profile.

        Args:
            title: Post title
            url: URL to post (the hosted video URL)
            flair_text: Optional flair (not supported via web)

        Returns:
            PostResult with success status and post info
        """
        if not self._connected:
            if not self.connect():
                return PostResult(success=False, error="Failed to connect to Reddit")

        try:
            # Navigate to submit page for user profile
            profile_sub = f"u_{self.username}"
            submit_url = f"https://old.reddit.com/r/{profile_sub}/submit"

            logger.info(f"Navigating to submit page: {submit_url}")
            self.page.goto(submit_url, wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Select "link" tab if present
            link_tab = self.page.locator('a.link-button, li.link-button a')
            if link_tab.count() > 0:
                link_tab.first.click()
                self._random_delay(0.5, 1)

            # Fill in URL
            url_input = self.page.locator('input[name="url"], textarea[name="url"]')
            if url_input.count() > 0:
                url_input.fill(url)
                self._random_delay(0.3, 0.7)
            else:
                return PostResult(success=False, error="Could not find URL input field")

            # Fill in title
            title_input = self.page.locator('textarea[name="title"], input[name="title"]')
            if title_input.count() > 0:
                title_input.fill(title)
                self._random_delay(0.3, 0.7)
            else:
                return PostResult(success=False, error="Could not find title input field")

            # Submit the post
            submit_btn = self.page.locator('button[type="submit"][name="submit"], input[type="submit"]')
            if submit_btn.count() > 0:
                submit_btn.first.click()
            else:
                return PostResult(success=False, error="Could not find submit button")

            # Wait for submission to complete
            self.page.wait_for_load_state('domcontentloaded', timeout=30000)
            self._random_delay(2, 3)

            # Check for errors
            error_elem = self.page.locator('.error')
            if error_elem.count() > 0:
                errors = [e.text_content() for e in error_elem.all() if e.text_content().strip()]
                if errors:
                    return PostResult(success=False, error="; ".join(errors))

            # Get the post URL from the redirected page
            current_url = self.page.url

            # Extract post ID from URL
            # URL format: https://old.reddit.com/r/u_username/comments/xxxxx/title/
            if '/comments/' in current_url:
                parts = current_url.split('/comments/')
                if len(parts) > 1:
                    post_id = parts[1].split('/')[0]
                    post_url = f"https://reddit.com/r/{profile_sub}/comments/{post_id}"

                    logger.info(f"Successfully posted to profile: {post_url}")
                    return PostResult(
                        success=True,
                        post_id=post_id,
                        post_url=post_url
                    )

            # Couldn't extract post ID but submission may have succeeded
            logger.warning(f"Post may have succeeded but couldn't extract ID. URL: {current_url}")
            return PostResult(
                success=False,
                error=f"Could not verify submission. Current URL: {current_url}"
            )

        except PlaywrightTimeout:
            return PostResult(success=False, error="Submission timed out")
        except Exception as e:
            logger.error(f"Post error: {e}")
            return PostResult(success=False, error=str(e))

    def crosspost_to_subreddit(
        self,
        source_post_id: str,
        subreddit: str,
        title: Optional[str] = None
    ) -> PostResult:
        """
        Crosspost an existing post to a subreddit.

        Args:
            source_post_id: ID of the source post
            subreddit: Target subreddit name (without r/)
            title: Optional new title

        Returns:
            PostResult with success status
        """
        if not self._connected:
            if not self.connect():
                return PostResult(success=False, error="Failed to connect to Reddit")

        try:
            # Navigate to crosspost page
            # Format: https://old.reddit.com/submit?crosspost_fullname=t3_xxxxx
            crosspost_url = f"https://old.reddit.com/submit?crosspost_fullname=t3_{source_post_id}"

            logger.info(f"Crossposting to r/{subreddit}: {crosspost_url}")
            self.page.goto(crosspost_url, wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Fill in subreddit
            sr_input = self.page.locator('input[name="sr"], input#sr-autocomplete')
            if sr_input.count() > 0:
                sr_input.fill(subreddit)
                self._random_delay(0.5, 1)

                # Wait for autocomplete and select if present
                autocomplete = self.page.locator('.sr-drop-down .sr-suggestion, .reddit-infobar')
                try:
                    autocomplete.first.wait_for(state='visible', timeout=3000)
                    # Try to click the matching suggestion
                    suggestion = self.page.locator(f'.sr-suggestion:has-text("{subreddit}")')
                    if suggestion.count() > 0:
                        suggestion.first.click()
                        self._random_delay(0.3, 0.5)
                except PlaywrightTimeout:
                    pass  # No autocomplete, continue
            else:
                return PostResult(success=False, error="Could not find subreddit input")

            # Update title if provided
            if title:
                title_input = self.page.locator('textarea[name="title"], input[name="title"]')
                if title_input.count() > 0:
                    title_input.fill('')  # Clear first
                    title_input.fill(title)
                    self._random_delay(0.3, 0.5)

            # Submit
            submit_btn = self.page.locator('button[type="submit"][name="submit"], input[type="submit"]')
            if submit_btn.count() > 0:
                submit_btn.first.click()
            else:
                return PostResult(success=False, error="Could not find submit button")

            # Wait for submission
            self.page.wait_for_load_state('domcontentloaded', timeout=30000)
            self._random_delay(2, 3)

            # Check for errors
            error_elem = self.page.locator('.error')
            if error_elem.count() > 0:
                errors = []
                for e in error_elem.all():
                    text = e.text_content().strip()
                    if text:
                        errors.append(text)

                if errors:
                    error_msg = "; ".join(errors)
                    # Parse common errors
                    if "not allowed" in error_msg.lower():
                        error_msg = f"Crossposts not allowed in r/{subreddit}"
                    elif "karma" in error_msg.lower():
                        error_msg = f"Insufficient karma for r/{subreddit}"
                    return PostResult(success=False, error=error_msg)

            # Get the crosspost URL
            current_url = self.page.url

            if '/comments/' in current_url:
                parts = current_url.split('/comments/')
                if len(parts) > 1:
                    post_id = parts[1].split('/')[0]
                    post_url = f"https://reddit.com/r/{subreddit}/comments/{post_id}"

                    logger.info(f"Successfully crossposted to r/{subreddit}: {post_url}")
                    return PostResult(
                        success=True,
                        post_id=post_id,
                        post_url=post_url
                    )

            return PostResult(
                success=False,
                error=f"Could not verify crosspost. URL: {current_url}"
            )

        except PlaywrightTimeout:
            return PostResult(success=False, error="Crosspost timed out")
        except Exception as e:
            logger.error(f"Crosspost error: {e}")
            return PostResult(success=False, error=str(e))

    def crosspost_to_subreddits(
        self,
        source_post_id: str,
        subreddits: List[str],
        title: Optional[str] = None,
        delay_between: float = 5.0
    ) -> CrosspostResults:
        """
        Crosspost to multiple subreddits with rate limiting.

        Args:
            source_post_id: ID of the source post
            subreddits: List of subreddit names
            title: Optional new title
            delay_between: Seconds between crossposts

        Returns:
            CrosspostResults with success/failure counts
        """
        results = CrosspostResults(total=len(subreddits))

        for i, subreddit in enumerate(subreddits):
            logger.info(f"Crossposting to r/{subreddit} ({i+1}/{len(subreddits)})")

            result = self.crosspost_to_subreddit(
                source_post_id=source_post_id,
                subreddit=subreddit,
                title=title
            )

            if result.success:
                results.successful += 1
                results.results.append({
                    "subreddit": subreddit,
                    "success": True,
                    "post_id": result.post_id,
                    "post_url": result.post_url
                })
            else:
                if "not allowed" in (result.error or "").lower():
                    results.skipped += 1
                else:
                    results.failed += 1

                results.results.append({
                    "subreddit": subreddit,
                    "success": False,
                    "error": result.error
                })

            # Rate limiting
            if i < len(subreddits) - 1:
                time.sleep(delay_between)

        logger.info(
            f"Crosspost complete: {results.successful} success, "
            f"{results.failed} failed, {results.skipped} skipped"
        )

        return results

    def delete_post(self, post_id: str) -> bool:
        """
        Delete a post.

        Args:
            post_id: Reddit post ID

        Returns:
            True if deletion successful
        """
        if not self._connected:
            if not self.connect():
                return False

        try:
            # Navigate to post
            post_url = f"https://old.reddit.com/comments/{post_id}"
            self.page.goto(post_url, wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Find and click delete button
            delete_btn = self.page.locator('a.delete-button, form.del-button a')
            if delete_btn.count() > 0:
                delete_btn.first.click()
                self._random_delay(0.5, 1)

                # Confirm deletion (if dialog appears)
                confirm_btn = self.page.locator('.yes, button:has-text("yes")')
                if confirm_btn.count() > 0:
                    confirm_btn.first.click()
                    self._random_delay(1, 2)

                logger.info(f"Deleted post {post_id}")
                return True
            else:
                logger.warning(f"Delete button not found for post {post_id}")
                return False

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def get_post_info(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a post.

        Args:
            post_id: Reddit post ID

        Returns:
            Post info dict or None
        """
        if not self._connected:
            if not self.connect():
                return None

        try:
            # Navigate to post
            post_url = f"https://old.reddit.com/comments/{post_id}"
            self.page.goto(post_url, wait_until='domcontentloaded', timeout=30000)
            self._random_delay(1, 2)

            # Extract info from page
            title_elem = self.page.locator('a.title')
            title = title_elem.text_content() if title_elem.count() > 0 else None

            score_elem = self.page.locator('.score.unvoted, .score.likes, .score.dislikes')
            score_text = score_elem.first.text_content() if score_elem.count() > 0 else "0"
            try:
                score = int(score_text.replace(',', '').split()[0])
            except (ValueError, IndexError):
                score = 0

            subreddit_elem = self.page.locator('a.subreddit')
            subreddit = subreddit_elem.text_content() if subreddit_elem.count() > 0 else None

            return {
                "id": post_id,
                "title": title,
                "url": self.page.url,
                "permalink": f"https://reddit.com/comments/{post_id}",
                "score": score,
                "subreddit": subreddit,
            }

        except Exception as e:
            logger.error(f"Get post info error: {e}")
            return None
