"""
Patreon video publisher using Playwright browser automation.

For publishing composed videos to Patreon creator pages.
Session cookies are persisted to avoid repeated logins.

First-time setup:
1. Call PatreonPublisher.interactive_login(niche, email) to open browser
2. Complete the login flow (Google OAuth, email verification, etc.)
3. Cookies are saved automatically for future headless use

Each niche has its own Patreon account with separate cookies.
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PatreonPublishResult:
    """Result from Patreon publish."""
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None


class PatreonPublisher:
    """
    Publish videos to Patreon using Playwright browser automation.

    Each niche has its own Patreon account with separate session cookies.

    First-time setup (run locally, not in Docker):
        python -c "from publishers import PatreonPublisher; PatreonPublisher.interactive_login('motivation', 'email@example.com')"

    After login, cookies are saved and publishing works headlessly.

    Usage:
        publisher = PatreonPublisher(niche='motivation')
        if publisher.connect():
            result = publisher.publish_video(
                "/path/to/video.mp4",
                title="My Video",
                description="Video description"
            )
            if result.success:
                print(f"Published: {result.post_url}")
        publisher.close()
    """

    COOKIES_DIR = "/data/patreon_cookies"
    LOCAL_COOKIES_DIR = "patreon_cookies"
    BASE_URL = "https://www.patreon.com"
    LOGIN_URL = "https://www.patreon.com/login"
    POSTS_URL = "https://www.patreon.com/posts/new"

    def __init__(self, niche: str, headless: bool = True):
        self.niche = niche
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._connected = False

    @property
    def cookies_file(self) -> str:
        """Get the cookies file path for this niche."""
        return os.path.join(self.COOKIES_DIR, f"{self.niche}_cookies.json")

    @property
    def local_cookies_file(self) -> str:
        """Get the local cookies file path for this niche."""
        return os.path.join(self.LOCAL_COOKIES_DIR, f"{self.niche}_cookies.json")

    @classmethod
    def interactive_login(cls, niche: str, email: str = None) -> bool:
        """
        Open a visible browser window for manual login.

        Call this method to set up the session for the first time.
        You'll need to complete the login flow manually.

        After successful login, cookies are saved for future headless use.

        Args:
            niche: The niche name (motivation, fitness, cooking, travel)
            email: Optional email hint for login

        Returns:
            True if login was successful
        """
        from playwright.sync_api import sync_playwright

        print("=" * 60)
        print(f"PATREON MANUAL LOGIN - {niche.upper()}")
        print("=" * 60)
        if email:
            print(f"Email: {email}")
        print("")
        print("A browser window will open. Please:")
        print("1. Log in to your Patreon account")
        print("2. If using Google OAuth, complete that flow")
        print("3. Once on your creator dashboard, the browser will close")
        print("=" * 60)
        print("")

        playwright = sync_playwright().start()

        # Use persistent context to avoid bot detection
        user_data_dir = f"/tmp/patreon_browser_{niche}"
        os.makedirs(user_data_dir, exist_ok=True)

        context = playwright.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            args=['--disable-blink-features=AutomationControlled']
        )
        page = context.new_page()
        page.set_default_timeout(120000)  # 2 minute timeout for manual actions

        try:
            # Navigate to login page
            page.goto(cls.LOGIN_URL, wait_until="domcontentloaded")
            time.sleep(3)

            # Wait for user to complete login (check every 5 seconds for up to 10 minutes)
            print("\nWaiting for you to complete login...")
            print("(Browser will close automatically when logged in)\n")

            max_wait = 600  # 10 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                time.sleep(5)

                try:
                    current_url = page.url

                    # Skip if still on login page or captcha
                    if "/login" in current_url.lower() or "challenge" in current_url.lower():
                        elapsed = int(time.time() - start_time)
                        remaining = max_wait - elapsed
                        print(f"Still on login page... ({remaining}s remaining)")
                        continue

                    # Check for successful login indicators
                    # After login, Patreon typically redirects to home or dashboard
                    if any(x in current_url for x in ["/home", "/dashboard", "/my-creators", "/c/"]):
                        # Double-check by looking for user avatar or create button
                        if page.query_selector('[data-tag="user-avatar"]') or \
                           page.query_selector('a[href*="/posts/new"]') or \
                           page.query_selector('[data-tag="sidebar-create-button"]'):
                            print("Login detected! Saving cookies...")
                            cls._save_cookies_static(context, niche)
                            print("Cookies saved successfully!")
                            return True

                    # If on posts/new page and can see editor, we're logged in
                    if "/posts/new" in current_url:
                        if page.query_selector('[data-tag="post-editor"]') or \
                           page.query_selector('div[contenteditable="true"]'):
                            print("Login successful! Saving cookies...")
                            cls._save_cookies_static(context, niche)
                            print("Cookies saved successfully!")
                            return True

                except Exception as e:
                    logger.debug(f"Check error: {e}")

                elapsed = int(time.time() - start_time)
                remaining = max_wait - elapsed
                print(f"Still waiting... ({remaining}s remaining)")

            print("Login timeout - 10 minutes elapsed")
            return False

        except Exception as e:
            print(f"Login error: {e}")
            return False
        finally:
            context.close()
            playwright.stop()

    @classmethod
    def _save_cookies_static(cls, context, niche: str) -> None:
        """Save cookies from a browser context (static method for interactive_login)."""
        cookies = context.cookies()

        # Save to both local and Docker paths
        save_paths = [
            os.path.join(cls.LOCAL_COOKIES_DIR, f"{niche}_cookies.json"),
            os.path.join(cls.COOKIES_DIR, f"{niche}_cookies.json"),
        ]

        for path in save_paths:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(cookies, f)
                print(f"Saved {len(cookies)} cookies to {path}")
            except Exception as e:
                logger.debug(f"Could not save to {path}: {e}")

    def connect(self) -> bool:
        """
        Initialize browser and load saved session.

        Returns True if connected and logged in.
        If not logged in, returns False - use interactive_login() first.
        """
        try:
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright().start()
            self.browser = self.playwright.firefox.launch(
                headless=self.headless,
                args=['--no-sandbox']
            )

            self.context = self.browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
            )

            # Load saved cookies
            self._load_cookies()

            self.page = self.context.new_page()
            self.page.set_default_timeout(60000)  # 60 second timeout

            # Check if we're logged in
            if self._check_logged_in():
                logger.info(f"Connected to Patreon for niche: {self.niche}")
                self._connected = True
                return True

            logger.warning(f"Not logged in to Patreon for {self.niche}. Session cookies missing or expired.")
            logger.warning("Run interactive_login() to set up a new session.")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to Patreon: {e}")
            return False

    def _check_logged_in(self) -> bool:
        """Check if we're logged in by visiting the site."""
        try:
            self.page.goto(self.BASE_URL, wait_until="domcontentloaded")
            time.sleep(2)

            # Look for logged-in indicators
            logged_in_indicators = [
                '[data-tag="user-menu"]',
                'a[href*="/my-creators"]',
                'a[href*="/dashboard"]',
                'a[href*="/posts/new"]',
                '[aria-label="User menu"]',
            ]

            for selector in logged_in_indicators:
                if self.page.query_selector(selector):
                    logger.debug(f"Found login indicator: {selector}")
                    return True

            # Try to access post creation page
            self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")
            time.sleep(2)

            # If redirected to login, we're not logged in
            if "/login" in self.page.url.lower():
                return False

            # Check for post creation UI
            if self.page.query_selector('button:has-text("Publish")') or \
               self.page.query_selector('[data-tag="post-editor"]'):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking login status: {e}")
            return False

    def _load_cookies(self):
        """Load saved session cookies."""
        cookie_paths = [
            self.cookies_file,
            self.local_cookies_file,
        ]

        for path in cookie_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        cookies = json.load(f)
                        if cookies:
                            self.context.add_cookies(cookies)
                            logger.info(f"Loaded {len(cookies)} cookies from {path}")
                            return
            except Exception as e:
                logger.debug(f"Could not load cookies from {path}: {e}")
        logger.warning(f"No session cookies found for {self.niche}. Run interactive_login() first.")

    def _save_cookies(self):
        """Save current session cookies."""
        try:
            cookies = self.context.cookies()
            Path(self.COOKIES_DIR).mkdir(parents=True, exist_ok=True)
            with open(self.cookies_file, 'w') as f:
                json.dump(cookies, f)
            logger.debug(f"Saved {len(cookies)} cookies to {self.cookies_file}")
        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")

    def publish_video(
        self,
        video_path: str,
        title: str,
        description: str = None,
        tags: list = None,
        public: bool = True
    ) -> PatreonPublishResult:
        """
        Publish a video to Patreon.

        Args:
            video_path: Path to video file
            title: Post title
            description: Optional post description
            tags: Optional list of tags
            public: If True, post is public. If False, patrons-only.

        Returns:
            PatreonPublishResult with success status and URL
        """
        if not self._connected:
            if not self.connect():
                return PatreonPublishResult(success=False, error="Not connected to Patreon")

        if not os.path.exists(video_path):
            return PatreonPublishResult(success=False, error=f"Video file not found: {video_path}")

        try:
            file_size = os.path.getsize(video_path)
            logger.info(f"Publishing video to Patreon: {video_path} ({file_size // 1024 // 1024}MB)")

            # Navigate to post creation page
            self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")
            time.sleep(3)

            # Wait for page to fully load
            self.page.wait_for_load_state("networkidle")

            # Set the post title
            self._set_title(title)
            time.sleep(1)

            # Set description if provided
            if description:
                self._set_description(description)
                time.sleep(1)

            # Upload the video
            upload_success = self._upload_video(video_path)
            if not upload_success:
                return PatreonPublishResult(success=False, error="Failed to upload video")

            # Wait for video to process
            time.sleep(5)

            # Set visibility
            self._set_visibility(public)
            time.sleep(1)

            # Click publish
            result = self._click_publish()

            if result.success:
                self._save_cookies()
                logger.info(f"Published to Patreon: {result.post_url}")

            return result

        except Exception as e:
            logger.error(f"Publish error: {e}")
            return PatreonPublishResult(success=False, error=str(e))

    def _set_title(self, title: str):
        """Set the post title."""
        try:
            title_selectors = [
                'input[placeholder*="title" i]',
                'input[name*="title" i]',
                '[data-tag="post-title-input"] input',
                '[aria-label*="title" i]',
            ]

            for selector in title_selectors:
                title_input = self.page.query_selector(selector)
                if title_input:
                    title_input.fill(title)
                    logger.debug(f"Set title: {title[:50]}...")
                    return

            # Try clicking on a title area first
            title_area = self.page.query_selector('h1, [class*="title"]')
            if title_area:
                title_area.click()
                time.sleep(0.5)
                self.page.keyboard.type(title)
                return

            logger.warning("Could not find title input field")

        except Exception as e:
            logger.warning(f"Failed to set title: {e}")

    def _set_description(self, description: str):
        """Set the post description/body."""
        try:
            desc_selectors = [
                'textarea[placeholder*="Write" i]',
                '[contenteditable="true"]',
                '[data-tag="post-editor"]',
                '.ProseMirror',
                '[role="textbox"]',
            ]

            for selector in desc_selectors:
                desc_input = self.page.query_selector(selector)
                if desc_input:
                    desc_input.click()
                    time.sleep(0.3)
                    # Use keyboard for contenteditable elements
                    self.page.keyboard.type(description)
                    logger.debug(f"Set description: {description[:50]}...")
                    return

            logger.warning("Could not find description input field")

        except Exception as e:
            logger.warning(f"Failed to set description: {e}")

    def _upload_video(self, video_path: str) -> bool:
        """Upload the video file."""
        try:
            # Look for file input or media upload button
            file_input = self.page.query_selector('input[type="file"]')

            if not file_input:
                # Try clicking media/attach button first
                media_buttons = [
                    'button[aria-label*="media" i]',
                    'button[aria-label*="attach" i]',
                    'button[aria-label*="video" i]',
                    '[data-tag="media-button"]',
                    'button:has-text("Add media")',
                    'button:has-text("Upload")',
                ]

                for selector in media_buttons:
                    button = self.page.query_selector(selector)
                    if button:
                        button.click()
                        time.sleep(1)
                        break

                # Now look for file input again
                file_input = self.page.query_selector('input[type="file"]')

            if file_input:
                file_input.set_input_files(video_path)
                logger.debug("Video file selected for upload")

                # Wait for upload to start and progress
                time.sleep(3)

                # Wait for upload to complete (check for progress indicators)
                max_wait = 300  # 5 minutes max for upload
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    # Check for upload complete indicators
                    if self.page.query_selector('video, [data-tag="video-preview"]'):
                        logger.debug("Video upload complete")
                        return True

                    # Check for progress
                    progress = self.page.query_selector('[class*="progress"], [role="progressbar"]')
                    if progress:
                        logger.debug("Upload in progress...")

                    time.sleep(3)

                logger.warning("Video upload timeout")
                return False

            logger.error("Could not find file upload input")
            return False

        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return False

    def _set_visibility(self, public: bool):
        """Set post visibility (public or patrons-only)."""
        try:
            # Look for visibility/access controls
            visibility_selectors = [
                '[data-tag="visibility-selector"]',
                'button:has-text("Public")',
                'button:has-text("Patrons")',
                '[aria-label*="visibility" i]',
                '[class*="visibility"]',
            ]

            for selector in visibility_selectors:
                element = self.page.query_selector(selector)
                if element:
                    element.click()
                    time.sleep(0.5)

                    if public:
                        public_option = self.page.query_selector('text=Public, [value="public"]')
                        if public_option:
                            public_option.click()
                    else:
                        patron_option = self.page.query_selector('text=Patrons, [value="patrons"]')
                        if patron_option:
                            patron_option.click()

                    logger.debug(f"Set visibility: {'public' if public else 'patrons-only'}")
                    return

        except Exception as e:
            logger.warning(f"Failed to set visibility: {e}")

    def _click_publish(self) -> PatreonPublishResult:
        """Click the publish button and wait for completion."""
        try:
            publish_selectors = [
                'button:has-text("Publish")',
                'button:has-text("Post")',
                'button[type="submit"]',
                '[data-tag="publish-button"]',
            ]

            for selector in publish_selectors:
                button = self.page.query_selector(selector)
                if button and button.is_visible():
                    button.click()
                    logger.debug("Publish button clicked")
                    break
            else:
                return PatreonPublishResult(success=False, error="Could not find publish button")

            # Wait for publish to complete
            time.sleep(5)

            # Check for success - usually redirects to the post page
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                current_url = self.page.url

                # Check if we're on a post page
                if "/posts/" in current_url and current_url != self.POSTS_URL:
                    post_id = current_url.split("/posts/")[-1].split("?")[0].split("/")[0]
                    return PatreonPublishResult(
                        success=True,
                        post_id=post_id,
                        post_url=current_url
                    )

                # Check for success message
                success_msg = self.page.query_selector('[class*="success"], [data-tag="publish-success"]')
                if success_msg:
                    # Try to extract post URL
                    post_link = self.page.query_selector('a[href*="/posts/"]')
                    if post_link:
                        href = post_link.get_attribute("href")
                        post_id = href.split("/posts/")[-1].split("?")[0]
                        return PatreonPublishResult(
                            success=True,
                            post_id=post_id,
                            post_url=f"https://www.patreon.com/posts/{post_id}"
                        )

                time.sleep(2)

            # If we get here, assume success but couldn't get URL
            logger.warning("Publish seemed to succeed but couldn't extract post URL")
            return PatreonPublishResult(success=True, post_url=self.page.url)

        except Exception as e:
            logger.error(f"Publish error: {e}")
            return PatreonPublishResult(success=False, error=str(e))

    def close(self):
        """Close browser and cleanup resources."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logger.debug("Patreon browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
