"""
Persistent Patreon browser session for interactive publishing.

The browser runs on a virtual display (Xvfb) with VNC access via noVNC.
This allows users to see and interact with the browser through their web browser
to handle CAPTCHAs and login.

Usage:
    from publishers.patreon_session import patreon_session_manager

    # Start session (async) - returns noVNC URL for browser access
    result = await patreon_session_manager.start_session("motivation")
    # Connect to result["novnc_url"] to see the browser

    # Publish using the session
    result = await patreon_session_manager.publish_video(video_path, title, description, tags)

    # Stop session when done
    await patreon_session_manager.stop_session()
"""
import asyncio
import base64
import os
import json
import logging
import random
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Human-like behavior utilities
# =============================================================================

async def human_delay(min_sec: float = 0.5, max_sec: float = 2.0):
    """Random delay to simulate human thinking/reaction time."""
    delay = random.uniform(min_sec, max_sec)
    await asyncio.sleep(delay)


async def human_short_delay():
    """Short delay for quick actions (0.1-0.4s)."""
    await asyncio.sleep(random.uniform(0.1, 0.4))


async def human_medium_delay():
    """Medium delay for reading/processing (0.8-2.5s)."""
    await asyncio.sleep(random.uniform(0.8, 2.5))


async def human_long_delay():
    """Longer delay for significant pauses (2-5s)."""
    await asyncio.sleep(random.uniform(2.0, 5.0))


def random_offset(base: int, variance: int = 5) -> int:
    """Add random pixel offset to coordinates for natural mouse movement."""
    return base + random.randint(-variance, variance)


async def human_type(page, text: str, min_delay: int = 30, max_delay: int = 120):
    """
    Type text with human-like variable speed.
    Includes occasional pauses and speed variations.
    """
    for i, char in enumerate(text):
        # Occasional longer pause (like thinking)
        if random.random() < 0.05:  # 5% chance
            await asyncio.sleep(random.uniform(0.3, 0.8))

        # Variable typing speed
        delay = random.randint(min_delay, max_delay)

        # Slow down at punctuation
        if char in '.,!?;:':
            delay = random.randint(80, 200)

        await page.keyboard.type(char, delay=delay)

        # Occasional micro-pause between words
        if char == ' ' and random.random() < 0.15:
            await asyncio.sleep(random.uniform(0.1, 0.3))


@dataclass
class PatreonPublishResult:
    """Result from Patreon publish."""
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None


class PatreonSessionManager:
    """
    Manages a persistent Patreon browser session using async playwright.

    The browser runs on a virtual display (Xvfb) with VNC access via noVNC.
    This allows users to see and interact with the browser through a web interface.
    Screenshots are also captured as backup.
    """

    BASE_URL = "https://www.patreon.com"
    LOGIN_URL = "https://www.patreon.com/login"
    POSTS_URL = "https://www.patreon.com/posts/new"
    SCREENSHOT_DIR = "/data/patreon_screenshots"

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.niche: Optional[str] = None
        self._lock = asyncio.Lock()
        self._started_at: Optional[datetime] = None
        self._publish_count = 0
        self._latest_screenshot: Optional[str] = None
        self._last_activity: Optional[str] = None
        self._vnc_active = False

        # Recording state
        self._recording_active = False
        self._recording_name: Optional[str] = None
        self._poll_task: Optional[asyncio.Task] = None

        # Ensure screenshot directory exists
        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)

    def get_status(self) -> Dict[str, Any]:
        """Get current session status (sync method for quick status checks)."""
        is_active = self.page is not None

        # Get VNC status if active
        vnc_info = {}
        if self._vnc_active:
            try:
                from utils.vnc_manager import vnc_manager
                vnc_status = vnc_manager.get_status()
                vnc_info = {
                    "vnc_active": vnc_status.get("active", False),
                    "novnc_port": vnc_status.get("novnc_port"),
                    "novnc_url": vnc_status.get("novnc_url"),
                }
            except Exception:
                vnc_info = {"vnc_active": False}

        return {
            "active": is_active,
            "niche": self.niche if is_active else None,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "publish_count": self._publish_count,
            "current_url": None,  # Can't get URL synchronously
            "last_activity": self._last_activity,
            "has_screenshot": self._latest_screenshot is not None,
            "recording_active": self._recording_active,
            "recording_name": self._recording_name,
            **vnc_info,
        }

    async def _take_screenshot(self, activity: str) -> Optional[str]:
        """Take a screenshot and return the path."""
        if not self.page:
            return None

        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"patreon_{self.niche}_{timestamp}.png"
            filepath = os.path.join(self.SCREENSHOT_DIR, filename)

            await self.page.screenshot(path=filepath)
            self._latest_screenshot = filepath
            self._last_activity = activity

            logger.info(f"Screenshot saved: {filepath} ({activity})")
            return filepath

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def get_latest_screenshot(self) -> Dict[str, Any]:
        """Get the latest screenshot as base64."""
        if not self._latest_screenshot or not os.path.exists(self._latest_screenshot):
            return {"available": False}

        try:
            with open(self._latest_screenshot, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            return {
                "available": True,
                "path": self._latest_screenshot,
                "activity": self._last_activity,
                "data": f"data:image/png;base64,{image_data}"
            }
        except Exception as e:
            logger.error(f"Failed to read screenshot: {e}")
            return {"available": False, "error": str(e)}

    # =========================================================================
    # Human-like interaction methods
    # =========================================================================

    async def _human_move_to(self, x: int, y: int, steps: int = 10):
        """
        Move mouse to coordinates in a natural curved path.
        Uses bezier-like movement with slight randomization.
        """
        if not self.page:
            return

        try:
            # Get current mouse position (approximate from viewport center if unknown)
            viewport = self.page.viewport_size or {'width': 1400, 'height': 1000}
            # Start from a reasonable position
            start_x = viewport['width'] // 2
            start_y = viewport['height'] // 2

            # Add slight randomness to target
            target_x = random_offset(x, 3)
            target_y = random_offset(y, 3)

            # Move in steps with slight curve
            for i in range(steps):
                progress = (i + 1) / steps
                # Add slight sine wave for natural curve
                curve_offset = int(5 * (1 - progress) * random.uniform(-1, 1))

                current_x = int(start_x + (target_x - start_x) * progress) + curve_offset
                current_y = int(start_y + (target_y - start_y) * progress)

                await self.page.mouse.move(current_x, current_y)
                await asyncio.sleep(random.uniform(0.01, 0.03))

        except Exception as e:
            logger.debug(f"Mouse move failed: {e}")

    async def _human_click(self, element, hover_time: float = None):
        """
        Click element with human-like behavior:
        1. Move mouse to element area
        2. Hover for a moment (like reading/considering)
        3. Click with slight delay
        """
        if not self.page or not element:
            return False

        try:
            # Get element bounding box
            box = await element.bounding_box()
            if not box:
                # Fallback to direct click
                await element.click()
                return True

            # Calculate click point with random offset within element
            click_x = int(box['x'] + box['width'] * random.uniform(0.3, 0.7))
            click_y = int(box['y'] + box['height'] * random.uniform(0.3, 0.7))

            # Move to element naturally
            await self._human_move_to(click_x, click_y)

            # Hover time (like reading the button)
            if hover_time is None:
                hover_time = random.uniform(0.2, 0.8)
            await asyncio.sleep(hover_time)

            # Click with slight position variance
            await self.page.mouse.click(
                random_offset(click_x, 2),
                random_offset(click_y, 2)
            )

            await human_short_delay()
            return True

        except Exception as e:
            logger.warning(f"Human click failed, falling back to direct click: {e}")
            try:
                await element.click()
                return True
            except:
                return False

    async def _human_scroll(self, direction: str = "down", amount: int = None):
        """
        Scroll page naturally with variable speed.
        direction: 'up' or 'down'
        """
        if not self.page:
            return

        try:
            if amount is None:
                amount = random.randint(100, 300)

            if direction == "up":
                amount = -amount

            # Scroll in smaller increments
            steps = random.randint(3, 6)
            for _ in range(steps):
                scroll_amount = amount // steps + random.randint(-20, 20)
                await self.page.mouse.wheel(0, scroll_amount)
                await asyncio.sleep(random.uniform(0.05, 0.15))

        except Exception as e:
            logger.debug(f"Human scroll failed: {e}")

    async def _random_mouse_movement(self):
        """Small random mouse movements to appear alive."""
        if not self.page:
            return

        try:
            viewport = self.page.viewport_size or {'width': 1400, 'height': 1000}
            x = random.randint(100, viewport['width'] - 100)
            y = random.randint(100, viewport['height'] - 100)
            await self.page.mouse.move(x, y)
        except:
            pass

    async def _human_type_in_element(self, element, text: str):
        """Click element and type text with human-like behavior."""
        if not element:
            return False

        try:
            # Click to focus with human behavior
            await self._human_click(element, hover_time=random.uniform(0.3, 0.6))
            await human_short_delay()

            # Clear existing content
            await self.page.keyboard.press('Control+a')
            await human_short_delay()
            await self.page.keyboard.press('Backspace')
            await human_short_delay()

            # Type with human-like speed
            await human_type(self.page, text)
            return True

        except Exception as e:
            logger.warning(f"Human type failed: {e}")
            return False

    async def _set_title_human(self, title: str) -> bool:
        """
        Set post title with human-like behavior.
        Uses natural mouse movements, variable typing speed, and realistic pauses.
        """
        if not self.page:
            return False

        try:
            # Strategy 1: Look for Patreon-specific title input selectors
            title_selectors = [
                '[data-tag="post-title"]',
                '[data-tag="post-title-input"]',
                '[placeholder*="title" i]',
                '[placeholder*="Title" i]',
                '[aria-label*="title" i]',
                'div[contenteditable="true"][data-placeholder*="title" i]',
                'input[name="title"]',
            ]

            for selector in title_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem and await elem.is_visible():
                        logger.info(f"Found title element with selector: {selector}")

                        # Human-like click on title field
                        await self._human_click(elem, hover_time=random.uniform(0.3, 0.7))
                        await human_short_delay()

                        # Clear existing content (human-like select all + delete)
                        await self.page.keyboard.press('Control+a')
                        await human_short_delay()
                        await self.page.keyboard.press('Backspace')
                        await human_short_delay()

                        # Type with human-like speed
                        await human_type(self.page, title)
                        logger.info(f"Title set via specific selector (human): {title[:40]}...")
                        return True
                except Exception as e:
                    logger.debug(f"Title selector {selector} failed: {e}")

            # Strategy 2: Get all contenteditable elements and find the title one
            all_editables = await self.page.query_selector_all('[contenteditable="true"]')
            editable_count = len(all_editables)
            logger.info(f"Found {editable_count} contenteditable elements on page")

            if editable_count == 0:
                logger.warning("No contenteditable elements found - trying position-based click")
                return await self._set_title_by_position_human(title)

            # Patreon's post editor: title is usually the first or has specific placeholder
            for i, elem in enumerate(all_editables):
                try:
                    is_visible = await elem.is_visible()
                    if not is_visible:
                        continue

                    # Check for title-like attributes
                    placeholder = await elem.get_attribute('data-placeholder') or ''
                    aria_label = await elem.get_attribute('aria-label') or ''

                    # If it looks like a title field
                    if 'title' in placeholder.lower() or 'title' in aria_label.lower() or i == 0:
                        logger.info(f"Selecting editable element {i} as title field (human)")

                        # Human-like click to focus
                        await self._human_click(elem, hover_time=random.uniform(0.2, 0.5))
                        await human_short_delay()

                        # Clear content (human-like)
                        await self.page.keyboard.press('Control+a')
                        await human_short_delay()
                        await self.page.keyboard.press('Backspace')
                        await human_short_delay()

                        # Type the title with human-like speed
                        await human_type(self.page, title)
                        logger.info(f"Title typed (human): {title[:40]}...")

                        # Human-like pause after typing (reviewing)
                        await human_short_delay()

                        # Verify it was set
                        new_content = await elem.text_content() or ''
                        if title[:20] in new_content:
                            logger.info("Title verified successfully")
                            return True
                        else:
                            logger.warning(f"Title may not have been set correctly. Content: {new_content[:40]}")

                        return True

                except Exception as e:
                    logger.debug(f"Editable element {i} failed: {e}")

            logger.warning("Could not set title via contenteditable - trying position-based")
            return await self._set_title_by_position_human(title)

        except Exception as e:
            logger.error(f"Human title setting failed: {e}")
            return await self._set_title_by_position_human(title)

    async def _set_title_by_position_human(self, title: str) -> bool:
        """Set title by clicking at a position with human-like behavior."""
        if not self.page:
            return False

        try:
            viewport = self.page.viewport_size
            if not viewport:
                logger.error("Could not get viewport size")
                return False

            # Click in upper-middle area where title input typically is
            click_x = viewport['width'] // 2 + random.randint(-50, 50)
            click_y = 200 + random.randint(-20, 20)

            logger.info(f"Position-based human click at ({click_x}, {click_y})")

            # Human-like: Move to position naturally
            await self._human_move_to(click_x, click_y)
            await human_short_delay()

            # Click
            await self.page.mouse.click(click_x, click_y)
            await human_short_delay()

            # Clear and type
            await self.page.keyboard.press('Control+a')
            await human_short_delay()
            await self.page.keyboard.press('Backspace')
            await human_short_delay()

            # Type with human-like speed
            await human_type(self.page, title)
            logger.info(f"Title typed via position-based click (human): {title[:40]}...")

            return True

        except Exception as e:
            logger.error(f"Position-based human title input failed: {e}")
            return False

    async def start_session(self, niche: str) -> Dict[str, Any]:
        """
        Start a persistent browser session with VNC display.

        The browser runs on a virtual display (Xvfb) with VNC access via noVNC.
        Users can connect to the noVNC URL to see and interact with the browser.
        Loads saved cookies if available for authentication.
        """
        logger.info(f"[SESSION] start_session called for {niche}")

        # Use timeout on lock acquisition
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            logger.error("[SESSION] Could not acquire lock within 5 seconds")
            return {"status": "error", "error": "Session busy, try again"}

        try:
            if self.page is not None:
                if self.niche == niche:
                    # Return existing session info with VNC URL
                    from utils.vnc_manager import vnc_manager
                    vnc_status = vnc_manager.get_status()
                    return {
                        "status": "already_running",
                        "niche": niche,
                        "novnc_port": vnc_status.get("novnc_port"),
                        "novnc_url": vnc_status.get("novnc_url"),
                    }
                else:
                    # Different niche - stop and restart
                    logger.info("[SESSION] Different niche, stopping existing session")
                    await self._stop_session_internal()

            try:
                # Start VNC display first
                logger.info("[SESSION] Starting VNC display...")
                from utils.vnc_manager import vnc_manager
                vnc_result = vnc_manager.start()
                if vnc_result.get("status") == "error":
                    return {"status": "error", "error": f"VNC failed: {vnc_result.get('error')}"}

                self._vnc_active = True
                display = vnc_result.get("display", ":99")
                novnc_port = vnc_result.get("novnc_port", 6080)
                logger.info(f"[SESSION] VNC started on display {display}, noVNC port {novnc_port}")

                # Set DISPLAY environment variable for browser
                os.environ["DISPLAY"] = display

                logger.info("[SESSION] Importing async playwright...")
                from playwright.async_api import async_playwright

                logger.info(f"[SESSION] Starting playwright for {niche}...")
                self.playwright = await async_playwright().start()
                logger.info("[SESSION] Playwright started")

                # Use persistent context to retain login state
                user_data_dir = f"/tmp/patreon_browser_{niche}"
                os.makedirs(user_data_dir, exist_ok=True)
                logger.info(f"[SESSION] User data dir: {user_data_dir}")

                # Launch NON-HEADLESS browser (visible on VNC display)
                # Enhanced anti-detection: realistic viewport, timezone, locale
                logger.info("[SESSION] Launching browser context (visible on VNC)...")

                # Randomize viewport slightly for uniqueness
                viewport_width = random.randint(1350, 1400)
                viewport_height = random.randint(950, 1050)

                self.context = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False,  # Visible browser on VNC display
                    viewport={'width': viewport_width, 'height': viewport_height},
                    # Use realistic, recent Chrome user agent
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    # Realistic locale and timezone
                    locale='en-US',
                    timezone_id='America/New_York',
                    # Color scheme preference (randomize)
                    color_scheme='light' if random.random() > 0.3 else 'dark',
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--no-sandbox',
                        '--disable-gpu',  # Helps with virtual display
                        '--start-maximized',
                        # Additional anti-detection args
                        '--disable-dev-shm-usage',
                        '--disable-infobars',
                        '--disable-extensions',
                        '--disable-plugins-discovery',
                    ],
                    # Permissions that a real browser would have
                    permissions=['geolocation'],
                )
                logger.info("[SESSION] Browser context launched")

                # Load saved cookies if available
                logger.info("[SESSION] Loading cookies...")
                await self._load_cookies(niche)

                logger.info("[SESSION] Creating new page...")
                self.page = await self.context.new_page()
                self.page.set_default_timeout(60000)  # 60 second timeout
                self.niche = niche
                self._started_at = datetime.utcnow()
                self._publish_count = 0
                self._latest_screenshot = None
                logger.info("[SESSION] Page created")

                # Anti-detection: Override navigator.webdriver property
                await self.page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    // Also hide automation indicators
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                    // Hide chrome driver detection
                    window.chrome = { runtime: {} };
                """)

                # Navigate to Patreon post creation page (forces creator view)
                # If not logged in, Patreon will redirect to login first
                logger.info("[SESSION] Navigating to Patreon posts page (creator view)...")

                # Human-like: Random initial delay before navigation
                await asyncio.sleep(random.uniform(1.0, 2.5))

                await self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")

                # Human-like: Wait as if looking at page
                await asyncio.sleep(random.uniform(2.0, 4.0))
                logger.info("[SESSION] Navigation complete")

                # Take initial screenshot
                logger.info("[SESSION] Taking screenshot...")
                await self._take_screenshot("session_started")

                # Check if logged in
                logger.info("[SESSION] Checking login status...")
                logged_in = await self._check_logged_in()

                logger.info(f"[SESSION] Session started for {niche}, logged_in={logged_in}")

                return {
                    "status": "started",
                    "niche": niche,
                    "logged_in": logged_in,
                    "novnc_port": novnc_port,
                    "novnc_url": vnc_result.get("novnc_url"),
                    "message": "Browser session started. Connect to noVNC to see and interact with the browser."
                }

            except Exception as e:
                logger.exception(f"[SESSION] Failed to start Patreon session: {e}")
                await self._stop_session_internal()
                return {"status": "error", "error": str(e)}
        finally:
            self._lock.release()
            logger.info("[SESSION] Lock released")

    async def _load_cookies(self, niche: str):
        """Load saved cookies for the niche."""
        cookie_paths = [
            f"/data/patreon_cookies/{niche}_cookies.json",
            f"patreon_cookies/{niche}_cookies.json",
        ]

        for path in cookie_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        cookies = json.load(f)
                    await self.context.add_cookies(cookies)
                    logger.info(f"Loaded {len(cookies)} cookies from {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load cookies from {path}: {e}")

        logger.warning(f"No cookies found for {niche}")

    async def _check_logged_in(self) -> bool:
        """Check if currently logged in to Patreon."""
        if not self.page:
            return False

        try:
            # Look for logged-in indicators
            indicators = [
                '[data-tag="user-menu"]',
                '[data-tag="user-avatar"]',
                'a[href*="/my-creators"]',
                'a[href*="/posts/new"]',
            ]

            for selector in indicators:
                if await self.page.query_selector(selector):
                    return True

            return False
        except Exception:
            return False

    async def stop_session(self) -> Dict[str, Any]:
        """Stop the browser session."""
        async with self._lock:
            if self.page is None:
                return {"status": "not_running"}

            niche = self.niche
            publish_count = self._publish_count
            await self._stop_session_internal()

            return {
                "status": "stopped",
                "niche": niche,
                "publish_count": publish_count
            }

    async def _stop_session_internal(self):
        """Internal method to stop session (must hold lock)."""
        try:
            if self.context:
                # Save cookies before closing
                await self._save_cookies()
                await self.context.close()
        except Exception as e:
            logger.error(f"Error closing context: {e}")

        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error(f"Error stopping playwright: {e}")

        # Stop VNC display
        if self._vnc_active:
            try:
                from utils.vnc_manager import vnc_manager
                vnc_manager.stop()
                logger.info("[SESSION] VNC display stopped")
            except Exception as e:
                logger.error(f"Error stopping VNC: {e}")
            self._vnc_active = False

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.niche = None
        self._started_at = None

    async def _save_cookies(self) -> int:
        """Save cookies from current session. Returns cookie count."""
        if not self.context or not self.niche:
            return 0

        try:
            cookies = await self.context.cookies()

            # Save to both paths for compatibility
            paths = [
                f"/data/patreon_cookies/{self.niche}_cookies.json",
                f"patreon_cookies/{self.niche}_cookies.json",
            ]

            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(cookies, f, indent=2)
                logger.info(f"Saved {len(cookies)} cookies to {path}")

            return len(cookies)

        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
            return 0

    async def save_cookies(self) -> int:
        """
        Public method to save cookies from current browser session.

        Call this after logging in via VNC to persist the session
        for headless publishing to use.

        Returns: Number of cookies saved (0 if failed)
        """
        if not self.context or not self.niche:
            logger.error("Cannot save cookies: no active session")
            return 0

        return await self._save_cookies()

    async def navigate_to_login(self) -> Dict[str, Any]:
        """Navigate to login page."""
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        try:
            await self.page.goto(self.LOGIN_URL, wait_until="domcontentloaded")
            await self._take_screenshot("navigated_to_login")
            return {"status": "ok", "url": self.page.url}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def navigate_to_post(self) -> Dict[str, Any]:
        """Navigate to new post page."""
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        try:
            await self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")
            await asyncio.sleep(2)
            logged_in = await self._check_logged_in()
            await self._take_screenshot("navigated_to_post")
            return {"status": "ok", "url": self.page.url, "logged_in": logged_in}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def check_login_status(self) -> Dict[str, Any]:
        """Check current login status."""
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        logged_in = await self._check_logged_in()
        return {
            "logged_in": logged_in,
            "current_url": self.page.url
        }

    async def try_solve_cloudflare(self) -> Dict[str, Any]:
        """
        Try to solve Cloudflare CAPTCHA challenge.
        Clicks the checkbox if present and waits for verification.
        Returns status of the attempt.
        """
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        try:
            await self._take_screenshot("before_cloudflare_solve")

            # Check if we're on a Cloudflare challenge page
            page_content = await self.page.content()
            is_cloudflare = "cloudflare" in page_content.lower() or "verify you are human" in page_content.lower()

            if not is_cloudflare:
                return {
                    "status": "not_needed",
                    "message": "Not on a Cloudflare challenge page",
                    "current_url": self.page.url
                }

            logger.info("Cloudflare challenge detected, attempting to solve...")

            # Try multiple approaches to click the checkbox
            checkbox_clicked = False

            # Approach 1: Look for the Cloudflare checkbox in an iframe
            frames = self.page.frames
            for frame in frames:
                try:
                    # Cloudflare checkbox selectors
                    checkbox_selectors = [
                        'input[type="checkbox"]',
                        '.ctp-checkbox-label',
                        '#challenge-stage input',
                        'label.ctp-checkbox-label',
                    ]
                    for selector in checkbox_selectors:
                        checkbox = await frame.query_selector(selector)
                        if checkbox:
                            await checkbox.click(timeout=5000)
                            checkbox_clicked = True
                            logger.info(f"Clicked Cloudflare checkbox via selector: {selector}")
                            break
                    if checkbox_clicked:
                        break
                except Exception as e:
                    logger.debug(f"Frame checkbox attempt failed: {e}")

            # Approach 2: Click at approximate position of checkbox (typically left-center)
            if not checkbox_clicked:
                try:
                    # The checkbox is usually in an iframe in the upper-left area
                    # Typical position: around (200, 270) based on standard Cloudflare layout
                    await self.page.mouse.click(200, 270)
                    checkbox_clicked = True
                    logger.info("Clicked at approximate Cloudflare checkbox position (200, 270)")
                except Exception as e:
                    logger.warning(f"Position click failed: {e}")

            if checkbox_clicked:
                # Wait for verification to complete
                await asyncio.sleep(5)
                await self._take_screenshot("after_cloudflare_click")

                # Check if we passed the challenge
                new_content = await self.page.content()
                still_cloudflare = "cloudflare" in new_content.lower() and "verify you are human" in new_content.lower()

                if not still_cloudflare:
                    # Wait a bit more for page to fully load
                    await asyncio.sleep(3)
                    await self._take_screenshot("cloudflare_solved")
                    return {
                        "status": "solved",
                        "message": "Cloudflare challenge appears to be solved",
                        "current_url": self.page.url
                    }
                else:
                    return {
                        "status": "still_challenged",
                        "message": "Clicked checkbox but still on challenge page. May need manual verification via VNC.",
                        "current_url": self.page.url
                    }
            else:
                return {
                    "status": "click_failed",
                    "message": "Could not click Cloudflare checkbox. Use VNC to solve manually.",
                    "current_url": self.page.url
                }

        except Exception as e:
            logger.error(f"Cloudflare solve attempt failed: {e}")
            await self._take_screenshot("cloudflare_error")
            return {
                "status": "error",
                "error": str(e),
                "current_url": self.page.url if self.page else None
            }

    async def _dismiss_modals(self) -> int:
        """
        Dismiss any modal dialogs blocking the page.
        Handles:
        - Feature announcement modals (X close button)
        - Privacy/cookie consent dialogs (Accept button)
        - Generic close buttons

        Returns the number of modals dismissed.
        """
        if not self.page:
            return 0

        dismissed = 0

        try:
            # Modal close button selectors (various Patreon modal types)
            close_selectors = [
                # Feature announcement modals - X button
                'button[aria-label="Close"]',
                'button[aria-label="close"]',
                '[data-tag="close-button"]',
                'button.close-button',
                '[aria-label="Close modal"]',
                # Generic close icons
                'button svg[aria-label="Close"]',
                'div[role="dialog"] button:has(svg)',
            ]

            for selector in close_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for elem in elements:
                        if await elem.is_visible():
                            await elem.click(timeout=2000)
                            await asyncio.sleep(0.5)
                            dismissed += 1
                            logger.info(f"Dismissed modal via selector: {selector}")
                except Exception as e:
                    logger.debug(f"Close selector {selector} failed: {e}")

            # Cookie consent / Privacy Settings dialog - click Accept
            accept_selectors = [
                'button:has-text("Accept")',
                'button:has-text("Accept all")',
                'button:has-text("Accept All")',
                'button:has-text("I Accept")',
                'button:has-text("Got it")',
                'button:has-text("OK")',
                '[data-tag="accept-cookies"]',
            ]

            for selector in accept_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem and await elem.is_visible():
                        await elem.click(timeout=2000)
                        await asyncio.sleep(0.5)
                        dismissed += 1
                        logger.info(f"Accepted dialog via selector: {selector}")
                        break  # Only click one accept button
                except Exception as e:
                    logger.debug(f"Accept selector {selector} failed: {e}")

            # Also try pressing Escape to close any remaining modals
            if dismissed == 0:
                try:
                    await self.page.keyboard.press('Escape')
                    await asyncio.sleep(0.3)
                    logger.debug("Pressed Escape to try closing modals")
                except Exception:
                    pass

            if dismissed > 0:
                logger.info(f"Dismissed {dismissed} modal(s)")
                await self._take_screenshot("after_dismiss_modals")

        except Exception as e:
            logger.warning(f"Error dismissing modals: {e}")

        return dismissed

    async def _set_title_fast(self, title: str) -> bool:
        """
        Fast title setting with simplified approach.
        Looks for Patreon-specific title elements first, then falls back to contenteditable.
        Returns True if title was set successfully.
        """
        if not self.page:
            return False

        try:
            # Strategy 1: Look for Patreon-specific title input selectors
            title_selectors = [
                '[data-tag="post-title"]',
                '[data-tag="post-title-input"]',
                '[placeholder*="title" i]',
                '[placeholder*="Title" i]',
                '[aria-label*="title" i]',
                'div[contenteditable="true"][data-placeholder*="title" i]',
                'input[name="title"]',
            ]

            for selector in title_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem and await elem.is_visible():
                        logger.info(f"Found title element with selector: {selector}")
                        await elem.click(timeout=3000)
                        await asyncio.sleep(0.3)

                        # Try to clear and fill
                        await elem.evaluate('el => el.textContent = ""')
                        await asyncio.sleep(0.1)
                        await self.page.keyboard.type(title, delay=15)
                        logger.info(f"Title set via specific selector: {title[:40]}...")
                        return True
                except Exception as e:
                    logger.debug(f"Title selector {selector} failed: {e}")

            # Strategy 2: Get all contenteditable elements and find the title one
            all_editables = await self.page.query_selector_all('[contenteditable="true"]')
            editable_count = len(all_editables)
            logger.info(f"Found {editable_count} contenteditable elements on page")

            if editable_count == 0:
                logger.warning("No contenteditable elements found - trying position-based click")
                return await self._set_title_by_position(title)

            # Patreon's post editor: title is usually the first or has specific placeholder
            for i, elem in enumerate(all_editables):
                try:
                    is_visible = await elem.is_visible()
                    if not is_visible:
                        continue

                    # Check for title-like attributes
                    placeholder = await elem.get_attribute('data-placeholder') or ''
                    aria_label = await elem.get_attribute('aria-label') or ''
                    text_content = await elem.text_content() or ''

                    logger.info(f"Editable {i}: placeholder='{placeholder[:30]}', aria='{aria_label[:30]}', content='{text_content[:20]}'")

                    # If it looks like a title field (short placeholder or first visible element)
                    if 'title' in placeholder.lower() or 'title' in aria_label.lower() or i == 0:
                        logger.info(f"Selecting editable element {i} as title field")

                        # Click to focus
                        await elem.click(timeout=3000)
                        await asyncio.sleep(0.3)

                        # Clear content using JavaScript (most reliable)
                        await elem.evaluate('el => { el.textContent = ""; el.innerHTML = ""; }')
                        await asyncio.sleep(0.1)

                        # Also select all and delete as backup
                        await self.page.keyboard.press('Meta+a')  # Mac
                        await asyncio.sleep(0.05)
                        await self.page.keyboard.press('Control+a')  # Windows/Linux
                        await asyncio.sleep(0.05)
                        await self.page.keyboard.press('Backspace')
                        await asyncio.sleep(0.1)

                        # Type the title
                        await self.page.keyboard.type(title, delay=15)
                        logger.info(f"Title typed: {title[:40]}...")

                        # Verify it was set
                        new_content = await elem.text_content() or ''
                        if title[:20] in new_content:
                            logger.info("Title verified successfully")
                            return True
                        else:
                            logger.warning(f"Title may not have been set correctly. Content: {new_content[:40]}")

                        return True

                except Exception as e:
                    logger.debug(f"Editable element {i} failed: {e}")

            logger.warning("Could not set title via contenteditable - trying position-based")
            return await self._set_title_by_position(title)

        except Exception as e:
            logger.error(f"Fast title setting failed: {e}")
            # Try position-based as fallback
            return await self._set_title_by_position(title)

    async def _set_title_by_position(self, title: str) -> bool:
        """Set title by clicking at a position where the title usually is."""
        if not self.page:
            return False

        try:
            viewport = self.page.viewport_size
            if not viewport:
                logger.error("Could not get viewport size")
                return False

            # Click in upper-middle area where title input typically is
            click_x = viewport['width'] // 2
            click_y = 200  # Near top of content area

            logger.info(f"Position-based click at ({click_x}, {click_y})")
            await self.page.mouse.click(click_x, click_y)
            await asyncio.sleep(0.2)

            # Clear and type
            await self.page.keyboard.press('Control+a')
            await asyncio.sleep(0.1)
            await self.page.keyboard.type(title, delay=10)
            logger.info(f"Title typed via position-based click: {title[:40]}...")

            return True

        except Exception as e:
            logger.error(f"Position-based title input failed: {e}")
            return False

    async def publish_video(
        self,
        video_path: str,
        title: str,
        description: Optional[str] = None,
        tags: Optional[list] = None
    ) -> PatreonPublishResult:
        """
        Publish a video using the browser session with human-like behavior.

        Uses Playwright's Locator API for resilient element detection,
        natural mouse movements, variable delays, and realistic interaction
        patterns to avoid bot detection.
        """
        if not self.page:
            return PatreonPublishResult(
                success=False,
                error="Browser session not started. Call start_session() first."
            )

        await self._take_screenshot("publish_starting")

        try:
            logger.info(f"Publishing video: {title}")
            current_url = self.page.url

            # Human-like: Move mouse around initially (like looking at page)
            await self._random_mouse_movement()
            await human_medium_delay()

            # Navigate to create post page
            if "/posts/new" not in current_url:
                logger.info(f"Navigating to {self.POSTS_URL}")
                await self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")
                await human_long_delay()
                await self._random_mouse_movement()
                await self._take_screenshot("navigated_to_post_page")

                # Check for redirects to login
                if "/login" in self.page.url.lower():
                    await self._take_screenshot("redirected_to_login")
                    return PatreonPublishResult(
                        success=False,
                        error="Not logged in. Redirected to login page. Please log in via VNC first."
                    )

            # Human-like: Small scroll to see the page
            await human_short_delay()
            await self._human_scroll("down", random.randint(50, 150))
            await human_medium_delay()

            # Dismiss any modal dialogs
            await self._dismiss_modals()
            await human_short_delay()
            await self._take_screenshot("before_upload")

            # Step 1: Find and trigger NATIVE VIDEO upload (not attachment!)
            # Patreon has two upload types:
            # - Native Video: Processed via Mux, streams in browser
            # - Attachment: Downloaded file, doesn't stream
            # We MUST use native video for streaming playback
            file_input = None

            # Strategy 1: Look for Video post type selector first
            # Patreon's post editor has post type buttons (Text, Image, Video, Audio, etc.)
            video_type_selectors = [
                self.page.get_by_role("button", name="Video"),
                self.page.get_by_role("tab", name="Video"),
                self.page.get_by_text("Video", exact=True),
                self.page.locator('[data-tag="video-post-type"]'),
                self.page.locator('[data-tag="post-type-video"]'),
                self.page.locator('[aria-label="Video post"]'),
                self.page.locator('button:has-text("Video"):not(:has-text("attachment"))'),
            ]

            video_mode_activated = False
            for selector in video_type_selectors:
                try:
                    if await selector.count() > 0 and await selector.first.is_visible():
                        logger.info("Found Video post type button - clicking to enable video mode")
                        await human_short_delay()
                        await selector.first.click()
                        await human_medium_delay()
                        video_mode_activated = True
                        await self._take_screenshot("video_mode_activated")
                        break
                except Exception as e:
                    logger.debug(f"Video type selector failed: {e}")
                    continue

            # Strategy 2: Look for video-specific upload buttons (NOT attachments!)
            # Prioritize explicit video upload over generic media/attachment
            video_upload_buttons = [
                # Video-specific (highest priority)
                self.page.get_by_role("button", name="Upload video"),
                self.page.get_by_role("button", name="Add video"),
                self.page.get_by_text("Upload video", exact=False),
                self.page.get_by_text("Add video", exact=False),
                self.page.locator('[data-tag*="video-upload"]'),
                self.page.locator('[aria-label*="upload video" i]'),
                self.page.locator('[aria-label*="add video" i]'),
                # Generic media (lower priority, but NOT attachment)
                self.page.get_by_role("button", name="Add media"),
                self.page.get_by_role("button", name="Upload"),
                self.page.get_by_text("Add media", exact=False),
                self.page.locator('[data-tag*="upload"]'),
                self.page.locator('[data-tag*="media"]'),
                self.page.locator('[aria-label*="upload" i]'),
                self.page.locator('[aria-label*="media" i]'),
            ]
            # NOTE: "Add attachment" is explicitly NOT in this list - it creates downloadable files

            for btn in video_upload_buttons:
                try:
                    if await btn.count() > 0 and await btn.first.is_visible():
                        btn_text = await btn.first.text_content() or ""
                        # Skip if this is an attachment button
                        if "attachment" in btn_text.lower():
                            logger.debug(f"Skipping attachment button: {btn_text}")
                            continue
                        logger.info(f"Clicking video upload button: {btn_text.strip()}")
                        await human_short_delay()
                        await btn.first.click()
                        await human_medium_delay()

                        # Now look for file input that may have appeared
                        file_input = self.page.locator('input[type="file"][accept*="video"]').first
                        if await file_input.count() > 0:
                            logger.info("Found video-specific file input")
                            break
                        # Fallback to any file input
                        file_input = self.page.locator('input[type="file"]').first
                        if await file_input.count() > 0:
                            break
                except Exception as e:
                    logger.debug(f"Video upload button attempt failed: {e}")
                    continue

            # Strategy 3: Look for hidden file input directly (video-specific preferred)
            if not file_input or await file_input.count() == 0:
                # Try video-specific input first
                file_input = self.page.locator('input[type="file"][accept*="video"]').first
                if await file_input.count() == 0:
                    file_input = self.page.locator('input[type="file"]').first
                    if await file_input.count() == 0:
                        file_input = None

            # Strategy 4: Look for drag-drop zone and click it
            if not file_input or await file_input.count() == 0:
                dropzone_selectors = [
                    self.page.get_by_text("Drag and drop", exact=False),
                    self.page.get_by_text("drop files", exact=False),
                    self.page.get_by_text("drop video", exact=False),
                    self.page.locator('[class*="dropzone"]'),
                    self.page.locator('[class*="upload-area"]'),
                    self.page.locator('[class*="video-upload"]'),
                ]
                for dropzone in dropzone_selectors:
                    try:
                        if await dropzone.count() > 0 and await dropzone.first.is_visible():
                            await dropzone.first.click()
                            await human_short_delay()
                            file_input = self.page.locator('input[type="file"]').first
                            if await file_input.count() > 0:
                                break
                    except Exception:
                        continue

            # Final check for file input
            if not file_input or await file_input.count() == 0:
                await self._take_screenshot("no_upload_button")
                return PatreonPublishResult(
                    success=False,
                    error="Could not find video upload input. Check screenshot for current UI state."
                )

            # Step 2: Upload the video file
            logger.info(f"Uploading video: {video_path}")
            await file_input.set_input_files(video_path)
            await self._take_screenshot("video_uploading")

            # Human-like: Random mouse movement while waiting
            await human_long_delay()
            await self._random_mouse_movement()

            # Step 3: Wait for video upload AND processing to complete (up to 10 minutes)
            # Native Patreon Video goes through Mux processing which takes additional time
            logger.info("Waiting for video upload and processing to complete...")
            upload_complete = False
            processing_started = False

            for i in range(120):  # 120 iterations * 5 seconds = 10 minutes max
                # Check for processing indicators (video is uploading/processing via Mux)
                processing_indicators = [
                    self.page.get_by_text("Processing", exact=False),
                    self.page.get_by_text("Uploading", exact=False),
                    self.page.get_by_text("Encoding", exact=False),
                    self.page.locator('[class*="processing"]'),
                    self.page.locator('[class*="progress"]'),
                    self.page.locator('progress'),  # Progress bar element
                ]

                for indicator in processing_indicators:
                    try:
                        if await indicator.count() > 0 and await indicator.first.is_visible():
                            if not processing_started:
                                logger.info("Video processing started (Mux encoding)")
                                processing_started = True
                            break
                    except Exception:
                        continue

                # Check for various completion indicators
                completion_indicators = [
                    self.page.locator('video'),  # Video preview appeared
                    self.page.locator('video[src]'),  # Video with source loaded
                    self.page.locator('[data-tag*="complete"]'),
                    self.page.locator('[class*="upload-complete"]'),
                    self.page.locator('[class*="upload-success"]'),
                    self.page.locator('[class*="video-preview"]'),
                    self.page.locator('[class*="video-player"]'),
                    self.page.get_by_text("Upload complete", exact=False),
                    self.page.get_by_text("Processing complete", exact=False),
                    self.page.get_by_text("Ready to publish", exact=False),
                ]

                for indicator in completion_indicators:
                    try:
                        if await indicator.count() > 0 and await indicator.first.is_visible():
                            # Extra check: if it's a video element, make sure it has a source
                            tag_name = await indicator.first.evaluate("el => el.tagName")
                            if tag_name == "VIDEO":
                                has_src = await indicator.first.evaluate("el => el.src || el.querySelector('source')")
                                if has_src:
                                    logger.info("Video upload/processing complete - video preview loaded")
                                    upload_complete = True
                                    break
                            else:
                                logger.info("Video upload/processing complete - found completion indicator")
                                upload_complete = True
                                break
                    except Exception:
                        continue

                if upload_complete:
                    break

                # Check for upload errors
                error_indicators = [
                    self.page.get_by_text("Upload failed", exact=False),
                    self.page.get_by_text("Error uploading", exact=False),
                    self.page.get_by_text("Processing failed", exact=False),
                    self.page.get_by_text("Video too large", exact=False),
                    self.page.get_by_text("Unsupported format", exact=False),
                    self.page.locator('[class*="upload-error"]'),
                    self.page.locator('[class*="error"]'),
                    self.page.locator('[data-tag*="error"]'),
                ]

                for error_ind in error_indicators:
                    try:
                        if await error_ind.count() > 0 and await error_ind.first.is_visible():
                            error_text = await error_ind.first.text_content()
                            # Skip if it's just a minor error class on some element
                            if error_text and len(error_text.strip()) > 5:
                                await self._take_screenshot("upload_error")
                                return PatreonPublishResult(
                                    success=False,
                                    error=f"Upload/processing failed: {error_text}"
                                )
                    except Exception:
                        continue

                # Human-like: Occasional mouse movements while waiting
                if random.random() < 0.3:
                    await self._random_mouse_movement()

                # Take progress screenshot every 30 seconds
                if i % 6 == 0:
                    await self._take_screenshot(f"upload_progress_{i*5}s")

                await asyncio.sleep(random.uniform(4.5, 5.5))

            if not upload_complete:
                await self._take_screenshot("upload_timeout")
                if processing_started:
                    logger.warning("Video processing timeout - Mux encoding may still be in progress")
                else:
                    logger.warning("Upload timeout - continuing anyway (video might still process)")

            # Human-like pause after upload completes (like checking it worked)
            await human_long_delay()
            await self._random_mouse_movement()

            # Set title - Patreon uses contenteditable divs in a rich text editor
            logger.info(f"Setting title: {title[:50]}...")
            title_set = False

            # Human-like: Small scroll and mouse movement before title
            await self._human_scroll("up", random.randint(30, 80))
            await human_medium_delay()

            # Take screenshot to see what's on the page
            await self._take_screenshot("before_title_set")

            # Wrap title setting in a timeout to prevent hanging
            try:
                title_set = await asyncio.wait_for(
                    self._set_title_human(title),  # Use human-like title setting
                    timeout=30.0  # 30 second timeout (human typing is slower)
                )
            except asyncio.TimeoutError:
                logger.warning("Title setting timed out after 30 seconds")
                title_set = False

            if not title_set:
                logger.warning("Could not set title - continuing without title")
                await self._take_screenshot("title_not_set")

            # Human-like pause after typing (like reviewing what was typed)
            await human_medium_delay()
            await self._random_mouse_movement()
            await self._take_screenshot("after_title_set")

            # Set description if provided
            if description:
                await human_medium_delay()
                desc_selectors = [
                    '[data-tag="post-body"]',
                    'textarea[placeholder*="description"]',
                    'div[contenteditable="true"]:not([data-tag="post-title"])',
                ]

                for selector in desc_selectors:
                    desc_elem = await self.page.query_selector(selector)
                    if desc_elem:
                        await self._human_type_in_element(desc_elem, description)
                        break

            # Step 5: Human-like pause before publishing (like reviewing the post)
            await human_long_delay()
            await self._random_mouse_movement()
            await human_medium_delay()
            await self._take_screenshot("ready_to_publish")

            # Dismiss any modals that may have appeared after upload
            await self._dismiss_modals()
            await human_short_delay()

            # Step 6: Click publish button with human-like behavior
            # Use Playwright's Locator API for more resilient element finding
            publish_buttons = [
                # Text-based (most resilient to UI changes)
                self.page.get_by_role("button", name="Publish"),
                self.page.get_by_role("button", name="Publish now"),
                self.page.get_by_role("button", name="Post"),
                self.page.get_by_role("button", name="Post now"),
                self.page.get_by_text("Publish", exact=True),
                self.page.get_by_text("Post now", exact=True),
                # Attribute-based
                self.page.locator('[data-tag*="publish"]'),
                self.page.locator('[data-tag*="post-button"]'),
                self.page.locator('[data-testid*="publish"]'),
                self.page.locator('button[type="submit"]'),
            ]

            publish_clicked = False
            await self._take_screenshot("before_publish_click")

            for publish_btn in publish_buttons:
                try:
                    if await publish_btn.count() > 0:
                        btn = publish_btn.first
                        is_visible = await btn.is_visible()
                        is_enabled = await btn.is_enabled()

                        if is_visible and is_enabled:
                            btn_text = await btn.text_content() or "unknown"
                            logger.info(f"Found publish button: '{btn_text.strip()}' - clicking...")

                            # Human-like: Hover briefly before clicking
                            await btn.hover()
                            await human_short_delay()
                            await btn.click()

                            logger.info("Publish button clicked successfully")
                            publish_clicked = True
                            break
                except Exception as e:
                    logger.debug(f"Publish button attempt failed: {e}")
                    continue

            # Fallback: Try pressing Enter as last resort
            if not publish_clicked:
                logger.warning("No publish button found via selectors, trying Enter key...")
                await self._take_screenshot("publish_button_not_found")
                try:
                    # Some forms accept Enter to submit
                    await self.page.keyboard.press("Enter")
                    logger.info("Pressed Enter as fallback publish action")
                    publish_clicked = True
                except Exception as e:
                    logger.warning(f"Enter key fallback failed: {e}")

            # Step 7: Wait for publish to complete
            await human_long_delay()
            await self._random_mouse_movement()
            await human_medium_delay()

            # Check for success indicators
            success_indicators = [
                self.page.get_by_text("Published", exact=False),
                self.page.get_by_text("Post created", exact=False),
                self.page.get_by_text("Successfully posted", exact=False),
            ]

            publish_success = False
            for indicator in success_indicators:
                try:
                    if await indicator.count() > 0 and await indicator.first.is_visible():
                        publish_success = True
                        logger.info("Found publish success indicator")
                        break
                except Exception:
                    continue

            await self._take_screenshot("publish_complete")

            # Try to get the post URL
            post_url = self.page.url
            if "/posts/" in post_url:
                post_id = post_url.split("/posts/")[-1].split("?")[0]
            else:
                post_id = None
                # URL might change after publish
                await asyncio.sleep(2)
                post_url = self.page.url
                if "/posts/" in post_url:
                    post_id = post_url.split("/posts/")[-1].split("?")[0]
                else:
                    post_url = None

            self._publish_count += 1

            if post_id:
                logger.info(f"Video published successfully: {post_url}")
                return PatreonPublishResult(
                    success=True,
                    post_id=post_id,
                    post_url=post_url
                )
            elif publish_clicked:
                # Button was clicked but we couldn't verify - assume success
                logger.info("Publish button clicked, assuming success (could not verify URL)")
                return PatreonPublishResult(
                    success=True,
                    post_id=None,
                    post_url=None
                )
            else:
                return PatreonPublishResult(
                    success=False,
                    error="Could not find or click publish button. Check screenshots."
                )

        except Exception as e:
            logger.exception(f"Error publishing video: {e}")
            await self._take_screenshot("publish_error")
            return PatreonPublishResult(
                success=False,
                error=str(e)
            )

    async def publish_with_embed(
        self,
        embed_url: str,
        title: str,
        description: Optional[str] = None,
        tags: Optional[list] = None
    ) -> PatreonPublishResult:
        """
        Publish a post with an embedded video link.

        Useful when you would rather keep the video on your own media host than
        upload it to Patreon: the post body contains the link and Patreon
        renders a preview/embed for it where supported.

        Args:
            embed_url: Public URL of the video (e.g. the self-hosted stream URL)
            title: Post title
            description: Optional post description
            tags: Optional list of tags
        """
        if not self.page:
            return PatreonPublishResult(
                success=False,
                error="Browser session not started. Call start_session() first."
            )

        await self._take_screenshot("embed_publish_starting")

        try:
            logger.info(f"Publishing embed post: {title} -> {embed_url}")
            current_url = self.page.url

            # Human-like: Move mouse around initially
            await self._random_mouse_movement()
            await human_medium_delay()

            # Navigate to create post page
            if "/posts/new" not in current_url:
                logger.info(f"Navigating to {self.POSTS_URL}")
                await self.page.goto(self.POSTS_URL, wait_until="domcontentloaded")
                await human_long_delay()
                await self._random_mouse_movement()
                await self._take_screenshot("navigated_to_post_page")

                # Check for redirects to login
                if "/login" in self.page.url.lower():
                    await self._take_screenshot("redirected_to_login")
                    return PatreonPublishResult(
                        success=False,
                        error="Not logged in. Please log in via VNC first."
                    )

            # Human-like: Small scroll
            await human_short_delay()
            await self._human_scroll("down", random.randint(50, 150))
            await human_medium_delay()

            # Dismiss any modal dialogs
            await self._dismiss_modals()
            await human_short_delay()
            await self._take_screenshot("before_embed_input")

            # Step 1: Set title first
            logger.info(f"Setting title: {title[:50]}...")
            await self._set_title_human(title)
            await human_medium_delay()
            await self._take_screenshot("title_set")

            # Step 2: Find the post body/content area and paste the embed URL
            # Patreon renders a link preview/embed for URLs pasted into the body
            logger.info(f"Adding embed URL: {embed_url}")

            body_selectors = [
                # Patreon post body contenteditable
                '[data-tag="post-body"]',
                '[data-tag="post-content"]',
                'div[contenteditable="true"]:not([data-tag="post-title"])',
                '[placeholder*="Write" i]',
                '[placeholder*="body" i]',
                '[aria-label*="body" i]',
                '[aria-label*="content" i]',
            ]

            body_element = None
            for selector in body_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem and await elem.is_visible():
                        body_element = elem
                        logger.info(f"Found body element with selector: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"Body selector {selector} failed: {e}")

            # If no specific body found, try to find second contenteditable (first is title)
            if not body_element:
                all_editables = await self.page.query_selector_all('[contenteditable="true"]')
                if len(all_editables) > 1:
                    body_element = all_editables[1]
                    logger.info("Using second contenteditable as body")
                elif len(all_editables) == 1:
                    # Single editable - might need to click below title
                    body_element = all_editables[0]
                    logger.info("Using only contenteditable as body")

            if body_element:
                # Click to focus on body
                await self._human_click(body_element, hover_time=random.uniform(0.2, 0.5))
                await human_short_delay()

                # Build the post content with embed URL
                # Format: Description (if any) + blank line + URL
                post_content = ""
                if description:
                    post_content = f"{description}\n\n"
                post_content += embed_url

                # Type the content with human-like speed
                await human_type(self.page, post_content)
                logger.info("Embed URL typed into post body")

                # Wait for Patreon to process and embed the URL
                # Link previews usually render after a moment
                await human_long_delay()
                await self._take_screenshot("embed_url_pasted")

                # Check if embed preview appeared
                embed_indicators = [
                    self.page.locator('iframe'),
                    self.page.locator('[class*="embed"]'),
                    self.page.locator('[class*="preview"]'),
                    self.page.locator('[class*="card"]'),
                ]

                embed_detected = False
                for indicator in embed_indicators:
                    try:
                        if await indicator.count() > 0 and await indicator.first.is_visible():
                            logger.info("Embed preview detected!")
                            embed_detected = True
                            break
                    except Exception:
                        continue

                if embed_detected:
                    await self._take_screenshot("embed_preview_loaded")
                else:
                    logger.warning("Embed preview not detected - link will still work")

            else:
                logger.warning("Could not find body element - trying position-based input")
                # Click below title area and type
                viewport = self.page.viewport_size
                if viewport:
                    await self.page.mouse.click(viewport['width'] // 2, 400)
                    await human_short_delay()
                    post_content = f"{description}\n\n{embed_url}" if description else embed_url
                    await human_type(self.page, post_content)

            await human_medium_delay()
            await self._random_mouse_movement()
            await self._take_screenshot("ready_to_publish_embed")

            # Dismiss any modals
            await self._dismiss_modals()
            await human_short_delay()

            # Step 3: Click publish button (same logic as publish_video)
            publish_buttons = [
                self.page.get_by_role("button", name="Publish"),
                self.page.get_by_role("button", name="Publish now"),
                self.page.get_by_role("button", name="Post"),
                self.page.get_by_role("button", name="Post now"),
                self.page.get_by_text("Publish", exact=True),
                self.page.get_by_text("Post now", exact=True),
                self.page.locator('[data-tag*="publish"]'),
                self.page.locator('[data-tag*="post-button"]'),
                self.page.locator('[data-testid*="publish"]'),
                self.page.locator('button[type="submit"]'),
            ]

            publish_clicked = False
            await self._take_screenshot("before_publish_click")

            for publish_btn in publish_buttons:
                try:
                    if await publish_btn.count() > 0:
                        btn = publish_btn.first
                        is_visible = await btn.is_visible()
                        is_enabled = await btn.is_enabled()

                        if is_visible and is_enabled:
                            btn_text = await btn.text_content() or "unknown"
                            logger.info(f"Found publish button: '{btn_text.strip()}' - clicking...")

                            await btn.hover()
                            await human_short_delay()
                            await btn.click()

                            logger.info("Publish button clicked successfully")
                            publish_clicked = True
                            break
                except Exception as e:
                    logger.debug(f"Publish button attempt failed: {e}")
                    continue

            if not publish_clicked:
                logger.warning("No publish button found, trying Enter key...")
                await self._take_screenshot("publish_button_not_found")
                try:
                    await self.page.keyboard.press("Enter")
                    publish_clicked = True
                except Exception as e:
                    logger.warning(f"Enter key fallback failed: {e}")

            # Wait for publish to complete
            await human_long_delay()
            await self._random_mouse_movement()
            await human_medium_delay()

            # Check for success
            success_indicators = [
                self.page.get_by_text("Published", exact=False),
                self.page.get_by_text("Post created", exact=False),
                self.page.get_by_text("Successfully posted", exact=False),
            ]

            for indicator in success_indicators:
                try:
                    if await indicator.count() > 0 and await indicator.first.is_visible():
                        logger.info("Found publish success indicator")
                        break
                except Exception:
                    continue

            await self._take_screenshot("embed_publish_complete")

            # Get post URL
            post_url = self.page.url
            if "/posts/" in post_url:
                post_id = post_url.split("/posts/")[-1].split("?")[0]
            else:
                post_id = None
                await asyncio.sleep(2)
                post_url = self.page.url
                if "/posts/" in post_url:
                    post_id = post_url.split("/posts/")[-1].split("?")[0]
                else:
                    post_url = None

            self._publish_count += 1

            if post_id:
                logger.info(f"Embed post published successfully: {post_url}")
                return PatreonPublishResult(
                    success=True,
                    post_id=post_id,
                    post_url=post_url
                )
            elif publish_clicked:
                logger.info("Publish button clicked, assuming success")
                return PatreonPublishResult(
                    success=True,
                    post_id=None,
                    post_url=None
                )
            else:
                return PatreonPublishResult(
                    success=False,
                    error="Could not find or click publish button. Check screenshots."
                )

        except Exception as e:
            logger.exception(f"Error publishing embed post: {e}")
            await self._take_screenshot("embed_publish_error")
            return PatreonPublishResult(
                success=False,
                error=str(e)
            )


    # =========================================================================
    # Action Recording Methods
    # =========================================================================

    async def start_recording(self, name: str) -> Dict[str, Any]:
        """
        Start recording user actions in the browser session.

        Args:
            name: Name for this recording (used for saving)

        Returns:
            Dict with status and recording info
        """
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        if self._recording_active:
            return {
                "status": "already_recording",
                "recording_name": self._recording_name
            }

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()

            # Inject and start recording
            result = await recorder.start_recording(self.page)

            if result.get("status") == "started":
                self._recording_active = True
                self._recording_name = name
                await self._take_screenshot("recording_started")

                logger.info(f"Recording started: {name}")
                return {
                    "status": "started",
                    "recording_name": name,
                    "niche": self.niche,
                    "message": "Recording started. Perform actions in the VNC browser, then stop recording."
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return {"status": "error", "error": str(e)}

    async def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and save the captured actions.

        Returns:
            Dict with status and saved recording info
        """
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        if not self._recording_active:
            return {"status": "not_recording"}

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()

            # Stop recording and get actions
            result = await recorder.stop_recording(self.page)
            actions = result.get("actions", [])

            # Save the recording
            save_result = recorder.save_recording(
                name=self._recording_name,
                actions=actions,
                niche=self.niche or "unknown"
            )

            self._recording_active = False
            recording_name = self._recording_name
            self._recording_name = None

            await self._take_screenshot("recording_stopped")

            logger.info(f"Recording stopped and saved: {recording_name} ({len(actions)} actions)")

            return {
                "status": "saved",
                "recording_name": recording_name,
                "action_count": len(actions),
                "save_result": save_result
            }

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self._recording_active = False
            self._recording_name = None
            return {"status": "error", "error": str(e)}

    async def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status including action count."""
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        if not self._recording_active:
            return {
                "recording_active": False,
                "action_count": 0
            }

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()

            # Get current actions without clearing
            actions = await recorder.poll_actions(self.page, clear=False)

            return {
                "recording_active": True,
                "recording_name": self._recording_name,
                "action_count": len(actions),
                "niche": self.niche
            }

        except Exception as e:
            return {
                "recording_active": self._recording_active,
                "error": str(e)
            }

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all saved recordings."""
        from utils.action_recorder import get_action_recorder
        return get_action_recorder().list_recordings()

    def get_recording(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific recording by name."""
        from utils.action_recorder import get_action_recorder
        recorder = get_action_recorder()
        recording = recorder.load_recording(name)

        if recording:
            return {
                "name": recording.name,
                "niche": recording.niche,
                "recorded_at": recording.recorded_at,
                "version": recording.version,
                "action_count": len(recording.actions),
                "actions": recording.actions
            }
        return None

    def delete_recording(self, name: str) -> bool:
        """Delete a recording by name."""
        from utils.action_recorder import get_action_recorder
        return get_action_recorder().delete_recording(name)

    async def replay_recording(
        self,
        name: str,
        variables: Optional[Dict[str, str]] = None,
        speed_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Replay a saved recording in the current browser session.

        Args:
            name: Name of the recording to replay
            variables: Dict of placeholder replacements (e.g., {"{{title}}": "My Video"})
            speed_multiplier: Speed up (>1) or slow down (<1) replay

        Returns:
            Dict with replay results
        """
        if not self.page:
            return {"status": "error", "error": "Session not started"}

        if self._recording_active:
            return {"status": "error", "error": "Cannot replay while recording is active"}

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()

            # Load the recording
            recording = recorder.load_recording(name)
            if not recording:
                return {"status": "error", "error": f"Recording not found: {name}"}

            await self._take_screenshot("replay_starting")

            # Replay the recording
            result = await recorder.replay_recording(
                page=self.page,
                recording=recording,
                variables=variables,
                speed_multiplier=speed_multiplier
            )

            await self._take_screenshot("replay_complete")

            logger.info(f"Replay complete: {name} - {result.get('success_count', 0)} succeeded")
            return result

        except Exception as e:
            logger.error(f"Failed to replay recording: {e}")
            await self._take_screenshot("replay_error")
            return {"status": "error", "error": str(e)}


# Global session manager instance
patreon_session_manager = PatreonSessionManager()
