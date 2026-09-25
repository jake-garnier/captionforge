"""
Headless-browser scrape of a Reddit user's profile page.

Used only to fetch the real follower count, which the public JSON API
hides (returns 0 for every account). Everything else about a user lives
in /user/<u>/about.json — see RedditJsonScraper.get_user_about.

Pattern:
    with RedditProfileScraper() as s:
        followers = s.get_follower_count("motivation_clips")
        # returns an int or None

Single shared browser context across all 3 accounts in one dispatch
cycle is much cheaper than launching a fresh one per call.
"""
import logging
import re
from typing import Optional

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

# Works for accounts with a few hundred followers down to single digits
# (the regex handles both).
# Reddit renders the follower number client-side inside the React app;
# domcontentloaded + ~4s of wall time gives the count time to populate.
_FOLLOWER_REGEX = re.compile(r"([\d,]+)\s*[Ff]ollowers?")
_PROFILE_URL_TEMPLATE = "https://www.reddit.com/user/{}/"
_PAGE_LOAD_TIMEOUT_MS = 30_000
_POST_LOAD_DELAY_MS = 4_500


class RedditProfileScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()

    def _start(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.firefox.launch(
            headless=self.headless,
            args=["--no-sandbox"],
        )
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        logger.info("RedditProfileScraper browser started")

    def _stop(self):
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass

    def get_follower_count(self, username: str) -> Optional[int]:
        """Return follower count, or None on any failure.

        Defensive on purpose: this is decorative data for analytics and
        a hang here must not block the karma snapshot path.
        """
        if not self._context:
            logger.warning("RedditProfileScraper not started; returning None")
            return None
        page = None
        try:
            page = self._context.new_page()
            url = _PROFILE_URL_TEMPLATE.format(username)
            page.goto(url, timeout=_PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
            page.wait_for_timeout(_POST_LOAD_DELAY_MS)
            html = page.content()
            m = _FOLLOWER_REGEX.search(html)
            if not m:
                logger.warning(f"No follower regex match for {username}")
                return None
            return int(m.group(1).replace(",", ""))
        except Exception as e:
            logger.warning(f"get_follower_count({username}) failed: {e}")
            return None
        finally:
            if page:
                try:
                    page.close()
                except Exception:
                    pass
