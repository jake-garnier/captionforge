"""Scrape comments on a single Patreon post via Playwright.

Patreon's v2 API does not expose post comments. The logged-in web app loads
them via XHR against the internal `/api/comments` (or `/api/posts/{id}` with
`include=comments,commenter`) JSON:API endpoint. We attach a response listener
to capture every JSON:API payload that contains comment resources, then walk
the `links.next` cursor.

Reads cookies persisted by the existing PatreonSession at:
  /data/patreon_cookies/{niche}_cookies.json

Failure surfaces:
  - PatreonScrapeError on any unrecoverable issue (no cookies, login expired,
    DataDome challenge). The caller is expected to convert this into a
    sync_alerts row and abort the cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from playwright.async_api import (
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logger = logging.getLogger(__name__)

COOKIE_PATHS = [
    "/data/patreon_cookies/{niche}_cookies.json",
    "patreon_cookies/{niche}_cookies.json",
]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
POST_URL_TEMPLATE = "https://www.patreon.com/posts/{post_id}"


class PatreonScrapeError(Exception):
    pass


@dataclass
class ScrapedComment:
    comment_id: str
    patreon_user_id: str
    body: str
    last_modified: Optional[datetime]


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _load_cookies_for(niche: str) -> list[dict]:
    for tmpl in COOKIE_PATHS:
        path = tmpl.format(niche=niche)
        if os.path.exists(path):
            try:
                cookies = json.loads(open(path).read())
                logger.info("Loaded %d Patreon cookies from %s", len(cookies), path)
                return cookies
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
    raise PatreonScrapeError(
        f"No Patreon cookies found for niche={niche}. "
        f"Run the existing /patreon/session flow to log in first."
    )


def _decode_comments(payload: dict) -> list[ScrapedComment]:
    """Pull comment resources out of a JSON:API response.

    The response may contain comments either in `data` (if the request was
    /api/comments) or in `included` (if it was /api/posts/{id}?include=comments).
    """
    comments: list[ScrapedComment] = []

    def consume(resource: dict):
        if (resource.get("type") or "").lower() != "comment":
            return
        cid = resource.get("id")
        attrs = resource.get("attributes") or {}
        rels = resource.get("relationships") or {}
        commenter = (rels.get("commenter") or {}).get("data") or {}
        user_id = commenter.get("id")
        body = attrs.get("body") or ""
        if not cid or not user_id:
            return
        comments.append(ScrapedComment(
            comment_id=str(cid),
            patreon_user_id=str(user_id),
            body=body,
            last_modified=_parse_iso(attrs.get("last_modified") or attrs.get("created")),
        ))

    data = payload.get("data")
    if isinstance(data, list):
        for r in data:
            consume(r)
    elif isinstance(data, dict):
        consume(data)

    for r in payload.get("included") or []:
        consume(r)

    return comments


async def _scrape_async(post_id: str, niche: str, *, timeout_seconds: int = 90) -> list[ScrapedComment]:
    cookies = _load_cookies_for(niche)

    captured: list[ScrapedComment] = []
    seen_comment_ids: set[str] = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        try:
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1366, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
            )
            await context.add_cookies(cookies)
            page = await context.new_page()
            page.set_default_timeout(timeout_seconds * 1000)

            async def on_response(resp: Response):
                url = resp.url
                if "/api/" not in url:
                    return
                # Heuristic: any internal JSON:API call may include comment resources.
                # Filter loosely to keep parsing cheap.
                if not any(token in url for token in ("comment", "comments", "/posts/")):
                    return
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    return
                try:
                    body = await resp.json()
                except Exception:
                    return
                for c in _decode_comments(body):
                    if c.comment_id in seen_comment_ids:
                        continue
                    seen_comment_ids.add(c.comment_id)
                    captured.append(c)

            page.on("response", on_response)

            await page.goto(POST_URL_TEMPLATE.format(post_id=post_id), wait_until="domcontentloaded")

            # Detect login expiry / DataDome.
            url_lower = page.url.lower()
            if "/login" in url_lower or "/signup" in url_lower:
                raise PatreonScrapeError(
                    f"Patreon redirected to {page.url} — session likely expired. "
                    f"Refresh cookies via /patreon/session for niche={niche}."
                )
            title = (await page.title()).lower()
            if any(s in title for s in ("just a moment", "access denied", "attention required")):
                raise PatreonScrapeError(
                    f"DataDome / Cloudflare challenge on post {post_id} (title={title!r})"
                )

            # Scroll to trigger lazy-loaded comment XHRs. Patreon paginates by
            # appending; load until we stop seeing new ones for two consecutive ticks.
            stable_ticks = 0
            for _ in range(40):  # hard cap ≈ 40 * 1.5s = 60s
                before = len(captured)
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                except Exception:
                    pass
                # Try clicking any "Load more comments" button if present.
                try:
                    btn = await page.query_selector('button:has-text("Load more")')
                    if btn:
                        await btn.click()
                except Exception:
                    pass
                await asyncio.sleep(1.5)
                if len(captured) == before:
                    stable_ticks += 1
                    if stable_ticks >= 2:
                        break
                else:
                    stable_ticks = 0

            return captured
        finally:
            try:
                await browser.close()
            except Exception:
                pass


def scrape_post_comments(post_id: str, niche: str, *, timeout_seconds: int = 90) -> list[ScrapedComment]:
    """Synchronous wrapper for use from Celery tasks."""
    if not post_id:
        raise PatreonScrapeError("scrape_post_comments called with empty post_id")
    try:
        return asyncio.run(_scrape_async(post_id, niche, timeout_seconds=timeout_seconds))
    except PatreonScrapeError:
        raise
    except PlaywrightTimeoutError as e:
        raise PatreonScrapeError(f"Playwright timeout scraping post {post_id}: {e}") from e
    except Exception as e:
        raise PatreonScrapeError(f"Unexpected error scraping post {post_id}: {e}") from e
