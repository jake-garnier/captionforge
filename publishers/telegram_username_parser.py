"""Parse a Telegram @username out of a Patreon comment body.

Per Q2/Q3:
  - Match must be exact case (Telegram usernames are case-insensitive in practice
    but we record the user's exact casing).
  - Telegram username rules: 5–32 chars, [a-zA-Z0-9_], must start with a letter.
  - First valid candidate wins.
  - Anything else returns None — caller raises a `comment_parse_failed` alert.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Optional

# Telegram official rules: 5–32 chars, [A-Za-z0-9_], must start with a letter,
# cannot end with underscore (we leave that loose — Telegram itself rejects).
_USERNAME_RE = re.compile(r"@?([A-Za-z][A-Za-z0-9_]{4,31})")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _strip_html(body: str) -> str:
    """Patreon comment bodies are HTML — pull plain text out."""
    return unescape(_TAG_STRIP_RE.sub(" ", body or ""))


def parse_telegram_username(body: str) -> Optional[str]:
    """Return the first valid @username, or None.

    Examples:
      "@alice123"               -> "alice123"
      "alice123"                -> "alice123"
      "Hi I am @bob_99 thanks"  -> "bob_99"
      "1abc"                    -> None  (doesn't start with letter)
      "@ab"                     -> None  (too short)
      "wow"                     -> None  (too short)
    """
    if not body:
        return None
    text = _strip_html(body)
    m = _USERNAME_RE.search(text)
    if not m:
        return None
    return m.group(1)
