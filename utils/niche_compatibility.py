"""
Niche compatibility checking for background video selection.

Thin wrappers over config/niche_rules.py so call sites don't need to know
about the two rule layers:

- `check_compatibility(searched_tag, niche)` — the coarse subreddit gate.
  Kept with its original signature; every legacy call site uses it.
- `check_tag_compatibility(ml_tags, niche)` — required/forbidden scene-tag
  rules on the VLM tags.
- `tag_preference_score(ml_tags, niche)` — soft ranking score from the
  niche's preferred clauses.
- `is_background_compatible(searched_tag, ml_tags, niche)` — both gates.
"""
from typing import Any, Dict, Optional

from config.niche_rules import (
    check_tag_rules,
    is_subreddit_compatible,
    score_tag_preferences,
)


def check_compatibility(searched_tag: Optional[str], niche: str) -> bool:
    """
    Check if a background video's source subreddit is compatible with a niche.

    Args:
        searched_tag: The subreddit the background video was scraped from
        niche: The niche to check against

    Returns:
        True if compatible
    """
    if not niche:
        return True
    return is_subreddit_compatible(searched_tag or "", niche)


def check_tag_compatibility(ml_tags: Optional[Dict[str, Any]], niche: str) -> bool:
    """
    Check a background video's VLM scene tags against a niche's
    required/forbidden rules. Untagged videos pass (nothing to judge).
    """
    if not niche:
        return True
    passes, _ = check_tag_rules(ml_tags, niche)
    return passes


def tag_preference_score(ml_tags: Optional[Dict[str, Any]], niche: str) -> int:
    """Soft ranking score: number of the niche's preferred clauses this BG satisfies."""
    if not niche:
        return 0
    return score_tag_preferences(ml_tags, niche)


def is_background_compatible(
    searched_tag: Optional[str],
    ml_tags: Optional[Dict[str, Any]],
    niche: str,
) -> bool:
    """Both gates: source subreddit allow-list AND scene-tag rules."""
    return check_compatibility(searched_tag, niche) and check_tag_compatibility(ml_tags, niche)
