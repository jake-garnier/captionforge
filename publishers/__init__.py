"""
Publishers module for posting composed videos to platforms.

- MediaHost / SelfHostedMediaHost: Make a composed video publicly reachable (default: served by this API)
- RedditPoster: Post to Reddit profile and crosspost (PRAW-based, requires API key)
- RedditPosterPlaywright: Post to Reddit (browser-based, no API key needed)
- PatreonPublisher: Post videos to Patreon (browser-based, per-niche accounts)
- PostponePublisher: Schedule Reddit link posts via Postpone API
"""
from .media_host import MediaHost, SelfHostedMediaHost, HostedMedia, MediaHostError, get_media_host
from .reddit_poster import RedditPoster, PostResult, CrosspostResults
from .reddit_poster_playwright import RedditPosterPlaywright
from .patreon_publisher import PatreonPublisher, PatreonPublishResult
from .postpone_publisher import PostponePublisher, PostponeScheduleResult

__all__ = [
    "MediaHost",
    "SelfHostedMediaHost",
    "HostedMedia",
    "MediaHostError",
    "get_media_host",
    "RedditPoster",
    "RedditPosterPlaywright",
    "PostResult",
    "CrosspostResults",
    "PatreonPublisher",
    "PatreonPublishResult",
    "PostponePublisher",
    "PostponeScheduleResult",
]
