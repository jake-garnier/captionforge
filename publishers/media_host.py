"""
Media host abstraction for publishing composed videos.

Reddit link posts and Postpone schedules need a public URL for each composed
video. The default implementation (``SelfHostedMediaHost``) simply points at
this API's own streaming endpoint, exposed publicly via ``MEDIA_BASE_URL``
(e.g. a Cloudflare tunnel). Swap in a third-party host by subclassing
``MediaHost`` and registering it in ``get_media_host()``.

Publishing flow:
    Composed video -> media host (self-hosted URL) -> Reddit profile post
    -> crosspost to subreddits
"""
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Route in api/video_composition.py that serves a composed video file.
COMPOSED_VIDEO_STREAM_PATH = "/composition/videos/{composed_video_id}/stream"


class MediaHostError(Exception):
    """Raised when a media host cannot publish a video."""


@dataclass
class HostedMedia:
    """Result of publishing a video to a media host."""
    url: str
    media_id: str


class MediaHost(ABC):
    """Pluggable interface for making a composed video publicly reachable."""

    #: Short identifier stored alongside jobs for diagnostics.
    name: str = "abstract"

    @abstractmethod
    def publish(
        self,
        video_path: str,
        title: str,
        composed_video_id: Optional[int] = None,
    ) -> HostedMedia:
        """
        Make ``video_path`` publicly reachable and return its URL + host-side id.

        Args:
            video_path: Local path of the composed mp4.
            title: Human-readable title (hosts that support metadata use it).
            composed_video_id: ID of the ``composed_videos`` row; the
                self-hosted implementation derives the URL from it.

        Raises:
            MediaHostError: the host could not publish the file.
        """

    def health_check(self) -> bool:
        """Return True when the host is configured well enough to publish."""
        return True


class SelfHostedMediaHost(MediaHost):
    """
    Serve composed videos straight from this API.

    Builds ``{MEDIA_BASE_URL}/composition/videos/{id}/stream``. Nothing is
    uploaded anywhere; the file just has to exist on disk and the API has to
    be reachable at ``MEDIA_BASE_URL``.
    """

    name = "self"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.MEDIA_BASE_URL or "").rstrip("/")

    def health_check(self) -> bool:
        return bool(self.base_url)

    def build_url(self, composed_video_id: int) -> str:
        if not self.base_url:
            raise MediaHostError(
                "MEDIA_BASE_URL is not configured — set it to the public URL of this API"
            )
        return self.base_url + COMPOSED_VIDEO_STREAM_PATH.format(
            composed_video_id=composed_video_id
        )

    def publish(
        self,
        video_path: str,
        title: str,
        composed_video_id: Optional[int] = None,
    ) -> HostedMedia:
        if composed_video_id is None:
            raise MediaHostError("SelfHostedMediaHost requires composed_video_id")
        if not video_path or not os.path.exists(video_path):
            raise MediaHostError(f"Video file not found: {video_path}")

        url = self.build_url(composed_video_id)
        logger.info(f"Self-hosting composed video {composed_video_id} at {url}")
        return HostedMedia(url=url, media_id=str(composed_video_id))


_MEDIA_HOSTS = {
    "self": SelfHostedMediaHost,
}


def get_media_host() -> MediaHost:
    """
    Factory: instantiate the media host selected by ``settings.MEDIA_HOST_TYPE``.

    Defaults to ``"self"``. Register additional implementations in
    ``_MEDIA_HOSTS`` to support third-party hosts.
    """
    host_type = (settings.MEDIA_HOST_TYPE or "self").lower()
    host_cls = _MEDIA_HOSTS.get(host_type)
    if host_cls is None:
        raise MediaHostError(
            f"Unknown MEDIA_HOST_TYPE '{host_type}'. Available: {sorted(_MEDIA_HOSTS)}"
        )
    return host_cls()
