"""
Video downloader using yt-dlp
Downloads videos from Reddit and other supported platforms
"""
import yt_dlp
import os
import hashlib
from typing import Dict, Optional
from config.settings import settings
from scrapers.exceptions import PermanentDownloadError, TemporaryDownloadError
import logging

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Video downloader with yt-dlp"""

    def __init__(self, output_path: Optional[str] = None, use_proxy: bool = True):
        """
        Initialize video downloader

        Args:
            output_path: Directory to save videos (default from settings)
            use_proxy: Whether to use proxy if configured in settings
        """
        self.output_path = output_path or settings.video_storage_path
        os.makedirs(self.output_path, exist_ok=True)

        # Configure proxy
        self.proxy_url = None
        if use_proxy and settings.proxy_enabled and settings.proxy_url:
            self.proxy_url = settings.proxy_url
            proxy_host = settings.proxy_url.split('@')[-1] if '@' in settings.proxy_url else settings.proxy_url
            logger.info(f"Video downloader initialized with proxy: {proxy_host}")
        else:
            logger.info(f"Video downloader initialized (output: {self.output_path})")

    def download(self, video_url: str, post_id: str) -> Dict:
        """
        Download video from URL

        Args:
            video_url: URL of video to download
            post_id: Reddit post ID for naming

        Returns:
            Dictionary with download metadata

        Raises:
            PermanentDownloadError: For errors that should not be retried (404, 410, dead domains)
            TemporaryDownloadError: For errors that may succeed on retry (timeouts, rate limits)
        """
        # Check for dead domains upfront
        if 'gfycat.com' in video_url.lower():
            raise PermanentDownloadError(
                "Gfycat service shut down in September 2023 - skipping permanently"
            )

        output_template = os.path.join(self.output_path, f"{post_id}.%(ext)s")

        ydl_opts = {
            # Format: prefer quality up to 1080p, but be flexible for Reddit videos
            # Reddit videos often need bestvideo+bestaudio or just best
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,

            # Quality/size limits
            'max_filesize': settings.download_max_file_size_mb * 1024 * 1024,

            # Post-processing - merge video/audio and convert to MP4
            'postprocessors': [
                {
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                },
            ],
            'merge_output_format': 'mp4',

            # Rate limiting
            'ratelimit': settings.download_rate_limit_mbps * 1024 * 1024,

            # Proxy configuration (supports http, https, socks5)
            'proxy': self.proxy_url,

            # Error handling
            'ignoreerrors': False,
            'no_abort_on_error': False,

            # Metadata
            'writethumbnail': False,
            'writeinfojson': False,
        }

        try:
            logger.info(f"Downloading video: {video_url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(video_url, download=False)

                # Check duration limit
                duration = info.get('duration', 0)
                if duration > settings.download_max_duration_seconds:
                    raise ValueError(
                        f"Video too long: {duration}s "
                        f"(max: {settings.download_max_duration_seconds}s)"
                    )

                # Download video
                info = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info)

                # If post-processor changed extension, update file_path
                if not os.path.exists(file_path):
                    # Try with .mp4 extension
                    file_path = os.path.splitext(file_path)[0] + '.mp4'

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Downloaded file not found: {file_path}")

                # Calculate file hash
                file_hash = self._calculate_file_hash(file_path)

                # Get file size
                file_size = os.path.getsize(file_path)

                result = {
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'title': info.get('title'),
                    'duration': duration,
                    'resolution': f"{info.get('width')}x{info.get('height')}" if info.get('width') else None,
                    'file_size': file_size,
                    'format': 'mp4',
                    'thumbnail': info.get('thumbnail'),
                    'uploader': info.get('uploader')
                }

                logger.info(
                    f"Download complete: {post_id} "
                    f"({file_size / 1024 / 1024:.2f} MB, {duration}s)"
                )

                return result

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            logger.error(f"Download failed for {video_url}: {e}")

            # Detect permanent errors that should not be retried
            permanent_errors = [
                'HTTP Error 404',  # Content not found
                'HTTP Error 410',  # Content gone
                'No address associated with hostname',  # Dead domain
            ]

            if any(err in error_msg for err in permanent_errors):
                raise PermanentDownloadError(f"Permanent download failure: {error_msg}") from e

            # All other yt-dlp errors are potentially retryable
            raise TemporaryDownloadError(f"Temporary download failure: {error_msg}") from e

        except ValueError as e:
            # Video too long/large - permanent error
            logger.error(f"Video validation failed for {video_url}: {e}")
            raise PermanentDownloadError(str(e)) from e

        except Exception as e:
            # Unknown errors - treat as temporary
            logger.error(f"Unexpected error downloading {video_url}: {e}")
            raise TemporaryDownloadError(f"Unexpected error: {e}") from e

    def get_info(self, video_url: str) -> Optional[Dict]:
        """
        Get video info without downloading

        Args:
            video_url: URL of video

        Returns:
            Dictionary with video metadata or None if failed
        """
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'proxy': self.proxy_url,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'width': info.get('width'),
                    'height': info.get('height'),
                    'filesize': info.get('filesize'),
                    'thumbnail': info.get('thumbnail'),
                    'formats': [
                        {
                            'format_id': f.get('format_id'),
                            'ext': f.get('ext'),
                            'resolution': f"{f.get('width')}x{f.get('height')}" if f.get('width') else 'audio',
                            'filesize': f.get('filesize')
                        }
                        for f in info.get('formats', [])
                    ]
                }

        except Exception as e:
            logger.error(f"Failed to get info for {video_url}: {e}")
            return None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in 64kb chunks
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def delete_video(self, file_path: str) -> bool:
        """Delete video file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted video: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
            return False
