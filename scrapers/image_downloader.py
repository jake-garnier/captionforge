"""
Image downloader for static images from Reddit and Imgur
Downloads images for OCR caption extraction
"""
import requests
import os
import hashlib
from typing import Dict, Optional, List
from config.settings import settings
from scrapers.exceptions import PermanentDownloadError, TemporaryDownloadError
import logging

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Image downloader for static images"""

    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize image downloader

        Args:
            output_path: Directory to save images (default from settings)
        """
        self.output_path = output_path or settings.video_storage_path
        os.makedirs(self.output_path, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        logger.info(f"Image downloader initialized (output: {self.output_path})")

    def download(self, image_url: str, post_id: str) -> Dict:
        """
        Download image from URL

        Args:
            image_url: URL of image to download
            post_id: Reddit post ID for naming

        Returns:
            Dictionary with download metadata

        Raises:
            PermanentDownloadError: For errors that should not be retried (404, etc.)
            TemporaryDownloadError: For errors that may succeed on retry
        """
        try:
            logger.info(f"Downloading image: {image_url}")

            response = self.session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            # Determine file extension from content-type or URL
            content_type = response.headers.get('content-type', '')
            ext = self._get_extension(image_url, content_type)

            file_path = os.path.join(self.output_path, f"{post_id}{ext}")

            # Download to file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)

            # Get file size
            file_size = os.path.getsize(file_path)

            # Get image dimensions (optional, requires PIL)
            resolution = self._get_image_resolution(file_path)

            result = {
                'file_path': file_path,
                'file_hash': file_hash,
                'title': None,
                'duration': 0,  # Images have no duration
                'resolution': resolution,
                'file_size': file_size,
                'format': ext.lstrip('.'),
                'thumbnail': None,
                'uploader': None
            }

            logger.info(
                f"Image download complete: {post_id} "
                f"({file_size / 1024:.1f} KB, {resolution or 'unknown'})"
            )

            return result

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            logger.error(f"HTTP error downloading image {image_url}: {e}")

            if status_code in [404, 410, 403]:
                raise PermanentDownloadError(f"HTTP {status_code}: {e}") from e
            raise TemporaryDownloadError(f"HTTP error: {e}") from e

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout downloading image {image_url}: {e}")
            raise TemporaryDownloadError(f"Timeout: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error downloading image {image_url}: {e}")
            raise TemporaryDownloadError(f"Request error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error downloading image {image_url}: {e}")
            raise TemporaryDownloadError(f"Unexpected error: {e}") from e

    def download_gallery(self, gallery_urls: List[str], post_id: str) -> Dict:
        """
        Download all images from a gallery

        Args:
            gallery_urls: List of image URLs in the gallery
            post_id: Reddit post ID for naming

        Returns:
            Dictionary with download metadata for primary image and list of all paths
        """
        downloaded_paths = []
        primary_result = None

        for idx, url in enumerate(gallery_urls):
            try:
                # Create unique ID for each gallery image
                item_id = f"{post_id}_gallery_{idx}"

                result = self.download(url, item_id)
                downloaded_paths.append(result['file_path'])

                # Use first image as primary
                if primary_result is None:
                    primary_result = result

            except Exception as e:
                logger.warning(f"Failed to download gallery image {idx}: {e}")
                continue

        if not primary_result:
            raise PermanentDownloadError("Failed to download any gallery images")

        # Add gallery info to result
        primary_result['gallery_paths'] = downloaded_paths
        primary_result['gallery_count'] = len(downloaded_paths)

        logger.info(f"Gallery download complete: {post_id} ({len(downloaded_paths)} images)")

        return primary_result

    def _get_extension(self, url: str, content_type: str) -> str:
        """Get file extension from URL or content-type"""
        # Try to get from URL first
        url_lower = url.lower()
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            if ext in url_lower:
                return ext if ext != '.jpeg' else '.jpg'

        # Fall back to content-type
        type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp',
            'image/gif': '.gif',
        }
        return type_map.get(content_type.split(';')[0], '.jpg')

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_image_resolution(self, file_path: str) -> Optional[str]:
        """Get image resolution (requires PIL)"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return f"{img.width}x{img.height}"
        except ImportError:
            logger.debug("PIL not available for resolution detection")
            return None
        except Exception as e:
            logger.debug(f"Could not get image resolution: {e}")
            return None

    def close(self):
        """Cleanup - close session"""
        if self.session:
            self.session.close()
