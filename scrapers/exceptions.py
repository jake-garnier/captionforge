"""
Custom exceptions for scraping operations
"""


class DownloadError(Exception):
    """Base class for download errors"""
    pass


class PermanentDownloadError(DownloadError):
    """
    Download error that should NOT be retried

    Examples:
    - HTTP 404 (content deleted/not found)
    - HTTP 410 (content permanently gone)
    - Dead domain (Gfycat shut down)
    - Content too large/long
    """
    pass


class TemporaryDownloadError(DownloadError):
    """
    Download error that MAY succeed if retried

    Examples:
    - Network timeouts
    - Rate limiting (HTTP 429)
    - Server errors (HTTP 5xx)
    - DNS resolution failures (except known dead domains)
    """
    pass
