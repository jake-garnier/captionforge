"""
Application configuration using pydantic-settings
Loads from environment variables and .env file
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "Captions Service"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Playwright Configuration (no API needed!)
    playwright_headless: bool = True
    playwright_timeout: int = 30000  # 30 seconds

    # Celery & Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # PostgreSQL
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "captions"
    postgres_user: str = "captionsuser"
    postgres_password: str

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Storage
    video_storage_path: str = "/data/videos"
    video_cache_max_age_days: int = 30

    # Scraping Configuration
    scraping_enabled: bool = True
    scraping_interval_hours: int = 8  # Increased to 8 hours (was 6) for rate limiting
    subreddit_targets: list[str] = ["GetMotivated"]  # Default target
    max_posts_per_scrape: int = 50  # Reduced to 50 (was 100) to avoid IP blocks
    download_max_file_size_mb: int = 100
    download_max_duration_seconds: int = 600  # 10 minutes
    max_pages_per_scrape: int = 5  # Maximum pages per scrape (anti-rate-limit)

    # Rate Limiting (for Playwright scraping)
    scrape_delay_min_seconds: float = 3.0  # Minimum delay between actions
    scrape_delay_max_seconds: float = 6.0  # Maximum delay between actions
    scrape_page_delay_min: float = 5.0  # Min delay between pages
    scrape_page_delay_max: float = 10.0  # Max delay between pages
    download_rate_limit_mbps: int = 5  # MB/s for video downloads

    # Proxy Configuration (e.g., proxy-seller.com)
    proxy_enabled: bool = False
    proxy_url: Optional[str] = None  # Format: http://user:pass@host:port or socks5://host:port
    proxy_type: str = "http"  # http, https, socks5

    # Flower (Celery monitoring)
    flower_port: int = 5555
    flower_basic_auth: Optional[str] = None  # Format: "username:password"

    # Telegram Scraping (Telethon - user account for private channels)
    # Get credentials from https://my.telegram.org
    TELEGRAM_API_ID: Optional[int] = None
    TELEGRAM_API_HASH: Optional[str] = None
    TELEGRAM_SESSION_PATH: str = "/data/telegram_session"  # Session file storage

    # Vast.ai Configuration (for cloud training)
    # Get API key from https://vast.ai/console/account/
    VASTAI_API_KEY: Optional[str] = None
    VASTAI_MAX_PRICE_PER_HOUR: float = 0.50  # Maximum $/hr for GPU rental
    VASTAI_MAX_TRAINING_HOURS: int = 12  # Maximum training time before timeout (increased for large models)
    VASTAI_PREFERRED_GPU: str = "RTX_4090"  # Preferred GPU model
    VASTAI_MIN_DISK_GB: int = 100  # Minimum disk space (need ~80GB for Mistral-Small-24B training)

    # Training Provider: "local" for local GPU training, "vastai" for cloud training
    # Use "vastai" for Mistral-Small-24B (requires RTX 4090 with 24GB VRAM)
    # Use "local" for smaller models that fit on local GPUs
    TRAINING_PROVIDER: str = "vastai"

    # Postpone API Configuration (Scheduled Reddit Posting)
    # Sign up at https://postpone.app (paid plan required)
    # Get API key from Settings > Integrations > Postpone API
    POSTPONE_API_KEY: Optional[str] = None

    # Media host (see publishers/media_host.py).
    # MEDIA_BASE_URL is the public base URL of this API: composed videos are served
    # from here so Reddit/Postpone link posts can point at them.
    # e.g. "https://captions.example.com" (Cloudflare tunnel) or "http://<vm-ip>:8000"
    MEDIA_BASE_URL: Optional[str] = None
    # Which MediaHost implementation to use: "self" (default) serves videos from this API.
    MEDIA_HOST_TYPE: str = "self"

    # Host-machine access from inside containers (llama-server control, GPU host SSH).
    LLM_HOST_IP: str = "host.docker.internal"
    LLM_HOST_SSH_USER: str = "ubuntu"
    LLM_HOST_SSH_PASSWORD: Optional[str] = None   # prefer LLM_HOST_SSH_KEY_PATH
    LLM_HOST_SSH_KEY_PATH: Optional[str] = None

    # Public noVNC URL shown in the UI for interactive browser logins
    NOVNC_PUBLIC_URL: str = "http://localhost:6080/vnc.html?autoconnect=true"

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def VIDEO_STORAGE_PATH(self) -> str:
        """Alias for video storage path"""
        return self.video_storage_path


# Create global settings instance
settings = Settings()
