"""
Add Reddit as a source for background videos.

Changes:
- Add source_type column to background_videos (default 'reddit') for installs
  created before the column existed
- Add reddit_post_id, reddit_subreddit, reddit_score columns
- Create reddit_background_subreddits table (per-niche background subreddit config)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from database.db import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Add Reddit background video support."""
    with engine.connect() as conn:
        # Check if migration already ran
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'background_videos' AND column_name = 'source_type'
        """))
        if result.fetchone():
            logger.info("source_type column already exists, skipping migration")
            return

        # 1. Add source_type column (default 'reddit' for existing rows)
        logger.info("Adding source_type column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN source_type VARCHAR(20) DEFAULT 'reddit' NOT NULL
        """))

        # 2. Add reddit-specific columns
        logger.info("Adding reddit_post_id column...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN reddit_post_id VARCHAR(50) UNIQUE
        """))

        logger.info("Adding reddit_subreddit column...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN reddit_subreddit VARCHAR(100)
        """))

        logger.info("Adding reddit_score column...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN reddit_score INTEGER
        """))

        # 3. Add indexes
        logger.info("Creating indexes...")
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bg_source_type ON background_videos(source_type)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bg_reddit_post_id ON background_videos(reddit_post_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bg_reddit_subreddit ON background_videos(reddit_subreddit)
        """))

        # 4. Create reddit_background_subreddits table
        logger.info("Creating reddit_background_subreddits table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reddit_background_subreddits (
                id SERIAL PRIMARY KEY,
                subreddit VARCHAR(100) UNIQUE NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                min_score INTEGER DEFAULT 100,
                min_duration INTEGER DEFAULT 10,
                max_duration INTEGER DEFAULT 60,
                batch_size INTEGER DEFAULT 10,
                scrape_stage VARCHAR(20) DEFAULT 'top_all',
                last_pagination_url TEXT,
                posts_scraped INTEGER DEFAULT 0,
                videos_downloaded INTEGER DEFAULT 0,
                videos_failed INTEGER DEFAULT 0,
                last_scrape_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reddit_bg_subreddit ON reddit_background_subreddits(subreddit)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reddit_bg_stage ON reddit_background_subreddits(scrape_stage)
        """))

        conn.commit()
        logger.info("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
