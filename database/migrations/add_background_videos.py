"""
Migration: Add background_videos table for background video storage

Background videos are the visual layer composed under generated captions.
They are scraped from the per-niche background subreddits configured in
`reddit_background_subreddits` (see add_reddit_background_videos.py) and then
run through the watermark filter and VLM tagging.

Run with: docker-compose exec api python database/migrations/add_background_videos.py
"""
import sys
sys.path.insert(0, '/app')

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Create background_videos table."""

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS background_videos (
                id SERIAL PRIMARY KEY,
                -- video source (reddit, or a future custom importer)
                source_type VARCHAR(20) DEFAULT 'reddit' NOT NULL,
                reddit_post_id VARCHAR(50) UNIQUE,
                reddit_subreddit VARCHAR(100),
                reddit_score INTEGER,
                source_url TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                thumbnail_path TEXT,
                file_hash VARCHAR(64) UNIQUE,

                -- Video metadata
                duration_seconds INTEGER,
                width INTEGER,
                height INTEGER,
                file_size_bytes BIGINT,

                -- Source engagement metrics (Reddit score is stored as views)
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,

                -- Content metadata
                tags JSONB,
                searched_tag VARCHAR(100),
                username VARCHAR(100),
                created_at_source TIMESTAMP WITH TIME ZONE,

                -- Processing status
                download_status VARCHAR(20) DEFAULT 'pending',
                error_message TEXT,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))

        # Indexes
        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_bg_source_type ON background_videos(source_type)",
            "CREATE INDEX IF NOT EXISTS idx_bg_reddit_post_id ON background_videos(reddit_post_id)",
            "CREATE INDEX IF NOT EXISTS idx_bg_reddit_subreddit ON background_videos(reddit_subreddit)",
            "CREATE INDEX IF NOT EXISTS idx_bg_videos_status ON background_videos(download_status)",
            "CREATE INDEX IF NOT EXISTS idx_bg_videos_views ON background_videos(views)",
            "CREATE INDEX IF NOT EXISTS idx_bg_videos_searched_tag ON background_videos(searched_tag)",
            "CREATE INDEX IF NOT EXISTS idx_bg_videos_tags ON background_videos USING GIN(tags jsonb_path_ops)",
            "CREATE INDEX IF NOT EXISTS idx_bg_videos_created_at ON background_videos(created_at)",
        ):
            conn.execute(text(stmt))

        conn.commit()
        print("Created background_videos table")

    print("Migration complete: background_videos table created")


if __name__ == "__main__":
    run_migration()
