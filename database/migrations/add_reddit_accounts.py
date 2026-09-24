"""
Migration: Add reddit_accounts table and hosted_url to composed_videos.

Creates:
- reddit_accounts table for storing Reddit credentials per niche
- Adds hosted_url column to composed_videos (public media-host URL of the video)
- Adds reddit_post_url column to composed_videos for tracking posted URLs

Run with: docker-compose exec api python database/migrations/add_reddit_accounts.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from database.db import engine


def run_migration():
    """Run the migration."""
    with engine.connect() as conn:
        # Create reddit_accounts table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reddit_accounts (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50) NOT NULL UNIQUE,
                username VARCHAR(100) NOT NULL,
                password VARCHAR(255) NOT NULL,
                client_id VARCHAR(100) NOT NULL,
                client_secret VARCHAR(255) NOT NULL,
                user_agent VARCHAR(255),
                is_enabled BOOLEAN DEFAULT TRUE,
                subreddits TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        print("Created reddit_accounts table")

        # Create index on niche
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reddit_accounts_niche
            ON reddit_accounts(niche);
        """))
        print("Created index on reddit_accounts.niche")

        # Add hosted_url column to composed_videos
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS hosted_url TEXT;
        """))
        print("Added hosted_url column to composed_videos")

        # Add reddit_post_url column to composed_videos
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS reddit_post_url TEXT;
        """))
        print("Added reddit_post_url column to composed_videos")

        # Add reddit_posted_at column to composed_videos
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS reddit_posted_at TIMESTAMP WITH TIME ZONE;
        """))
        print("Added reddit_posted_at column to composed_videos")

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
