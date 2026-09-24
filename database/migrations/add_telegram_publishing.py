#!/usr/bin/env python3
"""Add Telegram bots, channels, and publish jobs tables."""

import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from database.db import engine


def run_migration():
    """Create telegram_bots, telegram_channels, and telegram_publish_jobs tables."""

    with engine.connect() as conn:
        # Create telegram_bots table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telegram_bots (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50) UNIQUE NOT NULL,
                bot_username VARCHAR(100) NOT NULL,
                bot_token VARCHAR(200) NOT NULL,
                bot_name VARCHAR(100),
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ Created telegram_bots table")

        # Create telegram_channels table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telegram_channels (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50) UNIQUE NOT NULL,
                channel_id VARCHAR(50) NOT NULL,
                channel_username VARCHAR(100),
                channel_name VARCHAR(200),
                discussion_group_id VARCHAR(50),
                bot_id INTEGER REFERENCES telegram_bots(id) ON DELETE SET NULL,
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ Created telegram_channels table")

        # Create telegram_publish_jobs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telegram_publish_jobs (
                id SERIAL PRIMARY KEY,
                composed_video_id INTEGER REFERENCES composed_videos(id) ON DELETE CASCADE,
                niche VARCHAR(50) NOT NULL,
                channel_id INTEGER REFERENCES telegram_channels(id),
                bot_id INTEGER REFERENCES telegram_bots(id),
                status VARCHAR(50) DEFAULT 'pending',
                telegram_message_id INTEGER,
                telegram_post_url VARCHAR(500),
                caption_text TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """))
        print("✓ Created telegram_publish_jobs table")

        # Create indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telegram_publish_jobs_status
            ON telegram_publish_jobs(status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telegram_publish_jobs_niche
            ON telegram_publish_jobs(niche)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telegram_publish_jobs_composed_video
            ON telegram_publish_jobs(composed_video_id)
        """))
        print("✓ Created indexes")

        conn.commit()
        print("\n✅ Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
