#!/usr/bin/env python3
"""Add Telegram scrape channels table for scraping videos from private channels."""

import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from database.db import engine


def run_migration():
    """Create telegram_scrape_channels table."""

    with engine.connect() as conn:
        # Create telegram_scrape_channels table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telegram_scrape_channels (
                id SERIAL PRIMARY KEY,
                channel_id VARCHAR(100) UNIQUE NOT NULL,
                channel_username VARCHAR(100),
                channel_name VARCHAR(200),
                is_enabled BOOLEAN DEFAULT TRUE,
                batch_size INTEGER DEFAULT 25,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_scrape_at TIMESTAMP
            )
        """))
        print("Created telegram_scrape_channels table")

        # Create index on channel_id
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telegram_scrape_channels_channel_id
            ON telegram_scrape_channels(channel_id)
        """))
        print("Created index on channel_id")

        # Create index on is_enabled for dispatcher queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telegram_scrape_channels_enabled
            ON telegram_scrape_channels(is_enabled)
        """))
        print("Created index on is_enabled")

        conn.commit()
        print("\nMigration completed successfully!")


if __name__ == "__main__":
    run_migration()
