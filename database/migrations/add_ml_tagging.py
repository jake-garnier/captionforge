"""
Add ML tagging columns to background_videos table for VLM-based content analysis.

Status values:
- pending: Awaiting ML tagging
- processing: Currently being tagged
- completed: Tagging complete, ml_tags populated
- error: Tagging failed
- skipped: Video was rejected by filter (not eligible for tagging)
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
    """Add ML tagging columns to background_videos."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'background_videos' AND column_name = 'ml_tagging_status'
        """))

        if result.fetchone():
            logger.info("ml_tagging_status column already exists, skipping")
            return

        # Add ml_tags column (JSON for structured tags)
        logger.info("Adding ml_tags column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN ml_tags JSON
        """))

        # Add ml_tagging_status column
        logger.info("Adding ml_tagging_status column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN ml_tagging_status VARCHAR(20) DEFAULT 'pending'
        """))

        # Add ml_tagging_checked_at timestamp
        logger.info("Adding ml_tagging_checked_at column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN ml_tagging_checked_at TIMESTAMP
        """))

        # Add ml_tagging_error column for error messages
        logger.info("Adding ml_tagging_error column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN ml_tagging_error TEXT
        """))

        # Create index for efficient filtering queries
        logger.info("Creating index on ml_tagging_status...")
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_background_videos_ml_tagging_status
            ON background_videos(ml_tagging_status)
        """))

        conn.commit()
        logger.info("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
