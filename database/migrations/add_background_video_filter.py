"""
Add filter_status field to background_videos table for watermark detection.

Status values:
- pending: Awaiting OCR filter check
- approved: No watermark/text detected, video is usable
- rejected: Watermark/text detected, video should be deleted
- error: OCR check failed
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
    """Add filter_status and filter_text columns to background_videos."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'background_videos' AND column_name = 'filter_status'
        """))

        if result.fetchone():
            logger.info("filter_status column already exists, skipping")
            return

        # Add filter_status column (pending, approved, rejected, error)
        logger.info("Adding filter_status column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN filter_status VARCHAR(20) DEFAULT 'pending'
        """))

        # Add filter_text column to store detected text (for debugging)
        logger.info("Adding filter_text column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN filter_text TEXT
        """))

        # Add filter_checked_at timestamp
        logger.info("Adding filter_checked_at column to background_videos...")
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN filter_checked_at TIMESTAMP
        """))

        # Create index for efficient filtering queries
        logger.info("Creating index on filter_status...")
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_background_videos_filter_status
            ON background_videos(filter_status)
        """))

        conn.commit()
        logger.info("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
