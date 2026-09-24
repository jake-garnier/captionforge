"""
Migration: Add searched_tag column to background_videos table.

Tracks which tag was used to find each video (for strict tag matching).
"""
import logging
from sqlalchemy import text
from database.db import get_db_context

logger = logging.getLogger(__name__)


def run_migration():
    """Add searched_tag column to background_videos."""
    with get_db_context() as db:
        # Check if column exists
        result = db.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'background_videos' AND column_name = 'searched_tag'
        """))
        if result.fetchone():
            logger.info("Column 'searched_tag' already exists")
            return

        # Add the column
        db.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN searched_tag VARCHAR(100)
        """))

        # Add index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_background_videos_searched_tag
            ON background_videos (searched_tag)
        """))

        db.commit()
        logger.info("Added 'searched_tag' column to background_videos table")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migration()
