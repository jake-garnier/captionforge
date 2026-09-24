"""
Migration: Add edited_caption_text field to composed_videos table

This allows users to edit the caption text for a composed video and use
the edited version when recomposing with a new background video.
"""

import logging
from database.db import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add edited_caption_text column to composed_videos table."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'composed_videos' AND column_name = 'edited_caption_text'
        """))
        if result.fetchone():
            logger.info("Column edited_caption_text already exists, skipping migration")
            return

        # Add the column
        logger.info("Adding edited_caption_text column to composed_videos table...")
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN edited_caption_text TEXT
        """))
        conn.commit()
        logger.info("Migration completed successfully")


if __name__ == "__main__":
    migrate()
