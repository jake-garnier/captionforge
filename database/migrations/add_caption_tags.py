"""
Migration: Add tags field to generated_captions table

Adds a JSON field to store extracted activity tags from caption content.
These tags enable matching captions to appropriate background videos during composition.

Run with: docker-compose exec api python database/migrations/add_caption_tags.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from database.db import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add tags column to generated_captions table."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'generated_captions' AND column_name = 'tags'
        """))

        if result.fetchone():
            logger.info("Column 'tags' already exists in generated_captions table")
            return

        # Add the tags column (JSON array of activity tags)
        logger.info("Adding 'tags' column to generated_captions table...")
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN tags JSON DEFAULT '[]'::json
        """))

        conn.commit()
        logger.info("Successfully added 'tags' column to generated_captions table")


if __name__ == "__main__":
    migrate()
