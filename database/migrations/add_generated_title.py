"""
Migration: Add generated_title field to generated_captions table

Adds a VARCHAR(300) field to store LLM-generated Reddit post titles.
Titles are generated alongside captions during the generation job.

Run with: docker-compose exec api python database/migrations/add_generated_title.py
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
    """Add generated_title column to generated_captions table."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'generated_captions' AND column_name = 'generated_title'
        """))

        if result.fetchone():
            logger.info("Column 'generated_title' already exists in generated_captions table")
            return

        # Add the generated_title column
        logger.info("Adding 'generated_title' column to generated_captions table...")
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN generated_title VARCHAR(300) DEFAULT NULL
        """))

        conn.commit()
        logger.info("Successfully added 'generated_title' column to generated_captions table")


if __name__ == "__main__":
    migrate()
