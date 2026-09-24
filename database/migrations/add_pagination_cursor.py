"""
Migration: Add last_pagination_url to scraping_progress table
Enables cursor-based pagination tracking to avoid re-scraping same pages
"""
from database.db import get_db_context
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upgrade():
    """Add last_pagination_url column"""
    with get_db_context() as db:
        try:
            # Add last_pagination_url column
            db.execute(text("""
                ALTER TABLE scraping_progress
                ADD COLUMN IF NOT EXISTS last_pagination_url TEXT
            """))

            db.commit()
            logger.info("✅ Added last_pagination_url column to scraping_progress table")

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            db.rollback()
            raise


def downgrade():
    """Remove last_pagination_url column"""
    with get_db_context() as db:
        try:
            # Remove last_pagination_url column
            db.execute(text("""
                ALTER TABLE scraping_progress
                DROP COLUMN IF EXISTS last_pagination_url
            """))

            db.commit()
            logger.info("✅ Removed last_pagination_url column from scraping_progress table")

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            db.rollback()
            raise


if __name__ == "__main__":
    logger.info("Running migration: add_pagination_cursor")
    upgrade()
    logger.info("Migration completed successfully")
