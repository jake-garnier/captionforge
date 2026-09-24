"""
Migration: Backfill scraped_captions.upvotes from videos table

The scraped_captions.upvotes field was being set to 0 instead of
copying from the parent video. This migration fixes existing records.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from database.db import engine, get_db_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Update scraped_captions.upvotes from videos.upvotes"""

    with get_db_context() as db:
        # Count records that need updating
        count_sql = text("""
            SELECT COUNT(*) FROM scraped_captions sc
            JOIN videos v ON sc.video_id = v.id
            WHERE (sc.upvotes IS NULL OR sc.upvotes = 0)
            AND v.upvotes IS NOT NULL AND v.upvotes > 0
        """)
        result = db.execute(count_sql)
        count = result.scalar()

        logger.info(f"Found {count} captions to update with video upvotes")

        if count == 0:
            logger.info("No captions need updating")
            return

        # Update scraped_captions.upvotes from videos.upvotes
        update_sql = text("""
            UPDATE scraped_captions
            SET upvotes = videos.upvotes
            FROM videos
            WHERE scraped_captions.video_id = videos.id
            AND (scraped_captions.upvotes IS NULL OR scraped_captions.upvotes = 0)
            AND videos.upvotes IS NOT NULL AND videos.upvotes > 0
        """)

        result = db.execute(update_sql)
        db.commit()

        logger.info(f"Updated {result.rowcount} caption upvote values")

        # Verify the update
        verify_sql = text("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN upvotes > 0 THEN 1 END) as with_upvotes,
                AVG(CASE WHEN upvotes > 0 THEN upvotes END) as avg_upvotes
            FROM scraped_captions
        """)
        result = db.execute(verify_sql)
        stats = result.fetchone()

        logger.info(f"Verification: {stats[0]} total captions, {stats[1]} with upvotes > 0, avg: {stats[2]:.0f if stats[2] else 0}")


if __name__ == "__main__":
    logger.info("Starting caption upvotes backfill migration...")
    run_migration()
    logger.info("Migration complete!")
