#!/usr/bin/env python3
"""
Migration: Add videos_downloaded tracking to scraping_progress table

This migration adds fields to track actual successful video downloads
vs posts examined, allowing the scraper to set targets based on
unique videos collected rather than posts processed.
"""
import psycopg2
import os

def run_migration():
    """Run the migration to add videos_downloaded tracking"""

    # Database connection from environment
    db_config = {
        'dbname': os.getenv('POSTGRES_DB', 'captions'),
        'user': os.getenv('POSTGRES_USER', 'captionsuser'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432')
    }

    print("Connecting to database...")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    try:
        print("Adding videos_downloaded tracking columns...")

        # Add new columns
        cur.execute("""
            ALTER TABLE scraping_progress
            ADD COLUMN IF NOT EXISTS videos_downloaded INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS target_videos INTEGER DEFAULT 500,
            ADD COLUMN IF NOT EXISTS videos_failed INTEGER DEFAULT 0;
        """)

        # Add helpful comment
        cur.execute("""
            COMMENT ON COLUMN scraping_progress.posts_scraped IS
            'Total posts examined (includes duplicates, 404s, etc)';

            COMMENT ON COLUMN scraping_progress.videos_downloaded IS
            'Actual unique videos successfully downloaded and stored';

            COMMENT ON COLUMN scraping_progress.target_videos IS
            'Target number of unique videos to collect before stopping';

            COMMENT ON COLUMN scraping_progress.videos_failed IS
            'Videos that failed to download (404s, duplicates, etc)';
        """)

        # Backfill videos_downloaded from existing data
        print("Backfilling videos_downloaded from current database state...")
        cur.execute("""
            UPDATE scraping_progress sp
            SET videos_downloaded = (
                SELECT COUNT(DISTINCT v.id)
                FROM videos v
                WHERE v.source_subreddit = sp.subreddit
                AND v.processing_status IN ('completed', 'caption_extracted')
            );
        """)

        # Backfill videos_failed
        cur.execute("""
            UPDATE scraping_progress sp
            SET videos_failed = sp.posts_scraped - sp.videos_downloaded;
        """)

        conn.commit()

        # Show current state
        print("\nCurrent scraping progress after migration:")
        cur.execute("""
            SELECT
                subreddit,
                posts_scraped,
                videos_downloaded,
                videos_failed,
                target_videos,
                scraping_active
            FROM scraping_progress;
        """)

        rows = cur.fetchall()
        for row in rows:
            print(f"""
  Subreddit: r/{row[0]}
    Posts examined: {row[1]}
    Videos downloaded: {row[2]} ({row[2]/row[1]*100:.1f}% success)
    Videos failed: {row[3]}
    Target: {row[4]} videos
    Active: {row[5]}
            """)

        print("✅ Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
