#!/usr/bin/env python3
"""
Migration: Add batch_size and description to scraping_progress table

This migration adds configuration fields that were previously stored in
config/subreddits.yaml directly to the database, eliminating the need for
the config file.
"""
import psycopg2
import os

def run_migration():
    """Run the migration to add subreddit config fields"""

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
        print("Adding batch_size and description columns...")

        # Add new columns
        cur.execute("""
            ALTER TABLE scraping_progress
            ADD COLUMN IF NOT EXISTS batch_size INTEGER DEFAULT 25,
            ADD COLUMN IF NOT EXISTS description VARCHAR(255);
        """)

        # Add helpful comments
        cur.execute("""
            COMMENT ON COLUMN scraping_progress.batch_size IS
            'Number of posts to fetch per scrape batch';

            COMMENT ON COLUMN scraping_progress.description IS
            'Optional description for this subreddit scraper';
        """)

        # Backfill video counts from actual data
        print("Backfilling video counts from videos table...")
        cur.execute("""
            UPDATE scraping_progress sp
            SET
                videos_downloaded = COALESCE((
                    SELECT COUNT(*) FROM videos v
                    WHERE v.source_subreddit = sp.subreddit
                ), 0),
                posts_scraped = COALESCE((
                    SELECT COUNT(*) FROM videos v
                    WHERE v.source_subreddit = sp.subreddit
                ), 0)
            WHERE sp.videos_downloaded = 0 OR sp.posts_scraped = 0;
        """)

        conn.commit()

        # Show current state
        print("\nCurrent scraping_progress table after migration:")
        cur.execute("""
            SELECT
                subreddit,
                videos_downloaded,
                posts_scraped,
                batch_size,
                target_min_score,
                scraping_active
            FROM scraping_progress;
        """)

        rows = cur.fetchall()
        for row in rows:
            print(f"  r/{row[0]}: videos={row[1]}, posts={row[2]}, batch={row[3]}, min_score={row[4]}, active={row[5]}")

        print("\n✅ Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
