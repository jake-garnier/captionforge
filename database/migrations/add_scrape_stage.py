#!/usr/bin/env python3
"""
Migration: Add scrape_stage to scraping_progress table

Implements progressive depth scraping strategy:
- Starts at 'top_all' (top posts of all time)
- Progresses through time filters when pagination ends
- Eventually settles on 'new' for ongoing scraping

Stage progression: top_all -> top_year -> top_month -> top_week -> top_day -> new
"""
import psycopg2
import os

# Valid scrape stages in order of progression
SCRAPE_STAGES = ['top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new']

def run_migration():
    """Run the migration to add scrape_stage field"""

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
        print("Adding scrape_stage column to scraping_progress...")

        # Add scrape_stage column with default 'top_all'
        cur.execute("""
            ALTER TABLE scraping_progress
            ADD COLUMN IF NOT EXISTS scrape_stage VARCHAR(20) DEFAULT 'top_all';
        """)

        # Add check constraint to ensure valid stage values
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'scraping_progress_stage_check'
                ) THEN
                    ALTER TABLE scraping_progress
                    ADD CONSTRAINT scraping_progress_stage_check
                    CHECK (scrape_stage IN ('top_all', 'top_year', 'top_month', 'top_week', 'top_day', 'new'));
                END IF;
            END $$;
        """)

        # Add helpful comment
        cur.execute("""
            COMMENT ON COLUMN scraping_progress.scrape_stage IS
            'Current scraping stage: top_all -> top_year -> top_month -> top_week -> top_day -> new';
        """)

        # Check existing subreddits - if they have significant progress, assume they're past initial scraping
        # and set them to 'new' stage
        print("Checking existing subreddits for stage initialization...")
        cur.execute("""
            SELECT subreddit, videos_downloaded, scraping_active
            FROM scraping_progress;
        """)

        rows = cur.fetchall()
        for subreddit, videos_downloaded, scraping_active in rows:
            # If subreddit has collected significant videos, it's likely past initial deep scraping
            # Set to 'new' stage since it's been scraping for a while
            if videos_downloaded > 100:
                print(f"  r/{subreddit}: {videos_downloaded} videos - setting to 'new' stage")
                cur.execute("""
                    UPDATE scraping_progress
                    SET scrape_stage = 'new'
                    WHERE subreddit = %s;
                """, (subreddit,))
            else:
                print(f"  r/{subreddit}: {videos_downloaded} videos - keeping at 'top_all' stage")

        conn.commit()

        # Show final state
        print("\nCurrent scraping_progress table after migration:")
        cur.execute("""
            SELECT
                subreddit,
                scrape_stage,
                videos_downloaded,
                target_min_score,
                scraping_active
            FROM scraping_progress
            ORDER BY subreddit;
        """)

        rows = cur.fetchall()
        for row in rows:
            print(f"  r/{row[0]}: stage={row[1]}, videos={row[2]}, min_score={row[3]}, active={row[4]}")

        print("\n" + "="*50)
        print("Stage progression order:")
        print("  " + " -> ".join(SCRAPE_STAGES))
        print("="*50)
        print("\n Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f" Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
