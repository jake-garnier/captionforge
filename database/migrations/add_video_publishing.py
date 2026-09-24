"""
Add video publishing workflow tables.

Tables:
- video_publish_jobs: Main publish job tracking (media host + Reddit posting)
- reddit_crossposts: Track individual crosspost to each subreddit

Migration: Run once to add tables
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Add video publishing tables."""
    with engine.connect() as conn:
        # Check if table already exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'video_publish_jobs'
            );
        """))
        exists = result.scalar()

        if exists:
            print("video_publish_jobs table already exists, skipping...")
            return

        # Create video_publish_jobs table
        conn.execute(text("""
            CREATE TABLE video_publish_jobs (
                id SERIAL PRIMARY KEY,
                composed_video_id INTEGER NOT NULL REFERENCES composed_videos(id) ON DELETE CASCADE,

                -- Content info
                title VARCHAR(300) NOT NULL,
                niche VARCHAR(50),
                tags TEXT,  -- Comma-separated tags for the Reddit post

                -- Status tracking
                status VARCHAR(30) DEFAULT 'pending' NOT NULL,
                -- Status values: pending, hosting_media, posting_profile, waiting_crosspost, crossposting, completed, failed, cancelled

                -- Media host result (public URL the Reddit post links to)
                hosted_media_id VARCHAR(100),
                hosted_url TEXT,
                hosted_at TIMESTAMP WITH TIME ZONE,

                -- Reddit profile post (populated per-niche from reddit_accounts at job creation)
                profile_subreddit VARCHAR(100),
                profile_post_id VARCHAR(50),
                profile_post_url TEXT,
                profile_posted_at TIMESTAMP WITH TIME ZONE,

                -- Crosspost scheduling
                crosspost_delay_minutes INTEGER DEFAULT 30,
                crosspost_scheduled_at TIMESTAMP WITH TIME ZONE,
                crosspost_started_at TIMESTAMP WITH TIME ZONE,
                crosspost_completed_at TIMESTAMP WITH TIME ZONE,

                -- Celery task tracking
                celery_task_id VARCHAR(100),

                -- Error handling
                error_message TEXT,
                error_stage VARCHAR(50),  -- Which stage failed: media_host, profile, crosspost

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE
            );

            CREATE INDEX idx_video_publish_jobs_status ON video_publish_jobs(status);
            CREATE INDEX idx_video_publish_jobs_composed_video ON video_publish_jobs(composed_video_id);
            CREATE INDEX idx_video_publish_jobs_niche ON video_publish_jobs(niche);
            CREATE INDEX idx_video_publish_jobs_crosspost_scheduled ON video_publish_jobs(crosspost_scheduled_at);
        """))
        print("Created video_publish_jobs table")

        # Create reddit_crossposts table
        conn.execute(text("""
            CREATE TABLE reddit_crossposts (
                id SERIAL PRIMARY KEY,
                publish_job_id INTEGER NOT NULL REFERENCES video_publish_jobs(id) ON DELETE CASCADE,

                -- Target subreddit
                subreddit VARCHAR(100) NOT NULL,

                -- Status tracking
                status VARCHAR(30) DEFAULT 'pending' NOT NULL,
                -- Status values: pending, posting, posted, failed, skipped

                -- Post info
                post_id VARCHAR(50),
                post_url TEXT,

                -- Error handling
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                posted_at TIMESTAMP WITH TIME ZONE
            );

            CREATE INDEX idx_reddit_crossposts_publish_job ON reddit_crossposts(publish_job_id);
            CREATE INDEX idx_reddit_crossposts_status ON reddit_crossposts(status);
            CREATE INDEX idx_reddit_crossposts_subreddit ON reddit_crossposts(subreddit);
        """))
        print("Created reddit_crossposts table")

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
