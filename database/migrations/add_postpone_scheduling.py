"""
Add Postpone scheduling tables.

Tables:
- postpone_schedule_jobs: Track Postpone API scheduling status per composed video

Also adds approval_status and approved_at columns to composed_videos table.

Migration: Run once to add tables
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Add Postpone scheduling tables and composed_videos approval columns."""
    with engine.connect() as conn:
        # Check if table already exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'postpone_schedule_jobs'
            );
        """))
        exists = result.scalar()

        if exists:
            print("postpone_schedule_jobs table already exists, skipping table creation...")
        else:
            # Create postpone_schedule_jobs table
            conn.execute(text("""
                CREATE TABLE postpone_schedule_jobs (
                    id SERIAL PRIMARY KEY,
                    composed_video_id INTEGER NOT NULL REFERENCES composed_videos(id) ON DELETE CASCADE,

                    -- Content info
                    title VARCHAR(300) NOT NULL,
                    niche VARCHAR(50),
                    reddit_username VARCHAR(100) NOT NULL,

                    -- Scheduling info
                    scheduled_date DATE NOT NULL,
                    base_post_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    stagger_minutes INTEGER DEFAULT 10,

                    -- Subreddits (JSON array of subreddit names)
                    target_subreddits JSON NOT NULL,

                    -- Postpone tracking
                    postpone_post_id VARCHAR(100),
                    postpone_response JSON,

                    -- Status: pending, scheduling, scheduled, failed, cancelled
                    status VARCHAR(30) DEFAULT 'pending' NOT NULL,

                    -- Media host result (public URL the Reddit link post points to)
                    hosted_media_id VARCHAR(100),
                    hosted_url TEXT,

                    -- Celery task tracking
                    celery_task_id VARCHAR(100),

                    -- Error handling
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,

                    -- Timestamps
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    scheduled_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE
                );

                CREATE INDEX idx_postpone_jobs_status ON postpone_schedule_jobs(status);
                CREATE INDEX idx_postpone_jobs_composed_video ON postpone_schedule_jobs(composed_video_id);
                CREATE INDEX idx_postpone_jobs_niche ON postpone_schedule_jobs(niche);
                CREATE INDEX idx_postpone_jobs_scheduled_date ON postpone_schedule_jobs(scheduled_date);
                CREATE INDEX idx_postpone_jobs_celery_task ON postpone_schedule_jobs(celery_task_id);
            """))
            print("Created postpone_schedule_jobs table")

        # Add approval columns to composed_videos (idempotent)
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'composed_videos' AND column_name = 'approval_status'
            );
        """))
        has_approval = result.scalar()

        if not has_approval:
            conn.execute(text("""
                ALTER TABLE composed_videos
                ADD COLUMN approval_status VARCHAR(30) DEFAULT 'pending',
                ADD COLUMN approved_at TIMESTAMP WITH TIME ZONE;

                CREATE INDEX idx_composed_videos_approval ON composed_videos(approval_status);
            """))
            print("Added approval_status and approved_at columns to composed_videos")
        else:
            print("approval_status column already exists on composed_videos, skipping...")

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
