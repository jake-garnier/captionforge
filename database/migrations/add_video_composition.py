"""
Migration: Add video_composition_jobs and composed_videos tables

Run with: docker-compose exec api python database/migrations/add_video_composition.py
"""
import sys
sys.path.insert(0, '/app')

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Create video_composition_jobs and composed_videos tables."""

    with engine.connect() as conn:
        # Create video_composition_jobs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS video_composition_jobs (
                id SERIAL PRIMARY KEY,

                -- Job identification
                job_name VARCHAR(200) NOT NULL,
                celery_task_id VARCHAR(100),

                -- Source configuration
                caption_source VARCHAR(50) DEFAULT 'generated',
                caption_status_filter VARCHAR(30),
                min_upvotes INTEGER DEFAULT 0,
                background_tag_filter VARCHAR(100),

                -- Composition settings
                target_count INTEGER DEFAULT 10,

                -- Progress tracking
                status VARCHAR(30) DEFAULT 'pending',
                progress_percent FLOAT DEFAULT 0.0,
                videos_composed INTEGER DEFAULT 0,
                videos_failed INTEGER DEFAULT 0,
                videos_skipped INTEGER DEFAULT 0,

                -- Error handling
                error_message TEXT,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE
            );
        """))

        # Create indexes for video_composition_jobs
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_video_composition_jobs_status
            ON video_composition_jobs(status);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_video_composition_jobs_celery_task_id
            ON video_composition_jobs(celery_task_id);
        """))

        # Create composed_videos table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS composed_videos (
                id SERIAL PRIMARY KEY,

                -- Source references
                composition_job_id INTEGER REFERENCES video_composition_jobs(id) ON DELETE SET NULL,
                generated_caption_id INTEGER REFERENCES generated_captions(id) ON DELETE SET NULL,
                scraped_caption_id INTEGER REFERENCES scraped_captions(id) ON DELETE SET NULL,
                background_video_id INTEGER NOT NULL REFERENCES background_videos(id) ON DELETE CASCADE,

                -- Output file
                storage_path TEXT NOT NULL,
                file_size_bytes BIGINT,
                duration_seconds FLOAT,
                resolution VARCHAR(50),

                -- Composition metadata
                caption_chunks INTEGER,
                status VARCHAR(30) DEFAULT 'completed',

                -- Quality/review
                is_favorite BOOLEAN DEFAULT FALSE,
                is_published BOOLEAN DEFAULT FALSE,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))

        # Create indexes for composed_videos
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_composed_videos_status
            ON composed_videos(status);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_composed_videos_job_id
            ON composed_videos(composition_job_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_composed_videos_favorite
            ON composed_videos(is_favorite);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_composed_videos_published
            ON composed_videos(is_published);
        """))

        conn.commit()

    print("Migration complete: video_composition_jobs and composed_videos tables created")


if __name__ == "__main__":
    run_migration()
