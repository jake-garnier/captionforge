"""
Migration: Add Patreon credentials table for per-niche publishing.

Each niche can have its own Patreon account credentials for publishing videos.

Run with:
    docker-compose exec api python database/migrations/add_patreon_credentials.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text, inspect
from config.settings import settings


def run_migration():
    """Add patreon_credentials and patreon_publish_jobs tables."""
    engine = create_engine(settings.database_url)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.connect() as conn:
        # Create patreon_credentials table
        if 'patreon_credentials' not in existing_tables:
            conn.execute(text("""
                CREATE TABLE patreon_credentials (
                    id SERIAL PRIMARY KEY,
                    niche VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    cookies_path TEXT,
                    is_configured BOOLEAN DEFAULT FALSE,
                    last_login_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.execute(text("CREATE INDEX idx_patreon_credentials_niche ON patreon_credentials(niche)"))
            print("Created patreon_credentials table")
        else:
            print("patreon_credentials table already exists")

        # Create patreon_publish_jobs table
        if 'patreon_publish_jobs' not in existing_tables:
            conn.execute(text("""
                CREATE TABLE patreon_publish_jobs (
                    id SERIAL PRIMARY KEY,
                    composed_video_id INTEGER NOT NULL REFERENCES composed_videos(id) ON DELETE CASCADE,
                    credential_id INTEGER REFERENCES patreon_credentials(id) ON DELETE SET NULL,
                    niche VARCHAR(50) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    tags TEXT,

                    -- Status: pending, uploading, posted, failed, cancelled
                    status VARCHAR(30) DEFAULT 'pending',

                    -- Patreon post info
                    patreon_post_id VARCHAR(100),
                    patreon_post_url TEXT,
                    posted_at TIMESTAMP WITH TIME ZONE,

                    -- Celery task tracking
                    celery_task_id VARCHAR(100),

                    -- Error handling
                    error_message TEXT,

                    -- Timestamps
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE
                )
            """))
            conn.execute(text("CREATE INDEX idx_patreon_publish_jobs_status ON patreon_publish_jobs(status)"))
            conn.execute(text("CREATE INDEX idx_patreon_publish_jobs_niche ON patreon_publish_jobs(niche)"))
            conn.execute(text("CREATE INDEX idx_patreon_publish_jobs_video ON patreon_publish_jobs(composed_video_id)"))
            print("Created patreon_publish_jobs table")
        else:
            print("patreon_publish_jobs table already exists")

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
