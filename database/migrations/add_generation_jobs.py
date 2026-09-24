"""
Migration: Add generation_jobs table and update generated_captions

Run with: docker-compose exec api python database/migrations/add_generation_jobs.py
"""
import sys
sys.path.insert(0, '/app')

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Create generation_jobs table and update generated_captions."""

    with engine.connect() as conn:
        # Create generation_jobs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS generation_jobs (
                id SERIAL PRIMARY KEY,
                model_id INTEGER NOT NULL REFERENCES trained_models(id) ON DELETE CASCADE,

                -- Job identification
                job_name VARCHAR(200) NOT NULL,
                celery_task_id VARCHAR(100),

                -- Generation configuration
                num_captions INTEGER NOT NULL,
                prompt TEXT DEFAULT 'Generate a caption:',
                temperature FLOAT DEFAULT 0.9,
                top_p FLOAT DEFAULT 0.95,
                max_new_tokens INTEGER DEFAULT 300,
                repetition_penalty FLOAT DEFAULT 1.15,

                -- Progress tracking
                status VARCHAR(30) DEFAULT 'pending',
                progress_percent FLOAT DEFAULT 0.0,
                captions_generated INTEGER DEFAULT 0,

                -- Error handling
                error_message TEXT,
                error_traceback TEXT,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE
            );
        """))

        # Create indexes for generation_jobs
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_generation_jobs_celery_task_id ON generation_jobs(celery_task_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_generation_jobs_model_id ON generation_jobs(model_id);
        """))

        # Alter generated_captions table - make video_id nullable
        conn.execute(text("""
            ALTER TABLE generated_captions
            ALTER COLUMN video_id DROP NOT NULL;
        """))

        # Add generation_job_id column if not exists
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='generated_captions' AND column_name='generation_job_id'
                ) THEN
                    ALTER TABLE generated_captions
                    ADD COLUMN generation_job_id INTEGER REFERENCES generation_jobs(id) ON DELETE SET NULL;
                END IF;
            END $$;
        """))

        # Add is_favorite column if not exists
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='generated_captions' AND column_name='is_favorite'
                ) THEN
                    ALTER TABLE generated_captions
                    ADD COLUMN is_favorite BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        """))

        # Add temperature column if not exists
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='generated_captions' AND column_name='temperature'
                ) THEN
                    ALTER TABLE generated_captions
                    ADD COLUMN temperature FLOAT;
                END IF;
            END $$;
        """))

        # Add top_p column if not exists
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='generated_captions' AND column_name='top_p'
                ) THEN
                    ALTER TABLE generated_captions
                    ADD COLUMN top_p FLOAT;
                END IF;
            END $$;
        """))

        # Add max_tokens column if not exists
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='generated_captions' AND column_name='max_tokens'
                ) THEN
                    ALTER TABLE generated_captions
                    ADD COLUMN max_tokens INTEGER;
                END IF;
            END $$;
        """))

        # Increase llm_model column size if needed
        conn.execute(text("""
            ALTER TABLE generated_captions
            ALTER COLUMN llm_model TYPE VARCHAR(100);
        """))

        # Create index on generation_job_id
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_generated_captions_job_id ON generated_captions(generation_job_id);
        """))

        # Create index on is_favorite
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_generated_captions_favorite ON generated_captions(is_favorite);
        """))

        conn.commit()

    print("Migration complete: generation_jobs table created and generated_captions updated")


if __name__ == "__main__":
    run_migration()
