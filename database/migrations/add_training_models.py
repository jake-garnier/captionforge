"""
Migration: Add trained_models and training_jobs tables

Run with: docker-compose exec api python database/migrations/add_training_models.py
"""
import sys
sys.path.insert(0, '/app')

from database.db import engine
from sqlalchemy import text

def run_migration():
    """Create trained_models and training_jobs tables."""

    with engine.connect() as conn:
        # Create trained_models table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trained_models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,

                -- Base model info
                base_model VARCHAR(200) NOT NULL,
                adapter_path TEXT NOT NULL,

                -- Training data source
                source_subreddits JSONB NOT NULL,
                training_samples INTEGER NOT NULL,
                min_upvotes_filter INTEGER DEFAULT 0,

                -- Training hyperparameters
                hyperparameters JSONB,

                -- Training metrics
                final_loss FLOAT,
                validation_loss FLOAT,
                training_duration_seconds INTEGER,

                -- Status
                status VARCHAR(20) DEFAULT 'ready',
                is_loaded BOOLEAN DEFAULT FALSE,
                last_loaded_at TIMESTAMP WITH TIME ZONE,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))

        # Create indexes for trained_models
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trained_models_name ON trained_models(name);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trained_models_status ON trained_models(status);
        """))

        # Create training_jobs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                id SERIAL PRIMARY KEY,
                model_id INTEGER REFERENCES trained_models(id) ON DELETE SET NULL,

                -- Job identification
                job_name VARCHAR(200) NOT NULL,
                celery_task_id VARCHAR(100),

                -- Training configuration
                base_model VARCHAR(200) NOT NULL,
                source_subreddits JSONB NOT NULL,
                min_upvotes INTEGER DEFAULT 0,
                min_caption_length INTEGER DEFAULT 50,
                max_caption_length INTEGER DEFAULT 2000,

                -- Hyperparameters
                lora_rank INTEGER DEFAULT 16,
                lora_alpha INTEGER DEFAULT 32,
                learning_rate FLOAT DEFAULT 0.0002,
                num_epochs INTEGER DEFAULT 3,
                batch_size INTEGER DEFAULT 1,
                gradient_accumulation_steps INTEGER DEFAULT 4,
                max_seq_length INTEGER DEFAULT 512,
                warmup_ratio FLOAT DEFAULT 0.1,

                -- Progress tracking
                status VARCHAR(30) DEFAULT 'pending',
                progress_percent FLOAT DEFAULT 0.0,
                current_epoch INTEGER DEFAULT 0,
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER,
                current_loss FLOAT,
                best_loss FLOAT,

                -- Data statistics
                total_samples INTEGER,
                train_samples INTEGER,
                val_samples INTEGER,

                -- Results
                final_train_loss FLOAT,
                final_val_loss FLOAT,
                training_duration_seconds INTEGER,

                -- Error handling
                error_message TEXT,
                error_traceback TEXT,

                -- Timestamps
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE
            );
        """))

        # Create indexes for training_jobs
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_celery_task_id ON training_jobs(celery_task_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON training_jobs(model_id);
        """))

        conn.commit()

    print("Migration complete: trained_models and training_jobs tables created")


if __name__ == "__main__":
    run_migration()
