"""
Migration: Add Niche Tracking for Pipeline Automation

Adds 'niche' column to relevant tables for per-niche model training,
generation, and composition tracking.

Also adds:
- last_used_at to background_videos (for cooldown tracking)
- daily_quota_tracking table (for daily video quota per niche)
- pipeline_state_log table (for audit trail)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Run the migration to add niche tracking columns."""
    with engine.connect() as conn:
        # ============================================================
        # Add 'niche' column to training_jobs
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE training_jobs
                ADD COLUMN IF NOT EXISTS niche VARCHAR(50)
            """))
            logger.info("Added 'niche' column to training_jobs")
        except Exception as e:
            logger.warning(f"training_jobs.niche: {e}")

        # Add index for niche queries
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_jobs_niche
                ON training_jobs (niche)
            """))
            logger.info("Created index on training_jobs.niche")
        except Exception as e:
            logger.warning(f"training_jobs niche index: {e}")

        # ============================================================
        # Add 'niche' column to trained_models
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE trained_models
                ADD COLUMN IF NOT EXISTS niche VARCHAR(50)
            """))
            logger.info("Added 'niche' column to trained_models")
        except Exception as e:
            logger.warning(f"trained_models.niche: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_trained_models_niche
                ON trained_models (niche)
            """))
            logger.info("Created index on trained_models.niche")
        except Exception as e:
            logger.warning(f"trained_models niche index: {e}")

        # ============================================================
        # Add 'niche' column to generated_captions
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE generated_captions
                ADD COLUMN IF NOT EXISTS niche VARCHAR(50)
            """))
            logger.info("Added 'niche' column to generated_captions")
        except Exception as e:
            logger.warning(f"generated_captions.niche: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_generated_captions_niche
                ON generated_captions (niche)
            """))
            logger.info("Created index on generated_captions.niche")
        except Exception as e:
            logger.warning(f"generated_captions niche index: {e}")

        # ============================================================
        # Add 'niche' column to video_composition_jobs
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE video_composition_jobs
                ADD COLUMN IF NOT EXISTS niche VARCHAR(50)
            """))
            logger.info("Added 'niche' column to video_composition_jobs")
        except Exception as e:
            logger.warning(f"video_composition_jobs.niche: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_video_composition_jobs_niche
                ON video_composition_jobs (niche)
            """))
            logger.info("Created index on video_composition_jobs.niche")
        except Exception as e:
            logger.warning(f"video_composition_jobs niche index: {e}")

        # ============================================================
        # Add 'niche' column to composed_videos
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE composed_videos
                ADD COLUMN IF NOT EXISTS niche VARCHAR(50)
            """))
            logger.info("Added 'niche' column to composed_videos")
        except Exception as e:
            logger.warning(f"composed_videos.niche: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_composed_videos_niche
                ON composed_videos (niche)
            """))
            logger.info("Created index on composed_videos.niche")
        except Exception as e:
            logger.warning(f"composed_videos niche index: {e}")

        # ============================================================
        # Add 'last_used_at' column to background_videos
        # For tracking cooldown period before reusing backgrounds
        # ============================================================
        try:
            conn.execute(text("""
                ALTER TABLE background_videos
                ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP WITH TIME ZONE
            """))
            logger.info("Added 'last_used_at' column to background_videos")
        except Exception as e:
            logger.warning(f"background_videos.last_used_at: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_background_videos_last_used_at
                ON background_videos (last_used_at)
            """))
            logger.info("Created index on background_videos.last_used_at")
        except Exception as e:
            logger.warning(f"background_videos last_used_at index: {e}")

        # ============================================================
        # Create daily_quota_tracking table
        # Tracks videos composed per day per niche
        # ============================================================
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_quota_tracking (
                    id SERIAL PRIMARY KEY,
                    niche VARCHAR(50) NOT NULL,
                    date DATE NOT NULL,
                    videos_composed INTEGER DEFAULT 0,
                    videos_target INTEGER DEFAULT 3,
                    quota_met BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(niche, date)
                )
            """))
            logger.info("Created daily_quota_tracking table")
        except Exception as e:
            logger.warning(f"daily_quota_tracking table: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_daily_quota_niche_date
                ON daily_quota_tracking (niche, date)
            """))
            logger.info("Created index on daily_quota_tracking")
        except Exception as e:
            logger.warning(f"daily_quota_tracking index: {e}")

        # ============================================================
        # Create pipeline_state_log table
        # Audit trail for pipeline state transitions
        # ============================================================
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pipeline_state_log (
                    id SERIAL PRIMARY KEY,
                    state VARCHAR(50) NOT NULL,
                    previous_state VARCHAR(50),
                    niche VARCHAR(50),
                    triggered_by VARCHAR(100),
                    reason TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            logger.info("Created pipeline_state_log table")
        except Exception as e:
            logger.warning(f"pipeline_state_log table: {e}")

        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_state_log_created
                ON pipeline_state_log (created_at DESC)
            """))
            logger.info("Created index on pipeline_state_log")
        except Exception as e:
            logger.warning(f"pipeline_state_log index: {e}")

        conn.commit()
        logger.info("Migration completed successfully!")


def verify_migration():
    """Verify all columns and tables were created."""
    with engine.connect() as conn:
        # Check columns exist
        checks = [
            ("training_jobs", "niche"),
            ("trained_models", "niche"),
            ("generated_captions", "niche"),
            ("video_composition_jobs", "niche"),
            ("composed_videos", "niche"),
            ("background_videos", "last_used_at"),
        ]

        for table, column in checks:
            result = conn.execute(text(f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name = '{column}'
            """))
            if result.fetchone():
                logger.info(f"✓ {table}.{column} exists")
            else:
                logger.error(f"✗ {table}.{column} missing!")

        # Check tables exist
        tables = ["daily_quota_tracking", "pipeline_state_log"]
        for table in tables:
            result = conn.execute(text(f"""
                SELECT table_name FROM information_schema.tables
                WHERE table_name = '{table}'
            """))
            if result.fetchone():
                logger.info(f"✓ {table} table exists")
            else:
                logger.error(f"✗ {table} table missing!")


if __name__ == "__main__":
    logger.info("Running niche tracking migration...")
    run_migration()
    logger.info("\nVerifying migration...")
    verify_migration()
