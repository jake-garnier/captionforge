"""
Migration: Add GPU selection and concurrent training support

Adds:
- target_gpu column to training_jobs table (0 or 1)
- target_gpu column to trained_models table (which GPU it was trained on)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text
from config.settings import settings

def run_migration():
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Add target_gpu to training_jobs (default GPU 0)
        try:
            conn.execute(text("""
                ALTER TABLE training_jobs
                ADD COLUMN IF NOT EXISTS target_gpu INTEGER DEFAULT 0
            """))
            conn.commit()
            print("Added target_gpu column to training_jobs table")
        except Exception as e:
            print(f"Column target_gpu may already exist in training_jobs: {e}")

        # Add target_gpu to trained_models
        try:
            conn.execute(text("""
                ALTER TABLE trained_models
                ADD COLUMN IF NOT EXISTS target_gpu INTEGER DEFAULT 0
            """))
            conn.commit()
            print("Added target_gpu column to trained_models table")
        except Exception as e:
            print(f"Column target_gpu may already exist in trained_models: {e}")

        print("Migration completed successfully!")

if __name__ == "__main__":
    run_migration()
