"""
Migration: add ml_tags_version column and mark all approved reddit BGs
for v2 re-tagging with the Qwen2.5-VL-7B-Instruct pipeline.

Safe to run multiple times. Run with:
  docker-compose exec api python database/migrations/add_ml_tags_v2.py
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text

from database.db import engine


def run_migration():
    with engine.connect() as conn:
        # 1) Add ml_tags_version column if missing
        conn.execute(text("""
            ALTER TABLE background_videos
            ADD COLUMN IF NOT EXISTS ml_tags_version INTEGER DEFAULT 1
        """))
        conn.commit()
        print("Added ml_tags_version column (if not already present)")

        # 2) Reset all approved reddit backgrounds to pending for v2 re-tagging
        result = conn.execute(text("""
            UPDATE background_videos
            SET ml_tagging_status = 'pending',
                ml_tags = NULL,
                ml_tags_version = NULL,
                ml_tagging_error = NULL
            WHERE filter_status = 'approved'
              AND source_type = 'reddit'
              AND (ml_tags_version IS NULL OR ml_tags_version < 2)
        """))
        conn.commit()
        print(f"Marked {result.rowcount} reddit backgrounds for v2 re-tagging")


if __name__ == "__main__":
    run_migration()
