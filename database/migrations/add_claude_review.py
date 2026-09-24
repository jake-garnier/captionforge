"""
Migration: add claude_review_* columns to composed_videos.

Stage 4 of the quality pipeline (README, "What it does"). Visual review is performed
by Claude Code in an interactive session (NOT via the Anthropic API).
The reviewer reads a brief, looks at 3 keyframes per video, and writes
a verdict. These columns hold the result.

Safe to run multiple times.

Run with:
  docker-compose exec api python database/migrations/add_claude_review.py
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text

from database.db import engine


def run_migration():
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_status VARCHAR(20)
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_scores JSONB
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_issues JSONB
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_verdict VARCHAR(20)
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_notes TEXT
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_model VARCHAR(50)
        """))
        conn.execute(text("""
            ALTER TABLE composed_videos
            ADD COLUMN IF NOT EXISTS claude_review_at TIMESTAMP WITH TIME ZONE
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_composed_videos_claude_review_status
            ON composed_videos (claude_review_status)
        """))
        conn.commit()
        print("Added claude_review_* columns to composed_videos")


if __name__ == "__main__":
    run_migration()
