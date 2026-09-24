"""
Migration: Add upvote tracking to videos table
Run this to add upvotes and last_upvote_check columns
"""
from sqlalchemy import create_engine, Column, Integer, TIMESTAMP
from sqlalchemy.sql import func
from config.settings import settings


def upgrade():
    """Add upvote tracking columns to videos table"""
    from sqlalchemy import text

    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Add upvotes column (nullable initially for existing records)
        conn.execute(text("""
            ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS upvotes INTEGER DEFAULT 0
        """))

        # Add last_upvote_check column to track when we last updated upvotes
        conn.execute(text("""
            ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS last_upvote_check TIMESTAMP WITH TIME ZONE
        """))

        # Add index on upvotes for sorting
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_videos_upvotes ON videos(upvotes DESC)
        """))

        conn.commit()

    print("✅ Added upvotes tracking columns to videos table")


def downgrade():
    """Remove upvote tracking columns"""
    from sqlalchemy import text

    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_videos_upvotes"))
        conn.execute(text("ALTER TABLE videos DROP COLUMN IF EXISTS upvotes"))
        conn.execute(text("ALTER TABLE videos DROP COLUMN IF EXISTS last_upvote_check"))
        conn.commit()

    print("✅ Removed upvote tracking columns from videos table")


if __name__ == "__main__":
    print("Running migration: add_upvotes_tracking")
    upgrade()
