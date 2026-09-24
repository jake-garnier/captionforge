"""
Migration: Add media_type and gallery_image_count fields to videos table
Supports image posts, galleries, and GIFs in addition to videos
"""
from sqlalchemy import create_engine, text
from config.settings import settings


def upgrade():
    """Add media_type and gallery_image_count columns to videos table"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Add media_type column with default 'video' for existing records
        conn.execute(text("""
            ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS media_type VARCHAR(20) DEFAULT 'video'
        """))

        # Add gallery_image_count column
        conn.execute(text("""
            ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS gallery_image_count INTEGER
        """))

        # Create index on media_type for filtering
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_videos_media_type ON videos (media_type)
        """))

        conn.commit()

    print("Added media_type and gallery_image_count columns to videos table")


def downgrade():
    """Remove media_type and gallery_image_count columns"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ix_videos_media_type"))
        conn.execute(text("ALTER TABLE videos DROP COLUMN IF EXISTS media_type"))
        conn.execute(text("ALTER TABLE videos DROP COLUMN IF EXISTS gallery_image_count"))
        conn.commit()

    print("Removed media_type and gallery_image_count columns from videos table")


if __name__ == "__main__":
    print("Running migration: add_media_type")
    upgrade()
