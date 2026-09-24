"""
Migration: Add scraping progress tracking table
Run this to create the scraping_progress table
"""
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, Boolean
from sqlalchemy.sql import func
from database.models import Base
from config.settings import settings


def upgrade():
    """Add scraping_progress table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    # Create scraping_progress table
    scraping_progress = Table(
        'scraping_progress',
        metadata,
        Column('id', Integer, primary_key=True, index=True),
        Column('subreddit', String(100), nullable=False, unique=True, index=True),
        Column('last_post_id', String(50), nullable=True),  # Last successfully scraped post ID
        Column('last_post_score', Integer, nullable=True),  # Score of last scraped post
        Column('posts_scraped', Integer, default=0),  # Total posts scraped
        Column('target_min_score', Integer, default=300),  # Stop scraping below this score
        Column('scraping_active', Boolean, default=True),  # Whether to continue scraping
        Column('last_scrape_at', TIMESTAMP(timezone=True), nullable=True),
        Column('created_at', TIMESTAMP(timezone=True), server_default=func.now()),
        Column('updated_at', TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
    )

    metadata.create_all(engine)
    print("✅ Created scraping_progress table")


def downgrade():
    """Remove scraping_progress table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    scraping_progress = Table('scraping_progress', metadata)
    scraping_progress.drop(engine)
    print("✅ Dropped scraping_progress table")


if __name__ == "__main__":
    print("Running migration: add_scraping_progress")
    upgrade()
