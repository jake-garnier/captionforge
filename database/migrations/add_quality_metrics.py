"""
Migration: Add quality metrics columns to scraped_captions table

Run with: docker-compose exec api python database/migrations/add_quality_metrics.py
"""
import sys
sys.path.insert(0, '/app')

from database.db import engine
from sqlalchemy import text


def run_migration():
    """Add quality metrics columns to scraped_captions table."""

    with engine.connect() as conn:
        # Add compression_ratio column
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='scraped_captions' AND column_name='compression_ratio'
                ) THEN
                    ALTER TABLE scraped_captions
                    ADD COLUMN compression_ratio FLOAT;
                END IF;
            END $$;
        """))

        # Add slide_count_raw column
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='scraped_captions' AND column_name='slide_count_raw'
                ) THEN
                    ALTER TABLE scraped_captions
                    ADD COLUMN slide_count_raw INTEGER;
                END IF;
            END $$;
        """))

        # Add slide_count_final column
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='scraped_captions' AND column_name='slide_count_final'
                ) THEN
                    ALTER TABLE scraped_captions
                    ADD COLUMN slide_count_final INTEGER;
                END IF;
            END $$;
        """))

        # Add unique_word_ratio column
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='scraped_captions' AND column_name='unique_word_ratio'
                ) THEN
                    ALTER TABLE scraped_captions
                    ADD COLUMN unique_word_ratio FLOAT;
                END IF;
            END $$;
        """))

        # Add extraction_metadata column (JSONB for PostgreSQL)
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='scraped_captions' AND column_name='extraction_metadata'
                ) THEN
                    ALTER TABLE scraped_captions
                    ADD COLUMN extraction_metadata JSONB;
                END IF;
            END $$;
        """))

        conn.commit()

    print("Migration complete: quality metrics columns added to scraped_captions")


if __name__ == "__main__":
    run_migration()
