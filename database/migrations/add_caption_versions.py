"""
Migration script to add raw_ocr_text, rule_based_text, and llm_refined_text columns
to the scraped_captions table.

Run this script once to update the existing database schema.
"""
from sqlalchemy import text
from database.db import engine


def upgrade():
    """Add new columns for storing all 3 caption processing stages"""
    with engine.connect() as conn:
        # Add raw_ocr_text column
        conn.execute(text("""
            ALTER TABLE scraped_captions
            ADD COLUMN IF NOT EXISTS raw_ocr_text TEXT
        """))

        # Add rule_based_text column
        conn.execute(text("""
            ALTER TABLE scraped_captions
            ADD COLUMN IF NOT EXISTS rule_based_text TEXT
        """))

        # Add llm_refined_text column
        conn.execute(text("""
            ALTER TABLE scraped_captions
            ADD COLUMN IF NOT EXISTS llm_refined_text TEXT
        """))

        conn.commit()
        print("✓ Successfully added raw_ocr_text, rule_based_text, and llm_refined_text columns")


def downgrade():
    """Remove the new columns (rollback)"""
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE scraped_captions
            DROP COLUMN IF EXISTS raw_ocr_text,
            DROP COLUMN IF EXISTS rule_based_text,
            DROP COLUMN IF EXISTS llm_refined_text
        """))

        conn.commit()
        print("✓ Successfully removed caption version columns")


if __name__ == "__main__":
    print("Running migration: add_caption_versions")
    upgrade()
    print("Migration complete!")
