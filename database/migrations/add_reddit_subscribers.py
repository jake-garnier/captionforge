"""
Migration: add subscribers column to reddit_account_snapshots.

Reddit's public JSON API returns subscribers=0 for everyone. Real
follower counts have to be scraped from the rendered profile page
(see scrapers/reddit_profile_scraper.py). This column holds that
value per snapshot.

Safe to run multiple times.
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from database.db import engine


def run_migration():
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE reddit_account_snapshots
            ADD COLUMN IF NOT EXISTS subscribers INTEGER
        """))
        conn.commit()
    print("Added reddit_account_snapshots.subscribers column.")


if __name__ == "__main__":
    run_migration()
