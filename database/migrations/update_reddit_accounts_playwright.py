"""
Migration: Update reddit_accounts for Playwright-based authentication.

Removes PRAW API fields (password, client_id, client_secret, user_agent)
and adds Playwright session fields (cookies_path, is_logged_in, last_login_at).

Run with: docker-compose exec api python database/migrations/update_reddit_accounts_playwright.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from database.db import engine


def run_migration():
    """Run the migration."""
    with engine.connect() as conn:
        # Add new Playwright fields
        print("Adding Playwright session fields...")

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ADD COLUMN IF NOT EXISTS cookies_path VARCHAR(500);
        """))
        print("Added cookies_path column")

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ADD COLUMN IF NOT EXISTS is_logged_in BOOLEAN DEFAULT FALSE;
        """))
        print("Added is_logged_in column")

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP WITH TIME ZONE;
        """))
        print("Added last_login_at column")

        # Make PRAW fields nullable (for backwards compatibility)
        # We don't drop them in case there's data we might need
        print("Making PRAW fields nullable...")

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ALTER COLUMN password DROP NOT NULL;
        """))

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ALTER COLUMN client_id DROP NOT NULL;
        """))

        conn.execute(text("""
            ALTER TABLE reddit_accounts
            ALTER COLUMN client_secret DROP NOT NULL;
        """))

        print("Made password, client_id, client_secret nullable")

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
