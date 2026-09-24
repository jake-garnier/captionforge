"""
Migration: seed empty per-niche credential rows for fitness/travel/cooking

Motivation already has rows in reddit_accounts, patreon_credentials, and telegram_bots.
The other three niches had nothing — which means the multi-niche rollout
had no place to attach creds when you wire them up via the UI.

This migration inserts placeholder rows with is_enabled=False so the
UI/forms have something to update. NOT NULL string columns get sentinel
values prefixed `PLACEHOLDER_` so they're easy to grep when filling in.

Idempotent — uses ON CONFLICT DO NOTHING on the unique niche index.

Run with: docker-compose exec api python database/migrations/seed_niche_credential_rows.py
"""
import sys

sys.path.insert(0, "/app")

from database.db import engine
from sqlalchemy import text

NICHES = ["fitness", "travel", "cooking"]


def run_migration():
    with engine.connect() as conn:
        for niche in NICHES:
            # reddit_accounts
            conn.execute(
                text("""
                    INSERT INTO reddit_accounts
                        (niche, username, is_enabled, is_logged_in)
                    VALUES
                        (:niche, :username, FALSE, FALSE)
                    ON CONFLICT (niche) DO NOTHING
                """),
                {
                    "niche": niche,
                    "username": f"PLACEHOLDER_{niche}_REDDIT_USERNAME",
                },
            )
            # patreon_credentials
            conn.execute(
                text("""
                    INSERT INTO patreon_credentials
                        (niche, email, is_configured)
                    VALUES
                        (:niche, :email, FALSE)
                    ON CONFLICT (niche) DO NOTHING
                """),
                {
                    "niche": niche,
                    "email": f"placeholder+{niche}@example.invalid",
                },
            )
            # telegram_bots
            conn.execute(
                text("""
                    INSERT INTO telegram_bots
                        (niche, bot_username, bot_token, is_enabled)
                    VALUES
                        (:niche, :bot_username, :bot_token, FALSE)
                    ON CONFLICT (niche) DO NOTHING
                """),
                {
                    "niche": niche,
                    "bot_username": f"PLACEHOLDER_{niche}_BOT_USERNAME",
                    "bot_token": f"PLACEHOLDER_{niche}_BOT_TOKEN",
                },
            )
        conn.commit()
    print(f"Seeded placeholder credential rows (or skipped existing) for: {NICHES}")
    print("Each is is_enabled=False / is_configured=False — populate via UI then flip the flags.")


if __name__ == "__main__":
    run_migration()
