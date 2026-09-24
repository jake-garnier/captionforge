#!/usr/bin/env python3
"""Add Patreon→Telegram membership sync tables.

Creates:
  - patreon_subscribers: source of truth for (niche, patreon_user) → claimed @username + telegram state
  - telegram_join_requests: persisted state for incoming chat_join_request updates
  - sync_alerts: surface for the Membership tab "Issues" view

See docs/patreon_telegram_sync_plan.md for the full design.
"""

import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from database.db import engine


def run_migration():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patreon_subscribers (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50) NOT NULL,
                patreon_user_id VARCHAR(50) NOT NULL,
                patreon_member_id VARCHAR(50),
                full_name VARCHAR(200),
                email VARCHAR(200),
                patron_status VARCHAR(50),
                pledge_relationship_start TIMESTAMPTZ,
                last_charge_status VARCHAR(50),
                last_charge_date TIMESTAMPTZ,
                claimed_telegram_username VARCHAR(100),
                claimed_at TIMESTAMPTZ,
                comment_id VARCHAR(100),
                comment_last_modified TIMESTAMPTZ,
                telegram_user_id BIGINT,
                telegram_state VARCHAR(30) DEFAULT 'none' NOT NULL,
                last_synced_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_patreon_subscribers_niche_user UNIQUE (niche, patreon_user_id)
            )
        """))
        print("✓ Created patreon_subscribers table")

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_patreon_subscribers_niche
            ON patreon_subscribers(niche)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_patreon_subscribers_status
            ON patreon_subscribers(patron_status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_patreon_subscribers_claimed_username
            ON patreon_subscribers(LOWER(claimed_telegram_username))
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_patreon_subscribers_telegram_state
            ON patreon_subscribers(telegram_state)
        """))
        print("✓ Created patreon_subscribers indexes")

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telegram_join_requests (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50) NOT NULL,
                chat_id VARCHAR(50) NOT NULL,
                telegram_user_id BIGINT NOT NULL,
                telegram_username VARCHAR(100),
                first_name VARCHAR(200),
                last_name VARCHAR(200),
                status VARCHAR(30) DEFAULT 'pending' NOT NULL,
                received_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
                resolved_at TIMESTAMPTZ,
                matched_patreon_user_id VARCHAR(50),
                decline_reason VARCHAR(200),
                CONSTRAINT uq_join_requests_chat_user UNIQUE (chat_id, telegram_user_id)
            )
        """))
        print("✓ Created telegram_join_requests table")

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_join_requests_niche_status
            ON telegram_join_requests(niche, status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_join_requests_username
            ON telegram_join_requests(LOWER(telegram_username))
        """))
        print("✓ Created telegram_join_requests indexes")

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sync_alerts (
                id SERIAL PRIMARY KEY,
                niche VARCHAR(50),
                severity VARCHAR(20) DEFAULT 'warning' NOT NULL,
                category VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                context JSONB,
                status VARCHAR(20) DEFAULT 'open' NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
                resolved_at TIMESTAMPTZ
            )
        """))
        print("✓ Created sync_alerts table")

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_sync_alerts_status_created
            ON sync_alerts(status, created_at DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_sync_alerts_niche
            ON sync_alerts(niche)
        """))
        print("✓ Created sync_alerts indexes")

        conn.commit()
        print("\n✅ Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
