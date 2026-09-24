#!/usr/bin/env python3
"""Add workflows table for browser automation recordings."""

import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from database.db import engine


def run_migration():
    """Create workflows table for storing recorded browser automations."""

    with engine.connect() as conn:
        # Create workflows table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS workflows (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) UNIQUE NOT NULL,
                description TEXT,
                target_site VARCHAR(100),
                actions JSONB NOT NULL DEFAULT '[]'::jsonb,
                variables JSONB DEFAULT '[]'::jsonb,
                last_run_at TIMESTAMP WITH TIME ZONE,
                run_count INTEGER DEFAULT 0,
                last_run_status VARCHAR(50),
                last_run_error TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("Created workflows table")

        # Create index on name for lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_workflows_name
            ON workflows(name)
        """))
        print("Created index on name")

        # Create index on target_site for filtering
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_workflows_target_site
            ON workflows(target_site)
        """))
        print("Created index on target_site")

        conn.commit()
        print("\nMigration completed successfully!")


if __name__ == "__main__":
    run_migration()
