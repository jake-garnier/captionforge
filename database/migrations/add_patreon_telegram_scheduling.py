"""
Add scheduling fields to patreon_publish_jobs and telegram_publish_jobs.

Mirrors the Postpone backlog pattern: one post per niche per day, with the
dispatcher beat task promoting jobs from "scheduled" to "pending" when their
post time is within the dispatch window.

Columns added (idempotent):
- scheduled_date  TIMESTAMPTZ
- base_post_time  TIMESTAMPTZ

The default value of `status` is changed from 'pending' to 'scheduled' for new
rows. Existing rows are left untouched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import engine
from sqlalchemy import text


COLUMNS = [
    ("scheduled_date", "TIMESTAMP WITH TIME ZONE"),
    ("base_post_time", "TIMESTAMP WITH TIME ZONE"),
]
TABLES = ["patreon_publish_jobs", "telegram_publish_jobs"]


def _column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(text(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
        )
        """
    ), {"table": table, "column": column})
    return bool(result.scalar())


def run_migration():
    with engine.connect() as conn:
        for table in TABLES:
            for column, coltype in COLUMNS:
                if _column_exists(conn, table, column):
                    print(f"{table}.{column} already exists, skipping")
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
                print(f"Added {table}.{column}")

            # Index on scheduled_date for the beat-dispatcher query
            idx_name = f"idx_{table}_scheduled_date"
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}(scheduled_date)"
            ))

            # Update default for new rows. Existing rows keep their current status.
            conn.execute(text(
                f"ALTER TABLE {table} ALTER COLUMN status SET DEFAULT 'scheduled'"
            ))

        conn.commit()
        print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()
