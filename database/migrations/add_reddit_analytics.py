"""
Migration: add reddit analytics tables.

Adds three tables for the Analytics tab:
  - reddit_account_snapshots: per-account karma over time
  - reddit_posts: every Reddit post we've observed for tracked accounts
  - reddit_post_stats: per-post score/comments/ratio snapshots

Auto-discovers which accounts to track via config.automation_config
NICHE_CONFIGS[*].postpone_reddit_username at runtime, so this
migration creates no seed data.

Safe to run multiple times.

Run with:
  docker-compose exec api python database/migrations/add_reddit_analytics.py
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text

from database.db import engine


def run_migration():
    with engine.connect() as conn:
        # 1. reddit_account_snapshots — karma over time, one row per fetch
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reddit_account_snapshots (
                id BIGSERIAL PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                total_karma INTEGER,
                link_karma INTEGER,
                comment_karma INTEGER,
                awardee_karma INTEGER,
                awarder_karma INTEGER,
                is_suspended BOOLEAN NOT NULL DEFAULT FALSE,
                account_created_utc TIMESTAMPTZ,
                verified_email BOOLEAN,
                raw_about JSONB
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_account_snapshots_user_time
            ON reddit_account_snapshots (username, fetched_at DESC)
        """))

        # 2. reddit_posts — one row per (Reddit post id) ever observed.
        # composed_video_id is set when we can match the link to a hosted
        # video URL we generated. Nullable so external/manual posts also fit.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id BIGSERIAL PRIMARY KEY,
                reddit_post_id VARCHAR(40) NOT NULL UNIQUE,
                username VARCHAR(100) NOT NULL,
                subreddit VARCHAR(100) NOT NULL,
                title TEXT NOT NULL,
                link_url TEXT,
                permalink TEXT NOT NULL,
                created_utc TIMESTAMPTZ,
                is_video BOOLEAN DEFAULT FALSE,
                composed_video_id INTEGER REFERENCES composed_videos(id) ON DELETE SET NULL,
                first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                removed_at TIMESTAMPTZ,
                removed_reason VARCHAR(80)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_posts_username
            ON reddit_posts (username)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_posts_subreddit
            ON reddit_posts (subreddit)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_posts_composed_video
            ON reddit_posts (composed_video_id)
            WHERE composed_video_id IS NOT NULL
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_posts_user_created
            ON reddit_posts (username, created_utc DESC)
        """))

        # 3. reddit_post_stats — score/comments/ratio trajectory per post.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reddit_post_stats (
                id BIGSERIAL PRIMARY KEY,
                reddit_post_id VARCHAR(40) NOT NULL REFERENCES reddit_posts(reddit_post_id) ON DELETE CASCADE,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                score INTEGER,
                num_comments INTEGER,
                upvote_ratio REAL
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_reddit_post_stats_post_time
            ON reddit_post_stats (reddit_post_id, fetched_at DESC)
        """))

        conn.commit()

    print("Reddit analytics migration complete.")
    print("Tables created: reddit_account_snapshots, reddit_posts, reddit_post_stats")
    print("Indexes created.")


if __name__ == "__main__":
    run_migration()
