"""
Migration: add caption_candidates table for Stage 2 (BG-first generation).

Stage 2 of the quality pipeline (README, "What it does"). Each row is a caption generated
specifically for a particular background video — the model is grounded in
the BG's scene_description, subjects, activities, setting, mood and camera.
We generate K candidates per BG, Stage-3 judge each one inline, and the
highest-overall judge-pass candidate becomes the "winner" used for
composition.

Safe to run multiple times.

Run with:
  docker-compose exec api python database/migrations/add_caption_candidates.py
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text

from database.db import engine


def run_migration():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS caption_candidates (
                id SERIAL PRIMARY KEY,
                background_video_id INTEGER NOT NULL REFERENCES background_videos(id),
                niche VARCHAR(50) NOT NULL,
                prompt_version VARCHAR(20) NOT NULL,
                caption_text TEXT NOT NULL,
                llm_model VARCHAR(100),
                generation_temperature FLOAT,
                judge_scores JSONB,
                judge_issues JSONB,
                judge_pass BOOLEAN,
                judge_overall INTEGER,
                judge_status VARCHAR(20),
                status VARCHAR(20) DEFAULT 'candidate',
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_caption_candidates_bg
            ON caption_candidates (background_video_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_caption_candidates_niche
            ON caption_candidates (niche)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_caption_candidates_status
            ON caption_candidates (status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_caption_candidates_judge_overall
            ON caption_candidates (judge_overall)
        """))
        conn.commit()
        print("Created caption_candidates table + indexes")


if __name__ == "__main__":
    run_migration()
