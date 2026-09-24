"""
Migration: add LLM-judge columns to generated_captions.

Stage 3 of the quality pipeline (README, "What it does"). The judge is Mistral-Small-24B
(same model already loaded for generation) running with a scoring prompt.
Each caption gets grammar/flow/bg_consistency/appeal scores 1-10, an
overall score, an issues list, and a pass flag (all axes >= 7).

Safe to run multiple times.

Run with:
  docker-compose exec api python database/migrations/add_caption_judge.py
"""
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text

from database.db import engine


def run_migration():
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN IF NOT EXISTS judge_scores JSONB
        """))
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN IF NOT EXISTS judge_issues JSONB
        """))
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN IF NOT EXISTS judge_pass BOOLEAN
        """))
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN IF NOT EXISTS judge_status VARCHAR(20)
        """))
        conn.execute(text("""
            ALTER TABLE generated_captions
            ADD COLUMN IF NOT EXISTS judge_at TIMESTAMP WITH TIME ZONE
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_generated_captions_judge_status
            ON generated_captions (judge_status)
        """))
        conn.commit()
        print("Added judge columns to generated_captions")


if __name__ == "__main__":
    run_migration()
