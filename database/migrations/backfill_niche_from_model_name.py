"""
Migration: Backfill Niche Fields from Model Names

The niche tracking columns exist but were never populated for existing data.
This migration infers the niche from model names like 'motivation-lora-20251210-0016'.

Tables updated:
- trained_models
- training_jobs
- generated_captions
- generation_jobs
- video_composition_jobs
- composed_videos
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known niches to match against
NICHES = ['motivation', 'fitness', 'cooking', 'travel']


def run_migration():
    """Backfill niche columns based on model/job names."""
    with engine.connect() as conn:
        # ============================================================
        # 1. Backfill trained_models.niche from name
        # ============================================================
        logger.info("Backfilling trained_models.niche...")
        for niche in NICHES:
            result = conn.execute(text(f"""
                UPDATE trained_models
                SET niche = :niche
                WHERE niche IS NULL
                  AND (name ILIKE :pattern OR name ILIKE :pattern2)
            """), {
                "niche": niche,
                "pattern": f"{niche}-%",
                "pattern2": f"%-{niche}-%"
            })
            logger.info(f"  Updated {result.rowcount} models with niche='{niche}'")

        # ============================================================
        # 2. Backfill training_jobs.niche from job_name
        # ============================================================
        logger.info("Backfilling training_jobs.niche...")
        for niche in NICHES:
            result = conn.execute(text(f"""
                UPDATE training_jobs
                SET niche = :niche
                WHERE niche IS NULL
                  AND (job_name ILIKE :pattern OR job_name ILIKE :pattern2)
            """), {
                "niche": niche,
                "pattern": f"{niche}-%",
                "pattern2": f"%-{niche}-%"
            })
            logger.info(f"  Updated {result.rowcount} training jobs with niche='{niche}'")

        # ============================================================
        # 3. Backfill generated_captions.niche from llm_model name
        # ============================================================
        logger.info("Backfilling generated_captions.niche...")
        for niche in NICHES:
            result = conn.execute(text(f"""
                UPDATE generated_captions
                SET niche = :niche
                WHERE niche IS NULL
                  AND (llm_model ILIKE :pattern OR llm_model ILIKE :pattern2)
            """), {
                "niche": niche,
                "pattern": f"{niche}-%",
                "pattern2": f"%-{niche}-%"
            })
            logger.info(f"  Updated {result.rowcount} captions with niche='{niche}'")

        # ============================================================
        # 4. Backfill generation_jobs.niche from trained model
        # ============================================================
        logger.info("Backfilling generation_jobs.niche from linked model...")
        result = conn.execute(text("""
            UPDATE generation_jobs gj
            SET niche = tm.niche
            FROM trained_models tm
            WHERE gj.model_id = tm.id
              AND gj.niche IS NULL
              AND tm.niche IS NOT NULL
        """))
        logger.info(f"  Updated {result.rowcount} generation jobs from model niche")

        # Also try matching from job_name
        for niche in NICHES:
            result = conn.execute(text(f"""
                UPDATE generation_jobs
                SET niche = :niche
                WHERE niche IS NULL
                  AND (job_name ILIKE :pattern OR job_name ILIKE :pattern2)
            """), {
                "niche": niche,
                "pattern": f"{niche}-%",
                "pattern2": f"%-{niche}-%"
            })
            if result.rowcount > 0:
                logger.info(f"  Updated {result.rowcount} generation jobs with niche='{niche}' from job_name")

        # ============================================================
        # 5. Backfill video_composition_jobs.niche from job_name
        # ============================================================
        logger.info("Backfilling video_composition_jobs.niche...")
        for niche in NICHES:
            result = conn.execute(text(f"""
                UPDATE video_composition_jobs
                SET niche = :niche
                WHERE niche IS NULL
                  AND (job_name ILIKE :pattern OR job_name ILIKE :pattern2)
            """), {
                "niche": niche,
                "pattern": f"{niche}-%",
                "pattern2": f"%-{niche}-%"
            })
            if result.rowcount > 0:
                logger.info(f"  Updated {result.rowcount} composition jobs with niche='{niche}'")

        # ============================================================
        # 6. Backfill composed_videos.niche from generated caption
        # ============================================================
        logger.info("Backfilling composed_videos.niche from linked caption...")
        result = conn.execute(text("""
            UPDATE composed_videos cv
            SET niche = gc.niche
            FROM generated_captions gc
            WHERE cv.generated_caption_id = gc.id
              AND cv.niche IS NULL
              AND gc.niche IS NOT NULL
        """))
        logger.info(f"  Updated {result.rowcount} composed videos from caption niche")

        conn.commit()
        logger.info("Migration completed successfully!")


def verify_migration():
    """Show summary of niche distribution after migration."""
    with engine.connect() as conn:
        tables = [
            "trained_models",
            "training_jobs",
            "generated_captions",
            "generation_jobs",
            "video_composition_jobs",
            "composed_videos"
        ]

        for table in tables:
            try:
                result = conn.execute(text(f"""
                    SELECT niche, COUNT(*) as count
                    FROM {table}
                    GROUP BY niche
                    ORDER BY count DESC
                """))
                rows = result.fetchall()
                logger.info(f"\n{table}:")
                for row in rows:
                    niche = row[0] or "NULL"
                    count = row[1]
                    logger.info(f"  {niche}: {count}")
            except Exception as e:
                logger.warning(f"  Could not query {table}: {e}")


if __name__ == "__main__":
    logger.info("Running niche backfill migration...")
    run_migration()
    logger.info("\nVerifying migration results...")
    verify_migration()
