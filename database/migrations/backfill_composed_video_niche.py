"""
Backfill niche field across the data chain:
1. trained_models (from source_subreddits)
2. generated_captions (from trained_models via generation_jobs)
3. composed_videos (from generated_captions)
"""
import sys
sys.path.insert(0, '/app')

from sqlalchemy import create_engine, text
from config.settings import settings

def run_migration():
    """Backfill niche across the data chain."""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Step 1: Backfill trained_models from source_subreddits
        print("Backfilling trained_models...")
        result1 = conn.execute(text("""
            UPDATE trained_models SET niche = 'motivation'
            WHERE niche IS NULL AND source_subreddits::text ILIKE '%motivation%'
        """))
        result2 = conn.execute(text("""
            UPDATE trained_models SET niche = 'fitness'
            WHERE niche IS NULL AND source_subreddits::text ILIKE '%fitness%'
        """))
        result3 = conn.execute(text("""
            UPDATE trained_models SET niche = 'cooking'
            WHERE niche IS NULL AND source_subreddits::text ILIKE '%cooking%'
        """))
        result4 = conn.execute(text("""
            UPDATE trained_models SET niche = 'travel'
            WHERE niche IS NULL AND source_subreddits::text ILIKE '%travel%'
        """))
        print(f"  Updated trained_models: motivation={result1.rowcount}, fitness={result2.rowcount}, cooking={result3.rowcount}, travel={result4.rowcount}")

        # Step 2: Backfill generated_captions from trained_models
        print("Backfilling generated_captions...")
        result = conn.execute(text("""
            UPDATE generated_captions gc
            SET niche = tm.niche
            FROM generation_jobs gj
            JOIN trained_models tm ON gj.model_id = tm.id
            WHERE gc.generation_job_id = gj.id
              AND gc.niche IS NULL
              AND tm.niche IS NOT NULL
        """))
        print(f"  Updated generated_captions: {result.rowcount}")

        # Step 3: Backfill composed_videos from generated_captions
        print("Backfilling composed_videos...")
        result = conn.execute(text("""
            UPDATE composed_videos cv
            SET niche = gc.niche
            FROM generated_captions gc
            WHERE cv.generated_caption_id = gc.id
              AND cv.niche IS NULL
              AND gc.niche IS NOT NULL
        """))
        print(f"  Updated composed_videos: {result.rowcount}")

        conn.commit()

        # Check remaining nulls
        remaining = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM trained_models WHERE niche IS NULL) as tm_null,
                (SELECT COUNT(*) FROM generated_captions WHERE niche IS NULL) as gc_null,
                (SELECT COUNT(*) FROM composed_videos WHERE niche IS NULL) as cv_null
        """)).fetchone()
        print(f"\nRemaining NULL niche: trained_models={remaining[0]}, generated_captions={remaining[1]}, composed_videos={remaining[2]}")

if __name__ == "__main__":
    run_migration()
