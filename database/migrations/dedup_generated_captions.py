"""
Migration: collapse duplicate (niche, caption_text) rows in generated_captions
that were produced by the orchestrator's mirror loop before the dedup fix
in pipeline_orchestrator landed.

Typical situation this handles:
  a few hundred duplicate rows across a handful of (niche, text) clusters,
  some of them referenced by composed_videos rows.

Per-cluster strategy:
  - If any rows in the cluster are referenced by composed_videos.generated_caption_id,
    keep the one referenced by the most composed_videos rows (ties broken
    by lowest id). Delete the others.
  - Otherwise, keep the lowest-id row and delete the rest.

Idempotent — run again, find nothing to delete.

Run with: docker-compose exec api python database/migrations/dedup_generated_captions.py
"""
import sys

sys.path.insert(0, "/app")

from database.db import engine
from sqlalchemy import text


def run_migration():
    with engine.connect() as conn:
        # Find duplicate clusters: (niche, md5(caption_text)) with > 1 row
        clusters = conn.execute(text("""
            SELECT niche, md5(caption_text) AS text_hash, COUNT(*) AS n
            FROM generated_captions
            GROUP BY niche, md5(caption_text)
            HAVING COUNT(*) > 1
            ORDER BY n DESC
        """)).all()

        if not clusters:
            print("No duplicate (niche, caption_text) clusters found. Nothing to do.")
            return

        print(f"Found {len(clusters)} duplicate clusters totaling {sum(c.n for c in clusters)} rows")

        total_deleted = 0
        for cluster in clusters:
            # Get all ids in this cluster, with composed_videos ref count per row
            rows = conn.execute(text("""
                SELECT
                  gc.id,
                  COALESCE(cv_count.n, 0) AS composed_refs
                FROM generated_captions gc
                LEFT JOIN (
                  SELECT generated_caption_id, COUNT(*) AS n
                  FROM composed_videos
                  WHERE generated_caption_id IS NOT NULL
                  GROUP BY 1
                ) cv_count ON cv_count.generated_caption_id = gc.id
                WHERE gc.niche = :niche
                  AND md5(gc.caption_text) = :text_hash
                ORDER BY composed_refs DESC, gc.id ASC
            """), {"niche": cluster.niche, "text_hash": cluster.text_hash}).all()

            keep_id = rows[0].id  # most composed-refs wins, ties broken by lowest id
            delete_ids = [r.id for r in rows[1:]]

            if not delete_ids:
                continue

            # Sanity: don't delete a row that has any composed_videos refs.
            # If the keeper isn't the only one with refs, that's a complex
            # case — refuse to touch it and log so the operator can decide.
            others_with_refs = [r.id for r in rows[1:] if r.composed_refs > 0]
            if others_with_refs:
                print(
                    f"  cluster niche={cluster.niche} hash={cluster.text_hash[:8]} "
                    f"SKIPPED: keeper={keep_id} has refs but so do {others_with_refs}; "
                    f"manual review needed"
                )
                continue

            res = conn.execute(text("""
                DELETE FROM generated_captions WHERE id = ANY(:ids)
            """), {"ids": delete_ids})
            print(
                f"  cluster niche={cluster.niche} hash={cluster.text_hash[:8]}: "
                f"kept id={keep_id} (composed_refs={rows[0].composed_refs}), "
                f"deleted {res.rowcount} dups"
            )
            total_deleted += res.rowcount

        conn.commit()
        print(f"Done. Deleted {total_deleted} duplicate rows.")


if __name__ == "__main__":
    run_migration()
