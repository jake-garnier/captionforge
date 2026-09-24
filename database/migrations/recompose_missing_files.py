"""
Recover composed videos whose files are missing on disk.

A previous recompose flow deleted the old file before queueing the new
render, then the queued render task failed (GPU queue was backed up).
The DB still references the old path; the disk file is gone.

This script re-renders each affected video synchronously using its
already-assigned background. Runs in-process so it doesn't depend on
the celery GPU queue.

Usage:
  docker exec captions-api python database/migrations/recompose_missing_files.py 2837,2840,...
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, "/app")

from database.db import get_db_context
from database.models import ComposedVideo, BackgroundVideo, GeneratedCaption
from tasks.video_composition_tasks import create_composer_with_random_style


def run(video_ids):
    successes = 0
    failures = 0
    for vid in video_ids:
        with get_db_context() as db:
            cv = db.query(ComposedVideo).filter_by(id=vid).first()
            if not cv:
                print(f"  CV {vid}: not found, skipping")
                failures += 1
                continue

            caption_text = cv.edited_caption_text
            if not caption_text and cv.generated_caption_id:
                gc = db.query(GeneratedCaption).filter_by(id=cv.generated_caption_id).first()
                if gc:
                    caption_text = gc.caption_text
            if not caption_text:
                print(f"  CV {vid}: no caption text, skipping")
                failures += 1
                continue

            bg = db.query(BackgroundVideo).filter_by(id=cv.background_video_id).first()
            if not bg or not bg.storage_path or not os.path.exists(bg.storage_path):
                print(f"  CV {vid}: BG {cv.background_video_id} missing, skipping")
                failures += 1
                continue

            try:
                composer, style_name = create_composer_with_random_style()
                ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                result = composer.compose(
                    background_path=bg.storage_path,
                    caption=caption_text,
                    output_filename=f"recovered_{vid}_{bg.id}_{ts}.mp4",
                )
                if not result.success:
                    print(f"  CV {vid}: compose failed: {result.error}")
                    failures += 1
                    continue

                cv.storage_path = result.output_path
                cv.duration_seconds = result.duration
                cv.caption_chunks = result.caption_chunks
                cv.resolution = f"{bg.width}x{bg.height}" if bg.width else None
                cv.status = "completed"
                if result.output_path and os.path.exists(result.output_path):
                    cv.file_size_bytes = os.path.getsize(result.output_path)
                db.commit()

                successes += 1
                print(f"  CV {vid}: OK  bg={bg.id} (r/{bg.searched_tag}, {result.duration:.1f}s, style={style_name})")
            except Exception as e:
                print(f"  CV {vid}: exception {e}")
                failures += 1

    print(f"\nRecovery complete: {successes} ok, {failures} failed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python recompose_missing_files.py <comma-separated-ids>")
        sys.exit(1)
    ids = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    print(f"Recovering {len(ids)} composed videos: {ids}")
    run(ids)
