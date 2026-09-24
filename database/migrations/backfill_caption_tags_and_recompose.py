"""
Backfill action tags on approved generated captions and clear pending
composed videos for recomposition with action-tag matching.

1. Tags all approved generated_captions using the unified extract_tags()
2. Deletes pending (unapproved, unpublished) composed videos + their files
   so backgrounds become available for recomposition with better matching
3. Leaves approved/published composed videos untouched
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import get_db_context
from database.models import GeneratedCaption, ComposedVideo
from utils.generation_postprocessor import extract_tags
import json


def run():
    with get_db_context() as db:
        # === Step 1: Backfill tags on approved captions ===
        captions = db.query(GeneratedCaption).filter(
            GeneratedCaption.status == 'approved',
        ).all()

        tagged = 0
        already_tagged = 0
        for c in captions:
            if c.tags and json.dumps(c.tags) not in ('[]', 'null', 'None'):
                already_tagged += 1
                # Re-tag with new unified tags anyway (old tags used different vocabulary)

            if c.caption_text:
                new_tags = extract_tags(c.caption_text)
                c.tags = new_tags
                tagged += 1

        db.commit()
        print(f"Step 1: Tagged {tagged} approved captions ({already_tagged} had old tags, all re-tagged with unified set)")

        # === Step 2: Delete pending composed videos ===
        pending_composed = db.query(ComposedVideo).filter(
            ComposedVideo.approval_status != 'approved',
            ComposedVideo.is_published == False,
        ).all()

        deleted_files = 0
        deleted_records = 0
        for cv in pending_composed:
            # Delete the video file if it exists
            if cv.storage_path and os.path.exists(cv.storage_path):
                try:
                    os.remove(cv.storage_path)
                    deleted_files += 1
                except Exception as e:
                    print(f"  Warning: Could not delete {cv.storage_path}: {e}")

            db.delete(cv)
            deleted_records += 1

        db.commit()
        print(f"Step 2: Deleted {deleted_records} pending composed videos ({deleted_files} files removed)")

        # === Summary ===
        remaining = db.query(ComposedVideo).count()
        available_captions = db.query(GeneratedCaption).filter(
            GeneratedCaption.status == 'approved',
        ).count()
        print(f"\nSummary:")
        print(f"  Approved captions ready for composition: {available_captions}")
        print(f"  Remaining composed videos (approved/published): {remaining}")
        print(f"  Captions available for new composition: {available_captions - remaining}")


if __name__ == '__main__':
    run()
