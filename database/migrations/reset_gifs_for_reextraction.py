"""
Reset GIFs with no_caption_found status for re-extraction.

Run this migration after deploying GIF extraction support.
GIFs were previously only extracting the first frame, missing text on later frames.

Usage:
    docker-compose exec api python database/migrations/reset_gifs_for_reextraction.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import get_db_context
from database.models import Video
from utils.extraction_queue import publish_video_downloaded


def run_migration():
    """Reset GIFs with no_caption_found to downloaded status and queue for extraction."""
    with get_db_context() as db:
        # Find all GIFs with no_caption_found status
        gifs = db.query(Video).filter(
            Video.media_type == 'gif',
            Video.processing_status == 'no_caption_found'
        ).all()

        print(f"Found {len(gifs)} GIFs to reset for re-extraction")

        if not gifs:
            print("No GIFs to reset")
            return

        reset_count = 0
        queued_count = 0

        for gif in gifs:
            # Reset status to downloaded (makes it eligible for extraction)
            gif.processing_status = 'downloaded'
            gif.gallery_added_at = None  # Remove from gallery until re-extracted
            reset_count += 1

            # Queue for extraction
            try:
                publish_video_downloaded(
                    video_id=gif.id,
                    metadata={
                        'media_type': 'gif',
                        'subreddit': gif.source_subreddit,
                        'post_id': gif.source_post_id,
                        'reason': 'gif_reextraction'
                    }
                )
                queued_count += 1
            except Exception as e:
                print(f"  Warning: Could not queue GIF {gif.id}: {e}")

        db.commit()
        print(f"✓ Reset {reset_count} GIFs to 'downloaded' status")
        print(f"✓ Queued {queued_count} GIFs for extraction")
        print("\nGIFs will be processed by the extraction dispatcher.")


if __name__ == "__main__":
    run_migration()
