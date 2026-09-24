"""
Recompose approved composed videos with action-tag matched backgrounds.

Ensures each approved video gets a UNIQUE background — no reuse within
the approved set.
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import get_db_context
from database.models import GeneratedCaption, ComposedVideo, BackgroundVideo
from utils.generation_postprocessor import extract_tags
from video_generator import VideoComposer, TextChunker, TimingEngine
from config.niche_rules import is_subreddit_compatible


def find_best_background(db, caption_text, caption_tags, niche, used_bg_ids):
    """Find the best action-tag-matched background, excluding already-used ones."""
    chunker = TextChunker()
    timing = TimingEngine()
    chunks = chunker.chunk(caption_text)
    caption_duration = timing.get_total_duration(chunks)
    caption_action_set = set(caption_tags) if caption_tags else set()

    query = db.query(BackgroundVideo).filter(
        BackgroundVideo.filter_status == 'approved',
        BackgroundVideo.storage_path.isnot(None),
        BackgroundVideo.source_type == 'reddit',
        BackgroundVideo.ml_tagging_status == 'completed',
        BackgroundVideo.duration_seconds.isnot(None),
    )

    # Exclude all already-used backgrounds
    if used_bg_ids:
        query = query.filter(~BackgroundVideo.id.in_(list(used_bg_ids)))

    backgrounds = query.all()

    best = None
    best_score = -999

    for bg in backgrounds:
        if not bg.storage_path or not os.path.exists(bg.storage_path):
            continue
        if not bg.duration_seconds or bg.duration_seconds < caption_duration:
            continue

        if niche and not is_subreddit_compatible(bg.searched_tag, niche):
            continue

        score = min(bg.views / 10000, 10) if bg.views else 0

        if caption_action_set and bg.ml_tags:
            bg_actions = set(bg.ml_tags.get('activities', []))
            if bg_actions:
                overlap = caption_action_set & bg_actions
                if overlap:
                    score += len(overlap) * 5
                else:
                    score -= 3

        if score > best_score:
            best_score = score
            best = bg

    return best, best_score


def run():
    with get_db_context() as db:
        approved = db.query(ComposedVideo).filter(
            ComposedVideo.approval_status == 'approved',
        ).all()

        print(f"Found {len(approved)} approved composed videos to recompose")

        composer = VideoComposer(output_dir="/data/composed_videos")
        recomposed = 0
        skipped = 0
        errors = 0

        # Track used backgrounds across ALL approved videos to ensure uniqueness
        used_bg_ids = set()

        for cv in approved:
            caption = db.query(GeneratedCaption).filter_by(id=cv.generated_caption_id).first()
            if not caption or not caption.caption_text:
                print(f"  [{cv.id}] SKIP: no caption found")
                skipped += 1
                continue

            caption_tags = caption.tags if caption.tags else extract_tags(caption.caption_text)
            niche = cv.niche or caption.niche

            # Find best background excluding all already-used ones
            new_bg, score = find_best_background(
                db, caption.caption_text, caption_tags, niche, used_bg_ids
            )

            if not new_bg:
                print(f"  [{cv.id}] SKIP: no unique compatible background found")
                skipped += 1
                continue

            bg_actions = new_bg.ml_tags.get('activities', []) if new_bg.ml_tags else []
            print(f"  [{cv.id}] caption_tags={caption_tags} -> BG {new_bg.id} (score={score:.1f}, actions={bg_actions})")

            try:
                filename = f"recomposed_{cv.id}_{new_bg.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
                result = composer.compose(
                    background_path=new_bg.storage_path,
                    caption=caption.caption_text,
                    output_filename=filename,
                )

                if not result.success:
                    print(f"  [{cv.id}] ERROR composing: {result.error}")
                    errors += 1
                    continue

                # Delete old file
                old_path = cv.storage_path
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)

                # Update record
                cv.background_video_id = new_bg.id
                cv.storage_path = result.output_path
                cv.duration_seconds = result.duration
                cv.caption_chunks = result.caption_chunks
                cv.file_size_bytes = os.path.getsize(result.output_path) if result.output_path else None
                cv.hosted_url = None
                cv.reddit_post_url = None
                cv.reddit_posted_at = None

                new_bg.last_used_at = datetime.utcnow()

                # Mark this background as used
                used_bg_ids.add(new_bg.id)

                db.commit()
                recomposed += 1
                print(f"  [{cv.id}] OK: {result.output_path} ({result.duration:.1f}s)")

            except Exception as e:
                print(f"  [{cv.id}] ERROR: {e}")
                errors += 1
                db.rollback()

        print(f"\nDone: {recomposed} recomposed, {skipped} skipped, {errors} errors")
        print(f"Unique backgrounds used: {len(used_bg_ids)}")


if __name__ == '__main__':
    run()
