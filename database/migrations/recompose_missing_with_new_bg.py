"""
Recover composed videos whose recompose failed because the assigned BG
is too short for the (edited) caption. Picks a NEW background that fits.
"""
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, "/app")

from database.db import get_db_context
from database.models import ComposedVideo, BackgroundVideo, GeneratedCaption
from config.niche_rules import is_subreddit_compatible
from video_generator import TextChunker, TimingEngine
from tasks.video_composition_tasks import create_composer_with_random_style


def run(video_ids):
    chunker = TextChunker()
    timing = TimingEngine()
    successes = 0
    failures = 0

    with get_db_context() as db:
        used_bg_ids = set(
            r[0] for r in db.query(ComposedVideo.background_video_id).filter(
                ComposedVideo.background_video_id.isnot(None),
                ComposedVideo.status == "completed",
            ).distinct().all()
        )

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

            chunks = chunker.chunk(caption_text)
            required_duration = timing.get_total_duration(chunks) if chunks else 0
            niche = cv.niche or "motivation"

            candidates = db.query(BackgroundVideo).filter(
                BackgroundVideo.filter_status == "approved",
                BackgroundVideo.storage_path.isnot(None),
                BackgroundVideo.source_type == "reddit",
                BackgroundVideo.duration_seconds >= required_duration,
                ~BackgroundVideo.id.in_(used_bg_ids),
            ).all()

            scored = []
            for bg in candidates:
                if not bg.storage_path or not os.path.exists(bg.storage_path):
                    continue
                if is_subreddit_compatible(bg.searched_tag, niche):
                    score = min(bg.views / 10000, 10) if bg.views else 0
                    scored.append((bg, score))

            if not scored:
                print(f"  CV {vid}: no compatible BG long enough for {required_duration:.1f}s caption")
                failures += 1
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            new_bg = random.choice(scored[:10])[0]

            try:
                composer, style_name = create_composer_with_random_style()
                ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                result = composer.compose(
                    background_path=new_bg.storage_path,
                    caption=caption_text,
                    output_filename=f"recovered_newbg_{vid}_{new_bg.id}_{ts}.mp4",
                )
                if not result.success:
                    print(f"  CV {vid}: compose failed: {result.error}")
                    failures += 1
                    continue

                cv.background_video_id = new_bg.id
                cv.storage_path = result.output_path
                cv.duration_seconds = result.duration
                cv.caption_chunks = result.caption_chunks
                cv.resolution = f"{new_bg.width}x{new_bg.height}" if new_bg.width else None
                cv.status = "completed"
                cv.is_published = False
                if result.output_path and os.path.exists(result.output_path):
                    cv.file_size_bytes = os.path.getsize(result.output_path)
                db.commit()

                used_bg_ids.add(new_bg.id)
                successes += 1
                print(f"  CV {vid}: OK  new bg={new_bg.id} (r/{new_bg.searched_tag}, {result.duration:.1f}s, style={style_name})")
            except Exception as e:
                print(f"  CV {vid}: exception {e}")
                failures += 1

    print(f"\nRecovery-with-new-bg complete: {successes} ok, {failures} failed")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    print(f"Recomposing {len(ids)} videos with new longer BGs: {ids}")
    run(ids)
