"""
Video Composition Celery Tasks

Tasks for composing caption videos from background videos + text.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Tuple

from .celery_app import celery_app
from database.db import get_db_context
from database.models import (
    VideoCompositionJob, ComposedVideo, GeneratedCaption,
    ScrapedCaption, BackgroundVideo, Video
)
from video_generator import (
    VideoComposer, TextChunker, TimingEngine, TextRenderer,
    get_random_style, get_random_style_different_from, StylePreset
)

logger = logging.getLogger(__name__)

# Output directory for composed videos
OUTPUT_DIR = "/data/composed_videos"


def create_composer_with_style(style: StylePreset, output_dir: str = OUTPUT_DIR) -> VideoComposer:
    """Create a VideoComposer with a specific style preset."""
    renderer = TextRenderer(
        font_path=style.font_path,
        font_size=style.font_size,
        text_color=style.text_color,
        outline_color=style.outline_color,
        outline_width=style.outline_width,
        shadow_color=style.shadow_color,
        shadow_offset=style.shadow_offset,
        glow_color=style.glow_color,
        glow_radius=style.glow_radius,
        padding_bottom=style.padding_bottom,
    )
    return VideoComposer(
        chunker=TextChunker(),
        timing=TimingEngine(),
        renderer=renderer,
        output_dir=output_dir
    )


def create_composer_with_random_style(output_dir: str = OUTPUT_DIR) -> tuple:
    """Create a VideoComposer with a random style preset.

    Returns:
        Tuple of (VideoComposer, style_name)
    """
    style = get_random_style()
    return create_composer_with_style(style, output_dir), style.name


@celery_app.task(bind=True, name='tasks.video_composition_tasks.run_composition_job', queue='composition')
def run_composition_job(self, job_id: int):
    """
    Run a batch video composition job.

    Matches captions with compatible background videos and composes them.
    """
    logger.info(f"Starting composition job {job_id}")

    with get_db_context() as db:
        job = db.query(VideoCompositionJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Composition job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        try:
            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            db.commit()

            # Initialize composer with random style for this job
            composer, style_name = create_composer_with_random_style()
            logger.info(f"Using style '{style_name}' for composition job {job_id}")

            # Get captions based on source
            captions = _get_captions(db, job)
            if not captions:
                job.status = "failed"
                job.error_message = "No captions found matching criteria"
                job.completed_at = datetime.utcnow()
                db.commit()
                return {"status": "error", "message": "No captions found"}

            logger.info(f"Found {len(captions)} captions for job {job_id}")

            # Get available background videos
            backgrounds = _get_background_videos(db, job)
            if not backgrounds:
                job.status = "failed"
                job.error_message = "No background videos available"
                job.completed_at = datetime.utcnow()
                db.commit()
                return {"status": "error", "message": "No backgrounds found"}

            logger.info(f"Found {len(backgrounds)} background videos for job {job_id}")

            # Track which backgrounds we've used to avoid duplicates
            used_backgrounds = set()
            composed_count = 0
            failed_count = 0
            skipped_count = 0

            # Process captions
            for caption_data in captions:
                if composed_count >= job.target_count:
                    break

                caption_id, caption_text, caption_type, caption_tags, caption_niche = caption_data

                # Find compatible background(s) - may return multiple for concatenation
                background_list, bg_warnings = _find_compatible_backgrounds(
                    composer, caption_text, backgrounds, used_backgrounds,
                    caption_tags, niche=caption_niche, allow_concatenation=True
                )

                if not background_list:
                    skipped_count += 1
                    logger.debug(f"No compatible background(s) for caption {caption_id}")
                    continue

                # Log warnings if any
                if bg_warnings:
                    logger.warning(
                        f"Caption {caption_id} background selection warnings: {bg_warnings}"
                    )

                # Compose the video (single or multi-background)
                try:
                    if len(background_list) == 1:
                        # Single background - use standard compose
                        background = background_list[0]
                        result = composer.compose(
                            background_path=background.storage_path,
                            caption=caption_text,
                            output_filename=f"composed_{job_id}_{caption_id}_{background.id}.mp4"
                        )
                        bg_ids = [background.id]
                        primary_bg = background
                    else:
                        # Multiple backgrounds - use compose_multi
                        bg_paths = [bg.storage_path for bg in background_list]
                        bg_ids = [bg.id for bg in background_list]
                        bg_id_str = "_".join(str(bid) for bid in bg_ids[:3])  # Limit filename length
                        result = composer.compose_multi(
                            background_paths=bg_paths,
                            caption=caption_text,
                            output_filename=f"composed_{job_id}_{caption_id}_{bg_id_str}.mp4"
                        )
                        primary_bg = background_list[0]  # Use first as primary for metadata
                        logger.info(f"Using {len(background_list)} concatenated backgrounds for caption {caption_id}")

                    if result.success:
                        # Create composed video record
                        composed = ComposedVideo(
                            composition_job_id=job.id,
                            generated_caption_id=caption_id if caption_type == "generated" else None,
                            scraped_caption_id=caption_id if caption_type == "scraped" else None,
                            background_video_id=primary_bg.id,  # Primary background
                            storage_path=result.output_path,
                            duration_seconds=result.duration,
                            caption_chunks=result.caption_chunks,
                            resolution=f"{primary_bg.width}x{primary_bg.height}",
                            status="completed",
                            niche=caption_niche
                        )

                        # Get file size
                        if result.output_path and os.path.exists(result.output_path):
                            composed.file_size_bytes = os.path.getsize(result.output_path)

                        db.add(composed)
                        composed_count += 1

                        # Mark all used backgrounds
                        for bg_id in bg_ids:
                            used_backgrounds.add(bg_id)

                        logger.info(f"Composed video {composed_count}/{job.target_count} for job {job_id}")
                    else:
                        failed_count += 1
                        if "too long" in (result.error or "").lower():
                            skipped_count += 1
                            failed_count -= 1
                        logger.warning(f"Composition failed for caption {caption_id}: {result.error}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error composing caption {caption_id}: {e}")

                # Update progress
                total_processed = composed_count + failed_count + skipped_count
                job.progress_percent = min(100.0, (total_processed / job.target_count) * 100)
                job.videos_composed = composed_count
                job.videos_failed = failed_count
                job.videos_skipped = skipped_count
                db.commit()

            # Finalize job
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress_percent = 100.0
            db.commit()

            logger.info(f"Composition job {job_id} completed: {composed_count} videos composed, {failed_count} failed, {skipped_count} skipped")

            return {
                "status": "completed",
                "videos_composed": composed_count,
                "videos_failed": failed_count,
                "videos_skipped": skipped_count
            }

        except Exception as e:
            logger.error(f"Composition job {job_id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.video_composition_tasks.compose_single_video_task', queue='composition')
def compose_single_video_task(
    self,
    caption_text: str,
    background_video_path: str,
    background_video_id: int,
    generated_caption_id: Optional[int] = None,
    scraped_caption_id: Optional[int] = None,
    composition_job_id: Optional[int] = None,
):
    """
    Compose a single video from caption text and background.
    """
    logger.info(f"Composing single video with background {background_video_id}")

    try:
        composer, style_name = create_composer_with_random_style()
        logger.info(f"Using style '{style_name}' for single composition")

        result = composer.compose(
            background_path=background_video_path,
            caption=caption_text,
            output_filename=f"single_{background_video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4"
        )

        if not result.success:
            logger.error(f"Single composition failed: {result.error}")
            return {"status": "error", "message": result.error}

        # Save to database
        with get_db_context() as db:
            bg = db.query(BackgroundVideo).filter_by(id=background_video_id).first()

            # Get niche from the caption
            niche = None
            if generated_caption_id:
                caption = db.query(GeneratedCaption).filter_by(id=generated_caption_id).first()
                if caption:
                    niche = caption.niche
            elif scraped_caption_id:
                caption = db.query(ScrapedCaption).filter_by(id=scraped_caption_id).first()
                if caption and caption.video and caption.video.subreddit:
                    from config.automation_config import get_automation_config
                    config = get_automation_config()
                    niche = config.get_niche_for_subreddit(caption.video.subreddit)

            composed = ComposedVideo(
                composition_job_id=composition_job_id,
                generated_caption_id=generated_caption_id,
                scraped_caption_id=scraped_caption_id,
                background_video_id=background_video_id,
                storage_path=result.output_path,
                duration_seconds=result.duration,
                caption_chunks=result.caption_chunks,
                resolution=f"{bg.width}x{bg.height}" if bg else None,
                status="completed",
                niche=niche
            )

            if result.output_path and os.path.exists(result.output_path):
                composed.file_size_bytes = os.path.getsize(result.output_path)

            db.add(composed)
            db.commit()

            logger.info(f"Single video composed: {result.output_path}")

            return {
                "status": "completed",
                "video_id": composed.id,
                "output_path": result.output_path,
                "duration": result.duration
            }

    except Exception as e:
        logger.error(f"Single composition error: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.video_composition_tasks.recompose_video_task', queue='composition')
def recompose_video_task(
    self,
    composed_video_id: int,
    caption_text: str,
    background_video_id: int,
    background_video_path: str
):
    """
    Recompose an existing video with a new background.

    Updates the existing ComposedVideo record with new background and output.
    """
    logger.info(f"Recomposing video {composed_video_id} with new background {background_video_id}")

    try:
        composer, style_name = create_composer_with_random_style()
        logger.info(f"Using style '{style_name}' for recomposition")

        result = composer.compose(
            background_path=background_video_path,
            caption=caption_text,
            output_filename=f"recomposed_{composed_video_id}_{background_video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4"
        )

        if not result.success:
            logger.error(f"Recomposition failed: {result.error}")
            return {"status": "error", "message": result.error}

        # Update the existing composed video record
        with get_db_context() as db:
            composed = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
            if not composed:
                logger.error(f"ComposedVideo {composed_video_id} not found")
                return {"status": "error", "message": "Composed video not found"}

            bg = db.query(BackgroundVideo).filter_by(id=background_video_id).first()

            # Capture old path BEFORE we overwrite storage_path so we can
            # delete it after the DB commit succeeds. This avoids the race
            # where deleting up-front leaves DB+disk out of sync if the
            # render later fails.
            old_path = composed.storage_path

            # Update the record with new composition
            composed.background_video_id = background_video_id
            composed.storage_path = result.output_path
            composed.duration_seconds = result.duration
            composed.caption_chunks = result.caption_chunks
            composed.resolution = f"{bg.width}x{bg.height}" if bg else None
            composed.status = "completed"
            composed.is_published = False  # Reset published status since video changed

            if result.output_path and os.path.exists(result.output_path):
                composed.file_size_bytes = os.path.getsize(result.output_path)

            db.commit()

            # Now that the new file is on disk and the DB points at it,
            # safely remove the old file. If this fails the worst case is
            # a stranded file (no DB row points to it) — much better than
            # the prior race that lost playable files.
            if old_path and old_path != result.output_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    logger.info(f"Deleted old composed file after successful recompose: {old_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old file {old_path}: {e}")

            logger.info(f"Video {composed_video_id} recomposed with new background: {result.output_path}")

            # Outro persistence: if the previous file was an outro-baked
            # version (filename starts with "outro_"), the user already
            # decided this video gets the CTA tail. Re-bake it here so a
            # caption-cleanup or BG-swap recompose doesn't silently strip
            # the outro. The append_outro path is idempotent for niches
            # without reddit_outro_text configured (returns 'skipped').
            old_was_outroed = bool(old_path and os.path.basename(old_path).startswith("outro_"))

        # Out of the with-block — DB commits done. If the old file was
        # outroed, queue the outro re-bake against the new file. Run
        # synchronously so the caller (and downstream upload tasks) sees
        # the final outroed file as storage_path.
        if old_was_outroed:
            logger.info(
                f"Video {composed_video_id} previously had an outro; re-baking after recompose"
            )
            outro_result = append_outro_task.apply(args=[composed_video_id]).get()
            if outro_result.get("status") not in ("completed", "skipped"):
                logger.warning(
                    f"Video {composed_video_id} outro re-bake failed after recompose: "
                    f"{outro_result.get('message')}"
                )

        return {
            "status": "completed",
            "video_id": composed_video_id,
            "output_path": result.output_path,
            "duration": result.duration,
            "new_background_id": background_video_id,
        }

    except Exception as e:
        logger.error(f"Recomposition error: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.video_composition_tasks.append_outro_task', queue='composition')
def append_outro_task(self, composed_video_id: int):
    """
    Append a per-niche CTA/watermark outro segment to a composed video.

    Used by the Reddit publish path so Reddit-bound videos carry a Patreon
    plug at the end. The outro text comes from NicheConfig.reddit_outro_text.
    If the niche has no outro text configured, this is a successful no-op
    (so the publish flow can call this unconditionally).

    The outro background uses leftover frames from the original BG (the
    footage that was trimmed off when the caption ended); if that's not
    enough, it pads with black. Replaces the composed video's storage_path
    with the new file and deletes the old one.

    Note: After this runs, ComposedVideo.duration_seconds covers caption +
    outro. That's intentional for the published artifact, but means later
    "recompose with same BG" decisions should still validate against the
    original BG's full length, not the composed video's duration. The
    recompose endpoint already does this correctly (it checks
    BackgroundVideo.duration_seconds, not ComposedVideo.duration_seconds).
    """
    from video_generator.composer import append_outro_to_video
    from config.automation_config import get_automation_config

    logger.info(f"Appending outro to composed video {composed_video_id}")

    try:
        with get_db_context() as db:
            composed = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
            if not composed:
                return {"status": "error", "message": "Composed video not found"}

            niche = composed.niche
            if not niche:
                return {"status": "skipped", "message": "Video has no niche; nothing to do"}

            # Look up the outro text for this niche.
            cfg = get_automation_config()
            niche_cfg = cfg._niches.get(niche)
            outro_text = niche_cfg.reddit_outro_text if niche_cfg else None
            if not outro_text:
                return {
                    "status": "skipped",
                    "message": f"No reddit_outro_text configured for niche '{niche}'",
                }

            composed_path = composed.storage_path
            if not composed_path or not os.path.exists(composed_path):
                return {"status": "error", "message": f"Composed video file missing: {composed_path}"}

            # Pull the original BG so we can use its leftover frames as the
            # outro backdrop. None is fine — the function falls back to black.
            bg_path: Optional[str] = None
            bg_offset = composed.duration_seconds or 0
            if composed.background_video_id:
                bg = db.query(BackgroundVideo).filter_by(id=composed.background_video_id).first()
                if bg and bg.storage_path and os.path.exists(bg.storage_path):
                    bg_path = bg.storage_path

            output_path = os.path.join(
                OUTPUT_DIR,
                f"outro_{composed_video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4",
            )

            result = append_outro_to_video(
                composed_video_path=composed_path,
                bg_path=bg_path,
                bg_offset_seconds=bg_offset,
                outro_text=outro_text,
                output_path=output_path,
            )

            if not result.success:
                logger.error(f"Outro append failed for {composed_video_id}: {result.error}")
                return {"status": "error", "message": result.error}

            old_path = composed.storage_path
            composed.storage_path = result.output_path
            composed.duration_seconds = result.duration
            if result.output_path and os.path.exists(result.output_path):
                composed.file_size_bytes = os.path.getsize(result.output_path)
            db.commit()

            if old_path and old_path != result.output_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception as e:
                    logger.warning(f"Failed to delete old composed file {old_path}: {e}")

            logger.info(f"Outro appended to composed video {composed_video_id}: {result.output_path}")
            return {
                "status": "completed",
                "video_id": composed_video_id,
                "output_path": result.output_path,
                "duration": result.duration,
            }

    except Exception as e:
        logger.error(f"append_outro_task error: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.video_composition_tasks.restyle_video_task', queue='composition')
def restyle_video_task(self, composed_video_id: int):
    """
    Restyle an existing video with a new random text style.

    Keeps the same caption and background, but re-renders with a different
    random style preset.
    """
    logger.info(f"Restyling video {composed_video_id} with new text style")

    try:
        with get_db_context() as db:
            composed = db.query(ComposedVideo).filter_by(id=composed_video_id).first()
            if not composed:
                logger.error(f"ComposedVideo {composed_video_id} not found")
                return {"status": "error", "message": "Composed video not found"}

            # Get the background video
            bg = db.query(BackgroundVideo).filter_by(id=composed.background_video_id).first()
            if not bg or not bg.storage_path or not os.path.exists(bg.storage_path):
                logger.error(f"Background video not found or missing file")
                return {"status": "error", "message": "Background video not found"}

            # Get the caption text (prefer edited, then original)
            caption_text = composed.edited_caption_text
            if not caption_text:
                if composed.generated_caption_id:
                    gen_caption = db.query(GeneratedCaption).filter_by(id=composed.generated_caption_id).first()
                    caption_text = gen_caption.caption_text if gen_caption else None
                elif composed.scraped_caption_id:
                    sc_caption = db.query(ScrapedCaption).filter_by(id=composed.scraped_caption_id).first()
                    if sc_caption:
                        caption_text = sc_caption.llm_refined_text or sc_caption.rule_based_text or sc_caption.caption_text

            if not caption_text:
                logger.error(f"Could not find caption text for video {composed_video_id}")
                return {"status": "error", "message": "Caption text not found"}

            # Create composer with a random style
            composer, style_name = create_composer_with_random_style()
            logger.info(f"Using style '{style_name}' for restyle")

            result = composer.compose(
                background_path=bg.storage_path,
                caption=caption_text,
                output_filename=f"restyled_{composed_video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4"
            )

            if not result.success:
                logger.error(f"Restyle failed: {result.error}")
                return {"status": "error", "message": result.error}

            # Delete old file if exists
            old_path = composed.storage_path
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    logger.info(f"Deleted old video file: {old_path}")
                except Exception as e:
                    logger.warning(f"Could not delete old file {old_path}: {e}")

            # Update the record
            composed.storage_path = result.output_path
            composed.duration_seconds = result.duration
            composed.caption_chunks = result.caption_chunks
            composed.is_published = False  # Reset published status

            if result.output_path and os.path.exists(result.output_path):
                composed.file_size_bytes = os.path.getsize(result.output_path)

            db.commit()

            logger.info(f"Video {composed_video_id} restyled with style '{style_name}': {result.output_path}")

            return {
                "status": "completed",
                "video_id": composed.id,
                "output_path": result.output_path,
                "duration": result.duration,
                "style_name": style_name
            }

    except Exception as e:
        logger.error(f"Restyle error: {e}")
        return {"status": "error", "message": str(e)}


def _get_captions(db, job: VideoCompositionJob) -> List[Tuple[int, str, str, List[str], str]]:
    """
    Get captions based on job configuration.

    Returns list of (caption_id, caption_text, caption_type, tags, niche) tuples.
    Tags are activity tags extracted from the caption for background matching.
    """
    results = []

    if job.caption_source == "generated":
        query = db.query(GeneratedCaption)

        if job.caption_status_filter:
            query = query.filter(GeneratedCaption.status == job.caption_status_filter)

        # Filter by niche if specified on the job
        if job.niche:
            query = query.filter(GeneratedCaption.niche == job.niche)

        # Only get captions not already used in composed videos
        used_ids = db.query(ComposedVideo.generated_caption_id).filter(
            ComposedVideo.generated_caption_id.isnot(None)
        ).all()
        used_ids = [id[0] for id in used_ids]

        if used_ids:
            query = query.filter(~GeneratedCaption.id.in_(used_ids))

        captions = query.limit(job.target_count * 3).all()  # Get extra in case some are skipped

        for c in captions:
            if c.caption_text and len(c.caption_text.strip()) > 10:
                # Get tags from database (for generated captions)
                caption_tags = c.tags if c.tags else []
                results.append((c.id, c.caption_text, "generated", caption_tags, c.niche))

    else:  # scraped
        query = db.query(ScrapedCaption)

        if job.min_upvotes > 0:
            query = query.filter(ScrapedCaption.upvotes >= job.min_upvotes)

        # Filter by niche if specified (via video's subreddit)
        if job.niche:
            from config.automation_config import get_automation_config
            config = get_automation_config()
            niche_config = config.niches.get(job.niche)
            if niche_config:
                # Join with Video table and filter by subreddits for this niche
                query = query.join(Video, ScrapedCaption.video_id == Video.id)
                query = query.filter(Video.subreddit.in_(niche_config.subreddits))

        # Only get captions not already used
        used_ids = db.query(ComposedVideo.scraped_caption_id).filter(
            ComposedVideo.scraped_caption_id.isnot(None)
        ).all()
        used_ids = [id[0] for id in used_ids]

        if used_ids:
            query = query.filter(~ScrapedCaption.id.in_(used_ids))

        # Prefer LLM-refined text
        captions = query.order_by(ScrapedCaption.upvotes.desc()).limit(job.target_count * 3).all()

        for c in captions:
            text = c.llm_refined_text or c.rule_based_text or c.caption_text
            if text and len(text.strip()) > 10:
                # Extract tags on-the-fly for scraped captions
                from utils.generation_postprocessor import extract_tags
                caption_tags = extract_tags(text)
                # Try to get niche from video's subreddit
                niche = None
                if c.video and c.video.subreddit:
                    from config.automation_config import get_automation_config
                    config = get_automation_config()
                    niche = config.get_niche_for_subreddit(c.video.subreddit)
                results.append((c.id, text, "scraped", caption_tags, niche))

    return results


def _get_background_videos(db, job: VideoCompositionJob) -> List[BackgroundVideo]:
    """
    Get available background videos based on job filters.

    Excludes backgrounds already used in existing composed videos.
    When a composed video is deleted, the background becomes available again.
    """
    # Get IDs of backgrounds already used in composed videos
    used_background_ids = db.query(ComposedVideo.background_video_id).filter(
        ComposedVideo.background_video_id.isnot(None)
    ).all()
    used_background_ids = [id[0] for id in used_background_ids]

    query = db.query(BackgroundVideo).filter(
        BackgroundVideo.filter_status == 'approved',
        BackgroundVideo.storage_path.isnot(None),
        BackgroundVideo.source_type == 'reddit',
    )

    # Exclude already-used backgrounds
    if used_background_ids:
        query = query.filter(~BackgroundVideo.id.in_(used_background_ids))
        logger.info(f"Excluding {len(used_background_ids)} already-used backgrounds")

    if job.background_tag_filter:
        # tags is a JSON array, check if filter tag is in the array
        query = query.filter(BackgroundVideo.tags.contains([job.background_tag_filter]))

    # Order by views/quality
    return query.order_by(BackgroundVideo.views.desc()).all()


def _find_compatible_backgrounds(
    composer: VideoComposer,
    caption_text: str,
    backgrounds: List[BackgroundVideo],
    used_backgrounds: set,
    caption_tags: List[str] = None,
    niche: str = None,
    allow_concatenation: bool = True
) -> Tuple[List[BackgroundVideo], List[str]]:
    """
    Find background video(s) compatible with niche rules and caption content.

    If a single background is long enough, returns [single_bg].
    If no single background is long enough but concatenation is allowed,
    returns multiple backgrounds that together meet the duration requirement.

    Matching priority:
    1. Single niche-compatible + tag match + duration match (best)
    2. Single niche-compatible + duration match (good)
    3. Multiple niche-compatible backgrounds concatenated (fallback)

    Args:
        composer: VideoComposer instance
        caption_text: Text to compose
        backgrounds: Available background videos
        used_backgrounds: Set of already-used background IDs
        caption_tags: Activity tags extracted from the caption (see utils/generation_postprocessor.extract_tags)
        niche: Niche for compatibility checking (e.g., "motivation", "cooking")
        allow_concatenation: Whether to allow concatenating multiple videos

    Returns:
        Tuple of (list of BackgroundVideos, list of warnings)
    """
    from video_generator import TextChunker, TimingEngine
    from config.niche_rules import is_subreddit_compatible

    chunker = TextChunker()
    timing = TimingEngine()

    chunks = chunker.chunk(caption_text)
    caption_duration = timing.get_total_duration(chunks)

    # Build caption activity set for scoring
    caption_action_set = set(caption_tags) if caption_tags else set()

    # Filter and score backgrounds by subreddit compatibility + activity tag matching
    all_scored = []
    long_enough_scored = []

    for bg in backgrounds:
        if bg.id in used_backgrounds:
            continue

        if not bg.storage_path or not os.path.exists(bg.storage_path):
            continue

        if not bg.duration_seconds:
            continue

        # Check niche compatibility via subreddit matching
        if niche and not is_subreddit_compatible(bg.searched_tag, niche):
            continue

        # Base score from views (0-10)
        score = min(bg.views / 10000, 10) if bg.views else 0

        # Tag matching score against the VLM scene tags' "activities" list
        # (schema in tasks/ml_tagging.py). The other scene fields (setting,
        # mood, subjects, camera) are used by the niche rules and by Stage 2
        # BG-first generation; here we only need the activity overlap.
        tag_match_score = 0
        if caption_action_set and bg.ml_tags:
            bg_actions_raw = bg.ml_tags.get('activities', []) if isinstance(bg.ml_tags, dict) else []
            bg_actions = {a for a in bg_actions_raw if a != 'other'} if isinstance(bg_actions_raw, list) else set()
            if bg_actions:
                overlap = caption_action_set & bg_actions
                if overlap:
                    # +5 per matching activity (rewards specific matches)
                    tag_match_score = len(overlap) * 5
                else:
                    # Penalty when caption has clear activities but bg has none in common
                    tag_match_score = -3

        score += tag_match_score

        entry = {
            'background': bg,
            'combined_score': score,
            'tag_match_score': tag_match_score,
            'duration': bg.duration_seconds
        }

        all_scored.append(entry)
        if bg.duration_seconds >= caption_duration:
            long_enough_scored.append(entry)

    # Sort by score descending
    all_scored.sort(key=lambda x: x['combined_score'], reverse=True)
    long_enough_scored.sort(key=lambda x: x['combined_score'], reverse=True)

    # First try: single background that's long enough
    if long_enough_scored:
        best = long_enough_scored[0]
        bg = best['background']
        bg_actions = bg.ml_tags.get('activities', []) if bg.ml_tags else []
        logger.info(
            f"Selected single background {bg.id} for niche '{niche}' "
            f"(score={best['combined_score']:.2f}, tag_match={best['tag_match_score']}, "
            f"bg_actions={bg_actions}, caption_tags={list(caption_action_set)}, "
            f"duration={bg.duration_seconds:.1f}s >= {caption_duration:.1f}s needed)"
        )
        return [bg], []

    # Second try: concatenate multiple compatible backgrounds
    if allow_concatenation and all_scored:
        logger.info(
            f"No single background long enough ({caption_duration:.1f}s needed), "
            f"attempting concatenation from {len(all_scored)} compatible backgrounds"
        )

        # Greedily select backgrounds until we have enough duration
        selected = []
        total_duration = 0
        selected_ids = set()

        for entry in all_scored:
            if total_duration >= caption_duration:
                break
            bg = entry['background']
            if bg.id not in selected_ids:
                selected.append(bg)
                selected_ids.add(bg.id)
                total_duration += bg.duration_seconds

        if total_duration >= caption_duration:
            logger.info(
                f"Selected {len(selected)} backgrounds for concatenation "
                f"(total={total_duration:.1f}s >= {caption_duration:.1f}s needed): "
                f"{[bg.id for bg in selected]}"
            )
            return selected, []
        else:
            logger.warning(
                f"Even with concatenation, only {total_duration:.1f}s available "
                f"from {len(all_scored)} compatible backgrounds ({caption_duration:.1f}s needed)"
            )
            return [], [f"Insufficient compatible background duration: {total_duration:.1f}s < {caption_duration:.1f}s needed"]

    # No compatible backgrounds at all
    if not all_scored:
        logger.warning(
            f"No compatible backgrounds for niche '{niche}' - "
            f"all candidates violated niche rules or had no duration"
        )
        return [], [f"No backgrounds compatible with {niche} niche rules"]

    return [], ["No backgrounds available with sufficient duration"]


def _find_compatible_background(
    composer: VideoComposer,
    caption_text: str,
    backgrounds: List[BackgroundVideo],
    used_backgrounds: set,
    caption_tags: List[str] = None,
    niche: str = None
) -> Tuple[Optional[BackgroundVideo], List[str]]:
    """
    Find a single background video compatible with niche rules and caption content.

    This is a wrapper for backwards compatibility that returns just the first
    background from _find_compatible_backgrounds.

    Returns:
        Tuple of (BackgroundVideo or None, list of warnings)
    """
    bgs, warnings = _find_compatible_backgrounds(
        composer, caption_text, backgrounds, used_backgrounds,
        caption_tags, niche, allow_concatenation=False
    )
    return bgs[0] if bgs else None, warnings
