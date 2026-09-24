"""
Text detection filter for Reddit background videos.

Uses the same OCR model (Qwen2-VL-2B) as caption extraction to detect
any text overlays in background videos. Videos with ANY detected text
are rejected - background videos should be clean with no burned-in
creator watermarks, handles, site URLs, captions, or logos.

IMPROVED: Now crops and analyzes corners separately since watermarks
typically appear in corners (especially bottom-right).

GPU Routing: Configurable via Redis keys:
- watermark_filter:enabled - Enable/disable filtering
- watermark_filter:gpu - Which GPU(s) to use: "0", "1", or "both"
"""
import os
import re
import cv2
import logging
import redis
from datetime import datetime
from typing import List, Tuple, Optional
from tasks.celery_app import celery_app
from database.db import get_db_context
from database.models import BackgroundVideo
from config.settings import settings

# Redis keys for filter control (must match api/background_videos.py)
WATERMARK_FILTER_ENABLED_KEY = "watermark_filter:enabled"
WATERMARK_FILTER_GPU_KEY = "watermark_filter:gpu"


def get_redis_client():
    return redis.from_url(settings.celery_broker_url)

logger = logging.getLogger(__name__)

# Number of frames to sample from video
FRAMES_TO_SAMPLE = 3

# Corner crop settings - what percentage of frame to crop for each corner
CORNER_SIZE_RATIO = 0.25  # 25% of width/height for each corner


def extract_sample_frames(video_path: str, num_frames: int = 3) -> List[any]:
    """
    Extract sample frames from video at evenly spaced intervals.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of CV2 image frames
    """
    frames = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            num_frames = max(1, total_frames)

        # Calculate frame positions (avoid first and last few frames which may be black)
        start_offset = max(1, int(total_frames * 0.1))  # Skip first 10%
        end_offset = max(1, int(total_frames * 0.1))    # Skip last 10%
        usable_frames = total_frames - start_offset - end_offset

        if usable_frames < num_frames:
            # Video too short, just use middle frame
            positions = [total_frames // 2]
        else:
            step = usable_frames // (num_frames + 1)
            positions = [start_offset + step * (i + 1) for i in range(num_frames)]

        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()

    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")

    return frames


def extract_corners(frame, size_ratio: float = CORNER_SIZE_RATIO) -> List[Tuple[any, str]]:
    """
    Extract corner crops from a frame.

    Args:
        frame: CV2 image frame
        size_ratio: What percentage of width/height to crop (0.25 = 25%)

    Returns:
        List of (cropped_frame, corner_name) tuples
    """
    h, w = frame.shape[:2]
    corner_h = int(h * size_ratio)
    corner_w = int(w * size_ratio)

    corners = []

    # Top-left
    corners.append((frame[0:corner_h, 0:corner_w], "top-left"))

    # Top-right
    corners.append((frame[0:corner_h, w-corner_w:w], "top-right"))

    # Bottom-left
    corners.append((frame[h-corner_h:h, 0:corner_w], "bottom-left"))

    # Bottom-right (most common watermark location)
    corners.append((frame[h-corner_h:h, w-corner_w:w], "bottom-right"))

    return corners


def upscale_image(frame, scale: float = 2.0) -> any:
    """
    Upscale an image to make small text more visible.

    Args:
        frame: CV2 image
        scale: Scale factor (2.0 = double size)

    Returns:
        Upscaled CV2 image
    """
    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def run_ocr_on_image(image, gpu_id: int = 0) -> str:
    """
    Run OCR on a single image using Qwen2-VL model.

    Uses the same simple prompt as caption extraction - just asks to read text.
    Returns empty string if no text found.

    Args:
        image: CV2 image or PIL Image
        gpu_id: GPU to use

    Returns:
        Extracted text from image (empty if none found)
    """
    try:
        from scrapers.caption_extractor_qwen2vl import get_qwen2vl_model
        import torch
        from PIL import Image
        import numpy as np
        import re

        model, processor = get_qwen2vl_model(gpu_id)

        # Convert CV2 frame to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Same simple prompt as caption extraction
        prompt = (
            "Read all the text in this image carefully. "
            "Output only the text content, preserving the reading order "
            "from top to bottom, left to right. "
            "Do not add any explanations or descriptions."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        device = model.device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode output
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Clean output - same as caption extractor
        result = result.strip()

        # Remove excessive repeated characters
        result = re.sub(r'(.)\1{4,}', r'\1\1', result)

        # Filter lines that are just punctuation/symbols
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and re.search(r'[a-zA-Z0-9]', stripped):
                cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines)

    except Exception as e:
        logger.error(f"OCR error on image: {e}")
        return ""


def is_garbage_text(text: str) -> bool:
    """
    Check if OCR output is likely hallucinated garbage rather than real text.

    The Qwen2-VL model sometimes hallucinates when no text exists:
    - Random short words like "TUMI", "TUMS", "YUM"
    - Coordinate patterns like "(0,0),(999,999)"
    - Single characters or symbols

    Returns True if text looks like garbage (should be ignored).
    """
    import re

    text = text.strip()
    if not text:
        return True

    # Pattern 1: Coordinate-like patterns (numbers in parentheses)
    # e.g., "(0,0),(999,999)" or "(474,271),(999,444)"
    coord_pattern = r'^\s*[\(\[\{]?\d+\s*,\s*\d+[\)\]\}]?\s*$'
    if re.match(coord_pattern, text):
        return True

    # Pattern 2: Text that's ONLY coordinate patterns separated by delimiters
    # Remove all coordinate-like patterns and see what's left
    cleaned = re.sub(r'\(\d+,\d+\)', '', text)
    cleaned = re.sub(r'\[\d+,\d+\]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # If after removing coordinates, very little remains, it's garbage
    if len(cleaned) < 3:
        return True

    # Pattern 3: Known hallucination words (short nonsense)
    # These are common OCR hallucinations when no text exists
    hallucination_words = {
        'tum', 'tums', 'tumi', 'yum', 'mum', 'gum', 'hum', 'bum', 'rum',
        'um', 'mm', 'hmm', 'uh', 'ah', 'oh', 'eh',
        'the', 'a', 'i', 'it', 'is', 'at', 'to', 'of', 'in', 'on',
        'no', 'yes', 'ok', 'hi', 'bye',
    }

    # Split into words and check
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        return True

    # If ALL words are short hallucination words, it's garbage
    real_words = [w for w in words if w not in hallucination_words and len(w) > 2]
    if not real_words:
        return True

    # Pattern 4: Text is too short to be a real watermark (less than 4 meaningful chars)
    alpha_only = re.sub(r'[^a-zA-Z]', '', text)
    if len(alpha_only) < 4:
        return True

    # Pattern 5: Random letter sequences (no vowels or all vowels)
    vowels = set('aeiouAEIOU')
    if alpha_only:
        has_vowel = any(c in vowels for c in alpha_only)
        all_vowels = all(c in vowels for c in alpha_only)
        if not has_vowel or all_vowels:
            return True

    return False


# Known watermark fingerprints. Force-reject on substring or regex match
# even if is_garbage_text would otherwise dismiss the OCR output. Built
# from Stage 4 review runs — roughly a quarter of the first two batches
# were banned for burned-in watermarks that slipped past the generic path.
#
# Two reasons to keep this list spelled out rather than rely on the generic
# "any non-garbage text → reject" path:
#   1. OCR sometimes returns the watermark fragmented (e.g. "STOCK" alone
#      from "shutterstock") which is_garbage_text dismisses as a single
#      short word but is in fact a known commercial source.
#   2. Documents which sources we've already lost compositions to, so
#      future false-negatives can be triaged against this list quickly.
KNOWN_WATERMARK_SUBSTRINGS = {
    # Video-editor export watermarks (case-insensitive substring match)
    "capcut", "inshot", "kapwing", "filmora", "clideo", "kinemaster",
    "canva", "picsart", "veed.io", "invideo", "powerdirector", "splice",
    # Stock-footage / licensing watermarks
    "shutterstock", "getty images", "gettyimages", "pexels", "pixabay",
    "storyblocks", "envato", "artgrid", "videvo", "stock footage", "istock",
    # Platform / creator-handle overlays
    "tiktok", "instagram", "youtube", "twitch", "snapchat", "linktr.ee",
    "patreon.com", "ko-fi.com", "buymeacoffee",
    # Generic CTA overlays
    "subscribe", "follow me", "follow for more", "link in bio", "www.",
}

# Regex families. Any match → force reject. These cover the long tail
# beyond the named-watermark list.
KNOWN_WATERMARK_REGEXES = [
    # Reddit user-handle overlays (u/username, U/username)
    re.compile(r'\b[Uu]/[a-z0-9_]{3,}\b'),
    # Social handles (3+ chars to avoid garbage)
    re.compile(r'@[A-Za-z0-9_]{3,}'),
    # Generic ".com/slug" patterns — covers sites we haven't enumerated by name.
    re.compile(r'\b[a-z0-9-]{3,}\.(com|net|org|io|tv)/[a-z0-9_-]+', re.IGNORECASE),
    # Bare "www." domains
    re.compile(r'\bwww\.[a-z0-9-]+\.[a-z]{2,}\b', re.IGNORECASE),
    # patreon.com/<slug> — calls out a recurring family by name even
    # though the generic rule above catches it, for log readability.
    re.compile(r'patreon\.com/[a-z0-9_]+', re.IGNORECASE),
    # tr.ee / linktr.ee / similar link-aggregator slugs
    re.compile(r'(tr|linktr)\.ee/[a-z0-9_]+', re.IGNORECASE),
]

def matches_known_watermark(text: str) -> Optional[str]:
    """
    Check if `text` contains a known watermark fingerprint. Returns the
    matched pattern (for logging) or None.

    Used as a force-reject path in check_for_watermark — bypasses
    is_garbage_text so partial OCR fragments still get caught.
    """
    if not text:
        return None
    text_lower = text.lower()
    for needle in KNOWN_WATERMARK_SUBSTRINGS:
        if needle in text_lower:
            return needle
    for pattern in KNOWN_WATERMARK_REGEXES:
        m = pattern.search(text)
        if m:
            return f"regex:{pattern.pattern} → {m.group(0)}"
    return None


def check_for_watermark(texts: List[str]) -> Tuple[bool, str]:
    """
    Check if any real text was detected in the video frames/corners.

    Two-pass logic:
      1. Force-reject if any known watermark fingerprint matches (even
         if is_garbage_text would otherwise dismiss the fragment).
      2. Otherwise, fall back to the existing "any non-garbage text →
         reject" path.

    Real watermarks are things like "SHUTTERSTOCK", "@username", "www.example.com".

    Args:
        texts: List of OCR results from frames/corners

    Returns:
        Tuple of (has_text, detected_text)
    """
    # Strip OCR-region prefixes like "frame0_full: " so substring/regex
    # matching sees the raw text only.
    cleaned_texts = []
    for t in texts:
        if t and t.strip():
            if ": " in t:
                text_part = t.split(": ", 1)[1].strip()
            else:
                text_part = t.strip()
            if text_part:
                cleaned_texts.append(text_part)

    # Pass 1: known-watermark force-reject. Any match in any frame/corner.
    for text_part in cleaned_texts:
        match = matches_known_watermark(text_part)
        if match:
            logger.info(f"Known watermark fingerprint matched ({match}): {text_part[:200]}")
            return True, text_part

    # Pass 2: existing garbage-filtered behavior.
    all_text = [t for t in cleaned_texts if not is_garbage_text(t)]

    if not all_text:
        logger.debug("No real text detected (filtered out garbage)")
        return False, ""

    # Real text found = reject
    combined = " | ".join(all_text)
    logger.info(f"Real text detected: {combined[:200]}...")
    return True, combined


@celery_app.task(bind=True, queue='gpu', max_retries=2)
def filter_background_video(self, video_id: int, gpu_id: int = None):
    """
    Check a background video for watermarks using OCR.

    IMPROVED: Now analyzes corners separately (upscaled 2x) since watermarks
    typically appear in corners, especially bottom-right.

    If watermark/text is detected, marks video as rejected and deletes
    the file. If clean, marks as approved.

    Args:
        video_id: BackgroundVideo ID to check
        gpu_id: GPU to use (auto-detected if None)

    Returns:
        Dict with filter results
    """
    # Auto-detect GPU from environment if not specified
    if gpu_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        gpu_id = int(cuda_visible.split(',')[0]) if cuda_visible else 0

    logger.info(f"Filtering background video {video_id} on GPU {gpu_id}")

    try:
        with get_db_context() as db:
            video = db.query(BackgroundVideo).filter_by(id=video_id).first()

            if not video:
                logger.warning(f"Background video {video_id} not found")
                return {'status': 'error', 'message': 'Video not found'}

            if not os.path.exists(video.storage_path):
                error_msg = f"Video file not found: {video.storage_path}"
                logger.warning(error_msg)
                video.filter_status = 'error'
                video.filter_checked_at = datetime.utcnow()
                video.filter_text = error_msg
                db.commit()
                return {'status': 'error', 'message': error_msg}

            # Extract sample frames
            logger.info(f"Extracting {FRAMES_TO_SAMPLE} frames from video...")
            frames = extract_sample_frames(video.storage_path, FRAMES_TO_SAMPLE)

            if not frames:
                error_msg = f"Could not extract frames from {video.storage_path}"
                logger.warning(error_msg)
                video.filter_status = 'error'
                video.filter_checked_at = datetime.utcnow()
                video.filter_text = error_msg
                db.commit()
                return {'status': 'error', 'message': error_msg}

            # Collect all OCR results
            all_ocr_results = []

            # Process each frame
            for frame_idx, frame in enumerate(frames):
                logger.info(f"Processing frame {frame_idx + 1}/{len(frames)}...")

                # 1. Run OCR on full frame first
                full_result = run_ocr_on_image(frame, gpu_id)
                all_ocr_results.append(f"frame{frame_idx}_full: {full_result}")
                logger.debug(f"Frame {frame_idx} full OCR: {full_result[:100]}...")

                # 2. Extract and analyze corners (upscaled for better detection)
                corners = extract_corners(frame)
                for corner_img, corner_name in corners:
                    # Upscale corner 2x to make small text more visible
                    upscaled = upscale_image(corner_img, scale=2.0)
                    corner_result = run_ocr_on_image(upscaled, gpu_id)
                    all_ocr_results.append(f"frame{frame_idx}_{corner_name}: {corner_result}")
                    logger.debug(f"Frame {frame_idx} {corner_name} OCR: {corner_result[:100]}...")

            # Check for watermarks across all results
            has_watermark, detected_text = check_for_watermark(all_ocr_results)

            # Update video status
            video.filter_checked_at = datetime.utcnow()
            video.filter_text = detected_text[:2000] if detected_text else None

            if has_watermark:
                logger.info(f"Video {video_id} REJECTED - watermark detected: {detected_text[:100]}")
                video.filter_status = 'rejected'
                db.commit()

                return {
                    'status': 'rejected',
                    'video_id': video_id,
                    'reddit_post_id': video.reddit_post_id,
                    'reason': 'watermark_detected',
                    'detected_text': detected_text[:500]
                }
            else:
                logger.info(f"Video {video_id} APPROVED - no watermark")
                video.filter_status = 'approved'
                db.commit()

                return {
                    'status': 'approved',
                    'video_id': video_id,
                    'reddit_post_id': video.reddit_post_id,
                }

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"Error filtering video {video_id}: {error_msg}\n{error_trace}")

        # Mark as error in database BEFORE retry
        try:
            with get_db_context() as db:
                video = db.query(BackgroundVideo).filter_by(id=video_id).first()
                if video:
                    video.filter_status = 'error'
                    video.filter_checked_at = datetime.utcnow()
                    video.filter_text = f"{error_msg}\n\nPath: {video.storage_path if video else 'unknown'}"[:500]
                    db.commit()
                    logger.info(f"Marked video {video_id} as error with message: {error_msg[:100]}")
        except Exception as db_err:
            logger.error(f"Failed to save error status to database: {db_err}")

        # Only retry if we haven't exceeded max retries
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        else:
            logger.error(f"Max retries exceeded for video {video_id}")
            return {'status': 'error', 'video_id': video_id, 'message': error_msg}


@celery_app.task(queue='gpu')
def filter_pending_background_videos(batch_size: int = 10, gpu_id: int = None):
    """
    Batch process pending background videos through watermark filter.

    Args:
        batch_size: Number of videos to process
        gpu_id: GPU to use

    Returns:
        Dict with batch results
    """
    # Auto-detect GPU
    if gpu_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        gpu_id = int(cuda_visible.split(',')[0]) if cuda_visible else 0

    with get_db_context() as db:
        pending_videos = db.query(BackgroundVideo).filter(
            BackgroundVideo.filter_status == 'pending',
            BackgroundVideo.download_status == 'completed'
        ).limit(batch_size).all()

        if not pending_videos:
            return {'status': 'no_pending', 'processed': 0}

        video_ids = [v.id for v in pending_videos]

    # Queue individual filter tasks
    results = []
    for video_id in video_ids:
        task = filter_background_video.delay(video_id, gpu_id)
        results.append({'video_id': video_id, 'task_id': task.id})

    return {
        'status': 'dispatched',
        'queued': len(results),
        'tasks': results
    }


@celery_app.task(queue='maintenance')
def get_filter_stats():
    """Get statistics on background video filtering."""
    with get_db_context() as db:
        from sqlalchemy import func

        total = db.query(func.count(BackgroundVideo.id)).scalar() or 0
        pending = db.query(func.count(BackgroundVideo.id)).filter(
            BackgroundVideo.filter_status == 'pending'
        ).scalar() or 0
        approved = db.query(func.count(BackgroundVideo.id)).filter(
            BackgroundVideo.filter_status == 'approved'
        ).scalar() or 0
        rejected = db.query(func.count(BackgroundVideo.id)).filter(
            BackgroundVideo.filter_status == 'rejected'
        ).scalar() or 0
        errors = db.query(func.count(BackgroundVideo.id)).filter(
            BackgroundVideo.filter_status == 'error'
        ).scalar() or 0

        return {
            'total': total,
            'pending': pending,
            'approved': approved,
            'rejected': rejected,
            'errors': errors,
            'filter_rate': f"{(rejected / (rejected + approved) * 100):.1f}%" if (rejected + approved) > 0 else "0%"
        }


# === GPU-Specific Filter Tasks ===
# These tasks route to specific GPU workers to prevent model loading on both GPUs

@celery_app.task(bind=True, queue='gpu_status_0', max_retries=2)
def filter_background_video_gpu0(self, video_id: int):
    """
    Filter a background video for watermarks using GPU 0 only.

    Routes to gpu_status_0 queue which only GPU 0 worker listens to.
    Also updates GPU status cache after completion.
    """
    # Force GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    result = _run_filter(self, video_id, gpu_id=0)

    # Update GPU status cache (piggyback on filter task since they run frequently)
    try:
        from tasks.maintenance_tasks import _get_worker_gpu_status_for_cache, GPU_STATUS_CACHE_KEY_PREFIX, GPU_STATUS_CACHE_TTL
        import redis
        import json
        from config.settings import settings

        status = _get_worker_gpu_status_for_cache()
        status["index"] = 0
        status["worker_gpu"] = 0

        r = redis.from_url(settings.celery_broker_url)
        r.setex(
            f"{GPU_STATUS_CACHE_KEY_PREFIX}0",
            GPU_STATUS_CACHE_TTL,
            json.dumps(status)
        )
        logger.debug(f"Updated GPU 0 cache after filter task")
    except Exception as e:
        logger.debug(f"Could not update GPU 0 cache: {e}")

    return result


@celery_app.task(bind=True, queue='gpu_status_1', max_retries=2)
def filter_background_video_gpu1(self, video_id: int):
    """
    Filter a background video for watermarks using GPU 1 only.

    Routes to gpu_status_1 queue which only GPU 1 worker listens to.
    Also updates GPU status cache after completion.
    """
    # Force GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Worker already has NVIDIA_VISIBLE_DEVICES=1
    result = _run_filter(self, video_id, gpu_id=0)  # Use device 0 within the worker's context

    # Update GPU status cache (piggyback on filter task since they run frequently)
    try:
        from tasks.maintenance_tasks import _get_worker_gpu_status_for_cache, GPU_STATUS_CACHE_KEY_PREFIX, GPU_STATUS_CACHE_TTL
        import redis
        import json
        from config.settings import settings

        status = _get_worker_gpu_status_for_cache()
        status["index"] = 1
        status["worker_gpu"] = 1

        r = redis.from_url(settings.celery_broker_url)
        r.setex(
            f"{GPU_STATUS_CACHE_KEY_PREFIX}1",
            GPU_STATUS_CACHE_TTL,
            json.dumps(status)
        )
        logger.debug(f"Updated GPU 1 cache after filter task")
    except Exception as e:
        logger.debug(f"Could not update GPU 1 cache: {e}")

    return result


def _run_filter(task_self, video_id: int, gpu_id: int = 0):
    """
    Core filter logic shared by GPU-specific tasks.

    Args:
        task_self: Celery task instance (for retries)
        video_id: BackgroundVideo ID to check
        gpu_id: GPU to use (0 within worker's CUDA context)

    Returns:
        Dict with filter results
    """
    logger.info(f"Filtering background video {video_id} on GPU {gpu_id}")

    try:
        with get_db_context() as db:
            video = db.query(BackgroundVideo).filter_by(id=video_id).first()

            if not video:
                logger.warning(f"Background video {video_id} not found")
                return {'status': 'error', 'message': 'Video not found'}

            if not os.path.exists(video.storage_path):
                error_msg = f"Video file not found: {video.storage_path}"
                logger.warning(error_msg)
                video.filter_status = 'error'
                video.filter_checked_at = datetime.utcnow()
                video.filter_text = error_msg
                db.commit()
                return {'status': 'error', 'message': error_msg}

            # Extract sample frames
            logger.info(f"Extracting {FRAMES_TO_SAMPLE} frames from video...")
            frames = extract_sample_frames(video.storage_path, FRAMES_TO_SAMPLE)

            if not frames:
                error_msg = f"Could not extract frames from {video.storage_path}"
                logger.warning(error_msg)
                video.filter_status = 'error'
                video.filter_checked_at = datetime.utcnow()
                video.filter_text = error_msg
                db.commit()
                return {'status': 'error', 'message': error_msg}

            # Collect all OCR results
            all_ocr_results = []

            # Process each frame
            for frame_idx, frame in enumerate(frames):
                logger.info(f"Processing frame {frame_idx + 1}/{len(frames)}...")

                # 1. Run OCR on full frame first
                full_result = run_ocr_on_image(frame, gpu_id)
                all_ocr_results.append(f"frame{frame_idx}_full: {full_result}")
                logger.debug(f"Frame {frame_idx} full OCR: {full_result[:100]}...")

                # 2. Extract and analyze corners (upscaled for better detection)
                corners = extract_corners(frame)
                for corner_img, corner_name in corners:
                    # Upscale corner 2x to make small text more visible
                    upscaled = upscale_image(corner_img, scale=2.0)
                    corner_result = run_ocr_on_image(upscaled, gpu_id)
                    all_ocr_results.append(f"frame{frame_idx}_{corner_name}: {corner_result}")
                    logger.debug(f"Frame {frame_idx} {corner_name} OCR: {corner_result[:100]}...")

            # Check for watermarks across all results
            has_watermark, detected_text = check_for_watermark(all_ocr_results)

            # Update video status
            video.filter_checked_at = datetime.utcnow()
            video.filter_text = detected_text[:2000] if detected_text else None

            if has_watermark:
                logger.info(f"Video {video_id} REJECTED - watermark detected: {detected_text[:100]}")
                video.filter_status = 'rejected'
                db.commit()

                return {
                    'status': 'rejected',
                    'video_id': video_id,
                    'reddit_post_id': video.reddit_post_id,
                    'reason': 'watermark_detected',
                    'detected_text': detected_text[:500]
                }
            else:
                logger.info(f"Video {video_id} APPROVED - no watermark")
                video.filter_status = 'approved'
                db.commit()

                return {
                    'status': 'approved',
                    'video_id': video_id,
                    'reddit_post_id': video.reddit_post_id,
                }

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"Error filtering video {video_id}: {error_msg}\n{error_trace}")

        # Mark as error in database BEFORE retry
        try:
            with get_db_context() as db:
                video = db.query(BackgroundVideo).filter_by(id=video_id).first()
                if video:
                    video.filter_status = 'error'
                    video.filter_checked_at = datetime.utcnow()
                    video.filter_text = f"{error_msg}\n\nPath: {video.storage_path if video else 'unknown'}"[:500]
                    db.commit()
                    logger.info(f"Marked video {video_id} as error with message: {error_msg[:100]}")
        except Exception as db_err:
            logger.error(f"Failed to save error status to database: {db_err}")

        # Only retry if we haven't exceeded max retries
        if task_self.request.retries < task_self.max_retries:
            raise task_self.retry(exc=e, countdown=60)
        else:
            logger.error(f"Max retries exceeded for video {video_id}")
            return {'status': 'error', 'video_id': video_id, 'message': error_msg}


# === Filter Dispatcher (Checks Redis settings and routes to appropriate GPU) ===

@celery_app.task(queue='maintenance')
def dispatch_filter_batch(batch_size: int = 10):
    """
    Dispatch pending videos for watermark filtering based on Redis configuration.

    Checks:
    - watermark_filter:enabled - Must be "1" to process
    - watermark_filter:gpu - Routes to "0", "1", or "both" GPUs

    This decouples filtering from scraping - you can scrape without filtering,
    or filter without scraping.
    """
    r = get_redis_client()

    # Check if filtering is enabled
    if r.get(WATERMARK_FILTER_ENABLED_KEY) != b"1":
        logger.debug("Watermark filtering is disabled, skipping dispatch")
        return {'status': 'disabled', 'message': 'Filtering disabled via Redis flag'}

    # Get GPU setting (default to GPU 0 only)
    filter_gpu = (r.get(WATERMARK_FILTER_GPU_KEY) or b"0").decode()
    logger.info(f"Filter dispatch: GPU setting = {filter_gpu}")

    # Get pending videos
    with get_db_context() as db:
        pending_videos = db.query(BackgroundVideo).filter(
            BackgroundVideo.filter_status == 'pending',
            BackgroundVideo.download_status == 'completed'
        ).limit(batch_size).all()

        if not pending_videos:
            logger.debug("No pending videos to filter")
            return {'status': 'no_pending', 'processed': 0}

        video_ids = [v.id for v in pending_videos]
        logger.info(f"Dispatching {len(video_ids)} videos for filtering on GPU: {filter_gpu}")

    # Dispatch to appropriate GPU(s)
    results = []

    if filter_gpu == "0":
        # GPU 0 only
        for video_id in video_ids:
            task = filter_background_video_gpu0.delay(video_id)
            results.append({'video_id': video_id, 'task_id': task.id, 'gpu': '0'})

    elif filter_gpu == "1":
        # GPU 1 only
        for video_id in video_ids:
            task = filter_background_video_gpu1.delay(video_id)
            results.append({'video_id': video_id, 'task_id': task.id, 'gpu': '1'})

    elif filter_gpu == "both":
        # Round-robin between GPUs
        for i, video_id in enumerate(video_ids):
            if i % 2 == 0:
                task = filter_background_video_gpu0.delay(video_id)
                results.append({'video_id': video_id, 'task_id': task.id, 'gpu': '0'})
            else:
                task = filter_background_video_gpu1.delay(video_id)
                results.append({'video_id': video_id, 'task_id': task.id, 'gpu': '1'})

    else:
        # Default to GPU 0
        logger.warning(f"Unknown GPU setting '{filter_gpu}', defaulting to GPU 0")
        for video_id in video_ids:
            task = filter_background_video_gpu0.delay(video_id)
            results.append({'video_id': video_id, 'task_id': task.id, 'gpu': '0'})

    return {
        'status': 'dispatched',
        'gpu_setting': filter_gpu,
        'queued': len(results),
        'tasks': results
    }
