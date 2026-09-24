"""
VLM scene tagging for background videos (llama-server backend).

Uses InternVL3-14B (GGUF) served by llama-server via the "vlm-internvl3"
profile (see utils/llama_server_control.PROFILES). Same multi-GPU
tensor-split pattern we use for Mistral generation, just a different
model. This replaced an earlier in-process transformers load, which
couldn't fit a mid-size VLM in FP16 without CPU spill.

Runs on celery-worker-ml-tagging. On each batch we:
  1. Snapshot + pause competing GPU workloads (via orchestrator).
  2. Stop whichever llama-server profile is running (likely "llm").
  3. Start the VLM profile.
  4. POST each video's sampled frames as data-URL image parts to
     /v1/chat/completions with the tagging prompt.
  5. Parse the returned JSON into the scene schema and persist.
  6. Stop the VLM profile and restart "llm".
  7. Restore competing workload flags.

Scene schema stored on background_videos.ml_tags (ML_TAGS_VERSION = 4):
{
  "subjects": [{"type": "person", "description": "runner in a red jacket"}],
  "activities": ["running"],              # from VALID_ACTIVITIES
  "setting": "outdoors",                  # from VALID_SETTINGS
  "mood": "energetic",                    # from VALID_MOODS
  "camera": "handheld",                   # from VALID_CAMERA
  "text_on_screen": false,                # burned-in captions / watermarks / UI
  "scene_description": "A runner on a forest trail at sunrise, handheld tracking shot."
}

Niche compatibility rules (config/niche_rules.py) key off setting,
activities, subjects[].type, mood, camera and text_on_screen. The BG-first
generator (tasks/bg_first_generation.py) grounds its prompt in the same
fields plus scene_description.

Redis control:
- ml_tagging:enabled  "1" enables the dispatcher (gates auto-batching)
- ml_tagging:running  lock while a batch is active (2h TTL)
"""
import base64
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import redis
import requests

from config.settings import settings
from database.db import get_db_context
from database.models import BackgroundVideo
from tasks.celery_app import celery_app
from utils.generation_postprocessor import UNIFIED_ACTIVITY_TAGS

logger = logging.getLogger(__name__)

# === Redis keys ===
ML_TAGGING_ENABLED_KEY = "ml_tagging:enabled"
ML_TAGGING_LOCK_KEY = "ml_tagging:running"
ML_TAGGING_LOCK_TTL = 7200  # 2h max lock

# Keys the orchestrator flips in disable_all_gpu_workloads(). We snapshot
# them before each batch and restore after so we don't trample user state.
COMPETING_WORKLOAD_KEYS = [
    "extraction:enabled",
    "ocr:gpu0:enabled",
    "ocr:gpu1:enabled",
    "llm:batch:enabled",
    "watermark_filter:enabled",
]


def get_redis_client() -> redis.Redis:
    return redis.from_url(settings.celery_broker_url)


# === Tagging config ===

# Bump this whenever the schema or prompt changes in a way that makes old
# tags unusable. The dispatcher re-tags any BG whose stored version is
# older. History: v1 = flat activity list, v2/v3 = earlier structured
# schemas, v4 = the generic scene schema documented above.
ML_TAGS_VERSION = 4

# 6 frames per video. InternVL3-14B emits ~768 vision tokens per frame so
# 6 frames = ~4600 tokens + prompt, which fits in ctx=16384 with headroom.
FRAMES_TO_SAMPLE = 6

# Vocabularies — single source of truth the matcher + generator read.
VALID_SUBJECT_TYPES = [
    "person", "group", "animal", "food", "vehicle", "landscape", "object",
]

# Activities share one vocabulary with caption-side tag extraction
# (utils/generation_postprocessor.UNIFIED_ACTIVITY_TAGS) so composition can
# score caption/background overlap directly. "other" is the VLM's escape
# hatch for footage with no recognisable activity.
VALID_ACTIVITIES = [*UNIFIED_ACTIVITY_TAGS, "other"]

VALID_SETTINGS = [
    "kitchen", "gym", "outdoors", "city", "beach", "road", "studio", "home", "other",
]

VALID_MOODS = [
    "energetic", "calm", "dramatic", "cozy", "inspiring", "funny", "neutral",
]

VALID_CAMERA = ["static", "handheld", "drone", "timelapse"]

# Activities that only make sense when a person, group, or animal is
# actually visible. Used by the normalization pass to drop activities the
# model asserted without a matching subject (e.g. "running" on a pure
# landscape clip). Cooking/baking/eating are deliberately excluded — a
# hands-only recipe clip legitimately has a food subject and no person.
SUBJECT_BOUND_ACTIVITIES = {
    "running", "lifting", "stretching", "yoga", "climbing", "cycling",
    "swimming", "hiking", "walking", "talking", "dancing", "working",
    "reading", "posing",
}

# Per-niche description of what a source subreddit *usually* posts. Injected
# into the prompt as a prior — the model still has to look at the frames,
# and the prompt states clearly that the hint is not a guarantee.
NICHE_SOURCE_HINTS: Dict[str, str] = {
    "motivation": (
        "Scenic clips: nature, weather, aviation, satisfying processes, hiking "
        "trails. Very often there are NO people at all — just landscapes, skies, "
        "or machinery. Do not invent a person if none is visible."
    ),
    "fitness": (
        "Gym, running, climbing, and calisthenics clips. Usually one person or a "
        "small group exercising. Equipment (barbells, ropes, walls) is common."
    ),
    "cooking": (
        "Recipe and kitchen clips. Expect food close-ups, hands, utensils, and "
        "pans; a full person is often not in frame. Subjects are usually food."
    ),
    "travel": (
        "Travel, drone, road-trip, and van-life clips. Cities, coastlines, roads, "
        "and landmarks; the camera is often a drone or handheld walking shot."
    ),
}


def _prompt_hint_for_subreddit(searched_tag: Optional[str]) -> str:
    """Generate a short source-specific hint to inject into the prompt.

    Hints describe the *type* of content the subreddit posts, NOT what is
    in any particular clip. The model has to look at the actual frames to
    decide what's happening.
    """
    if not searched_tag:
        return ""
    from config.niche_rules import get_niche_for_subreddit

    niche = get_niche_for_subreddit(searched_tag)
    hint = NICHE_SOURCE_HINTS.get(niche or "")
    if not hint:
        return ""
    return f"This clip was scraped from r/{searched_tag}. Typical content there: {hint}"


def build_tagging_prompt(searched_tag: Optional[str] = None) -> str:
    """Build the tagging prompt with an optional source-subreddit hint."""
    hint_block = _prompt_hint_for_subreddit(searched_tag)
    hint_section = f"\n=== SOURCE HINT ===\n\n{hint_block}\n" if hint_block else ""

    return f"""You are tagging {FRAMES_TO_SAMPLE} frames sampled from a short background video for a scene-matching database. Return ONE valid minified JSON object describing the scene. Be concise, accurate, and grounded ONLY in what you actually see. Prefer "other" / "neutral" over guessing.
{hint_section}
=== HARD RULES (apply these first) ===

RULE A (describe what you SEE, not what you expect):
List one subject entry per distinct visible subject. Landscape-only clips are common in this database — if no person is visible, do NOT add a person subject. The SOURCE HINT above says what the subreddit USUALLY posts; it is NOT a guarantee about this clip.

RULE B (activities need a matching subject):
Only list an activity if a subject that could perform it is visible. "running", "lifting", "hiking", "talking" etc. require a person, group, or animal. "cooking" / "baking" / "eating" may be tagged from hands-and-food shots. "driving" / "flying" may be tagged from a vehicle or an in-vehicle view. If nothing is happening (a static landscape), use ["relaxing"] or ["other"].

RULE C (text_on_screen is strict):
Set text_on_screen=true if ANY burned-in text is visible in ANY frame: captions, subtitles, watermarks, usernames, logos, app UI, timestamps. Small corner watermarks count. Pure video with no text = false.

RULE D (camera):
"drone" only for clearly aerial footage. "timelapse" only if motion is obviously sped up (clouds racing, traffic streaks). Handheld = visible shake / walking motion. Otherwise "static".

RULE E (mood is about the footage, not the topic):
Pick the mood the footage itself conveys — pacing, light, color, motion. A calm sunrise over mountains is "calm" or "inspiring", a crowded gym with fast cuts is "energetic", a rainy kitchen with warm light is "cozy".

=== VOCABULARIES (use ONLY these exact strings) ===

subjects — list of objects, one per distinct visible subject:
  type: one of {", ".join(VALID_SUBJECT_TYPES)}
  description: short phrase (max 8 words) describing that subject, e.g. "runner in a red jacket", "bowl of ramen", "coastal highway at dusk"

activities — list of ALL activities present (can be empty):
{", ".join(VALID_ACTIVITIES)}

setting — {", ".join(VALID_SETTINGS)}

mood — {", ".join(VALID_MOODS)}

camera — {", ".join(VALID_CAMERA)}

text_on_screen — boolean (true | false), see RULE C.

scene_description — ONE short sentence (max 25 words). What you actually see across the frames. Mention the main subject, what it is doing, and where.

=== OUTPUT ===

Return ONLY this minified JSON, no markdown, no preamble:
{{"subjects":[{{"type":"...","description":"..."}}],"activities":[...],"setting":"...","mood":"...","camera":"...","text_on_screen":true|false,"scene_description":"..."}}"""


# Static fallback prompt used only when subreddit isn't known (defensive).
TAGGING_PROMPT = build_tagging_prompt(None)


# === Frame extraction ===

def extract_tagging_frames(video_path: str, num_frames: int = FRAMES_TO_SAMPLE) -> List[Any]:
    """Extract evenly-spaced sample frames, skipping first/last 10%."""
    frames: List[Any] = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < num_frames:
            num_frames = max(1, total)

        start_offset = max(1, int(total * 0.1))
        end_offset = max(1, int(total * 0.1))
        usable = total - start_offset - end_offset

        if usable < num_frames:
            positions = [total // 2] if total > 0 else []
        else:
            step = usable // (num_frames + 1)
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


def _frame_to_data_url(frame: Any, max_dim: int = 512, jpeg_quality: int = 85) -> str:
    """Downscale a frame, JPEG-encode it, and wrap as a data: URL.

    The max_dim cap keeps vision-token count per image manageable. At 512px
    with the InternVL3 tiling settings we use that's a few hundred vision
    tokens per frame.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# === llama-server VLM client ===

def _post_vlm_once(
    content: List[Dict[str, Any]],
    llama_url: str,
    temperature: float,
    request_timeout: int,
) -> Optional[str]:
    """Single POST to llama-server chat completions. Returns raw text or None."""
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0.0 else 0.9,
    }
    try:
        resp = requests.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=request_timeout,
        )
        if resp.status_code != 200:
            logger.error(f"VLM HTTP {resp.status_code}: {resp.text[:500]}")
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"VLM request error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"VLM response malformed: {e}")
        return None


def _is_empty_or_malformed(raw: Optional[str]) -> bool:
    """Detect output that looks like a formatting failure or a non-answer."""
    if not raw:
        return True
    stripped = raw.strip()
    if not stripped:
        return True
    # No JSON at all
    if "{" not in stripped or "}" not in stripped:
        return True
    # Try a quick parse — if it doesn't shape like our schema, treat as failure
    parsed = _parse_tag_response(stripped)
    if parsed is None:
        return True
    # Catch the "nothing tagged + empty description" non-answer signature
    if (
        not parsed["subjects"]
        and not [a for a in parsed["activities"] if a != "other"]
        and not parsed["scene_description"]
    ):
        return True
    return False


def _call_vlm(
    frames: List[Any],
    llama_url: str,
    searched_tag: Optional[str] = None,
    request_timeout: int = 180,
) -> Optional[str]:
    """POST frames + prompt to llama-server chat completions. Return raw text.

    searched_tag is the source subreddit; if provided we inject a per-source
    hint block into the prompt to steer subject/activity labels.

    If the first attempt looks like a non-answer (empty / malformed / nothing
    tagged), retry ONCE with a slightly higher temperature so the model can
    break out of a deterministic failure mode.
    """
    prompt = build_tagging_prompt(searched_tag)
    content: List[Dict[str, Any]] = []
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": _frame_to_data_url(f)}})
    content.append({"type": "text", "text": prompt})

    raw = _post_vlm_once(content, llama_url, temperature=0.0, request_timeout=request_timeout)
    if not _is_empty_or_malformed(raw):
        return raw

    logger.warning(f"VLM first attempt empty/malformed for r/{searched_tag}, retrying with temperature=0.4")
    raw2 = _post_vlm_once(content, llama_url, temperature=0.4, request_timeout=request_timeout)
    return raw2 if raw2 else raw  # prefer the retry, fall back to whatever we had


# === JSON parsing / schema clamping ===

def _coerce_str(value: Any, vocabulary: List[str], default: Optional[str]) -> Optional[str]:
    if value is None:
        return default
    s = str(value).lower().strip()
    return s if s in vocabulary else default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _parse_tag_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse model output, clamp every field to the allowed vocabulary."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1:
            logger.warning(f"No JSON object in VLM output: {response_text[:200]}")
            return None

        raw = json.loads(response_text[start:end + 1])
        if not isinstance(raw, dict):
            return None

        # subjects: list of {type, description}; drop entries with an
        # unknown type, trim descriptions, cap the list.
        subjects: List[Dict[str, str]] = []
        subjects_raw = raw.get("subjects", []) or []
        if isinstance(subjects_raw, list):
            for entry in subjects_raw[:8]:
                if isinstance(entry, str):
                    entry = {"type": entry, "description": ""}
                if not isinstance(entry, dict):
                    continue
                s_type = _coerce_str(entry.get("type"), VALID_SUBJECT_TYPES, default=None)
                if s_type is None:
                    continue
                desc = str(entry.get("description", "") or "").strip()[:80]
                subjects.append({"type": s_type, "description": desc})

        activities_raw = raw.get("activities", []) or []
        if isinstance(activities_raw, str):
            activities_raw = [activities_raw]
        if isinstance(activities_raw, list):
            activities = sorted({
                a for a in (
                    _coerce_str(x, VALID_ACTIVITIES, default=None) for x in activities_raw
                ) if a is not None
            })
        else:
            activities = []

        setting = _coerce_str(raw.get("setting"), VALID_SETTINGS, default="other")
        mood = _coerce_str(raw.get("mood"), VALID_MOODS, default="neutral")
        camera = _coerce_str(raw.get("camera"), VALID_CAMERA, default="static")
        text_on_screen = _coerce_bool(raw.get("text_on_screen"), default=False)

        scene_description = str(raw.get("scene_description", "") or "").strip()
        if len(scene_description) > 400:
            scene_description = scene_description[:400]

        return {
            "subjects": subjects,
            "activities": activities,
            "setting": setting,
            "mood": mood,
            "camera": camera,
            "text_on_screen": text_on_screen,
            "scene_description": scene_description,
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Parse error: {e}, response: {response_text[:300]}")
        return None


# === Post-processing / consistency normalization ===

def _normalize_tags(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic consistency fixes to parsed VLM output.

    The model is asked to ground everything in what it sees, but it still
    occasionally asserts an activity with no subject that could perform it
    (e.g. "hiking" on a landscape-only clip). We drop those rather than
    invent a subject, and synthesize a scene description if the model left
    it blank so downstream prompt grounding always has something to use.
    """
    out = dict(parsed)
    out["subjects"] = [dict(s) for s in parsed.get("subjects", [])][:6]
    activities = list(parsed.get("activities", []))

    has_actor = any(s["type"] in ("person", "group", "animal") for s in out["subjects"])
    if not has_actor:
        activities = [a for a in activities if a not in SUBJECT_BOUND_ACTIVITIES]

    if not activities:
        # A static scene with no visible activity reads as "relaxing" for
        # landscape/food/object-only clips; otherwise fall back to "other".
        activities = ["relaxing"] if out["subjects"] and not has_actor else ["other"]

    out["activities"] = sorted(set(activities))[:6]

    if not out.get("scene_description"):
        out["scene_description"] = _build_scene_description_from_tags(out)
    return out


def _build_scene_description_from_tags(tags: Dict[str, Any]) -> str:
    """Synthesize a one-sentence scene description from the tags. Used when
    the model omitted scene_description or the normalization pass changed
    the tags enough that the original no longer matches.
    """
    subjects = tags.get("subjects") or []
    activities = [a for a in (tags.get("activities") or []) if a != "other"]
    setting = tags.get("setting") or "other"
    camera = tags.get("camera") or "static"

    if subjects:
        labels = [s["description"] or s["type"] for s in subjects]
        if len(labels) == 1:
            subject_phrase = labels[0]
        elif len(labels) == 2:
            subject_phrase = f"{labels[0]} and {labels[1]}"
        else:
            subject_phrase = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    else:
        subject_phrase = "an empty scene"

    parts = [subject_phrase]
    if activities:
        parts.append(", ".join(activities))
    if setting and setting != "other":
        location = {
            "outdoors": "outdoors", "city": "in a city", "beach": "on a beach",
            "road": "on the road", "home": "at home",
        }.get(setting, f"in a {setting}")
        parts.append(location)
    if camera == "drone":
        parts.append("(aerial shot)")
    elif camera == "timelapse":
        parts.append("(timelapse)")

    return " ".join(parts).strip().capitalize() + "."


# === Per-video tagging ===

def tag_background_video(video_id: int, llama_url: str) -> Dict[str, Any]:
    """Tag a single background video. Assumes the llama-server VLM profile is running."""
    with get_db_context() as db:
        video = db.query(BackgroundVideo).filter_by(id=video_id).first()
        if not video:
            return {"status": "error", "error": f"Video {video_id} not found"}

        if not video.storage_path or not os.path.exists(video.storage_path):
            video.ml_tagging_status = "error"
            video.ml_tagging_error = "File not found"
            video.ml_tagging_checked_at = datetime.utcnow()
            db.commit()
            return {"status": "error", "error": "File not found"}

        searched_tag = video.searched_tag
        video.ml_tagging_status = "processing"
        db.commit()

        try:
            frames = extract_tagging_frames(video.storage_path)
            if not frames:
                video.ml_tagging_status = "error"
                video.ml_tagging_error = "Could not extract frames"
                video.ml_tagging_checked_at = datetime.utcnow()
                db.commit()
                return {"status": "error", "error": "No frames extracted"}

            raw = _call_vlm(frames, llama_url, searched_tag=searched_tag)
            if raw is None:
                video.ml_tagging_status = "error"
                video.ml_tagging_error = "VLM HTTP/transport failure"
                video.ml_tagging_checked_at = datetime.utcnow()
                db.commit()
                return {"status": "error", "error": "VLM call failed"}

            parsed = _parse_tag_response(raw)
            if parsed is None:
                video.ml_tagging_status = "error"
                video.ml_tagging_error = f"Could not parse VLM output: {raw[:200]}"
                video.ml_tagging_checked_at = datetime.utcnow()
                db.commit()
                return {"status": "error", "error": "Parse failure", "raw": raw[:500]}

            corrected = _normalize_tags(parsed)

            video.ml_tags = corrected
            video.ml_tags_version = ML_TAGS_VERSION
            video.ml_tagging_status = "completed"
            video.ml_tagging_error = None
            video.ml_tagging_checked_at = datetime.utcnow()
            db.commit()

            logger.info(
                f"Tagged video {video_id} (r/{searched_tag}): "
                f"setting={corrected['setting']}, mood={corrected['mood']}, "
                f"camera={corrected['camera']}, activities={corrected['activities']}, "
                f"subjects={[s['type'] for s in corrected['subjects']]}, "
                f"text={corrected['text_on_screen']}, "
                f"desc='{corrected['scene_description'][:80]}'"
            )
            return {"status": "success", "video_id": video_id, "tags": corrected}

        except Exception as e:
            logger.error(f"Error tagging video {video_id}: {e}")
            video.ml_tagging_status = "error"
            video.ml_tagging_error = str(e)[:500]
            video.ml_tagging_checked_at = datetime.utcnow()
            db.commit()
            return {"status": "error", "error": str(e)}


# === GPU exclusivity helpers ===

def _snapshot_and_pause_competing_work(r: redis.Redis) -> Dict[str, Optional[bytes]]:
    snapshot: Dict[str, Optional[bytes]] = {k: r.get(k) for k in COMPETING_WORKLOAD_KEYS}
    from tasks.pipeline_orchestrator import disable_all_gpu_workloads
    disable_all_gpu_workloads()
    return snapshot


def _restore_competing_work(r: redis.Redis, snapshot: Dict[str, Optional[bytes]]) -> None:
    for key, prior in snapshot.items():
        if prior is None:
            r.delete(key)
        else:
            r.set(key, prior)
    logger.info("Restored competing GPU work flags")


# === Celery tasks ===

@celery_app.task(bind=True, queue="ml_tagging", time_limit=7200, soft_time_limit=7000)
def run_tagging_batch(self, video_ids: List[int]):
    """Tag a batch using the llama-server VLM profile.

    Lifecycle:
    1. Acquire ml_tagging:running lock.
    2. Pause competing GPU workloads (OCR, watermark filter, LLM batch, extraction).
    3. Stop llama-server (whatever profile is running).
    4. Start the VLM profile.
    5. Tag each video via chat completion.
    6. Stop the VLM profile, restart the llm profile.
    7. Restore competing flags.
    """
    r = get_redis_client()

    if not r.set(ML_TAGGING_LOCK_KEY, "1", nx=True, ex=ML_TAGGING_LOCK_TTL):
        return {"status": "locked"}

    results: List[Dict[str, Any]] = []
    flag_snapshot: Dict[str, Optional[bytes]] = {}

    try:
        from utils.llama_server_control import (
            LLAMA_SERVER_URL,
            is_server_running,
            start_server,
            stop_server,
        )

        flag_snapshot = _snapshot_and_pause_competing_work(r)

        # Stop any currently-running llama-server profile
        if is_server_running():
            logger.info("Stopping current llama-server profile before loading the VLM...")
            stop_server()
            time.sleep(15)  # CUDA needs time to release VRAM cleanly between profiles

        # Start the VLM profile. InternVL3-14B replaced a 7B VLM here: it is
        # markedly better at counting subjects and at hedging to "other" /
        # "neutral" on genuinely ambiguous frames instead of guessing.
        logger.info("Starting llama-server vlm-internvl3 profile (InternVL3-14B GGUF)...")
        started = start_server(profile="vlm-internvl3")
        if not started:
            raise RuntimeError("Failed to start llama-server VLM profile")

        # Tag every video
        for i, vid in enumerate(video_ids):
            logger.info(f"Tagging {i+1}/{len(video_ids)}: video {vid}")
            results.append(tag_background_video(vid, LLAMA_SERVER_URL))

        success = sum(1 for x in results if x.get("status") == "success")
        errors = sum(1 for x in results if x.get("status") == "error")
        logger.info(f"Batch complete: {success} success, {errors} errors")

    except Exception as e:
        logger.error(f"Tagging batch failed: {e}")
        results.append({"status": "batch_error", "error": str(e)})

    finally:
        # Always stop the VLM and restore llm (generation server) + flags
        try:
            from utils.llama_server_control import stop_server, start_server, is_server_running
            stop_server()
            time.sleep(30)  # CUDA needs time to fully release VRAM before reloading the 24B
            logger.info("Restarting llama-server llm profile...")
            for attempt in range(3):
                if start_server(profile="llm"):
                    break
                logger.warning(f"llm profile restart attempt {attempt+1} failed, sleeping 30s and retrying...")
                stop_server()  # Clean up any stuck process
                time.sleep(30)
        except Exception as e:
            logger.error(f"Failed to restore llm profile after tagging: {e}")

        if flag_snapshot:
            _restore_competing_work(r, flag_snapshot)

        r.delete(ML_TAGGING_LOCK_KEY)

    return {
        "status": "completed",
        "total": len(video_ids),
        "success": sum(1 for x in results if x.get("status") == "success"),
        "errors": sum(1 for x in results if x.get("status") == "error"),
    }


@celery_app.task(queue="maintenance")
def dispatch_ml_tagging_batch(batch_size: int = 30):
    """Collect untagged / stale-version backgrounds and send a batch to the tagging worker."""
    r = get_redis_client()

    enabled = r.get(ML_TAGGING_ENABLED_KEY)
    if enabled is None:
        r.set(ML_TAGGING_ENABLED_KEY, "1")
        enabled = b"1"
    if enabled != b"1":
        return {"status": "disabled"}

    if r.get(ML_TAGGING_LOCK_KEY):
        return {"status": "locked"}

    with get_db_context() as db:
        from sqlalchemy import or_, case
        # Pick BGs that need a tag pass — either never tagged ('pending')
        # OR tagged at an older ML_TAGS_VERSION. Order:
        #   1. status='pending' first (never-tagged BGs are highest priority
        #      because nothing downstream can use them yet),
        #   2. then v<ML_TAGS_VERSION BGs (re-tags) by id DESC (newest first
        #      — those are most likely to actually be used soon, and the
        #      oldest BGs in the corpus are most likely to be retired by
        #      reuse cooldowns anyway).
        # Without this priority order, a version bump dumps the whole corpus
        # onto the dispatcher and the ORDER BY id ASC starves the handful
        # of fresh 'pending' BGs that show up daily from the scraper.
        priority = case(
            (BackgroundVideo.ml_tagging_status == "pending", 0),
            else_=1,
        )
        pending = db.query(BackgroundVideo).filter(
            BackgroundVideo.filter_status == "approved",
            BackgroundVideo.storage_path.isnot(None),
            BackgroundVideo.source_type == "reddit",
            or_(
                BackgroundVideo.ml_tagging_status == "pending",
                (BackgroundVideo.ml_tagging_status == "completed") & (
                    (BackgroundVideo.ml_tags_version == None) |  # noqa: E711
                    (BackgroundVideo.ml_tags_version < ML_TAGS_VERSION)
                ),
            ),
        ).order_by(priority.asc(), BackgroundVideo.id.desc()).limit(batch_size).all()

        if not pending:
            return {"status": "no_pending", "queued": 0}

        video_ids = [v.id for v in pending]

    logger.info(f"Dispatching {len(video_ids)} videos for ML tagging (v{ML_TAGS_VERSION})")
    task = run_tagging_batch.delay(video_ids)
    return {"status": "dispatched", "queued": len(video_ids), "task_id": task.id}


@celery_app.task(queue="maintenance")
def reset_transient_tagging_errors(max_per_run: int = 200):
    """Reset videos that errored with transient transport failures so they can
    be retried. Runs on a beat schedule. Most "VLM HTTP/transport failure"
    errors are transient (mmproj edge cases or temporary server flakes) — a
    re-tag attempt on a different batch usually succeeds.
    """
    transient_patterns = (
        "VLM HTTP/transport failure",
        "VLM call failed",
    )
    with get_db_context() as db:
        from sqlalchemy import text, or_
        # Build OR conditions for each transient pattern
        clauses = " OR ".join([f"ml_tagging_error LIKE '{p}%'" for p in transient_patterns])
        result = db.execute(text(f"""
            UPDATE background_videos
            SET ml_tagging_status = 'pending',
                ml_tags = NULL,
                ml_tags_version = NULL,
                ml_tagging_error = NULL
            WHERE id IN (
                SELECT id FROM background_videos
                WHERE filter_status = 'approved'
                  AND source_type = 'reddit'
                  AND ml_tagging_status = 'error'
                  AND ({clauses})
                LIMIT :max_per_run
            )
        """), {"max_per_run": max_per_run})
        db.commit()
        count = result.rowcount
        if count:
            logger.info(f"Reset {count} transient tagging errors back to pending")
        return {"status": "success", "reset_count": count}


@celery_app.task(queue="maintenance")
def reset_all_reddit_bgs_for_retag():
    """One-shot: mark every approved reddit background for re-tagging at the current schema version."""
    with get_db_context() as db:
        from sqlalchemy import text
        result = db.execute(text("""
            UPDATE background_videos
            SET ml_tagging_status = 'pending',
                ml_tags = NULL,
                ml_tags_version = NULL,
                ml_tagging_error = NULL
            WHERE filter_status = 'approved'
              AND source_type = 'reddit'
              AND (ml_tags_version IS NULL OR ml_tags_version < :target_version)
        """), {"target_version": ML_TAGS_VERSION})
        db.commit()
        count = result.rowcount
        logger.info(f"Reset {count} reddit backgrounds for v{ML_TAGS_VERSION} re-tagging")
        return {"status": "success", "reset_count": count}
