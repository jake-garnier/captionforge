"""
Stage 2 of the quality pipeline (README, "What it does"): BG-first caption generation.

Old flow: generate captions blind, then post-hoc match them to whatever BG
happens to be available. Result: gym-crowd captions on a solo-runner BG,
recipe tips over a drone shot of a coastline, etc.

New flow:
  1. Pick a BG that's niche-compatible and not on cooldown.
  2. Build a prompt grounded in that BG's ml_tags (scene_description,
     subjects, activities, setting, mood, camera).
  3. Generate K candidates targeted at this scene.
  4. Judge each candidate inline with Stage 3 (Mistral-Small judge prompt).
  5. Highest-overall judge-pass candidate becomes status='winner';
     others stay status='candidate'.
  6. Composition selects winners and renders.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import or_

from tasks.celery_app import celery_app
from utils.llama_server_control import is_server_running, generate as llama_generate, get_loaded_model, restart_server

logger = logging.getLogger(__name__)


PROMPT_VERSION = "bg_first_v3"  # v3: per-BG dynamic word budget + hard max_tokens cap


# Per-niche style guidance. Mirrors the old prompts in automation_config.py
# but is now a *suffix* — the bulk of the prompt is built dynamically from
# the BG's ml_tags so the model has actual scene grounding.
NICHE_STYLE: dict[str, str] = {
    # Per-niche tone descriptions. Intentionally do NOT include literal
    # phrase examples — empirically the model copies
    # any concrete strings here verbatim, leading to 15-25% of outputs
    # repeating prompt-supplied phrases. Describe the vibe and POV; let
    # the model invent the wording.
    "motivation": (
        "Style: motivational second-person voice. Mix a blunt truth with "
        "encouragement and one concrete next action. The narrator is a "
        "straight-talking coach; the 'you' is someone on the edge of "
        "quitting or starting. Use vivid specific scenarios."
    ),
    "fitness": (
        "Style: training-floor voice. 'You' are mid-workout. Mix a practical "
        "cue (breathing, tempo, form), the reason it matters, and a push to "
        "finish the set. The narrator is a coach or your own inner voice. "
        "Use vivid specific scenarios — not generic."
    ),
    "cooking": (
        "Style: home-cook voice. 'You' are at the stove. One concrete "
        "technique or ingredient tip, why it works, and what the result "
        "tastes, smells, or looks like. Sensory and specific — heat, "
        "texture, timing. Use vivid specific scenarios — not generic."
    ),
    "travel": (
        "Style: wanderlust with a practical edge. 'You' are on the road or "
        "about to be. A specific kind of place or moment, one sensory "
        "detail, and a nudge to actually go. The narrator is a friend who "
        "has been there. Use vivid specific scenarios — not generic."
    ),
}


# Hard rules baked into every prompt — same list the Stage 3 judge enforces,
# stated upfront so the model doesn't generate trash that gets rejected.
HARD_RULES = """Hard rules — break any one and your output is rejected:
- 40-100 words. 2-5 sentences. Build an arc, don't be one-liner.
- Every sentence ends with proper punctuation (. ! ?). No mid-sentence periods.
- Second person ("you"). Don't switch to first person.
- PUNCTUATION: ONLY commas, periods, question marks, exclamation marks, and straight double quotes (") for dialogue. Banned: em-dash (—), en-dash (–), spaced hyphen ( - ), semicolon (;), apostrophe ('), curly quotes (' ' " "), ampersand (&), ellipsis character (…). The renderer cannot display em-dashes/semicolons/apostrophes correctly and they get stripped to commas/dropped, so writing them produces garbled output. Write 'dont', 'cant', 'youre', 'its', 'coachs' (no apostrophe). Write 'and' (not '&'). Use commas where you would use a semicolon.
- NO "Caption:", "Read all text", "com/", "Join now", URLs, or promotional text.
- Don't ask rhetorical questions to end. End on a concrete sensory image.
- Don't repeat the same word/phrase 3+ times.
- Don't list tags or use markdown.
- Use vivid, specific imagery. No generic praise.

REGISTER (very important): Write plain, direct, Reddit-style prose.
Use concrete everyday words appropriate to the niche (reps, sets, the
trail, the pan, the road, the alarm clock). NOT abstractions. This is
sharp and specific — not literary, not a greeting card. Phrases to
AVOID (instant rejection if you stack 2+ of them):
- "blurs into darkness", "whispers across", "cascading", "shimmering"
- "golden hour bathes", "lazy stripe of sunlight", "soul awakens"
- "savor every", "symphony of", "dances across", "painted with light"
- "the clock strikes/ticks past midnight"
- "a testament to", "in that moment, everything changed"
You are not writing a novel. You are writing a Reddit caption that gets
upvoted on a niche-specific captions sub.

SCENE GROUNDING: Do NOT invent colors, brands, place names, weather, or
times of day that arent in the scene description. The video will be
matched at composition time by SETTING, ACTIVITIES, and SUBJECTS only,
not by those specifics. So:
- DO NOT write "red jacket", "blue yoga mat", "at Yosemite", "on a
  Peloton", "with a Le Creuset", "at 5:03 AM", "in the snow".
- INSTEAD use generic terms: "your jacket", "the trail", "the pan",
  "the road", "before sunrise".
- The reader's BG might be a forest trail or a coastal road — your
  caption has to read right against EITHER.

Opener diversity (very important):
- Do not start two consecutive captions with the same construction.
- Avoid "greeting card" openers — clock/time scene-setters, soft-focus
  travelogue intros, weather/lighting tableau.
- Open with a specific moment, situation, or sensory detail concrete to
  THIS scene's tags. Use the clip's activities, setting, and subjects as
  the opener anchor.
- Vary sentence structure: not every caption should be commands. Mix
  observation, command, internal monologue, dialogue.
- DO NOT copy any phrase from these instructions verbatim. The opener
  must come from the scene, not from this prompt."""


def _format_subjects(subjects: list) -> str:
    """Turn [{"type":"person","description":"runner in a jacket"}, ...] into
    'runner in a jacket and bowl of ramen'. Falls back to the type when the
    description is empty."""
    labels = []
    for s in subjects or []:
        if isinstance(s, dict):
            labels.append(s.get("description") or s.get("type") or "")
        elif isinstance(s, str):
            labels.append(s)
    labels = [l for l in labels if l]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def build_bg_grounded_prompt(
    niche: str,
    ml_tags: dict,
    bg_subreddit: Optional[str],
    bg_duration_seconds: Optional[float] = None,
) -> str:
    """
    Build a Mistral-Small prompt that grounds caption generation in this
    specific BG's scene. The model produces captions that reference what
    the video actually shows AND fit within the BG's duration.
    """
    if not ml_tags:
        # Fallback when ml_tags missing — fall back gracefully to niche-only.
        scene_desc = "(scene description unavailable; write a generic caption for this niche)"
        activities_str = "unknown"
        setting = "unknown"
        mood = "unknown"
        camera = "unknown"
        subjects_str = "(unknown)"
        people_visible = None
    else:
        scene_desc = ml_tags.get("scene_description") or "(unavailable)"
        activities = [a for a in (ml_tags.get("activities") or []) if a != "other"]
        activities_str = ", ".join(activities) if activities else "none visible"
        setting = ml_tags.get("setting") or "unknown"
        mood = ml_tags.get("mood") or "unknown"
        camera = ml_tags.get("camera") or "unknown"
        subjects = ml_tags.get("subjects") or []
        subjects_str = _format_subjects(subjects) or "(unknown)"
        people_visible = any(
            isinstance(s, dict) and s.get("type") in ("person", "group") for s in subjects
        )

    style = NICHE_STYLE.get(niche, "Direct, specific, conversational.")
    bg_hint = f"\nBG source: r/{bg_subreddit}" if bg_subreddit else ""

    # Don't clip caption length per-BG. Aim for the full 40-100 word range
    # always — a 2-4 sentence caption with a real arc reads infinitely
    # better than a single-line "Just keep going" mode-collapsed
    # output. If the BG is too short to fit a real caption, that's a
    # composition-side concern (skip the BG, or pick a longer one); don't
    # neuter the writer.
    # Per-BG word budget tied to duration. Renderer reads ~150 WPM (~2.5 wps)
    # and adds a 3s read buffer. We give the model a tight target so the
    # output actually fits the BG it was generated for. The LoRA was trained
    # on 200-word Reddit captions, so without this constraint it produces
    # captions too long for the BG and they get discarded at composition.
    if bg_duration_seconds and bg_duration_seconds > 0:
        target_max = int(max(20, min((float(bg_duration_seconds) - 3.0) * 2.5, 100)))
        target_min = max(15, target_max - 30)
        length_constraint = (
            f"\nLENGTH (HARD): {target_min}-{target_max} words. The video is "
            f"{bg_duration_seconds:.0f} seconds long; a longer caption is "
            f"REJECTED automatically. Aim for a tight arc (setup, action, "
            f"kicker) within {target_max} words. Stop at {target_max} words "
            f"even mid-thought; an unfinished short caption is acceptable."
        )
    else:
        length_constraint = (
            "\nLENGTH: 40-80 words across 2-4 sentences. Build a small arc "
            "(setup, action, kicker)."
        )

    # People-visible guard. The #1 mismatch we saw in Stage 4 review was
    # captions narrating a person ("watch her form", "he laces up") over a
    # landscape or food-only BG. Bake the constraint in.
    if people_visible is None:
        people_constraint = ""
    elif people_visible:
        people_constraint = (
            "\nA person is visible in the video. The caption may describe what "
            "they are doing, but keep it consistent with the listed activities."
        )
    else:
        people_constraint = (
            "\nIMPORTANT: NO person is visible in this video — it is scenery, "
            "food, a vehicle, or objects only. Do NOT narrate someone on screen "
            "('watch him', 'she grabs the bar'). Address the reader directly "
            "and let the scene stand on its own."
        )

    return f"""You are writing a single Reddit-style video caption.

{style}

The caption MUST match this specific video:
- Scene: {scene_desc}
- Subjects: {subjects_str}
- Activities: {activities_str}
- Setting: {setting}
- Mood: {mood}
- Camera: {camera}{bg_hint}
{length_constraint}
{people_constraint}

{HARD_RULES}

Now write ONE caption. Output nothing but the caption text."""


def _ensure_llm_profile_with_lora(niche: str) -> bool:
    """
    Make sure llama-server is on the 'llm' profile (Mistral-Small) with this
    niche's LoRA adapter loaded. If a different model is loaded (e.g. VLM
    profile from ml-tagging), stop and restart with the correct config.

    Returns True if the server is correctly configured after this call.
    """
    if is_server_running():
        loaded = get_loaded_model()
        loaded_id = (loaded or {}).get("id", "") if loaded else ""
        # Mistral GGUF filename contains 'Mistral-Small' — that's our LLM profile.
        if "Mistral-Small" in loaded_id or "mistral-small" in loaded_id.lower():
            logger.info(f"[BG-FIRST] llama-server already on LLM profile ({loaded_id})")
            # NOTE: we don't currently introspect whether the *correct* LoRA
            # is hot-loaded. If the niche was switched mid-day, the operator
            # would need to manually restart. For now we trust the most
            # recently set adapter is correct for the niche in flight.
            return True
        logger.warning(
            f"[BG-FIRST] llama-server has wrong model loaded ({loaded_id}); "
            f"restarting with LLM profile + {niche} LoRA"
        )
    else:
        logger.warning(f"[BG-FIRST] llama-server not running; starting LLM profile + {niche} LoRA")

    ok = restart_server(niche=niche, profile="llm")
    if not ok:
        logger.error(f"[BG-FIRST] Failed to start llama-server LLM profile for {niche}")
        return False
    logger.info(f"[BG-FIRST] llama-server now on LLM profile + {niche} LoRA")
    return True


# How far back to look for duplicate captions. The current motivation LoRA is
# mode-collapsed onto ~3 stories, so we want a generous window — a 14d look-
# back catches all of the recent attractor outputs. If the LoRA gets retrained
# and starts producing more variety this can shrink without changing behavior.
DEDUP_LOOKBACK_DAYS = 14
# How many leading words count as the "same caption". Two captions that share
# the first 12 words almost certainly diverged later only because of sampling
# noise around an already-collapsed opener. Reject those — they're the
# duplicate-perception failure mode the user reported.
DEDUP_PREFIX_WORDS = 12


def _caption_dedup_key(text: str) -> str:
    """
    Stable key for near-duplicate detection.

    Lowercased, punctuation-stripped, first DEDUP_PREFIX_WORDS words. This
    catches the "same opener, slight reword" collapse where the LoRA picks
    the same attractor and then tail words drift slightly because of
    sampling. We don't try to be cute with semantic dedup — exact prefix
    match is fast and the failure pattern is exact-prefix.
    """
    import re as _re
    cleaned = _re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = cleaned.split()[:DEDUP_PREFIX_WORDS]
    return " ".join(words)


# --- Corpus-level n-gram diversity ---------------------------------------
# When picking which of N candidates becomes the BG's winner, the raw
# judge_overall score is "is this caption good in isolation". That misses
# the global problem we hit at scale: the LoRA collapses onto a small set
# of attractor phrases ('one more rep', 'the road ahead', 'low and slow
# until it falls apart') and even with judge_overall=10 each, the 50th caption
# reading the same as the 1st is bad output.
#
# REPETITION_NGRAM_SIZE: which n-gram length to compare on. 4 is the
#   sweet spot — 3-grams are too noisy (every caption shares some), 5+
#   only catches long verbatim copies. 4-grams catch attractor phrases
#   like "one more rep and then" and "the road ahead of you".
# REPETITION_LOOKBACK_DAYS: how far back to mine winner captions for the
#   "what does the corpus already sound like" baseline.
# REPETITION_MAX_PENALTY: max points deducted from judge_overall per
#   100% overlap. judge_overall is on a 1-10 scale, so 3.0 means a fully
#   recycled caption can drop from 10 to 7 — significant but not enough
#   to elevate a judge=4 caption over a judge=8 one. The intent is to
#   tie-break, not override quality.
REPETITION_NGRAM_SIZE = 4
REPETITION_LOOKBACK_DAYS = 14
REPETITION_MAX_PENALTY = 3.0


def _ngrams(text: str, n: int) -> set[str]:
    """Lowercased, punctuation-stripped n-grams as a set (no repeats per caption)."""
    import re as _re
    cleaned = _re.sub(r"[^a-z0-9\s]", " ", text.lower())
    toks = cleaned.split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _load_recent_winner_ngrams(db, niche: str) -> set[str]:
    """
    Mine n-grams from recent winning captions for this niche so we can
    penalize candidates that recycle them.

    Pulls from caption_candidates with status IN ('winner', 'composing',
    'composed') — those are the captions actually exiting the funnel.
    Discarded/candidate rows aren't part of "what the corpus sounds like".
    """
    from datetime import timedelta
    from database.models import CaptionCandidate

    cutoff = datetime.utcnow() - timedelta(days=REPETITION_LOOKBACK_DAYS)
    rows = (
        db.query(CaptionCandidate.caption_text)
        .filter(
            CaptionCandidate.niche == niche,
            CaptionCandidate.status.in_(("winner", "composing", "composed")),
            CaptionCandidate.created_at > cutoff,
        )
        .all()
    )
    out: set[str] = set()
    for (txt,) in rows:
        if not txt:
            continue
        out |= _ngrams(txt, REPETITION_NGRAM_SIZE)
    return out


def _repetition_score(text: str, baseline_ngrams: set[str]) -> float:
    """
    Fraction of this caption's 4-grams that already appear in the recent-
    winners baseline. 0.0 = entirely fresh phrasing, 1.0 = every 4-gram
    has been seen before.
    """
    cand = _ngrams(text, REPETITION_NGRAM_SIZE)
    if not cand:
        return 0.0
    overlap = sum(1 for g in cand if g in baseline_ngrams)
    return overlap / len(cand)


def _is_duplicate_caption(db, niche: str, text: str) -> bool:
    """
    True if a caption with the same dedup key was already saved for this
    niche in the last DEDUP_LOOKBACK_DAYS. Cheap: pulls only recent rows
    for this niche and computes the key in Python (avoids needing a
    PostgreSQL functional index just for this).
    """
    from datetime import timedelta
    from database.models import GeneratedCaption

    key = _caption_dedup_key(text)
    if not key:
        # Empty key = caption too short to dedup on; let it through and let
        # the min-words filter below reject it if needed.
        return False

    cutoff = datetime.utcnow() - timedelta(days=DEDUP_LOOKBACK_DAYS)
    recent = (
        db.query(GeneratedCaption.caption_text)
        .filter(
            GeneratedCaption.niche == niche,
            GeneratedCaption.generated_at > cutoff,
        )
        .all()
    )
    for (prev_text,) in recent:
        if _caption_dedup_key(prev_text) == key:
            return True
    return False


def _generate_for_bg_impl(background_video_id: int, niche: str, n_candidates: int = 10) -> dict:
    """Plain-function body so we can call from another task without going through celery dispatch."""
    from database.db import get_db_context
    from database.models import BackgroundVideo, CaptionCandidate
    from tasks.caption_judge import judge_caption, apply_judge_result
    from utils.generation_postprocessor import clean_generated_caption, should_reject_caption

    # Make sure we're talking to Mistral-Small with the niche LoRA, not the
    # VLM profile that ml-tagging might have left loaded.
    if not _ensure_llm_profile_with_lora(niche):
        return {"status": "error", "reason": "could_not_load_llm_profile"}

    if not is_server_running():
        logger.error(f"[BG-FIRST] llama-server not up; can't generate for BG {background_video_id}")
        return {"status": "error", "reason": "server_not_running"}

    with get_db_context() as db:
        bg = db.query(BackgroundVideo).filter_by(id=background_video_id).first()
        if not bg:
            return {"status": "error", "reason": "bg_not_found"}

        ml_tags = bg.ml_tags or {}
        bg_subreddit = bg.searched_tag
        bg_duration = bg.duration_seconds

        prompt_body = build_bg_grounded_prompt(niche, ml_tags, bg_subreddit, bg_duration)
        formatted = f"<s>[INST] {prompt_body} [/INST]\n"

        # Hard-cap max_tokens so the LoRA-trained model physically cannot
        # run past the BG's word budget. Mistral tokenizes narrative English
        # at ~1.3 tokens/word, but the LoRA was trained on long captions and
        # ignores prompt-level length instructions. We need the token cap to
        # actually clip the output. Use 1.0 multiplier (= word_budget tokens
        # ≈ 0.77 × word_budget words at 1.3 t/w) — generous floor 60, hard
        # ceiling 130 (≈ 100 words max regardless of BG length, which fits
        # the largest BG we'd compose against without bloat).
        if bg_duration and bg_duration > 0:
            word_budget = max(20, int((float(bg_duration) - 3.0) * 2.5))
            max_tokens_per_caption = max(60, min(word_budget + 10, 130))
        else:
            max_tokens_per_caption = 100

        produced = 0
        judged_pass = 0
        judged_fail = 0
        judge_failed = 0
        best_candidate_id: Optional[int] = None
        best_effective_score = -1.0  # judge_overall - REPETITION_MAX_PENALTY * rep_score
        best_overall_for_log = -1
        best_rep_score_for_log = 0.0

        # Mine recent winners once per BG cycle. ~10-100ms, vs running it
        # per-candidate which would be 10x that for no benefit.
        recent_ngrams = _load_recent_winner_ngrams(db, niche)
        logger.debug(
            f"[BG-FIRST] BG {background_video_id} ({niche}): "
            f"loaded {len(recent_ngrams)} recent-winner {REPETITION_NGRAM_SIZE}-grams"
        )

        for i in range(n_candidates):
            # Diversity knobs walked back from a previous over-correction.
            # Setting temp=1.15 / top_p=0.98 / rep_pen=1.4 broke the attractor
            # stories but flushed the model out of the niche LoRA's plain
            # Reddit register into base Mistral's literary register — output
            # read like a novel ("lazy stripe of sunlight", "golden hour
            # bathes the trail"). The high rep_pen actively forbade the
            # niche's everyday vocabulary (reps, trail, pan) once it appeared,
            # forcing thesaurus alternatives.
            #
            # Current settings: temp=1.05 (just above baseline 1.0), top_p=0.95
            # (back to original), rep_pen=1.15 (low — the dedup_then_resample
            # below + per-call seed are doing the variety work, not the
            # rep_pen). seed=None forces a fresh RNG per call so identical
            # prompts don't collapse to identical outputs.
            raw = llama_generate(
                prompt=formatted,
                max_tokens=max_tokens_per_caption,
                temperature=1.05,
                repetition_penalty=1.15,
                top_p=0.95,
                seed=None,
            )
            if not raw:
                continue

            text = clean_generated_caption(raw.strip(), aggressive=True, niche=niche)
            if not text:
                continue

            # Cheap pre-filter — saves a judge call on obvious gibberish.
            rejected, reason = should_reject_caption(text)
            if rejected:
                logger.debug(f"[BG-FIRST] BG {background_video_id} cand {i+1}: pre-rejected ({reason})")
                continue

            # Reject too-short captions (mode-collapsed one-liners).
            # Threshold scales with BG: a 30s BG needs ≥15 words; a 60s BG
            # needs ≥25. Always accept if model produced 75% of the budget.
            word_count = len(text.split())
            if bg_duration and bg_duration > 0:
                min_words = max(12, int(((float(bg_duration) - 3.0) * 2.5) * 0.5))
            else:
                min_words = 25
            if word_count < min_words:
                logger.debug(
                    f"[BG-FIRST] BG {background_video_id} cand {i+1}: "
                    f"too short ({word_count}w < {min_words}), dropping"
                )
                continue
            # Don't reject for being "too long for the BG" here — that's
            # composition's job. The writer should produce a real caption.

            # Near-duplicate guard: reject if a caption with the same opener
            # was already saved for this niche in the last DEDUP_LOOKBACK_DAYS.
            # Forces the model to keep sampling rather than persisting the
            # 4th copy of the same story. Loop keeps trying until we hit a
            # novel one or run out of candidate slots.
            if _is_duplicate_caption(db, niche, text):
                logger.info(
                    f"[BG-FIRST] BG {background_video_id} cand {i+1}: "
                    f"duplicate of recent caption (key='{_caption_dedup_key(text)}') — resampling"
                )
                continue

            candidate = CaptionCandidate(
                background_video_id=background_video_id,
                niche=niche,
                prompt_version=PROMPT_VERSION,
                caption_text=text,
                llm_model="mistral-small-24b-instruct",
                generation_temperature=1.0,
                judge_status="pending",
                status="candidate",
            )
            db.add(candidate)
            db.flush()  # populate candidate.id

            # Stage 3 judge inline.
            try:
                judge_result = judge_caption(text)
                apply_judge_result(candidate, judge_result)
                if judge_result is None:
                    judge_failed += 1
                else:
                    overall = judge_result["scores"]["overall"]
                    candidate.judge_overall = overall
                    if judge_result["passed"]:
                        judged_pass += 1
                        # Corpus-level repetition penalty. Captions whose
                        # 4-grams overlap heavily with recent winners get
                        # demoted before the winner pick, so the corpus
                        # doesn't collapse onto attractor phrases.
                        rep_score = _repetition_score(text, recent_ngrams)
                        effective_score = overall - REPETITION_MAX_PENALTY * rep_score
                        if effective_score > best_effective_score:
                            best_effective_score = effective_score
                            best_candidate_id = candidate.id
                            best_overall_for_log = overall
                            best_rep_score_for_log = rep_score
                    else:
                        judged_fail += 1
                        candidate.status = "discarded"
            except Exception as e:
                logger.warning(f"[BG-FIRST] judge threw on BG {background_video_id} cand {i+1}: {e}")
                candidate.judge_status = "failed"
                judge_failed += 1

            produced += 1
            db.commit()

        # Promote the best (judge_overall - repetition_penalty) candidate to 'winner'.
        if best_candidate_id is not None:
            winner = db.query(CaptionCandidate).filter_by(id=best_candidate_id).first()
            if winner:
                winner.status = "winner"
                db.commit()

    if best_candidate_id is not None:
        winner_summary = (
            f"winner_overall={best_overall_for_log} "
            f"rep_score={best_rep_score_for_log:.2f} "
            f"effective={best_effective_score:.2f}"
        )
    else:
        winner_summary = "winner=none"
    logger.info(
        f"[BG-FIRST] BG {background_video_id} ({niche}): produced={produced}, "
        f"judge_pass={judged_pass}, judge_fail={judged_fail}, judge_errored={judge_failed}, "
        f"{winner_summary}"
    )
    return {
        "status": "ok",
        "background_video_id": background_video_id,
        "produced": produced,
        "judge_pass": judged_pass,
        "judge_fail": judged_fail,
        "winner_overall": best_overall_for_log if best_candidate_id else None,
    }


@celery_app.task(
    bind=True,
    name="tasks.bg_first_generation.generate_for_bg",
    queue="maintenance",
    time_limit=900,
    soft_time_limit=850,
)
def generate_for_bg(self, background_video_id: int, niche: str, n_candidates: int = 10):
    """Celery wrapper around _generate_for_bg_impl for direct dispatch."""
    return _generate_for_bg_impl(background_video_id, niche, n_candidates)


@celery_app.task(
    bind=True,
    name="tasks.bg_first_generation.run_bg_first_cycle",
    queue="maintenance",
    time_limit=7200,
    soft_time_limit=7000,
)
def run_bg_first_cycle(self, niche: str, n_bgs: int = 30, candidates_per_bg: int = 10):
    """
    One full BG-first generation cycle for a niche:
      - pick `n_bgs` un-targeted, niche-compatible BGs
      - for each, generate `candidates_per_bg` candidates and pick a winner

    The orchestrator calls this when a niche needs more captions to compose.
    """
    from database.db import get_db_context
    from database.models import BackgroundVideo, CaptionCandidate, ComposedVideo
    from config.niche_rules import is_subreddit_compatible, check_tag_rules, score_tag_preferences

    # CRITICAL: ensure llama-server is on the LLM profile with this niche's
    # LoRA adapter loaded. The VLM profile (InternVL3-14B) used for ml_tagging
    # might still be loaded from an earlier batch — we must swap it for
    # Mistral-Small + the niche LoRA before generating, otherwise we generate
    # captions with a generic vision model and the model thinks fine.
    if not _ensure_llm_profile_with_lora(niche):
        logger.error(f"[BG-FIRST CYCLE] {niche}: cannot start cycle, llama-server LLM profile failed to load")
        return {"status": "error", "niche": niche, "reason": "llm_profile_load_failed"}

    with get_db_context() as db:
        # BGs that already have a winner candidate for this niche — skip.
        bgs_with_winner = db.query(CaptionCandidate.background_video_id).filter(
            CaptionCandidate.niche == niche,
            CaptionCandidate.status == "winner",
        ).distinct().all()
        bgs_with_winner_ids = [r[0] for r in bgs_with_winner]

        # BGs already used in a composed video — skip.
        used_bgs = db.query(ComposedVideo.background_video_id).filter(
            ComposedVideo.background_video_id.isnot(None),
        ).distinct().all()
        used_bg_ids = [r[0] for r in used_bgs]

        excluded = set(bgs_with_winner_ids + used_bg_ids)

        # Approved Reddit BGs only, ml_tags must be present so we can ground.
        # Min duration: 30s. v2 prompt asks for 40-100 word captions which
        # require 19-43s of video time at ~150 WPM. 30s minimum lets a 60-word
        # caption fit comfortably; we have ~3.6k BGs >= 30s so the pool is
        # plenty deep.
        #
        # Skip BGs flagged ml_tags.text_on_screen=true. Stage 4 reviews
        # surfaced compositions with two competing text layers (burned-in
        # source captions + rendered overlays). Reject these at selection
        # time. Keeps text_on_screen=false AND missing/null — only drops 'true'.
        text_on_screen_value = BackgroundVideo.ml_tags["text_on_screen"].astext
        candidates_query = db.query(BackgroundVideo).filter(
            BackgroundVideo.filter_status == "approved",
            BackgroundVideo.source_type == "reddit",
            BackgroundVideo.ml_tags.isnot(None),
            BackgroundVideo.duration_seconds.isnot(None),
            BackgroundVideo.duration_seconds >= 30,
            or_(text_on_screen_value.is_(None), text_on_screen_value != "true"),
        ).order_by(BackgroundVideo.id.desc())
        candidates = candidates_query.all()
        logger.info(
            f"[BG-FIRST CYCLE] {niche}: {len(candidates)} eligible candidate BGs "
            f"after text_on_screen filter"
        )

        # Two gates plus a soft ranking, all from config/niche_rules.py:
        #   1. source-subreddit allow-list,
        #   2. required/forbidden scene-tag rules,
        #   3. preferred clauses → most-preferred BGs first (ties keep id DESC).
        eligible: list[tuple[int, int]] = []
        rejected_by_rules = 0
        for bg in candidates:
            if bg.id in excluded:
                continue
            if not is_subreddit_compatible(bg.searched_tag, niche):
                continue
            passes, _reasons = check_tag_rules(bg.ml_tags, niche)
            if not passes:
                rejected_by_rules += 1
                continue
            eligible.append((score_tag_preferences(bg.ml_tags, niche), bg.id))
        eligible.sort(key=lambda t: (-t[0], -t[1]))
        chosen = [bg_id for _, bg_id in eligible[:n_bgs]]

        logger.info(
            f"[BG-FIRST CYCLE] {niche}: selected {len(chosen)} BGs (target {n_bgs}); "
            f"{rejected_by_rules} rejected by scene-tag rules"
        )

    if not chosen:
        return {"status": "ok", "message": "no eligible BGs", "niche": niche, "n_bgs": 0}

    # Process serially — llama-server has only one slot. Parallelizing
    # candidate generation would just queue them anyway. Iterating in this
    # task is fine because we have a 2h time_limit.
    summary = {"bgs_processed": 0, "winners": 0, "no_winner": 0}
    try:
        for bg_id in chosen:
            # Call the plain function so the loop blocks (one llama-server slot).
            result = _generate_for_bg_impl(
                background_video_id=bg_id,
                niche=niche,
                n_candidates=candidates_per_bg,
            )
            if isinstance(result, dict):
                summary["bgs_processed"] += 1
                if result.get("winner_overall") is not None:
                    summary["winners"] += 1
                else:
                    summary["no_winner"] += 1
    finally:
        # Always clear the sentinel so the orchestrator can advance, even
        # if we crashed mid-cycle.
        try:
            import redis
            from config.settings import settings
            redis.from_url(settings.celery_broker_url).delete(f"bg_first_cycle:{niche}")
        except Exception as e:
            logger.warning(f"[BG-FIRST CYCLE] {niche}: could not clear sentinel: {e}")

    logger.info(f"[BG-FIRST CYCLE] {niche} done: {summary}")
    return {"status": "ok", "niche": niche, **summary}
