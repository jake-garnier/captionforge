"""
LLM judge for generated captions. Stage 3 of the quality pipeline (README, "What it does").

Uses the same Mistral-Small-24B-Instruct that does generation (llama-server
"llm" profile, port 1234). Scores each caption on four 1-10 axes and
produces a pass/fail. We block the composition pipeline on judge_pass = True
so only quality captions get composed.

This module is stateless — it expects llama-server to be already running on
the "llm" profile. The pipeline orchestrator handles profile swaps and
this code just makes one HTTP call per caption.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from utils.llama_server_control import is_server_running, chat_completion

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


JUDGE_PROMPT_VERSION = "v5"  # bump whenever the rubric text below changes

# We deliberately ask for compact JSON to keep parsing reliable. Scores are
# integers 1-10 to make thresholding obvious.
SYSTEM_PROMPT = (
    "You are a strict editorial reviewer for short Reddit video captions "
    "(motivation, fitness, cooking, travel, and similar niches). "
    "You score captions on four axes 1-10 and return ONLY valid JSON. "
    "No prose, no markdown fences, no explanations outside JSON. "
    "Be honest — most captions are mediocre. A 7 means 'I would post this'. "
    "An 8 is good. 9-10 is reserved for genuinely excellent work."
)

USER_PROMPT_TEMPLATE = """Caption to score:
\"\"\"
{caption}
\"\"\"

Score on each axis 1-10:
- grammar: punctuation, capitalization, complete sentences, no typos.
- flow: reads naturally, no awkward repetition, ends on a strong image or line.
- bg_consistency: would plausibly accompany a short background clip for its niche; not generic; not narrating something a video can't show.
- appeal: how compelling and shareable the caption is for the niche audience; a specific, earned insight beats vague hype.

Reject signals — IF ANY OF THESE APPEAR, set the corresponding axis to 1-3 and DO NOT pass:
- ends with a rhetorical question ("right?", "isn't it?", "are you ready?")
- contains literal "Caption:" or "Caption 4b" or other slide/frame markers
- em-dashes (—), en-dashes (–), spaced hyphens ( - ), or semicolons (;) — these are post-stripped to commas anyway, so their presence means the writer ignored a hard rule. Set grammar=2.
- apostrophes (') in contractions like "you're", "don't", "coach's" — should be stripped to "youre", "dont", "coachs"
- repeats the same phrase or word stem
- generic praise with no concrete detail
- contradictions (says it is dawn and midnight, indoors and on a trail, etc)
- exceeds 120 words

PROMOTIONAL / TRAINING-DATA LEAKS — these AUTOMATICALLY fail the caption (set ALL axes <= 3):
- "Read all (the) text", "Output only the text", "preserving the reading order"
- "Want the. Full. Story." or any "1000+ CAPTIONS" / "Daily Updates" / "Access to."
- Promo codes ("SAVE25", "SPRING20", "20% off code:")
- URLs, ".com/", "join for free", "get a taste", "upgrading to"
- "BG source: r/..." (we leaked our prompt template)
- Multi-choice quiz format ("Choose. Your. Path.", "A) ... B) ... C) ...")
- Creator credits ("Edited. By. Studio Name", "Caption by ...")
- Bot-style stutter periods (e.g. "Want. The. Full. Story.")
- Anything that sounds like a Patreon / newsletter / channel tagline

GENERATION-PROMPT LEAKS — fragments of the writer's own system prompt
appearing inside the caption. Auto-fail with all axes <= 3:
- "Style:" or "STYLE:" anywhere in the caption (it's the prompt's
  per-niche style header)
- "Source:" or "BG source:" or "Scene:" or "Activities:" or "Setting:" or
  "Mood:" or "Camera:" labels (the prompt's BG-grounding header)
- "Hard arc" / "tight arc" / "setup, action, kicker" (length-rule
  echo)
- "motivational caption", "fitness caption", "cooking caption", "travel
  caption" — the prompt describes the niche category as a directive; if
  those literal category labels appear in caption text, the model is
  parroting the prompt rather than writing in voice
- "The video is N seconds", "is rejected automatically", "REJECTED
  automatically" (length-constraint echo)
- "do not write", "DO NOT" (negative-instruction echo)
- "second person" (POV-instruction echo)
- "vivid specific scenarios", "vivid, specific scenarios" (style
  directive echo)

FLORID / "GREETING CARD" REGISTER — these AUTOMATICALLY fail (set appeal=1, flow<=3, overall<=3):
The caption is supposed to read like a sharp Reddit post, not a greeting card or a novel. If the writing has slipped into florid, "literary" prose, fail it. Specific tells:
- "blurs into darkness", "whispers across", "cascading", "shimmering"
- "golden hour bathes", "lazy stripe of sunlight", "soul awakens"
- "savor every", "symphony of", "dances across", "painted with light"
- "the clock strikes/ticks past midnight"
- "a testament to", "in that moment, everything changed"
- clock/time openers in general
- Multi-clause sentences with semicolons or em-dashes mimicking literary style
A good caption uses plain, direct words appropriate to its niche. If the caption tip-toes around with abstractions and Victorian prose, it has fallen out of register and is unusable.

SCENE INVENTION — these AUTOMATICALLY fail bg_consistency to <=3 (the BG cant be matched against asserted specifics):
- Specific colors of clothing or gear: "red jacket", "blue yoga mat", "white apron"
- Named places, brands, or products not in the scene description: "at Yosemite", "on a Peloton", "with a Le Creuset"
- Exact times, dates, or weather the footage may not show: "at 5:03 AM", "in the snow"
The caption should use generic terms (your jacket, the trail, the pan, the road) so it stays compatible with any BG that matches the setting + activities + subjects tags.

Return ONLY this JSON shape:
{{
  "grammar": <int 1-10>,
  "flow": <int 1-10>,
  "bg_consistency": <int 1-10>,
  "appeal": <int 1-10>,
  "overall": <int 1-10>,
  "issues": [<short strings, empty list if none>]
}}"""


# Pass threshold: every axis must clear this AND overall must clear it.
PASS_THRESHOLD = 7


def _build_messages(caption: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption.strip())},
    ]


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of the model's response."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip code fences if the model wrapped despite instructions.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    # Find the first {...} block.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_score(val) -> Optional[int]:
    """Coerce model output to an int in [1, 10] or None."""
    try:
        n = int(round(float(val)))
    except (TypeError, ValueError):
        return None
    if n < 1 or n > 10:
        # Clamp into range — models occasionally drift.
        n = max(1, min(10, n))
    return n


def judge_caption(caption_text: str) -> Optional[dict]:
    """
    Score one caption with the LLM judge.

    Returns a dict with keys:
        scores: {"grammar", "flow", "bg_consistency", "appeal", "overall"}
        issues: list[str]
        passed: bool
        prompt_version: str
        raw: str (model's raw response, for debugging)

    Returns None if the judge call failed entirely (server down, malformed
    JSON, etc.). The caller should mark the row judge_status='failed' so
    the backlog scorer can retry it later.
    """
    if not is_server_running():
        logger.error("[JUDGE] llama-server not running; cannot judge caption")
        return None

    if not caption_text or len(caption_text.split()) < 3:
        # Don't waste tokens on obvious garbage. Score it as a hard fail.
        return {
            "scores": {"grammar": 1, "flow": 1, "bg_consistency": 1, "appeal": 1, "overall": 1},
            "issues": ["empty or trivially short"],
            "passed": False,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "raw": "<skipped>",
        }

    messages = _build_messages(caption_text)
    raw = chat_completion(messages, max_tokens=300, temperature=0.2)
    if not raw:
        logger.warning("[JUDGE] llama-server returned empty response")
        return None

    parsed = _extract_json(raw)
    if not parsed:
        logger.warning(f"[JUDGE] Could not parse JSON from response: {raw[:200]}")
        return None

    grammar = _normalize_score(parsed.get("grammar"))
    flow = _normalize_score(parsed.get("flow"))
    bg = _normalize_score(parsed.get("bg_consistency"))
    appeal = _normalize_score(parsed.get("appeal"))
    overall = _normalize_score(parsed.get("overall"))
    if None in (grammar, flow, bg, appeal, overall):
        logger.warning(f"[JUDGE] Missing/invalid score field in: {parsed}")
        return None

    issues_raw = parsed.get("issues") or []
    issues = [str(i)[:200] for i in issues_raw if isinstance(i, (str, int, float))][:10]

    scores = {
        "grammar": grammar,
        "flow": flow,
        "bg_consistency": bg,
        "appeal": appeal,
        "overall": overall,
    }

    passed = all(v >= PASS_THRESHOLD for v in scores.values())

    return {
        "scores": scores,
        "issues": issues,
        "passed": passed,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "raw": raw,
    }


def apply_judge_result(caption, result: Optional[dict]) -> None:
    """
    Mutate a GeneratedCaption row in-place from a judge result. Caller
    is responsible for committing the session.
    """
    if result is None:
        caption.judge_status = "failed"
        caption.judge_at = datetime.utcnow()
        return

    caption.judge_scores = result["scores"]
    caption.judge_issues = result["issues"]
    caption.judge_pass = bool(result["passed"])
    caption.judge_status = "judged"
    caption.judge_at = datetime.utcnow()


# ============================================================================
# Celery tasks
# ============================================================================


@celery_app.task(name="tasks.caption_judge.judge_backlog_batch", queue="gpu_llm",
                 time_limit=1800, soft_time_limit=1700)
def judge_backlog_batch(batch_size: int = 25):
    """
    Score un-judged generated_captions in batches.

    Picks rows where judge_status IS NULL or 'failed' (retry transient
    failures). Skips rows whose llama-server isn't running so we don't
    burn the batch on errors.

    Runs on the gpu_llm queue (GPU 0, where Mistral lives), same as the
    LLM refinement backlog. The pipeline orchestrator owns the broader
    GPU-config gate; this task only adds work to that queue.
    """
    if not is_server_running():
        logger.info("[JUDGE-BACKLOG] llama-server not running; skipping batch")
        return {"status": "skipped", "reason": "server_not_running"}

    from database.db import get_db_context
    from database.models import GeneratedCaption
    from sqlalchemy import or_

    judged = 0
    failed = 0
    with get_db_context() as db:
        rows = (
            db.query(GeneratedCaption)
            .filter(
                or_(
                    GeneratedCaption.judge_status.is_(None),
                    GeneratedCaption.judge_status == "failed",
                ),
                GeneratedCaption.caption_text.isnot(None),
            )
            .order_by(GeneratedCaption.id.desc())
            .limit(batch_size)
            .all()
        )

        if not rows:
            logger.info("[JUDGE-BACKLOG] No captions need judging")
            return {"status": "ok", "judged": 0, "failed": 0}

        for row in rows:
            try:
                result = judge_caption(row.caption_text)
                apply_judge_result(row, result)
                if result is None:
                    failed += 1
                else:
                    judged += 1
            except Exception as e:
                logger.warning(f"[JUDGE-BACKLOG] Caption {row.id} threw: {e}")
                row.judge_status = "failed"
                row.judge_at = datetime.utcnow()
                failed += 1
            db.commit()

    logger.info(f"[JUDGE-BACKLOG] judged={judged} failed={failed}")
    return {"status": "ok", "judged": judged, "failed": failed}
