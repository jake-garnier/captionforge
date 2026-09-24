#!/usr/bin/env python3
"""
Stage 4 visual review for composed videos. Runs inside Claude Code.

Workflow:
    1. Export:    docker-compose exec api python scripts/claude_review_videos.py export --niche motivation --limit 30
                  This pulls the next N pending composed videos, extracts 3
                  evenly-spaced keyframes per video, and writes a brief markdown
                  file per video into scripts/claude_review_batches/.

    2. Review:    A Claude Code session opens the batch directory, reads each
                  brief, looks at the 3 keyframes alongside the caption, and
                  writes a result JSONL line to scripts/claude_review_results/.
                  See CLAUDE_REVIEW_INSTRUCTIONS.md for the rubric.

    3. Writeback: docker-compose exec api python scripts/claude_review_videos.py writeback
                  Reads the result files and updates composed_videos rows.

The export step uses ffmpeg to grab keyframes at 25 / 50 / 75% of duration.
Frames go to scripts/claude_review_batches/<video_id>/frame_N.jpg, and the
brief is scripts/claude_review_batches/<video_id>/brief.md.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Both paths work whether we're inside the container or on the host repo.
# Inside the container the bind mount makes /app/ === host repo root.
DEFAULT_BATCH_DIR = "/app/scripts/claude_review_batches"
DEFAULT_RESULTS_DIR = "/app/scripts/claude_review_results"
COMPOSED_DIR = "/data/composed_videos"


def _ffprobe_duration(path: str) -> float:
    """Return duration in seconds via ffprobe. Returns 0 on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return float(result.stdout.strip() or 0)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def _extract_frame(video_path: str, timestamp: float, out_path: str) -> bool:
    """Extract one frame at the given timestamp (seconds). Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.2f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "3",
                "-vf", "scale='min(720,iw)':-2",
                out_path,
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _chown_to_host_owner(path: str):
    """
    The api container runs as root, but the bind-mounted repo on the host
    is owned by ubuntu (uid 1000). If we leave root-owned files in
    /app/scripts (which is the repo's scripts/ dir), the next GitHub
    Actions checkout fails with EACCES when it tries to clean the
    workspace. So after writing batches/results we chown back to 1000:1000.
    """
    if not os.path.exists(path):
        return
    try:
        # Walk the tree and chown everything. os.chown(uid=1000, gid=1000)
        # works when running as root inside the container.
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    os.chown(os.path.join(root, d), 1000, 1000)
                except (PermissionError, OSError):
                    pass
            for f in files:
                try:
                    os.chown(os.path.join(root, f), 1000, 1000)
                except (PermissionError, OSError):
                    pass
        try:
            os.chown(path, 1000, 1000)
        except (PermissionError, OSError):
            pass
    except Exception:
        # Best-effort. Don't fail the run if chown is unavailable.
        pass


def export_pending(niche: str = None, limit: int = 30, batch_dir: str = DEFAULT_BATCH_DIR):
    """Mark `limit` videos as claude_review_status='pending', extract frames, write briefs."""
    sys.path.insert(0, "/app")
    from database.db import get_db_context
    from database.models import ComposedVideo, GeneratedCaption, BackgroundVideo

    os.makedirs(batch_dir, exist_ok=True)

    with get_db_context() as db:
        query = db.query(ComposedVideo).filter(
            ComposedVideo.status == "completed",
            ComposedVideo.claude_review_status.is_(None),
        )
        if niche:
            query = query.filter(ComposedVideo.niche == niche)
        # Prefer videos that have already been approval-reviewed by the user
        # OR are pending — skip the rejected ones entirely.
        query = query.filter(
            ComposedVideo.approval_status.in_(["pending", "approved"])
        )
        rows = query.order_by(ComposedVideo.id.desc()).limit(limit).all()

        if not rows:
            print(f"No pending composed videos for niche={niche}")
            return 0

        manifest = []
        for cv in rows:
            video_id = cv.id
            video_dir = Path(batch_dir) / str(video_id)
            video_dir.mkdir(parents=True, exist_ok=True)

            video_path = cv.storage_path
            if not video_path.startswith("/"):
                video_path = os.path.join(COMPOSED_DIR, video_path)
            if not os.path.exists(video_path):
                print(f"  video {video_id}: file missing at {video_path}, skipping")
                shutil.rmtree(video_dir, ignore_errors=True)
                continue

            duration = cv.duration_seconds or _ffprobe_duration(video_path)
            if duration <= 0:
                print(f"  video {video_id}: bad duration, skipping")
                shutil.rmtree(video_dir, ignore_errors=True)
                continue

            # 3 frames at 25 / 50 / 75% so we never grab a black opening or
            # final-fade frame.
            timestamps = [duration * 0.25, duration * 0.50, duration * 0.75]
            frame_paths = []
            for i, ts in enumerate(timestamps, 1):
                frame_path = str(video_dir / f"frame_{i}.jpg")
                ok = _extract_frame(video_path, ts, frame_path)
                if ok:
                    frame_paths.append(frame_path)

            if len(frame_paths) < 2:
                print(f"  video {video_id}: only {len(frame_paths)}/3 frames extracted, skipping")
                shutil.rmtree(video_dir, ignore_errors=True)
                continue

            caption_text = cv.edited_caption_text
            gen_caption = None
            quality_score = None
            judge_pass = None
            judge_overall = None
            if cv.generated_caption_id:
                gen_caption = db.query(GeneratedCaption).filter_by(id=cv.generated_caption_id).first()
                if gen_caption:
                    if not caption_text:
                        caption_text = gen_caption.caption_text
                    quality_score = gen_caption.quality_score
                    judge_pass = gen_caption.judge_pass
                    if gen_caption.judge_scores:
                        judge_overall = gen_caption.judge_scores.get("overall")

            bg = db.query(BackgroundVideo).filter_by(id=cv.background_video_id).first()
            bg_subreddit = bg.searched_tag if bg else None
            bg_ml_tags = bg.ml_tags if bg and bg.ml_tags else None

            brief_path = video_dir / "brief.md"
            with brief_path.open("w") as f:
                f.write(f"# Composed video #{video_id} ({cv.niche})\n\n")
                f.write(f"**Duration:** {duration:.1f}s · **Resolution:** {cv.resolution or 'unknown'}\n\n")
                if quality_score is not None:
                    f.write(f"**Stage 2 quality_score:** {quality_score}\n")
                if judge_pass is not None:
                    f.write(f"**Stage 3 LLM judge:** {'PASS' if judge_pass else 'FAIL'} (overall {judge_overall}/10)\n")
                f.write(f"**BG source:** r/{bg_subreddit}\n" if bg_subreddit else "")
                if bg_ml_tags:
                    f.write(f"**BG ml_tags:** `{json.dumps(bg_ml_tags, separators=(',', ':'))}`\n")
                f.write("\n## Caption\n\n")
                f.write(f"```\n{(caption_text or '<missing>').strip()}\n```\n\n")
                f.write("## Frames\n\n")
                for fp in frame_paths:
                    rel = os.path.relpath(fp, video_dir.parent)
                    f.write(f"![frame]({rel})\n\n")
                f.write("## Reviewer instructions\n\n")
                f.write("Open frames, score 1-10 on the three axes in CLAUDE_REVIEW_INSTRUCTIONS.md, "
                        "then append one JSONL line to "
                        "`scripts/claude_review_results/results.jsonl`.\n")

            manifest.append({
                "id": video_id,
                "niche": cv.niche,
                "brief": str(brief_path),
                "frames": frame_paths,
            })

            cv.claude_review_status = "pending"

        db.commit()

        manifest_path = Path(batch_dir) / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        # Chown the whole batch dir back to the host user so GitHub Actions
        # can wipe the workspace cleanly on next deploy.
        _chown_to_host_owner(batch_dir)

        print(f"Exported {len(manifest)} videos to {batch_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"\nNext: open Claude Code and read CLAUDE_REVIEW_INSTRUCTIONS.md")
        return len(manifest)


def writeback(results_dir: str = DEFAULT_RESULTS_DIR):
    """
    Read JSONL result files and apply actions to composed_videos.

    Each result line looks like:
        {
          "id": 1234,
          "verdict": "pass" | "fail" | "maybe",
          "action": "pass" | "reject" | "edit_caption" | "swap_bg" | "edit_caption_and_swap_bg",
          "title": "<reddit post title written from caption+frames>",  # required
          "fixed_caption": "<text>",            # required when editing
          "ban_bg_source": true | false,        # set BackgroundVideo.filter_status='rejected'
          "scores": {...},
          "issues": [...],
          "notes": "...",
          "model": "claude-..."
        }

    Actions:
      pass                       — leave video as-is, mark reviewed.
      reject                     — set approval_status='rejected', mark reviewed.
                                   if ban_bg_source: also reject the BG.
      edit_caption               — write new caption text, queue recompose
                                   with use_same_background=True.
      swap_bg                    — queue recompose with use_same_background=False
                                   so a new compatible BG is picked.
      edit_caption_and_swap_bg   — combine the two: write caption, queue
                                   recompose with new BG.

    Recompose is queued via the existing celery task path so this script
    doesn't block; it returns once the row is updated and the task is
    enqueued. The user sees the freshly-composed video in the swipe queue
    once it lands.
    """
    sys.path.insert(0, "/app")
    from datetime import datetime
    from database.db import get_db_context
    from database.models import ComposedVideo, BackgroundVideo, GeneratedCaption

    results_path = Path(results_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"No results at {results_path}")
        return 0

    counts = {
        "pass": 0,
        "reject": 0,
        "edit_caption": 0,
        "swap_bg": 0,
        "edit_caption_and_swap_bg": 0,
        "ban_bg_source": 0,
        "titles_written": 0,
        "skipped": 0,
        "errors": 0,
    }
    recompose_jobs = []

    with get_db_context() as db:
        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  bad JSON line: {e}; line: {line[:100]}")
                    counts["errors"] += 1
                    continue

                cv = db.query(ComposedVideo).filter_by(id=entry["id"]).first()
                if not cv:
                    print(f"  video {entry['id']}: not found, skipping")
                    counts["skipped"] += 1
                    continue

                # Default action falls back to verdict-based heuristic for
                # backwards compat with the old score-only result schema.
                action = entry.get("action")
                if not action:
                    action = "reject" if entry.get("verdict") == "fail" else "pass"

                cv.claude_review_status = "reviewed"
                cv.claude_review_scores = entry.get("scores")
                cv.claude_review_issues = entry.get("issues") or []
                cv.claude_review_verdict = entry.get("verdict")
                cv.claude_review_notes = entry.get("notes")
                cv.claude_review_model = entry.get("model", "claude-code")
                cv.claude_review_at = datetime.utcnow()

                # --- per-action work ---
                if action == "pass":
                    counts["pass"] += 1

                elif action == "reject":
                    cv.approval_status = "rejected"
                    counts["reject"] += 1

                elif action in ("edit_caption", "edit_caption_and_swap_bg"):
                    fixed = (entry.get("fixed_caption") or "").strip()
                    if not fixed:
                        print(f"  video {cv.id}: action={action} but no fixed_caption, skipping caption edit")
                    else:
                        cv.edited_caption_text = fixed
                    counts[action] += 1
                    recompose_jobs.append((
                        cv.id,
                        # use_same_background = True for caption-only,
                        # False (pick new) for combined.
                        action == "edit_caption",
                    ))

                elif action == "swap_bg":
                    counts["swap_bg"] += 1
                    recompose_jobs.append((cv.id, False))

                else:
                    print(f"  video {cv.id}: unknown action '{action}', recording verdict only")
                    counts["errors"] += 1

                # Reddit post title generated by the reviewer using caption +
                # frames as context. Stored on the GeneratedCaption row so
                # _get_video_title() (api/postpone.py) picks it up at
                # schedule time. Overwrites any previous title — re-review
                # intentionally replaces it.
                title = (entry.get("title") or "").strip()
                if title and cv.generated_caption_id:
                    gen_caption = db.query(GeneratedCaption).filter_by(id=cv.generated_caption_id).first()
                    if gen_caption:
                        gen_caption.generated_title = title[:300]
                        counts["titles_written"] += 1

                # Optional BG source ban — applies regardless of action.
                if entry.get("ban_bg_source") and cv.background_video_id:
                    bg = db.query(BackgroundVideo).filter_by(id=cv.background_video_id).first()
                    if bg and bg.filter_status != "rejected":
                        bg.filter_status = "rejected"
                        # filter_text is where filter detection notes go.
                        bg.filter_text = (bg.filter_text or "") + (
                            f" | banned via Stage 4 review of composed video #{cv.id}: "
                            f"{(entry.get('notes') or '')[:120]}"
                        )
                        counts["ban_bg_source"] += 1

        db.commit()

    # Trigger recompose via the existing HTTP endpoint, which does the BG
    # selection / duration check / file-delete-on-success dance the celery
    # task alone can't do. The endpoint enqueues the celery task itself.
    if recompose_jobs:
        import requests
        api_base = os.environ.get("CAPTIONS_API_BASE", "http://localhost:8000")
        for video_id, use_same in recompose_jobs:
            try:
                r = requests.post(
                    f"{api_base}/composition/videos/{video_id}/recompose",
                    json={"use_same_background": use_same},
                    timeout=30,
                )
                if r.status_code == 200:
                    print(f"  video {video_id}: recompose queued (use_same_background={use_same})")
                else:
                    print(f"  video {video_id}: recompose HTTP {r.status_code}: {r.text[:200]}")
                    counts["errors"] += 1
            except Exception as e:
                print(f"  video {video_id}: recompose request failed: {e}")
                counts["errors"] += 1

    # Same chown rationale as export — keeps GitHub Actions checkout happy.
    _chown_to_host_owner(results_dir)

    print(
        f"Writeback summary: pass={counts['pass']} reject={counts['reject']} "
        f"edit_caption={counts['edit_caption']} swap_bg={counts['swap_bg']} "
        f"edit+swap={counts['edit_caption_and_swap_bg']} "
        f"banned_bgs={counts['ban_bg_source']} titles_written={counts['titles_written']} "
        f"skipped={counts['skipped']} errors={counts['errors']}"
    )
    return counts


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export", help="Pull pending videos and prepare review briefs")
    exp.add_argument("--niche", default=None, help="Filter by niche (motivation, fitness, ...)")
    exp.add_argument("--limit", type=int, default=30)
    exp.add_argument("--batch-dir", default=DEFAULT_BATCH_DIR)

    wb = sub.add_parser("writeback", help="Apply review JSONL results to DB")
    wb.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args()

    if args.cmd == "export":
        export_pending(niche=args.niche, limit=args.limit, batch_dir=args.batch_dir)
    elif args.cmd == "writeback":
        writeback(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
