# Run a visual review batch (Stage 4)

**Trigger:** in an interactive Claude Code session opened on this repo, say `run scripts/RUN_CLAUDE_REVIEW.md` — optionally with `niche=motivation limit=30`, or `all niches`.

**Claude Code then does everything below in one go.** The session itself is the reviewer: it looks at keyframes and scores them. Nothing here calls a model API.

What the review catches that the Stage-3 LLM judge cannot: captions that describe something the footage does not show, overlay text that is unreadable against the clip, footage that is watermarked or otherwise unpostable, and captions that drifted off-niche. The rubric and the JSON result shape are in [CLAUDE_REVIEW_INSTRUCTIONS.md](CLAUDE_REVIEW_INSTRUCTIONS.md); this file is the procedure around it.

## Where things live

`scripts/claude_review_videos.py` runs inside the `api` container and reads/writes under `/app/scripts/`, which is the repo checkout bind-mounted into the container. So on the VM the batch files are simply `<checkout>/scripts/claude_review_batches/` (gitignored). The script chowns them to uid 1000 so the deploy runner can clean the workspace.

```
export   → <batch-dir>/<video_id>/brief.md + frame_1.jpg frame_2.jpg frame_3.jpg, <batch-dir>/manifest.json
review   → <results-dir>/results.jsonl   (one JSON object per line, written by the reviewer)
writeback→ updates composed_videos, optionally queues recomposes via POST /composition/videos/{id}/recompose
```

CLI (`python scripts/claude_review_videos.py …`):

| Command | Flags | Default |
|---|---|---|
| `export` | `--niche <name>`, `--limit N`, `--batch-dir <path>` | all niches, 30, `/app/scripts/claude_review_batches` |
| `writeback` | `--results-dir <path>` (reads `<path>/results.jsonl`) | `/app/scripts/claude_review_results` |

`export` selects `composed_videos` with `status='completed'`, `claude_review_status IS NULL`, `approval_status IN ('pending','approved')`, newest first, extracts frames at 25/50/75 % of duration with ffmpeg, and marks each row `claude_review_status='pending'` so the next export skips it. Videos with a missing file or fewer than 2 extractable frames are skipped (and stay unreviewed).

`writeback` needs the API reachable at `CAPTIONS_API_BASE` (default `http://localhost:8000`, which is correct from inside the `api` container).

## Two ways to run it

Pick the block that matches where the Claude Code session is running. Placeholders: `$VM` = `user@vm-host` (an SSH alias is fine), `$CHECKOUT` = the compose project directory on the VM (the runner's `~/actions-runner/_work/captionforge/captionforge` in the reference deploy).

**A. Session on the VM** (in `$CHECKOUT`): run `docker compose exec -T api …` directly and read the batch from `scripts/claude_review_batches/`. No copying.

**B. Session on a workstation with SSH to the VM:**

```bash
ssh $VM "cd $CHECKOUT && docker compose exec -T api python scripts/claude_review_videos.py export --limit 30"
rm -rf /tmp/claude_review_local && rsync -az "$VM:$CHECKOUT/scripts/claude_review_batches/" /tmp/claude_review_local/
# … review locally, write /tmp/claude_review_local_results/results.jsonl …
ssh $VM "mkdir -p $CHECKOUT/scripts/claude_review_results"
scp /tmp/claude_review_local_results/results.jsonl "$VM:$CHECKOUT/scripts/claude_review_results/results.jsonl"
ssh $VM "cd $CHECKOUT && docker compose exec -T api python scripts/claude_review_videos.py writeback"
```

The examples below are written for **A**; wrap each `docker compose exec` in `ssh $VM "cd $CHECKOUT && …"` and add the rsync/scp hops for **B**.

---

## Single-batch mode

Use when the user passed `niche=` / `limit=`, asked for "one batch", or is testing.

### 1. Export

```bash
docker compose exec -T api python scripts/claude_review_videos.py export ${NICHE:+--niche $NICHE} --limit ${LIMIT:-30}
```

`Exported 0 videos` means the queue is empty for that niche — stop and say so.

### 2. Review

For every `scripts/claude_review_batches/<id>/`: read `brief.md` (caption, niche, BG subreddit, `ml_tags`, earlier-stage scores), look at all three frames, then score, choose a verdict and an action, fix the caption if it is fixable, and write a Reddit post title — exactly as [CLAUDE_REVIEW_INSTRUCTIONS.md](CLAUDE_REVIEW_INSTRUCTIONS.md) specifies. You are repairing the batch, not just grading it: the goal is that the Swipe tab shows fixed videos rather than rejected ones.

### 3. Write `results.jsonl`

```bash
mkdir -p scripts/claude_review_results
cat > scripts/claude_review_results/results.jsonl <<'JSONL'
{"id": 3350, "scores": {"caption_quality": 8, "caption_video_match": 8, "would_post": 8}, "issues": [], "verdict": "pass", "action": "pass", "title": "The 5am version of you is waiting on the one reading this", "notes": "clean", "model": "claude-code"}
{"id": 3347, "scores": {"caption_quality": 5, "caption_video_match": 7, "would_post": 5}, "issues": ["'Caption:' artifact in overlay"], "verdict": "maybe", "action": "edit_caption", "fixed_caption": "You keep waiting for motivation to show up first. It never does. Lace up, start the first set, and let it catch up to you.", "title": "Motivation shows up after you start, never before", "notes": "same clip is fine", "model": "claude-code"}
JSONL
```

One object per line, no array wrapper, no trailing commas. Validate before writeback:

```bash
python3 - <<'PY'
import json
ok = 0
for line in open("scripts/claude_review_results/results.jsonl"):
    line = line.strip()
    if not line: continue
    o = json.loads(line)
    assert {"id","scores","verdict","action","title"} <= o.keys(), o.get("id")
    assert o["verdict"] in ("pass","fail","maybe") and o["action"] in ("pass","reject","edit_caption","swap_bg","edit_caption_and_swap_bg")
    if o["action"].startswith("edit_caption"): assert o.get("fixed_caption","").strip(), f"missing fixed_caption on {o['id']}"
    ok += 1
print("OK", ok)
PY
```

### 4. Writeback

```bash
docker compose exec -T api python scripts/claude_review_videos.py writeback
```

Per row it sets `claude_review_status='reviewed'` plus scores/issues/verdict/notes/model/timestamp, stores `title` on the caption's `generated_title` (Postpone uses it at schedule time), and then by action:

| action | effect |
|---|---|
| `pass` | nothing else; the video is now eligible for the Swipe queue |
| `reject` | `approval_status='rejected'` |
| `edit_caption` | `edited_caption_text=fixed_caption`, recompose with the same background |
| `swap_bg` | recompose with a new niche-compatible background |
| `edit_caption_and_swap_bg` | both |
| `ban_bg_source: true` (any action) | `background_videos.filter_status='rejected'` so the clip is never reused |

Recomposes are queued Celery tasks; new versions land in a minute or two. Quote the script's `Writeback summary: …` line in the report.

### 5. Report

> Reviewed N videos: `<n>` pass · `<n>` caption edits · `<n>` BG swaps · `<n>` edit+swap · `<n>` rejected (`<n>` BG sources banned).
> Top failure modes: `<count>` unreadable overlay, `<count>` caption/footage mismatch, `<count>` watermark, `<count>` off-niche, `<count>` other.
> Code-fix follow-ups: `<e.g. "watermark filter missed '@handle' on 4 clips", "renderer clipped the last chunk on 2 videos">`.

---

## Parallel mode (all niches, subagents)

Default when no args are given or the user says "all niches". Drains the queue with up to **10 subagents**, each reviewing **≤ 30 videos from exactly one niche**.

### Constraints

- Hard cap: 10 subagents per run.
- **Exports are sequential.** The export SELECTs and UPDATEs `claude_review_status` in one transaction per run; two concurrent exports can claim the same rows. Run all exports before spawning anything.
- One isolated directory pair per slot: `scripts/claude_review_batches_<niche>_<slot>/` and `scripts/claude_review_results_<niche>_<slot>/`.
- Each subagent does its own review and its own writeback.

### 1. Count the queue

```bash
docker compose exec -T postgres psql -U captionsuser -d captions -c \
 "SELECT niche, COUNT(*) FROM composed_videos WHERE status='completed' AND claude_review_status IS NULL AND approval_status IN ('pending','approved') GROUP BY niche ORDER BY niche;"
```

### 2. Allocate slots

Greedy: 30 per slot per niche until a niche is drained or 10 slots are used. Example for cooking=33, travel=38, fitness=33, motivation=904 → cooking 2, travel 2, fitness 2, motivation 4 (274 videos).

### 3. Export one batch per slot, sequentially

```bash
for pair in cooking:0 cooking:1 travel:0 travel:1 fitness:0 fitness:1 motivation:0 motivation:1 motivation:2 motivation:3; do
  NICHE=${pair%%:*}; SLOT=${pair##*:}
  docker compose exec -T api python scripts/claude_review_videos.py export \
    --niche $NICHE --limit 30 --batch-dir /app/scripts/claude_review_batches_${NICHE}_${SLOT}
done
```

Drop any slot that exported 0 videos. (Mode B: rsync each slot's batch directory down afterwards; those copies can run in parallel.)

### 4. Spawn the subagents in one message

Use `subagent_type: general-purpose`, one Agent call per slot, all in a single response. Each prompt must be self-contained. Template (substitute `<NICHE>`, `<SLOT>`, and the batch path for your mode):

> You are reviewing composed caption videos for the `<NICHE>` niche (Stage 4 visual review). Batch directory: `<path to scripts/claude_review_batches_<NICHE>_<SLOT>>` — numbered subdirectories each holding `brief.md` and `frame_1.jpg`..`frame_3.jpg`. Read every directory.
>
> Follow the rubric in `scripts/CLAUDE_REVIEW_INSTRUCTIONS.md` (in this repo) exactly: score `caption_quality`, `caption_video_match`, `would_post` 1–10; pick `verdict` and `action`; set `fixed_caption` for edit actions; set `ban_bg_source: true` for watermarked or unusable clips; always write a `title`. Write one JSON object per line to `<path to scripts/claude_review_results_<NICHE>_<SLOT>>/results.jsonl`, validate it, then run
> `docker compose exec -T api python scripts/claude_review_videos.py writeback --results-dir /app/scripts/claude_review_results_<NICHE>_<SLOT>`
> (from the VM checkout; if you are remote, scp the file into that directory on the VM first).
>
> Report back, under 400 words: counts per action; failure-mode tallies; patterns worth a code fix; the video ids reviewed; and the script's `Writeback summary:` line.

### 5. Consolidate

Aggregate the subagent reports into one table (niche × reviewed / pass / edit / swap / edit+swap / reject), list the top failure modes and the code-fix follow-ups, then re-run the queue count from step 1 to show what remains.

---

## Failure modes

- `service "api" is not running` — check `docker compose ps`; if `api` is unhealthy, stop and tell the user.
- `Exported 0 videos` — queue empty for that filter; nothing to do.
- A frame is black or the overlay is mid-transition — judge from the other frames; if all three are unusable, skip the id (leave it out of `results.jsonl`; it stays `pending` and can be reset below).
- `No results at …/results.jsonl` on writeback — the results directory does not exist inside the container or the file was written elsewhere; create the directory under `scripts/` and re-run.
- Writeback prints `recompose HTTP 5xx` — the API rejected the recompose (usually no compatible background left for the niche); the review fields are still saved.

## Reset a batch (only when the user asks)

```bash
docker compose exec -T postgres psql -U captionsuser -d captions -c \
 "UPDATE composed_videos SET claude_review_status=NULL, claude_review_scores=NULL, claude_review_issues=NULL,
   claude_review_verdict=NULL, claude_review_notes=NULL, claude_review_model=NULL, claude_review_at=NULL
  WHERE id IN (<ids>);"
```
