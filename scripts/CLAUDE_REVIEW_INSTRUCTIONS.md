# Visual review rubric (Stage 4)

Stage 4 of the quality pipeline. The reviewer is an interactive Claude Code session looking at three keyframes and a brief per composed video, after the Stage-3 LLM judge has already scored the caption text. The procedure (export / writeback commands, single vs. parallel mode) is in [RUN_CLAUDE_REVIEW.md](RUN_CLAUDE_REVIEW.md); this file is what to look for and what to write.

You are not only scoring — you are **repairing the batch**. Edit captions that are fixable, swap backgrounds that do not fit, and reject-and-ban only what cannot be saved. The Swipe tab should be full of good videos, not rejections.

## What you get per video

`<batch-dir>/<id>/brief.md`:
- niche, duration, resolution
- Stage-2 heuristic `quality_score` and the Stage-3 judge verdict (`grammar`, `flow`, `bg_consistency`, `appeal`, `overall`, pass = all ≥ 7) when present
- the background clip's subreddit and its VLM `ml_tags` (`subjects`, `activities`, `setting`, `mood`, `camera`, `text_on_screen`)
- the caption text as composed (the edited text if one exists)

`frame_1.jpg`, `frame_2.jpg`, `frame_3.jpg` at 25 / 50 / 75 % of the runtime. These are frames of the *finished* video, so the rendered overlay is visible in them.

## What to check

1. **Legibility of the overlay.** Can the text be read at a glance on every frame? Watch for text over busy or bright regions, low contrast, clipped lines at the frame edge, chunks that are far too long for their on-screen time, and any rotated or mis-scaled render (a renderer bug — reject and flag it in your report).
2. **Caption / footage fit.** Does the caption plausibly belong on this clip? A quote about early-morning discipline over a sunrise hike fits; the same quote over a birthday-cake decorating clip does not. Anything the caption *asserts* about the scene must be consistent with the frames; loose thematic alignment is fine, contradiction is not.
3. **Timing.** From the three frames: is the caption still on its first chunk at 75 %? Has it run out before 50 %? Either means the chunking/timing did not match the clip length.
4. **Watermark and text leakage.** Creator handles (`@name`), site URLs, `subscribe`/`follow` banners, channel logos, or any burned-in text in the *background* footage. These make the clip unusable — reject and set `ban_bg_source: true` so it is never picked again.
5. **Niche appropriateness.** Is this something the niche's audience would expect from the account? A motivation clip that reads like a recipe tip, a fitness caption that describes nothing physical, or a caption with an off-tone joke should not go out under that account.
6. **Caption text quality** independent of the video: grammar, artifacts from OCR training data (`Caption:`, `Read more`, `link in bio`, stray `*`, `www.`, trailing fragments), repetition, generic filler, awkward line breaks that show in the overlay.

## Scores (1–10 each)

| Axis | Question | 8+ | 5–7 | ≤ 4 |
|---|---|---|---|---|
| `caption_quality` | Would this text stand on its own as a post? | I would upvote it | fine but forgettable | artifacts, grammar problems, generic, off-niche |
| `caption_video_match` | Does the caption fit these frames? | everything it implies is consistent | loosely related, nothing contradicted | contradicts the footage or is unrelated to it |
| `would_post` | Would I publish this exact video under the niche account? | yes, as is | with the fix below | no — unreadable, watermarked, wrong timing, wrong tone |

Be strict on `caption_video_match`. A "close enough" 6 is exactly the case Stage 4 exists to catch.

## Verdict and action

`verdict` is the quality call; `action` is what the writeback should do.

| Situation | `verdict` | `action` | Also set |
|---|---|---|---|
| All three scores ≥ 7 | `pass` | `pass` | — |
| Text is fixable (artifacts, grammar, a weak ending) and the clip fits | `fail` or `maybe` | `edit_caption` | `fixed_caption` |
| Text is fine but the clip does not fit (mismatch, timing, illegible on this footage) | `fail` | `swap_bg` | — |
| Both, but salvageable | `fail` | `edit_caption_and_swap_bg` | `fixed_caption` |
| Clip is unusable: watermark, burned-in text, logo, off-niche footage | `fail` | `reject` | `ban_bg_source: true` |
| Render bug (rotated / mis-scaled overlay) | `fail` | `reject` | flag in report |
| Middle ground you cannot resolve (5–6 on several axes) | `maybe` | `pass` | `notes` |

`maybe` still reaches the Swipe queue (below `pass`); `fail` does not. Use `maybe` sparingly — it pushes the decision onto the human swiping.

## Fixing captions

When you set `fixed_caption`:
- Remove artifacts: `Caption:`, `Read more`, `link in bio`, `follow for more`, URLs, `www.`, stray `*`, trailing fragments.
- Fix spelling and case (`u`/`ur` → `you`/`your`, `&` → `and`, random mid-sentence capitals).
- Replace em dashes and semicolons with commas or full stops (the renderer and the judge both penalise them).
- Keep the caption's idea and voice; do not rewrite it into a different message. Second person unless it is a quotation.
- 40–80 words. If it trails off, end on one concrete image or action.
- No emoji, no hashtags.

## Reddit post title (required on every row)

Postpone does not generate titles, so this is the only place one is written. Use the caption **and** the frames.
- 60–100 characters (hard cap 300).
- Reddit-native voice: a hook, a question, or a first-person line — not a sentence copied from the caption.
- Reference something visible (the sunrise, the empty gym, the pan on the stove) plus the caption's angle.
- Match the *fixed* caption if you edited it.
- No emoji, hashtags, or bracket tags.

Examples: `The alarm is not the problem, the snooze button is` · `Nobody is coming to do the first rep for you` · `Rest the dough or it will fight you the whole way` · `Book the ticket before you talk yourself out of it`.

## Result line

One JSON object per video, one per line, in `<results-dir>/results.jsonl`:

```jsonl
{"id": 1234, "scores": {"caption_quality": 8, "caption_video_match": 7, "would_post": 8}, "issues": ["minor: caption says 'trail' but frames show a road"], "verdict": "pass", "action": "pass", "title": "Nobody is coming to do the first rep for you", "notes": null, "model": "claude-code"}
{"id": 1235, "scores": {"caption_quality": 4, "caption_video_match": 8, "would_post": 4}, "issues": ["'Caption:' artifact rendered on screen", "ends mid-sentence"], "verdict": "fail", "action": "edit_caption", "fixed_caption": "You do not need a perfect plan. You need the next ten minutes. Put the phone face down, open the notebook, and start with the ugliest first draft you can stand.", "title": "Ten ugly minutes beat a perfect plan you never start", "notes": "same clip is fine", "model": "claude-code"}
{"id": 1236, "scores": {"caption_quality": 7, "caption_video_match": 2, "would_post": 2}, "issues": ["channel logo bottom-right on all frames"], "verdict": "fail", "action": "reject", "ban_bg_source": true, "title": "Every long climb starts with one boring step", "notes": "watermarked source", "model": "claude-code"}
```

Fields:
- required: `id`, `scores`, `verdict` (`pass` | `fail` | `maybe`), `action` (`pass` | `reject` | `edit_caption` | `swap_bg` | `edit_caption_and_swap_bg`), `title`
- required for `edit_caption*`: `fixed_caption`
- optional: `issues` (list of short strings), `notes`, `ban_bg_source` (bool), `model` (stored as `claude_review_model`; defaults to `claude-code`)

The writeback (`scripts/claude_review_videos.py writeback`) stores these on `composed_videos.claude_review_*`, writes the title to the caption's `generated_title`, and queues recomposes for the edit/swap actions. See RUN_CLAUDE_REVIEW.md for what each action does.

## Edge cases

- **Missing file / failed frame extraction** — the export skipped it; nothing to do.
- **One black or transitional frame** — judge from the others. If all three are unusable, leave the id out of the results so it stays `pending`.
- **Caption implies more than a clip can show** (a story, a future action) — fine as long as nothing on screen contradicts it.
- **Same background across several videos in the batch** — score each on its own, but if the clip is watermarked, ban it once and reject all of them.
