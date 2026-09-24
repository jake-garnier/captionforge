# API Reference

Route catalog for the CaptionForge API, grouped by router (`api/*.py`). Interactive docs are at `http://localhost:8000/docs` (FastAPI/Swagger). Examples use `http://localhost:8000`; substitute `https://captions.example.com` or `http://<vm-ip>:8000` when remote. Every route below exists in the current code — if a route is missing here, it was removed.

Conventions: `{niche}` is one of the keys in `NICHE_CONFIGS` (`motivation`, `fitness`, `cooking`, `travel`); JSON bodies are sent with `-H 'Content-Type: application/json'`.

## Contents

- [Core](#core) · [Scraping](#scraping) · [Scraper control](#scraper-control) · [Subreddits](#subreddits) · [Captions maintenance](#captions-maintenance)
- [Dashboard](#dashboard) · [Extraction queue](#extraction-queue) · [Reddit block detection](#reddit-block-detection) · [Proxy pool](#proxy-pool)
- [Training](#training) · [Training manager](#training-manager) · [GPU control](#gpu-control) · [llama-server](#llama-server)
- [Generation](#generation) · [Background videos](#background-videos) · [Reddit background subreddits](#reddit-background-subreddits) · [Video composition](#video-composition) · [Pipeline](#pipeline)
- [Publishing](#publishing) · [Postpone](#postpone) · [Reddit accounts](#reddit-accounts) · [Patreon](#patreon) · [Telegram](#telegram) · [Telegram scraping](#telegram-scraping)
- [Workflows](#workflows) · [Membership sync](#membership-sync) · [Analytics](#analytics)

---

## Core

`api/main.py`, `api/video_gallery.py`

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/health` | Health check (used by Docker and the deploy workflow) |
| GET | `/stats` | Aggregate counts (videos, captions) |
| GET | `/gallery/` | The operator UI (single page, all tabs) |
| GET | `/gallery/stream/{post_id}` | Stream a scraped video |
| GET | `/videos` | List scraped videos (paginated, filterable) |
| GET | `/videos/{video_id}` | One scraped video + captions |
| DELETE | `/videos/{video_id}` | Delete a scraped video and its file |
| GET | `/task/{task_id}` | Celery task status |
| POST | `/admin/reset-database` | Wipe application tables (destructive) |

```bash
curl http://localhost:8000/health
docker compose exec postgres psql -U captionsuser -d captions     # DB shell
```

## Scraping

`api/main.py`

| Method | Path | Description |
|---|---|---|
| POST | `/scrape/{subreddit}` | One-off Playwright/JSON scrape of a subreddit |
| POST | `/scrape/all` | Scrape every configured subreddit |
| POST | `/scrape/incremental/{subreddit}` | Run one incremental batch (25 posts, current stage) |
| GET | `/scrape/progress` | Incremental progress for all subreddits |
| GET | `/scrape/progress/{subreddit}` | Stage + cursor for one subreddit |
| POST | `/scrape/progress/{subreddit}/reset` | Restart at `top_all` |
| POST | `/upvotes/update/{video_id}` | Refresh one post's score |
| POST | `/upvotes/update/stale` | Refresh posts not updated recently |
| POST | `/upvotes/update/popular` | Refresh the most popular posts |

## Scraper control

`api/scraper_control.py` — prefix `/scraper/control`

| Method | Path | Description |
|---|---|---|
| GET | `/status` | Redis flags for scraping / upvotes / auto-disable |
| POST | `/enable`, `/disable` | Toggle scraping and upvote refresh together |
| POST | `/configure` | Set flags from a JSON body |
| POST | `/scraper/enable`, `/scraper/disable` | Scraping only |
| POST | `/upvotes/enable`, `/upvotes/disable` | Upvote refresh only |
| POST | `/auto-disable/enable`, `/auto-disable/disable` | Auto-disable scraping when Reddit blocks the proxies |

## Subreddits

`api/subreddit_management.py` — prefix `/subreddits`

| Method | Path | Description |
|---|---|---|
| GET | `/` | Subreddits with config + scraping progress |
| POST | `/` | Add: `{"name","min_score":300,"batch_size":25,"enabled":true,"scrape_stage":"top_all"}` |
| PUT | `/{name}` | Update `enabled` / `min_score` / `batch_size` / `description` |
| DELETE | `/{name}` | Remove |
| POST | `/{name}/toggle` | Flip `enabled` |
| POST | `/{name}/reset?stage=top_all` | Reset progress to a stage |
| GET | `/stages` | Stage list (`top_all → top_year → top_month → top_week → top_day → new`) |
| POST | `/{name}/scrape` | Trigger an incremental batch now |

```bash
curl -X POST http://localhost:8000/subreddits/ -H 'Content-Type: application/json' -d '{"name":"GetMotivated","min_score":300}'
```

## Captions maintenance

`api/main.py`

| Method | Path | Description |
|---|---|---|
| POST | `/captions/reextract/all?batch_size=50` | Re-run OCR on all videos |
| GET | `/captions/reextract/status` | Progress of the above |
| POST | `/captions/reextract/failed` | Re-run OCR on failed videos only |
| GET | `/captions/reextract/failed/status` | Progress |
| POST | `/captions/reextract/clean-and-run` | Clear captions, then re-extract |
| POST | `/captions/reprocess-rule-based` | Re-run the rule-based cleanup stage |
| GET | `/captions/reprocess-rule-based/status` | Progress |
| POST | `/captions/llm-refine` | Queue LLM refinement for unrefined captions |
| GET | `/captions/llm-refine/status` | Progress |

## Dashboard

`api/dashboard.py` — prefix `/dashboard`

| Method | Path | Description |
|---|---|---|
| GET | `/overview` | Counts and rates for the Dashboard tab |
| GET | `/scraping/progress`, `/scraping/schedule`, `/scraping/timeline` | Scraper views |
| POST | `/scraping/trigger/{task_name}` | Run a beat task by name |
| GET | `/upvotes/trending`, `/upvotes/stale` | Upvote views |
| GET | `/processing/pipeline` | Per-stage processing counts |
| GET | `/storage/breakdown` | Disk usage by directory |
| GET | `/celery/tasks` | Active/reserved Celery tasks |
| GET | `/captions/stats` | Caption stage statistics |
| GET | `/tasks` | HTML task monitor |
| GET | `/worker/status` | Heartbeats + hang detection |
| POST | `/worker/restart` | Restart a hung worker |
| GET | `/hang-incidents`, `/hang-incidents/{id}`, `/hang-incidents/analysis/patterns` | Hang diagnostics |
| GET | `/subreddit/{name}/metrics` | Per-subreddit metrics |

## Extraction queue

`api/dashboard.py`

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard/extraction/queue/status` | Queue depth |
| GET | `/dashboard/extraction/queue/detailed` | Queue contents |
| GET | `/dashboard/extraction/progress` | Extraction progress |
| POST | `/dashboard/extraction/dispatch/trigger` | Dispatch a batch now |
| POST | `/dashboard/extraction/orphans/process` | Queue videos that missed the queue |
| POST | `/dashboard/extraction/queue/requeue/{video_id}` | Requeue one video |
| POST | `/dashboard/extraction/queue/clear` | Empty the queue |

## Reddit block detection

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard/reddit/block-status` | Current block state per proxy |
| GET | `/dashboard/reddit/block-stats`, `/dashboard/reddit/block-history` | History |
| POST | `/dashboard/reddit/check-access` | Probe Reddit now |
| POST | `/dashboard/reddit/clear-block-status` | Clear the block flag |

## Proxy pool

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard/proxy/status` | Configured proxies |
| POST | `/dashboard/proxy/test` | Test the configured proxy |
| GET | `/dashboard/proxy/pool/status` | Subreddit → proxy assignments |
| POST | `/dashboard/proxy/pool/assign/{subreddit}?proxy_index=5` | Assign (specific index optional) |
| DELETE | `/dashboard/proxy/pool/assign/{subreddit}` | Unassign |
| POST | `/dashboard/proxy/pool/test/{index}`, `/dashboard/proxy/pool/test-all` | Test proxies |
| POST | `/dashboard/proxy/pool/clear` | Clear all assignments |

## Training

`api/training.py` — prefix `/training`. Cloud training on Vast.ai; see [training_runbook.md](training_runbook.md).

| Method | Path | Description |
|---|---|---|
| POST | `/trigger/{niche}?auto_destroy=true` | Export corpus → rent RTX 4090 → QLoRA → download → GGUF → restart llama-server. `auto_destroy=false` keeps the instance for diagnosis |
| GET | `/vastai/status/{task_id}` | Step-by-step progress of a cloud run |
| GET | `/adapters` | Adapters under `/data/lora_adapters/` (PEFT + GGUF presence) |
| GET | `/adapters/{niche}` | One adapter's files + LoRA config |
| GET | `/data/count/{niche}?min_upvotes=100` | Captions available for export; `ready_for_training` at ≥ 100 |
| POST | `/export-data` | Export the corpus without training |
| GET | `/data/status` | Exported train/val files present? |
| GET | `/model/status?niche=` | Legacy in-container model directory check |
| POST | `/start` | Legacy local LoRA training task |
| GET | `/status/{task_id}` | Status of a legacy task |

```bash
curl -X POST "http://localhost:8000/training/trigger/motivation?auto_destroy=false"
curl http://localhost:8000/training/vastai/status/<task_id>
```

## Training manager

`api/training_management.py` — prefix `/training-manager`. Legacy local (in-container) training jobs and the trained-model registry.

| Method | Path | Description |
|---|---|---|
| GET | `/data/subreddits` | Caption counts per subreddit |
| GET | `/data/subreddit/{name}/stats` | Per-subreddit corpus stats |
| GET | `/data/preview?subreddits=&min_upvotes=` | Sample training rows |
| POST | `/jobs` | Create a local training job |
| GET | `/jobs`, `/jobs/{job_id}` | List / detail |
| POST | `/jobs/{job_id}/cancel`, `/retry`, `/fix-stuck` | Job control |
| DELETE | `/jobs/{job_id}` | Delete |
| GET | `/models`, `/models/{model_id}` | Trained-model registry |
| POST | `/models/{model_id}/load?target_gpu=0`, `/unload` | Load/unload a PEFT model in a GPU worker |
| POST | `/models/{model_id}/generate` | Generate with a loaded model: `{"prompt": "..."}` |
| DELETE | `/models/{model_id}` | Delete a model |
| GET | `/gpu/status` | GPU memory per worker |
| POST | `/gpu/clear-cache` | Empty CUDA cache |
| GET | `/base-models` | Base models known to the trainer |

## GPU control

`api/gpu_control.py` — prefix `/training-manager/gpu-control`

| Method | Path | Description |
|---|---|---|
| GET | `/status`, `/gpus` | Flags and per-GPU memory |
| POST | `/gpus/refresh` | Refresh cached GPU status |
| POST | `/extraction/enable`, `/extraction/disable` | `extraction:enabled` |
| POST | `/ocr/{gpu_id}/enable`, `/ocr/{gpu_id}/disable` | `ocr:gpu{N}:enabled` |
| POST | `/llm/enable`, `/llm/disable` | `llm:batch:enabled` |
| POST | `/{gpu_id}/prepare-for-training`, `/prepare-all-for-training` | Disable workloads and unload models |
| POST | `/restore-extraction` | Re-enable after training |
| GET | `/{gpu_id}/training-readiness` | Is the GPU clear? |
| POST | `/force-unload-all`, `/deep-clean-all` | Unload models / aggressive cleanup |
| POST | `/purge-gpu-queue` | Drop queued `gpu` tasks |

## llama-server

`api/llama_server.py` — prefix `/llama-server`. Controls the host llama-server over SSH (`LLM_HOST_*` settings).

| Method | Path | Description |
|---|---|---|
| GET | `/status`, `/health` | Running? loaded model? |
| POST | `/start` | `{"niche": "motivation"}` or `{"lora_path": "..."}` — start with a LoRA (`llm` profile); empty body = base model |
| POST | `/stop` | Stop |
| POST | `/restart` | Same body as `/start`; restart, optionally with a different LoRA |
| POST | `/generate` | `{"prompt","max_tokens":512,"temperature":0.8,"niche"}` — quick smoke test |

## Generation

`api/generation.py` — prefix `/generation`

| Method | Path | Description |
|---|---|---|
| POST | `/jobs` | Create a legacy generation job `{"job_name","model_id","num_captions"}` |
| GET | `/jobs`, `/jobs/{job_id}` | List / detail |
| POST | `/jobs/{job_id}/cancel` | Cancel |
| DELETE | `/jobs/{job_id}`, `/jobs/cancelled/all` | Delete |
| GET | `/captions?status=pending_review` | List generated captions (filters: status, niche, …) |
| GET | `/captions/{caption_id}` | One caption |
| PATCH | `/captions/{caption_id}` | Update (`{"status":"approved"}`, text edits) |
| DELETE | `/captions/{caption_id}` | Delete |
| POST | `/captions/{caption_id}/favorite` | Toggle favourite |
| POST | `/captions/bulk-action` | `{"action":"approve","caption_ids":[…]}` |
| GET | `/captions/tag-stats`, `/captions/score-stats` | Aggregates |
| POST | `/captions/backfill-tags` → GET `/captions/backfill-tags/status/{task_id}` | Backfill caption tags |
| POST | `/captions/score-batch?limit=50` → GET `/captions/score-batch/status/{task_id}` | Heuristic quality scoring |
| POST | `/captions/rescore-all` → GET `/captions/rescore-all/status/{task_id}` | Re-score everything |
| GET | `/stats` | Generation statistics |

## Background videos

`api/background_videos.py` — prefix `/background-videos`. Stock clips scraped from Reddit; `filter_status` is set by the watermark filter, `ml_tags` by VLM tagging (dispatched from the orchestrator's idle path and controlled by the Redis flags `ml_tagging:enabled` / `ml_tagging:gpu`).

| Method | Path | Description |
|---|---|---|
| GET | `/` | List (`filter_status`, `tag`, `sort_by`, pagination) |
| GET | `/count`, `/stats` | Counts by status / source |
| GET | `/{video_id}` | Detail incl. `ml_tags` |
| GET | `/{video_id}/stream`, `/{video_id}/thumbnail` | Media |
| DELETE | `/{video_id}` | Delete one |
| DELETE | `/orphans` | Delete rows whose files are missing |
| DELETE | `/all?confirm=true` | Delete everything |
| GET | `/filter/stats` | Watermark-filter counts |
| POST | `/filter/process-pending?batch_size=10` | Run the filter on pending clips |
| POST | `/filter/retry-errors`, `/filter/reset-rejected` | Requeue |
| POST | `/filter/{video_id}` | Filter one clip |
| GET | `/filter/control/status` | `watermark_filter:enabled`, GPU selection |
| POST | `/filter/control/enable`, `/filter/control/disable`, `/filter/control/set-gpu?gpu=0` | Control |

## Reddit background subreddits

`api/reddit_background_videos.py` — prefix `/background-videos/reddit`

| Method | Path | Description |
|---|---|---|
| GET | `/subreddits/` | Configured background subreddits |
| POST | `/subreddits/` | Add: `{"name","min_score":100,"min_duration":10,"max_duration":60,"batch_size":10}` |
| PUT | `/subreddits/{name}` | Update |
| DELETE | `/subreddits/{name}` | Remove |
| POST | `/subreddits/{name}/toggle`, `/subreddits/{name}/reset`, `/subreddits/{name}/scrape` | Control |
| GET | `/subreddits/stages` | Stage list |
| GET | `/control/status` | `reddit_bg:enabled` |
| POST | `/control/enable`, `/control/disable` | Toggle background scraping |

## Video composition

`api/video_composition.py` — prefix `/composition`

| Method | Path | Description |
|---|---|---|
| POST | `/jobs` | Batch job `{"job_name","caption_source":"generated","target_count":10}` |
| GET | `/jobs`, `/jobs/{job_id}` | List / detail |
| POST | `/jobs/{job_id}/cancel` · DELETE `/jobs/{job_id}` | Control |
| GET | `/videos` | Composed videos (filters incl. `niche`, `approval_status`, `claude_review_verdict`) |
| GET | `/videos/{video_id}` | Detail (quality metrics, judge + review fields, `hosted_url`) |
| GET | `/videos/{video_id}/stream` | Serve the mp4 — also the public URL the self-hosted media host publishes (`MEDIA_BASE_URL` + this path) |
| GET | `/swipe-queue?niche=` | Stage-5 queue: reviewed `pass`/`maybe` videos, ordered verdict → judge pass → quality score |
| GET | `/stage4-pending-count` | Videos awaiting visual review |
| PATCH | `/videos/{video_id}` | Update fields (e.g. `approval_status`) |
| PUT | `/videos/{video_id}/caption` | Set `edited_caption_text` |
| POST | `/videos/{video_id}/recompose` | `{"use_same_background":false,"background_video_id":null}` — new BG picked by niche rules unless specified |
| POST | `/videos/{video_id}/append-outro` | Append the niche's Reddit outro |
| POST | `/videos/{video_id}/restyle` | Re-render with another style preset |
| DELETE | `/videos/{video_id}` | Delete |
| POST | `/single` | Compose one: `{"caption_text","background_video_id"}` |
| POST | `/preview-timing` | `{"caption_text"}` → chunk/timing preview |
| GET | `/debug-frames`, `/debug-frames/{frame_name}` | Renderer debug output |
| GET | `/stats` | Composition statistics |

## Pipeline

`api/pipeline.py` — prefix `/pipeline`. See [pipeline_state_machine.md](pipeline_state_machine.md).

| Method | Path | Description |
|---|---|---|
| GET | `/status`, `/state` | Status per niche / current state |
| POST | `/enable`, `/disable`, `/pause`, `/resume`, `/reset` | Loop control |
| GET | `/niches`, `/niches/{niche}` | Niche status |
| POST | `/trigger/generate/{niche}`, `/trigger/compose/{niche}`, `/trigger/score/{niche}` | Manual transitions / scoring |
| POST | `/trigger/train/{niche}`, `/trigger/check-training` | Legacy (auto-training is disabled; use `/training/trigger/{niche}`) |
| GET | `/quotas`, `/quotas/today` · POST `/quotas/{niche}/reset` | Daily quotas |
| GET | `/jobs/active`, `/jobs/{niche}` | Jobs |
| GET | `/config` · POST `/config/reload` | Automation config |
| GET | `/history`, `/history/stats?hours=24`, `/history/timeline?hours=24` | State log |
| GET | `/state-machine` | Machine-readable state definition |
| GET | `/failures`, `/failures/{niche}` · POST `/failures/{niche}/reset` | Failure counters |
| GET | `/data-integrity` | Cross-table consistency checks |

## Publishing

`api/publishing.py` — prefix `/publishing`. Direct flow: composed video → media host (self-hosted URL) → Reddit profile post → delayed crossposts.

| Method | Path | Description |
|---|---|---|
| POST | `/publish` | `{"composed_video_id","title":null,"tags":null,"crosspost_delay_minutes":30}` |
| GET | `/jobs`, `/jobs/{job_id}` | Jobs (`status`, `hosted_url`, `profile_post_url`, crossposts) |
| POST | `/jobs/{job_id}/cancel`, `/jobs/{job_id}/retry-crossposts` | Control |
| DELETE | `/jobs/{job_id}` | Delete |
| GET | `/videos/{video_id}/status` | Publish state for a composed video |
| GET | `/config/niches` | Target subreddits per niche |
| GET | `/stats` | Statistics |

## Postpone

`api/postpone.py` — prefix `/postpone`. Scheduled Reddit posting via the Postpone GraphQL API; `hosted_url` is filled by the media-host task before the schedule is pushed.

| Method | Path | Description |
|---|---|---|
| POST | `/approve`, `/reject` | `{"video_ids":[…]}` — bulk approval state |
| GET | `/approval-stats` | Pending / approved / scheduled / rejected per niche |
| POST | `/schedule-batch` | `{"niche","start_date","posting_hour_utc","posting_minute","stagger_minutes","max_videos"}` |
| POST | `/schedule-single` | `{"composed_video_id",…}` — what the Swipe tab's `→` calls |
| GET | `/jobs`, `/jobs/{job_id}` | Schedule jobs |
| POST | `/jobs/{job_id}/cancel`, `/retry`, `/publish-now` | Control |
| POST | `/jobs/{job_id}/set-hosted-url` | `{"hosted_url"}` — point a job at a video hosted elsewhere |
| POST | `/jobs/set-hosted-urls` | `{"jobs":[{"job_id","hosted_url"},…]}` |
| GET | `/calendar` | Scheduled posts by day |
| GET | `/health` | Postpone API reachable? |

## Reddit accounts

`api/reddit.py` — prefix `/reddit`. Playwright-managed accounts for direct posting.

| Method | Path | Description |
|---|---|---|
| GET | `/accounts`, `/accounts/{niche}` | Account rows |
| POST | `/accounts` | `{"niche","username","subreddits":"a,b"}` |
| PUT | `/accounts/{niche}` · DELETE `/accounts/{niche}` | Update / delete |
| GET | `/session/status` | Browser session state + noVNC URL |
| POST | `/session/start?niche=`, `/session/save-cookies`, `/session/stop` | Interactive login via noVNC |
| GET | `/session/screenshot` | Screenshot |
| POST | `/post` | `{"composed_video_id","title"}` — profile post + crossposts now |
| PATCH | `/videos/{video_id}/hosted-url?hosted_url=` | Manually set a composed video's public URL |

## Patreon

`api/patreon.py` — prefix `/patreon`

| Method | Path | Description |
|---|---|---|
| GET | `/credentials`, `/credentials/{niche}` | Credential rows |
| POST | `/credentials` | `{"niche","email"}` |
| DELETE | `/credentials/{niche}` | Delete |
| POST | `/credentials/{niche}/test-connection` | Verify the saved session |
| POST | `/publish` | `{"composed_video_id","title","description","tags","public":true,"publish_immediately":false}` |
| POST | `/schedule` | Schedule for a day (`scheduled_date`, hour/minute) |
| GET | `/jobs`, `/jobs/{job_id}` · GET `/stats` | Jobs |
| POST | `/jobs/{job_id}/publish-now`, `/cancel`, `/retry` · DELETE `/jobs/{job_id}` | Control |
| GET | `/session/status`, `/session/login-status` | Browser session |
| POST | `/session/start?niche=`, `/session/stop`, `/session/login`, `/session/solve-cloudflare` | Interactive login via noVNC |
| GET/POST | `/session/screenshot` | Screenshot |
| POST | `/session/save-cookies`, `/session/upload-cookies` | Persist / import cookies |
| POST | `/session/publish`, `/session/publish-video`, `/session/publish-job/{job_id}` | Publish through the live session |
| POST | `/session/publish-embed` | `{"embed_url","title"}` — post a link to the hosted video instead of uploading |
| POST | `/session/recording/start`, `/session/recording/stop` · GET `/session/recording/status` | Record browser actions |
| GET | `/recordings`, `/recordings/{name}` · DELETE `/recordings/{name}` · POST `/recordings/{name}/replay` | Saved recordings |

noVNC: `http://<host>:6080/vnc.html?autoconnect=true` (`NOVNC_PUBLIC_URL`).

## Telegram

`api/telegram.py` — prefix `/telegram`

| Method | Path | Description |
|---|---|---|
| GET | `/bots`, `/bots/{niche}` | Bot rows |
| POST | `/bots` | `{"niche","bot_username","bot_token","bot_name"}` |
| DELETE | `/bots/{niche}` | Delete |
| POST | `/bots/{niche}/verify` | `getMe` check |
| POST | `/bots/{niche}/discover-channels` | Channel ids the bot can see |
| GET | `/channels`, `/channels/{niche}` | Channel rows |
| POST | `/channels` | `{"niche","channel_id","channel_username","channel_name"}` |
| DELETE | `/channels/{niche}` · POST `/channels/{niche}/test` | Delete / send a test message |
| POST | `/publish/{composed_video_id}` | Schedule (default) or `{"publish_immediately":true}` |
| GET | `/jobs`, `/jobs/{job_id}` · GET `/stats` | Jobs |
| POST | `/jobs/{job_id}/publish-now`, `/retry` · DELETE `/jobs/{job_id}` | Control |

## Telegram scraping

`api/telegram_scraping.py` — prefix `/telegram/scrape` (Telethon user session; `TELEGRAM_API_ID` / `TELEGRAM_API_HASH`)

| Method | Path | Description |
|---|---|---|
| GET | `/auth/status` · POST `/auth/start`, `/auth/verify` · GET `/auth/dialogs` | Log in the user session, list dialogs |
| GET | `/control/status` · POST `/control/enable`, `/control/disable` | `telegram:scraping:enabled` |
| GET | `/channels`, `/channels/{channel_id}` | Scrape targets |
| POST | `/channels` · PUT `/channels/{channel_id}` · DELETE `/channels/{channel_id}` | Manage |
| POST | `/channels/{channel_id}/toggle`, `/scrape`, `/reset` · GET `/channels/{channel_id}/progress` | Control |
| GET | `/stats` | Statistics |

## Workflows

`api/workflows.py` — prefix `/workflows`. Generic VNC browser sessions and recorded action replay.

| Method | Path | Description |
|---|---|---|
| POST | `/session/start` (`{"start_url"}`), `/session/stop`, `/session/navigate?url=` | Session |
| GET | `/session/status` · GET/POST `/session/screenshot` | Inspect |
| POST | `/recording/start`, `/recording/stop` · GET `/recording/actions`, `/recording/status` | Record actions |
| GET | `/`, `/{workflow_id}` | Saved workflows |
| POST | `/` (`{"name","actions":[…],"variables":[…]}`) · PUT `/{workflow_id}` · DELETE `/{workflow_id}` | Manage |
| POST | `/{workflow_id}/execute` (`{"variables":{}}`), `/execute-actions` | Run |
| GET | `/execution/status` · POST `/execution/cancel` | Execution |
| POST | `/detect-variables` | Find variable placeholders in actions |

## Membership sync

`api/membership_sync.py` — prefix `/membership-sync`. Patreon → Telegram membership (the listener service and hourly beat ship disabled).

| Method | Path | Description |
|---|---|---|
| GET | `/status` | Last sync, counts |
| GET | `/alerts` · POST `/alerts/{alert_id}/resolve` | Sync alerts |
| GET | `/subscribers` | Known Patreon members |
| GET | `/join-requests` · POST `/join-requests/{request_id}/approve`, `/decline` | Pending Telegram join requests |
| POST | `/sync/trigger` | Run a reconciliation now |

## Analytics

`api/analytics.py` — prefix `/analytics`. Read-only views over the Reddit analytics tables.

| Method | Path | Description |
|---|---|---|
| GET | `/accounts` | Leaderboard of tracked accounts (from `postpone_reddit_username`) |
| GET | `/accounts/{username}` | Account detail |
| GET | `/accounts/{username}/karma-history` | Snapshots over time |
| GET | `/accounts/{username}/posts` | Post feed with stats |
| GET | `/accounts/{username}/subreddit-breakdown` | Performance by subreddit |
| GET | `/posts/{reddit_post_id}`, `/posts/{reddit_post_id}/history` | One post and its stat history |
| POST | `/accounts/{username}/backfill` | Pull full post history |
