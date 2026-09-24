# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CaptionForge: a Reddit captioned-video content pipeline. Incremental scraping → GPU OCR → LLM refinement → per-niche LoRA training → background-first caption generation → LLM judge → composition → agentic visual review → Swipe approval → publishing (Reddit/Postpone/Patreon/Telegram). The README is the user-facing guide; this file is the operator/agent orientation.

## Critical rules

- **Never run `docker compose down -v`.** It drops the named volumes (database, videos, adapters).
- **Deploy via git push.** The self-hosted runner in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) rebuilds only the affected services and skips GPU workers while a generation/composition job is active.
- **Verification is `python -m py_compile <file>`, then watch logs.** There is no test suite or linter config; `test_telegram.py` and `training/test_lora_model.py` are ad-hoc scripts.

## Runbooks

- [docs/add_new_niche_runbook.md](docs/add_new_niche_runbook.md) — adding a niche end to end
- [docs/training_runbook.md](docs/training_runbook.md) — Vast.ai LoRA training and silent-failure handling
- [docs/pipeline_state_machine.md](docs/pipeline_state_machine.md) — orchestrator transitions
- [docs/api_reference.md](docs/api_reference.md) — API catalog
- [scripts/RUN_CLAUDE_REVIEW.md](scripts/RUN_CLAUDE_REVIEW.md) — "run scripts/RUN_CLAUDE_REVIEW.md" starts a visual review batch in which *this* Claude Code session is the reviewer; rubric in [scripts/CLAUDE_REVIEW_INSTRUCTIONS.md](scripts/CLAUDE_REVIEW_INSTRUCTIONS.md)

## Architecture

**Services** (docker-compose.yml): `api` (FastAPI, port 8000, UI at `/gallery/`, noVNC 6080), `celery-worker` (GPU 0), `celery-worker-gpu-2` (GPU 1), `celery-worker-ml-tagging` (both GPUs, calls host llama-server), `celery-worker-composition` (CPU, prefork ×2), `celery-worker-scraper` (CPU, prefork ×5), `celery-beat` (mounts the Docker socket for watchdog restarts), `postgres` (pgvector), `redis`, `flower` (5555), `tunnel` (Cloudflare).

**Runtime**: Python 3.11 on `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`, Playwright (Firefox + Chromium), MoviePy/PIL/ffmpeg. The repo is bind-mounted at `/app`; `PYTHONDONTWRITEBYTECODE=1` avoids root-owned `__pycache__` breaking the runner's workspace cleanup.

**Models**: Qwen2-VL-2B (OCR, in-container, ~5 GB VRAM), Mistral-Small-24B-Instruct GGUF + per-niche LoRA (host `llama-server`, `llm` profile, tensor-split across GPUs), InternVL3-14B GGUF (host `llama-server`, `vlm-internvl3` profile, used by `ml_tagging`). Profiles are swapped over SSH by [utils/llama_server_control.py](utils/llama_server_control.py) using `LLM_HOST_*` settings.

### Queues and routing

```python
@celery_app.task(queue='scraping')       # I/O: downloads, Telegram scraping
@celery_app.task(queue='gpu')            # OCR extraction (both GPU workers)
@celery_app.task(queue='gpu_llm')        # LLM batch refinement + judging (GPU 0 only)
@celery_app.task(queue='ml_tagging')     # VLM tagging via host llama-server
@celery_app.task(queue='composition')    # CPU-bound MoviePy composition
@celery_app.task(queue='gpu_status_0/1') # per-GPU status caching
@celery_app.task(queue='gpu_training_0/1')
@celery_app.task(queue='maintenance')    # dispatchers, orchestrator, cleanup, watchdog
@celery_app.task(queue='publishing')     # media host, Reddit/Postpone/Patreon/Telegram posting
```

Broker: `visibility_timeout` 12 h, `task_time_limit` 1 h, `acks_late`, `reject_on_worker_lost`. Inline LLM refinement is disabled on GPU workers; all refinement runs as batches on `gpu_llm`.

**GPU quirk**: the GPU 1 container sets `CUDA_VISIBLE_DEVICES=0` while Docker assigns `device_ids: ['1']`, so physical GPU 1 appears as device 0 inside it.

### Code patterns

```python
# FastAPI routes
def endpoint(db: Session = Depends(get_db)): ...
# Celery tasks
with get_db_context() as db: ...
```

Redis feature flags (default `true` unless noted): `scraping:enabled`, `extraction:enabled`, `ocr:gpu0:enabled`, `ocr:gpu1:enabled`, `watermark_filter:enabled`, `ml_tagging:enabled`, `ml_tagging:gpu` (`0`/`1`/`both`), `llm:batch:enabled`, `pipeline:enabled` (default `false`), `telegram:scraping:enabled` (default `false`), `reddit_bg:enabled`.

### Pipeline

- **Incremental scraping**: `top_all → top_year → top_month → top_week → top_day → new`, state in `scraping_progress`, one proxy per subreddit (`utils/proxy_pool.py`, assignments in Redis `proxy_pool:assignments`), 25 posts per run every 10 min.
- **Extraction**: scraper pushes to a Redis queue; `dispatch-extraction-batch` (1 min) moves batches onto `gpu`. Three-stage captions in `scraped_captions`: `raw_ocr_text` → `rule_based_text` → `llm_refined_text`.
- **Orchestrator** ([tasks/pipeline_orchestrator.py](tasks/pipeline_orchestrator.py), config in [config/automation_config.py](config/automation_config.py)): `collecting → generation_pending → generating → composition_pending → composing → collecting`, per niche, round-robin by lowest quota progress. Transitions are logged to `pipeline_state_log`. Training is manual (`POST /training/trigger/{niche}`); the dormant `training_*` states remain only so old in-flight jobs can finish. When no niche is below quota, the `collecting` tick dispatches one VLM-tagging batch.
- **Background-first generation** ([tasks/bg_first_generation.py](tasks/bg_first_generation.py)): pick an untargeted, niche-compatible background (rules in [config/niche_rules.py](config/niche_rules.py) over the `ml_tags` schema: subjects/activities/setting/mood/camera/text_on_screen), prompt from its tags, generate K candidates into `caption_candidates`, judge inline, promote the best pass to `status='winner'`, mirror into `generated_captions`. Defaults 30 backgrounds × 10 candidates per cycle (`BG_FIRST_BGS_PER_CYCLE`, `BG_FIRST_CANDIDATES_PER_BG`); legacy blind `run_generation_job` remains behind `BG_FIRST_GENERATION=false`.
- **Judge** ([tasks/caption_judge.py](tasks/caption_judge.py)): grammar / flow / bg_consistency / appeal / overall, 1–10, pass = all ≥ 7. Inline plus the `judge-caption-backlog` beat (self-skips unless llama-server is on the `llm` profile).
- **Composition** (`video_generator/`): chunk → time by word count → PIL render → MoviePy overlay; captions needing > 30 s concatenate multiple backgrounds. Composition uses Reddit-sourced backgrounds only. The orchestrator finalizes BG-first composition jobs by counting `composed_videos` rows (stale after 180 s → PARTIAL, no mp4 after 300 s → failed).
- **Visual review**: `scripts/claude_review_videos.py export` writes `brief.md` + 3 keyframes per video; the reviewer writes JSONL; `writeback` fills `claude_review_*` on `composed_videos`. Per-run dirs under `scripts/` are gitignored.
- **Swipe**: `GET /composition/swipe-queue` orders by review verdict → judge score → quality score; `→` approves and schedules through Postpone `schedule-single`.
- **Publishing**: composed video → media host (`publishers/media_host.py`, default self-hosted at `MEDIA_BASE_URL`) → Reddit profile post → 30 min → crossposts. Postpone (`tasks/postpone_tasks.py`), Patreon (browser session), Telegram (Bot API). Analytics beats are offset from scraper dispatchers (:02/:17/:32/:47 snapshots, :07/:22/:37/:52 fresh post stats, 02:30 UTC aged, :12 removed-post detection).
- **Patreon → Telegram membership sync**: `services/telegram_join_listener.py` (own container, one instance per bot) + hourly `tasks/patreon_telegram_sync.py`. Both are commented out in compose/beat by default; uncomment to enable.

### Database

Models in [database/models.py](database/models.py). `init_db()` on API startup runs `create_all`, so new tables appear automatically; new columns need a migration. Migrations are standalone idempotent scripts (`ADD COLUMN IF NOT EXISTS`, `run_migration()` entrypoint; copy `add_claude_review.py`), run with `docker compose exec api python database/migrations/<name>.py`. Many files there are one-off backfills rather than schema changes.

### Deploy path detection

`*.md`/LICENSE/.gitignore → skip; `api/`, `scrapers/`, `config/`, `utils/`, `database/` → API + all workers; `tasks/`, `training/` → workers only; `Dockerfile`, compose, requirements, workflows → everything; anything else → safe rebuild all.

## Quick reference

```bash
docker compose up -d && curl localhost:8000/health
curl localhost:8000/scraper/control/status         # POST .../enable|disable
curl -X POST localhost:8000/subreddits/{name}/scrape
curl localhost:8000/pipeline/status                # POST /pipeline/enable
curl -X POST localhost:8000/training/trigger/{niche}; curl localhost:8000/training/vastai/status/{task_id}
curl localhost:8000/dashboard/worker/status        # hangs; POST .../worker/restart
curl localhost:8000/dashboard/reddit/block-status  # proxy blocks
docker compose exec postgres psql -U captionsuser -d captions
docker compose exec redis redis-cli LLEN gpu
```

## Adding a niche

1. `NicheConfig` entry in [config/automation_config.py](config/automation_config.py) (caption-source subreddits, keywords, prompt, Postpone account, flairs).
2. Background allow-list + tag rules in [config/niche_rules.py](config/niche_rules.py) (`NICHE_SUBREDDITS` is a flat dict; rules use `required` / `forbidden` / `preferred`).
3. Credential rows via the platform `POST` endpoints (`/reddit/accounts`, `/patreon/credentials`, `/telegram/bots`, `/telegram/channels`).
4. Train manually once the corpus exists. Full procedure: [docs/add_new_niche_runbook.md](docs/add_new_niche_runbook.md).

## Configuration

Pydantic settings in [config/settings.py](config/settings.py), loaded from `.env` (`.env.example` lists the keys). Proxy files `proxy_list.txt` / `list_proxyseller.txt` are gitignored and live only on the deploy host. Automation thresholds and per-niche overrides (`generation_temperature`, `generation_repetition_penalty`, `postpone_reddit_username`, `subreddit_flairs`) are in [config/automation_config.py](config/automation_config.py).
