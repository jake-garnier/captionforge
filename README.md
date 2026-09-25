# CaptionForge

**An end-to-end pipeline that turns Reddit's captioned-video culture into a content engine for any niche.**

CaptionForge scrapes captioned videos from the subreddits of a niche you choose (motivational quotes, fitness tips, recipes, travel clips, ...), extracts the on-screen text with a GPU vision model, fine-tunes a per-niche LoRA adapter on the corpus, generates new captions grounded in the visual content of stock background clips, composes them into finished videos, runs a multi-stage quality gate, and schedules the results to Reddit, Patreon, and Telegram.

It is a single Docker Compose stack: FastAPI + a single-page operator UI, Celery workers pinned to specific GPUs, PostgreSQL, Redis, and a host-side `llama-server` for LLM inference. It has run unattended for months on a two-GPU homelab VM with zero-downtime deploys from GitHub Actions.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Running locally](#running-locally)
4. [Deploying to a VM](#deploying-to-a-vm)
5. [Configuration reference](#configuration-reference)
6. [Adding a niche](#adding-a-niche)
7. [Training a LoRA adapter](#training-a-lora-adapter)
8. [Publishing integrations](#publishing-integrations)
9. [Operating it](#operating-it)
10. [Project layout](#project-layout)
11. [Docs](#docs)

---

## What it does

```
              ┌────────────────────────── CAPTION CORPUS ──────────────────────────┐
 Reddit  ───▶ │ incremental scraper ─▶ download ─▶ GPU OCR (Qwen2-VL) ─▶ rule-based │
 (per niche)  │   top_all→…→new         (proxy pool)   3 keyframes/clip    cleanup   │
              │                                        └─▶ LLM refinement (Mistral) │
              └────────────────────────────────────────────────────────────────────┘
                                             │  export + train/val split
                                             ▼
                          QLoRA fine-tune per niche on a rented RTX 4090 (Vast.ai)
                                             │  PEFT → GGUF, hot-loaded by llama-server
                                             ▼
 Reddit  ───▶ background clips ─▶ watermark filter ─▶ VLM tagging (InternVL3) ─┐
 (stock subs)                     (OCR-based)          subjects/setting/mood   │
                                                                               ▼
              ┌──────────────────────── GENERATION & QUALITY ────────────────────────┐
              │ pick untargeted, niche-compatible clip ─▶ prompt grounded in its tags │
              │ ─▶ K candidates (LoRA) ─▶ LLM judge (grammar/flow/consistency/appeal) │
              │ ─▶ winner ─▶ MoviePy composition (PIL text overlay, timing, chunking) │
              │ ─▶ optional agentic visual review (keyframes + brief) ─▶ Swipe queue  │
              └──────────────────────────────────────────────────────────────────────┘
                                             │  one keypress: approve + schedule
                                             ▼
                 media host (self-hosted URL) ─▶ Reddit profile post ─▶ crossposts
                 Postpone (scheduled Reddit) · Patreon · Telegram · Reddit analytics
```

Highlights that are worth reading the code for:

- **Incremental scraping state machine** (`tasks/incremental_scraping.py`): progressive depth `top_all → top_year → … → new` with a pagination cursor per subreddit, one dedicated proxy per subreddit, and automatic block detection/backoff.
- **Event-driven GPU extraction** (`tasks/extraction_dispatcher.py`, `utils/extraction_queue.py`): scraping and OCR are decoupled through a Redis queue; two GPU workers of different sizes pull from the same `gpu` queue, with per-GPU enable flags and idle-model unloading.
- **Background-first generation** (`tasks/bg_first_generation.py`): instead of generating captions blind and hunting for a matching clip, the pipeline starts from a tagged clip and asks the LoRA for captions that fit *that* footage. Every candidate is judged inline; the best judge-pass is promoted.
- **Per-niche orchestrator** (`tasks/pipeline_orchestrator.py`): a small state machine (`collecting → generation_pending → generating → composition_pending → composing`) with daily quotas, round-robin niche selection, an audit log of every transition, and idle-time dispatch of VLM tagging.
- **Agentic visual review** (`scripts/claude_review_videos.py`): exports a brief plus three keyframes per composed video for an interactive Claude Code session to score, then writes verdicts back. The Swipe tab orders its queue by review verdict → judge score → heuristic quality.
- **Zero-downtime deploys** (`.github/workflows/deploy.yml`): a self-hosted runner on the VM detects which services a push touched and rebuilds only those, skipping GPU workers while a generation or composition job is active.
- **Numbered quality stages.** Code and docs refer to the quality gate as stages 1–5: VLM tagging, background-first generation, LLM judge, visual review, Swipe approval.
- **Pluggable media host** (`publishers/media_host.py`): Reddit link posts need a public video URL; the default host is the API itself behind a Cloudflare tunnel, and the interface lets you drop in a third-party host.

---

## Architecture

### Services (`docker-compose.yml`)

| Service | Role | GPU | Queues |
|---|---|---|---|
| `api` | FastAPI REST API + operator UI at `/gallery/` (port 8000), noVNC for interactive browser logins (port 6080) | – | – |
| `celery-worker` | GPU 0 worker: OCR extraction, LLM batch refinement/judging, GPU 0 status | GPU 0 | `gpu`, `gpu_llm`, `gpu_status_0`, `gpu_training_0` |
| `celery-worker-gpu-2` | GPU 1 worker: OCR extraction, GPU 1 status | GPU 1 | `gpu`, `gpu_status_1`, `gpu_training_1` |
| `celery-worker-ml-tagging` | VLM tagging of background clips via host `llama-server` (tensor-parallel across both GPUs) | both | `ml_tagging` |
| `celery-worker-composition` | MoviePy/PIL/ffmpeg composition (CPU only, so OCR backlogs never block composes) | – | `composition` |
| `celery-worker-scraper` | I/O-bound scraping, dispatchers, orchestrator, publishing | – | `scraping`, `maintenance`, `publishing` |
| `celery-beat` | Periodic scheduler; the only service mounting the Docker socket (for watchdog auto-recovery) | – | – |
| `postgres` | `pgvector/pgvector:pg15` | – | – |
| `redis` | Broker, result backend, feature flags | – | – |
| `flower` | Celery monitoring (port 5555) | – | – |
| `tunnel` | Cloudflare tunnel for public access (optional) | – | – |

All Celery queues and the beat schedule are defined in `tasks/celery_app.py`. Every GPU worker uses the `solo` pool; CPU workers use `prefork`.

### External dependency: `llama-server` on the host

LLM inference (caption refinement, LoRA generation, judging) and VLM tagging are served by [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` running **on the host machine**, not in a container, so the two large models can be tensor-split across both GPUs and hot-swapped between profiles. Containers reach it at `http://host.docker.internal:1234`. `utils/llama_server_control.py` switches profiles (`llm` = Mistral-Small-24B-Instruct + per-niche LoRA, `vlm-internvl3` = InternVL3-14B) over SSH to the host.

### Data model

Key tables (SQLAlchemy models in `database/models.py`; the schema is created automatically on first API start, column additions ship as idempotent scripts in `database/migrations/`):

| Table | Purpose |
|---|---|
| `videos`, `scraped_captions` | Scraped posts and their three-stage captions (`raw_ocr_text` → `rule_based_text` → `llm_refined_text`) |
| `scraping_progress` | Per-subreddit incremental scraping stage + cursor |
| `background_videos` | Stock clips with `filter_status` (watermark filter) and `ml_tags` (VLM) |
| `trained_models`, `training_jobs` | LoRA registry and Vast.ai job tracking |
| `caption_candidates`, `generated_captions`, `generation_jobs` | Background-first generation output with judge scores |
| `composed_videos`, `video_composition_jobs` | Final videos, quality metrics, review verdicts |
| `video_publish_jobs`, `reddit_crossposts`, `postpone_schedule_jobs`, `patreon_publish_jobs`, `telegram_publish_jobs` | Publishing state per platform |
| `reddit_account_snapshots`, `reddit_posts`, `reddit_post_stats` | Analytics tab |
| `pipeline_state_log` | Audit trail of every orchestrator transition |

---

## Running locally

### Prerequisites

- Docker Engine 24+ with the Compose plugin
- An NVIDIA GPU with ≥ 8 GB VRAM and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-smi` must work inside a container). OCR and VLM tagging need a GPU; scraping, the UI, composition, and publishing run fine without one.
- ~40 GB free disk for images, models, and videos
- Optional: a second GPU. With one GPU, remove `celery-worker-gpu-2` from `docker-compose.yml`, change `celery-worker-ml-tagging` to `device_ids: ['0']`, and set the Redis flag `ocr:gpu1:enabled=false` (see [Operating it](#operating-it)).

### 1. Clone and configure

```bash
git clone https://github.com/jake-garnier/captionforge.git
cd captionforge
cp .env.example .env
```

Edit `.env`. For a first local run you only need:

```dotenv
POSTGRES_DB=captions
POSTGRES_USER=captionsuser
POSTGRES_PASSWORD=pick_something
FLOWER_BASIC_AUTH=admin:pick_something
# leave everything else empty for now
```

### 2. Start the stack

```bash
docker compose up -d --build        # first build takes 10–20 min (CUDA base image + Playwright browsers)
docker compose ps
curl http://localhost:8000/health
```

The API creates the schema on first start. Open the UI at **http://localhost:8000/gallery/**.

If you only have one GPU, comment out the `celery-worker-gpu-2` service and trim `celery-worker-ml-tagging` to `device_ids: ['0']` before `up`, or those containers will fail to schedule.

### 3. Add a subreddit and scrape

Either use the **Scrapers** tab, or:

```bash
curl -X POST http://localhost:8000/subreddits/ -H 'Content-Type: application/json' \
     -d '{"name": "GetMotivated", "min_score": 300}'
curl -X POST http://localhost:8000/scraper/control/enable
curl -X POST http://localhost:8000/subreddits/GetMotivated/scrape     # or wait for the 10-minute beat
curl http://localhost:8000/scraper/control/status
```

A subreddit belongs to whichever niche lists it as a caption source in `config/automation_config.py` (`GetMotivated` is in the shipped `motivation` example). Without proxies the scraper uses your own IP against Reddit's public JSON endpoints, which is fine for one or two subreddits. For many subreddits, set `PROXY_POOL` (see [Configuration](#configuration-reference)); each subreddit is pinned to its own proxy.

### 4. Watch OCR run

The extraction dispatcher moves downloaded videos onto the `gpu` queue every minute. Check progress in the **Extraction** tab or:

```bash
curl http://localhost:8000/dashboard/extraction/queue/status
docker compose logs -f celery-worker
```

The first OCR job downloads Qwen2-VL-2B (~5 GB) into the container's Hugging Face cache.

### 5. (Optional) LLM features: run `llama-server` on the host

Caption refinement, judging, generation, and VLM tagging all talk to `llama-server` at `http://host.docker.internal:1234`. Everything else works without it.

```bash
# on the host (paths match the defaults in utils/llama_server_control.py; override with LLM_HOST_LLAMA_CPP_DIR / LLM_HOST_MODELS_DIR)
sudo mkdir -p /opt/llama.cpp /opt/models && sudo chown $USER /opt/llama.cpp /opt/models
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
# download a GGUF, e.g. Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf (~14 GB) from Hugging Face, into /opt/models
/opt/llama.cpp/build/bin/llama-server --model /opt/models/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf \
    --host 0.0.0.0 --port 1234 --n-gpu-layers 35 --ctx-size 4096
curl http://localhost:1234/v1/models
```

`scripts/llama-server.service` is a systemd unit for the same thing (edit the `# EDIT ME` lines: user, paths, `MODEL_PATH`, `TENSOR_SPLIT`, `N_GPU_LAYERS`). The containers switch profiles over SSH, so set `LLM_HOST_IP`, `LLM_HOST_SSH_USER`, and `LLM_HOST_SSH_KEY_PATH` (or `LLM_HOST_SSH_PASSWORD`) in `.env` if you want VLM tagging or the LoRA hot-reload to work; otherwise leave them unset and those steps fail with a clear "host SSH not configured" error rather than hanging.

### Running without Docker (development)

There is no separate dev server; the repo is bind-mounted at `/app` inside the containers, so code edits take effect on the next container restart (`docker compose restart api`), and the UI HTML is served uncached so edits show on reload. There is no test suite; verification is `python -m py_compile <file>` plus watching the logs.

---

## Deploying to a VM

This is how the reference deployment runs: an Ubuntu 22.04 VM (Proxmox, two passed-through NVIDIA GPUs) with a GitHub Actions **self-hosted runner** that redeploys on every push to `main`.

### 1. Provision the VM

```bash
# Ubuntu 22.04, NVIDIA driver + container toolkit
sudo apt update && sudo apt install -y nvidia-driver-535 docker.io docker-compose-plugin git build-essential cmake
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker $USER && newgrp docker
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi   # must list your GPUs
```

### 2. Install `llama-server` as a service

Follow step 5 of the local guide to build llama.cpp and download the GGUF, then:

```bash
sudo cp scripts/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
sudo systemctl status llama-server
```

Create an SSH key the containers will use to control it and add the public key to `~/.ssh/authorized_keys` on the VM. Point `LLM_HOST_SSH_KEY_PATH` at the private key (mounted into the workers via the `vastai_ssh` volume at `/root/.ssh`) and set `LLM_HOST_IP` to the VM's LAN IP.

### 3. Register a self-hosted runner

In the GitHub repo: *Settings → Actions → Runners → New self-hosted runner*, pick Linux x64, and run the printed commands on the VM as the deploy user, in `~/actions-runner`. Install it as a service:

```bash
cd ~/actions-runner && sudo ./svc.sh install && sudo ./svc.sh start
```

The workflow checks out into `~/actions-runner/_work/<repo>/<repo>`; that checkout is what gets bind-mounted into the containers, so it is also where you edit `.env` and the gitignored proxy files.

### 4. First deploy

```bash
cd ~/actions-runner/_work/captionforge/captionforge   # after the first workflow run, or clone it there yourself
cp .env.example .env && nano .env                        # fill in DB password, integrations, MEDIA_BASE_URL, LLM_HOST_*
docker compose up -d --build
curl http://localhost:8000/health
```

From then on, `git push origin main` deploys. `.github/workflows/deploy.yml`:

- classifies changed paths (`api/`, `tasks/`, `database/`, `Dockerfile`, docs-only, ...) and rebuilds only the affected services;
- checks the orchestrator state and active GPU jobs first, and **skips GPU workers** while generation or composition is running so a deploy never kills a job;
- verifies health afterwards and lists any exited containers.

Never run `docker compose down -v`: it drops the named volumes (database, videos, adapters).

### 5. Public access (optional but needed for publishing)

Reddit link posts and Postpone schedules need a public URL for composed videos. Create a [Cloudflare tunnel](https://one.dash.cloudflare.com) routing your hostname to `http://api:8000`, put its token in `CLOUDFLARE_TUNNEL_TOKEN`, and set `MEDIA_BASE_URL=https://<your-hostname>`. The `tunnel` service in `docker-compose.yml` picks it up; no ports need to be opened on the VM.

### 6. Remote operation

Everything is driven by the HTTP API and the UI, so a VPN/overlay network (e.g. Tailscale) to the VM is all you need for remote administration. All examples below assume `http://localhost:8000` on the VM; substitute the VM address when remote.

---

## Configuration reference

All settings are Pydantic fields in `config/settings.py`, loaded from `.env`. The important ones:

| Variable | Purpose |
|---|---|
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | Database credentials (also used by the `postgres` container) |
| `FLOWER_BASIC_AUTH` | `user:pass` for the Flower dashboard |
| `PROXY_POOL` | Comma-separated `user:pass@host:port` proxies; or put one per line in `proxy_list.txt` (gitignored). One proxy is pinned per subreddit in Redis. |
| `LLM_POSTPROCESS_ENABLED` | Inline LLM refinement on GPU workers (off by default; batch refinement on `gpu_llm` handles it) |
| `LLM_HOST_IP`, `LLM_HOST_SSH_USER`, `LLM_HOST_SSH_KEY_PATH` / `LLM_HOST_SSH_PASSWORD` | How containers reach and control the host `llama-server` |
| `MEDIA_BASE_URL`, `MEDIA_HOST_TYPE` | Public base URL for composed videos; `self` (default) serves them from this API |
| `NOVNC_PUBLIC_URL` | Link the UI shows for interactive browser logins |
| `CLOUDFLARE_TUNNEL_TOKEN` | Tunnel token for the `tunnel` service |
| `VASTAI_API_KEY`, `VASTAI_MAX_PRICE_PER_HOUR`, `VASTAI_PREFERRED_GPU` | Cloud LoRA training |
| `POSTPONE_API_KEY`, `POSTPONE_POSTING_HOUR_UTC`, `POSTPONE_STAGGER_MINUTES` | Scheduled Reddit posting via [Postpone](https://postpone.app) |
| `REDDIT_POSTER_TYPE` (`playwright` \| `praw`), `REDDIT_USERNAME`, `REDDIT_PASSWORD`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | Direct Reddit posting |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | Telegram channel scraping (Telethon) |
| `PATREON_<NICHE>_CLIENT_ID` … `_POST_ID` | Per-niche Patreon OAuth client for the Patreon → Telegram membership sync |
| `AUTOMATION_VIDEOS_PER_NICHE_DAY`, `AUTOMATION_MIN_QUALITY_SCORE`, `BG_FIRST_BGS_PER_CYCLE`, `BG_FIRST_CANDIDATES_PER_BG` | Orchestrator quotas and generation batch sizes (`config/automation_config.py`) |
| `WATCHDOG_AUTO_RECOVERY` | Let `celery-beat` restart hung workers via the Docker socket |

Runtime feature flags live in Redis and are toggled from the UI or the API (`scraping:enabled`, `extraction:enabled`, `ocr:gpu0:enabled`, `ocr:gpu1:enabled`, `ml_tagging:enabled`, `llm:batch:enabled`, `pipeline:enabled`, `watermark_filter:enabled`, ...).

---

## Adding a niche

A niche is a bundle of caption-source subreddits, background-clip subreddits, a generation prompt, compatibility rules, and publishing targets. The four shipped examples are `motivation`, `fitness`, `cooking`, and `travel`. To add one:

1. **`config/automation_config.py`** — add a `NicheConfig` entry to `NICHE_CONFIGS`: caption-source subreddits, content `keywords`, the generation system prompt, sampling overrides, Postpone account, and any required subreddit flairs.
2. **`config/niche_rules.py`** — add the background-clip subreddit allow-list and the `required` / `forbidden` / `preferred` rules over the VLM tag schema (`subjects`, `activities`, `setting`, `mood`, `camera`, `text_on_screen`).
3. Add the subreddits in the **Scrapers** tab (or `POST /subreddits/`), let the corpus build, then train an adapter (below).
4. Create credential rows for the platforms you publish to via their `POST` endpoints (`/reddit/accounts`, `/patreon/credentials`, `/telegram/bots`, ...), or from the UI tabs.

The step-by-step version, including the Patreon/Telegram/Postpone side, is in [`docs/add_new_niche_runbook.md`](docs/add_new_niche_runbook.md).

---

## Training a LoRA adapter

Training is manual and runs on a rented RTX 4090 (about $0.40/hour, ~2–4 hours per niche):

```bash
curl -X POST http://localhost:8000/training/trigger/motivation         # export corpus, rent instance, train QLoRA
curl http://localhost:8000/training/vastai/status/<task_id>
curl http://localhost:8000/training-manager/models
```

The task exports LLM-refined captions for the niche, splits train/validation, launches a Vast.ai instance, runs `training/train_vastai.py` (QLoRA on Mistral-Small-24B, rank 32 / alpha 64), downloads the PEFT adapter, converts it to GGUF (`training/convert_peft_to_gguf.py`), and restarts `llama-server` with the new adapter. `docs/training_runbook.md` covers the failure modes worth knowing about (silent OOMs, instance churn, adapter conversion).

---

## Publishing integrations

| Platform | How | Where |
|---|---|---|
| Reddit (direct) | Playwright browser session (default, no API key) or PRAW; profile post then delayed crossposts | `publishers/reddit_poster*.py`, `tasks/publishing_tasks.py` |
| Reddit (scheduled) | [Postpone](https://postpone.app) GraphQL API; the Swipe tab's approve key schedules in one call | `publishers/postpone_publisher.py`, `tasks/postpone_tasks.py` |
| Patreon | Browser automation with a persisted session (log in once through noVNC) | `publishers/patreon_*.py`, `tasks/patreon_tasks.py` |
| Telegram | Bot API posting; optional Patreon → Telegram membership sync (join-request approval + hourly reconciliation) | `publishers/telegram_*.py`, `services/telegram_join_listener.py` |
| Analytics | Karma, follower, and per-post stats for each niche's Reddit account, scraped on an offset schedule | `tasks/reddit_analytics.py`, Analytics tab |

Composed videos are served publicly from `MEDIA_BASE_URL` so link posts have somewhere to point; implement `MediaHost` in `publishers/media_host.py` to use a third-party host instead.

---

## Operating it

```bash
docker compose ps && curl -s localhost:8000/health                      # stack health
curl localhost:8000/pipeline/status                                    # orchestrator state per niche
curl -X POST localhost:8000/pipeline/enable                            # turn on the automation loop
curl localhost:8000/dashboard/worker/status                            # hang detection
curl localhost:8000/dashboard/reddit/block-status                      # proxy block detection
docker compose exec redis redis-cli LLEN gpu                           # queue depth
docker compose exec postgres psql -U captionsuser -d captions          # DB shell
docker compose exec api python database/migrations/<name>.py          # run a migration
docker compose logs -f api celery-worker celery-beat
```

The **Dashboard** and **Tasks** tabs surface the same information, plus per-GPU memory, hang incidents, and beat history. Flower is at `:5555`.

Troubleshooting quick hits:

| Symptom | Check |
|---|---|
| GPU not detected in a worker | `docker compose exec celery-worker nvidia-smi`; confirm the container toolkit is configured |
| Worker hung | `GET /dashboard/worker/status`, then `POST /dashboard/worker/restart` (or enable `WATCHDOG_AUTO_RECOVERY`) |
| Reddit blocked | `GET /dashboard/reddit/block-status`; rotate proxies, `DEL proxy_pool:assignments` in Redis, re-enable the scraper |
| LLM steps skipped | `curl http://<host>:1234/v1/models` and `systemctl status llama-server` on the host |
| Playwright errors | `docker compose exec celery-worker-scraper playwright install firefox` |

---

## Project layout

```
api/                 FastAPI routers (one per domain) + gallery_with_dashboard.html (the whole UI)
tasks/               Celery tasks: scraping, extraction, generation, judging, composition, publishing, orchestrator, analytics, watchdog
scrapers/            Reddit JSON/Playwright scrapers, downloaders, Qwen2-VL OCR, caption post-processing
video_generator/     Chunking, timing, PIL rendering, MoviePy composition, style presets
publishers/          Reddit, Postpone, Patreon, Telegram, media host
training/            Corpus export, train/val split, Vast.ai training script, PEFT→GGUF conversion
config/              Pydantic settings, niche configs, niche compatibility rules
database/            SQLAlchemy models, session helpers, idempotent migration scripts
utils/               Proxy pool, extraction queue, llama-server control, VNC sessions, heartbeats
services/            Long-running non-Celery services (Telegram join-request listener)
scripts/             Visual review export/writeback, audits, llama-server systemd unit
docs/                Runbooks and reference
.github/workflows/   Zero-downtime deploy via self-hosted runner
```

## Docs

- [`docs/add_new_niche_runbook.md`](docs/add_new_niche_runbook.md) — adding a niche end to end
- [`docs/training_runbook.md`](docs/training_runbook.md) — Vast.ai LoRA training and its failure modes
- [`docs/pipeline_state_machine.md`](docs/pipeline_state_machine.md) — orchestrator state transitions
- [`docs/api_reference.md`](docs/api_reference.md) — API catalog
- [`scripts/RUN_CLAUDE_REVIEW.md`](scripts/RUN_CLAUDE_REVIEW.md) — running the agentic visual review
- [`CLAUDE.md`](CLAUDE.md) — orientation notes for AI coding agents working in this repo

## License

MIT — see [LICENSE](LICENSE).
