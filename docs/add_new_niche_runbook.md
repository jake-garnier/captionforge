# Add a new niche — runbook

Adding a niche touches a handful of systems: two config files, the scraper tables, and whichever publishing platforms you want the niche to post to (Reddit/Postpone, Telegram, Patreon). Downstream of that the pipeline is entirely data-driven — no further code changes are needed.

Use this as a checklist. Every step has a verification command. Phases run in order: **code → subreddits → external accounts → credential rows → corpus + training → enable**. The examples below walk through the shipped `motivation` niche as if you were adding it from scratch.

All `curl` examples target `http://localhost:8000` (run them on the VM). Substitute `https://captions.example.com` or `http://<vm-ip>:8000` when working remotely.

---

## What a niche owns

| Thing | Where it lives |
|---|---|
| Caption-source subreddits (captioned videos we scrape and OCR to build the training corpus) | `NicheConfig.subreddits` in [config/automation_config.py](../config/automation_config.py) |
| Content keywords (prompt grounding / search hints) | `NicheConfig.keywords` |
| Generation prompt, temperature, repetition penalty | `NicheConfig.generation_prompt`, `generation_temperature`, `generation_repetition_penalty` |
| Daily composition quota | `NicheConfig.daily_video_target` |
| Postpone Reddit account + optional outro CTA | `NicheConfig.postpone_reddit_username`, `reddit_outro_text` |
| Background-clip subreddits (un-captioned stock footage) | `NICHE_SUBREDDITS` in [config/niche_rules.py](../config/niche_rules.py) |
| Background compatibility rules over the VLM tags | `NICHE_TAG_RULES` in [config/niche_rules.py](../config/niche_rules.py) |
| Subreddit flairs (only for subs that require one) | `PostponeConfig.subreddit_flairs` in [config/automation_config.py](../config/automation_config.py) |
| A LoRA adapter | `/data/lora_adapters/<niche>/adapter.gguf` (produced by training) |
| Credential rows for Reddit / Telegram / Patreon | `reddit_accounts`, `telegram_bots`, `telegram_channels`, `patreon_credentials` tables |

The orchestrator round-robins across every niche with `enabled=True`, so a new niche enters rotation on the next beat tick after deploy.

---

## Phase 0 — pre-flight

Decide these before touching anything:

- [ ] **Niche name** — short, lowercase, letters/underscores only (e.g. `motivation`). It ends up in config keys, DB rows, adapter paths, and the Postpone account mapping, so changing it later is painful.
- [ ] **Caption-source subreddits** — subs whose posts are videos *with text burned in*. That text is what the OCR corpus is built from. (`GetMotivated`, `Motivation`, `quotes`, `DecidingToBeBetter`.)
- [ ] **Background-clip subreddits** — subs with short, clean, *un-captioned* footage that suits the niche. (`naturegifs`, `oddlysatisfying`, `aviation`, `hiking`.)
- [ ] **Target subreddits for posting** and whether any of them require a flair.
- [ ] **A Reddit account** with enough age/karma to post in those subs. Fresh accounts get filtered quickly; plan to warm the account up manually first.

---

## Phase 1 — code changes

### 1.1 `NicheConfig` entry

File: [config/automation_config.py](../config/automation_config.py), inside `AutomationConfig._load_defaults()`.

```python
motivation_prompt = """Write a short motivational caption (40-80 words MAX). Second person ("you").

RULES: 2-4 sentences. Every sentence ends with punctuation (. or ! or ?). No URLs or promotions.

STYLE: Mix a blunt truth with encouragement and one concrete next action. Use vivid, specific scenarios — not generic. Vary your opening line and word choices.

Caption:"""

self._niches["motivation"] = NicheConfig(
    name="motivation",                                   # MUST equal the dict key
    subreddits=["GetMotivated", "Motivation", "quotes", "DecidingToBeBetter"],  # caption sources
    keywords=["motivation", "discipline", "mindset", "habits"],
    enabled=True,
    daily_video_target=30,
    generation_prompt=motivation_prompt,
    generation_temperature=0.85,
    generation_repetition_penalty=1.5,
    postpone_reddit_username="motivation_clips",         # exact case as Postpone reports it (see 3.1)
    reddit_outro_text=None,                              # optional CTA appended to Reddit posts only
)
```

Watch-outs:
- `subreddits` here are **caption sources**, not background sources.
- The prompt is what the LoRA-loaded LLM sees at generation time; the BG-first flow prepends a scene description built from the background clip's `ml_tags` (see [tasks/bg_first_generation.py](../tasks/bg_first_generation.py)). Target 40–80 words so captions fit ~60 s of footage.
- `postpone_reddit_username` must match Postpone's `socialAccounts.username` exactly (case included). Fill it in after step 3.1 if you do not know it yet.

### 1.2 Background allow-list and tag rules

File: [config/niche_rules.py](../config/niche_rules.py).

```python
NICHE_SUBREDDITS["motivation"] = ["naturegifs", "oddlysatisfying", "aviation", "hiking"]

NICHE_TAG_RULES["motivation"] = {
    "required": [],
    "forbidden": [
        {"text_on_screen": True},        # burned-in text competes with our overlay
        {"mood": ["funny"]},
    ],
    "preferred": [
        {"mood": ["inspiring", "calm"]},
        {"subject_types_any": ["landscape"]},
        {"camera": ["drone", "timelapse"]},
        {"setting": ["outdoors"]},
    ],
}
```

- `NICHE_SUBREDDITS` is the coarse gate: a background clip is eligible only if it was scraped from one of these subs (`background_videos.searched_tag`).
- `NICHE_TAG_RULES` evaluates the VLM tags on `background_videos.ml_tags` (`subjects[].type`, `activities`, `setting`, `mood`, `camera`, `text_on_screen`). `required` clauses must all hold, any `forbidden` clause rejects, each `preferred` clause adds a ranking point. Supported clause keys are documented in the module docstring (`setting`, `mood`, `camera`, `activities_any`, `subject_types_any`, `subject_types_none`, `text_on_screen`, `any_of`).
- Untagged clips pass the tag layer trivially; the orchestrator tags clips during idle ticks, so the rules bite once tagging catches up.

### 1.3 Subreddit flairs (only if required)

Some subreddits reject posts without a flair. Sample the flairs a sub actually uses:

```bash
docker compose exec api python scripts/list_subreddit_flairs.py GetMotivated
```

Add the exact flair text to `PostponeConfig.subreddit_flairs` in [config/automation_config.py](../config/automation_config.py) (`"getmotivated": "[Video]"`). Match brackets, case, and trailing spaces exactly.

### 1.4 Commit, push, verify

```bash
python -m py_compile config/automation_config.py config/niche_rules.py
git add config/automation_config.py config/niche_rules.py
git commit -m "niche: add motivation"
git push origin main          # GitHub Actions redeploys api + workers (config/ change)
```

```bash
curl -s http://localhost:8000/pipeline/status | python3 -c "import json,sys; print(list(json.load(sys.stdin)['niches']))"
curl -s http://localhost:8000/pipeline/config | python3 -m json.tool | grep -A6 '"motivation"'
```

---

## Phase 2 — register the subreddits with the scrapers

The config lists tell the pipeline which subreddit *belongs* to which niche; the scrapers only run for subreddits that exist in their tables. Add them via the API (or the **Scrapers** / **Background Videos** tabs in the UI).

```bash
# Caption sources → scraping_progress / subreddit config
for s in GetMotivated Motivation quotes DecidingToBeBetter; do
  curl -s -X POST http://localhost:8000/subreddits/ -H 'Content-Type: application/json' \
       -d "{\"name\": \"$s\", \"min_score\": 300, \"batch_size\": 25}"
done
curl -X POST http://localhost:8000/scraper/control/enable

# Background clips → reddit_background_subreddits
for s in naturegifs oddlysatisfying aviation hiking; do
  curl -s -X POST http://localhost:8000/background-videos/reddit/subreddits/ -H 'Content-Type: application/json' \
       -d "{\"name\": \"$s\", \"min_score\": 100, \"min_duration\": 10, \"max_duration\": 60}"
done
curl -X POST http://localhost:8000/background-videos/reddit/control/enable
```

Kick the first run instead of waiting for the beat (`dispatch-scraper-tasks` and `dispatch-reddit-background-scraper` both run every 10 minutes):

```bash
curl -X POST http://localhost:8000/subreddits/GetMotivated/scrape
curl -X POST http://localhost:8000/background-videos/reddit/subreddits/naturegifs/scrape
```

Verify:

```bash
curl -s http://localhost:8000/scraper/control/status
curl -s http://localhost:8000/dashboard/extraction/queue/status       # OCR backlog
curl -s "http://localhost:8000/background-videos/count"
```

---

## Phase 3 — external accounts

None of this is automatable; it is clicking through web UIs.

### 3.1 Reddit account + Postpone

1. Create the Reddit account, give it a neutral bio, and post/comment from it by hand for a few days.
2. In Postpone: *Settings → Social Accounts → Connect Reddit*, log in as that account, finish OAuth.
3. Read back the exact username Postpone stores:

```bash
docker compose exec -T api python -c "
from publishers.postpone_publisher import PostponePublisher
from config.settings import settings
p = PostponePublisher(settings.POSTPONE_API_KEY)
for a in p._execute_query('query { socialAccounts { id username platform } }')['socialAccounts']:
    print(a['platform'], a['id'], a['username'])
"
```

Put that string in `postpone_reddit_username` (step 1.1) and push again if it changed.

### 3.2 Telegram bot + channel

1. In Telegram, message `@BotFather`: `/newbot` → display name → username ending in `bot`. Save the token.
2. Create the channel, add the bot as an admin with **Post Messages**, and post one message so the bot receives an update it can discover the channel from.

### 3.3 Patreon (optional — can be wired later)

1. Create the Patreon page.
2. Register an OAuth client at https://www.patreon.com/portal/registration/register-clients and note client ID, client secret, access token, refresh token.
3. If you use the Patreon → Telegram membership sync, pick the post whose comments collect join requests and note its post ID.

---

## Phase 4 — credential rows

Each platform has a `POST` endpoint that creates the per-niche row on demand (they upsert on `niche`). Alternatively pre-seed placeholders with [database/migrations/seed_niche_credential_rows.py](../database/migrations/seed_niche_credential_rows.py) after editing its `NICHES` list.

### 4.1 Reddit account row (direct Playwright posting)

```bash
curl -X POST http://localhost:8000/reddit/accounts -H 'Content-Type: application/json' \
     -d '{"niche": "motivation", "username": "motivation_clips", "subreddits": "GetMotivated,Motivation"}'

# Log in once through the browser session (noVNC at $NOVNC_PUBLIC_URL), then save cookies:
curl -X POST "http://localhost:8000/reddit/session/start?niche=motivation"
curl -X POST http://localhost:8000/reddit/session/save-cookies
curl -X POST http://localhost:8000/reddit/session/stop
```

Only needed if you post directly (`POST /reddit/post`, `POST /publishing/publish`). Postpone scheduling does not need it.

### 4.2 Telegram

```bash
curl -X POST http://localhost:8000/telegram/bots -H 'Content-Type: application/json' \
     -d '{"niche": "motivation", "bot_token": "<BotFather token>", "bot_username": "motivation_clips_bot", "bot_name": "Motivation Clips"}'

curl -X POST http://localhost:8000/telegram/bots/motivation/discover-channels   # returns channel_id like -100XXXXXXXXXX

curl -X POST http://localhost:8000/telegram/channels -H 'Content-Type: application/json' \
     -d '{"niche": "motivation", "channel_id": "-100XXXXXXXXXX"}'

curl -X POST http://localhost:8000/telegram/channels/motivation/test              # a test message should land in the channel
```

### 4.3 Patreon

```bash
curl -X POST http://localhost:8000/patreon/credentials -H 'Content-Type: application/json' \
     -d '{"niche": "motivation", "email": "you@example.com"}'

# Interactive login through noVNC, then persist the session:
curl -X POST "http://localhost:8000/patreon/session/start?niche=motivation"
curl -X POST http://localhost:8000/patreon/session/login
curl -X POST http://localhost:8000/patreon/session/save-cookies
curl -X POST http://localhost:8000/patreon/credentials/motivation/test-connection
```

For the membership sync, add the OAuth client to `.env` (the variable prefix is the upper-cased niche name):

```dotenv
PATREON_MOTIVATION_CLIENT_ID=...
PATREON_MOTIVATION_CLIENT_SECRET=...
PATREON_MOTIVATION_ACCESS_TOKEN=...
PATREON_MOTIVATION_REFRESH_TOKEN=...
PATREON_MOTIVATION_CAMPAIGN_ID=...    # optional, cached on first sync
PATREON_MOTIVATION_POST_ID=...
```

The `api` and worker services only read `.env` at container creation, so a restart is not enough:

```bash
docker compose up -d --force-recreate --no-deps api celery-worker-scraper
```

The sync itself (`telegram-join-listener` service + `sync-patreon-telegram` beat) ships commented out in `docker-compose.yml` / `tasks/celery_app.py`; uncomment both to enable it.

---

## Phase 5 — corpus, training, first generation

### 5.1 Let the corpus build

Scraping pulls 25 posts per subreddit per 10-minute tick, and only a fraction of posts yield OCR-able captions. Expect a few days for a new niche with four source subs.

```bash
curl -s "http://localhost:8000/training/data/count/motivation?min_upvotes=100"
```

The export needs **≥ 100** refined captions to start; a useful adapter wants **500+**.

### 5.2 Train the adapter

```bash
curl -X POST "http://localhost:8000/training/trigger/motivation?auto_destroy=false"
curl http://localhost:8000/training/vastai/status/<task_id>
```

Takes 2–6 hours on a rented RTX 4090. Follow [training_runbook.md](training_runbook.md) — it covers the pre-flight checks and the silent-failure modes. When it finishes, `/data/lora_adapters/motivation/adapter.gguf` exists and llama-server has been restarted with it:

```bash
curl -s http://localhost:8000/training/adapters/motivation | python3 -m json.tool
curl -s http://localhost:8000/llama-server/status
```

### 5.3 Smoke-test generation and composition

```bash
curl -X POST http://localhost:8000/pipeline/enable
curl -X POST http://localhost:8000/pipeline/trigger/generate/motivation
watch -n 30 'curl -s http://localhost:8000/pipeline/state'
```

The orchestrator picks the trigger up on its next 5-minute tick and walks `generation_pending → generating → composition_pending → composing → collecting`. Composed videos appear in the **Video Gallery** tab; after a Stage-4 review ([../scripts/RUN_CLAUDE_REVIEW.md](../scripts/RUN_CLAUDE_REVIEW.md)) they show up in **Swipe**, where `→` approves and schedules through Postpone.

---

## Phase 6 — post-launch

- **Analytics** auto-discovers the account from `postpone_reddit_username`; it appears in the Analytics tab within ~20 minutes. Backfill history with `POST /analytics/accounts/motivation_clips/backfill`.
- **Quota**: `daily_video_target` caps composed videos per day; reset a day with `POST /pipeline/quotas/motivation/reset`.
- **Tag rules tuning**: if the wrong kind of clip keeps getting picked, add one clause to `NICHE_TAG_RULES` rather than filtering at a call site.

---

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Niche name differs between the two config files | Niche shows in `/pipeline/status` but never composes (no eligible backgrounds) | Same key in `NICHE_CONFIGS` and `NICHE_SUBREDDITS` |
| Subreddits only in config, not in the scraper tables | `total_captions` stays 0 | `POST /subreddits/` and `POST /background-videos/reddit/subreddits/` |
| `postpone_reddit_username` case wrong | Postpone schedule fails with "Reddit account is not connected" | Re-run the `socialAccounts` query and copy the exact string |
| `.env` edited but not re-read | New Patreon vars missing | `docker compose up -d --force-recreate --no-deps api` |
| Fresh Reddit account | Posts publish but never appear in the sub | Warm the account up manually; check in a private window |
| Corpus too small | Training aborts with "Not enough training data" | Wait; ≥ 100 to start, 500+ for a usable adapter |
| Background subs untagged | Rules never fire, random-looking clips | Tagging runs on idle orchestrator ticks; check `ml_tagging:enabled` in Redis |

---

## Quick checklist

- [ ] `config/automation_config.py`: `NicheConfig` + prompt (+ flairs if needed)
- [ ] `config/niche_rules.py`: `NICHE_SUBREDDITS` + `NICHE_TAG_RULES`
- [ ] `git push` and confirm the niche in `/pipeline/status`
- [ ] `POST /subreddits/` for each caption source; `POST /background-videos/reddit/subreddits/` for each background sub
- [ ] Reddit account created and warmed up; connected in Postpone; `postpone_reddit_username` matches
- [ ] Telegram: bot + channel; `POST /telegram/bots`, `.../discover-channels`, `POST /telegram/channels`, `.../test`
- [ ] (Optional) `POST /reddit/accounts` + session login; `POST /patreon/credentials` + session login; Patreon env vars + force-recreate
- [ ] Corpus ≥ 100 (ideally 500+): `GET /training/data/count/{niche}`
- [ ] `POST /training/trigger/{niche}?auto_destroy=false`; smoke-test the adapter
- [ ] `POST /pipeline/enable`, `POST /pipeline/trigger/generate/{niche}`
- [ ] Run a Stage-4 review, swipe, confirm the account appears in Analytics

Roughly 1–2 hours of hands-on work spread across a week; the wall time is scraping and training.
