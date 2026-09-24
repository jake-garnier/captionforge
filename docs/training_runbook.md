# LoRA training runbook (Vast.ai)

Step-by-step procedure for training a per-niche LoRA adapter on a rented Vast.ai GPU, written so it can be run cold. Training is **manual only** — the orchestrator never triggers it. Each run costs real money (RTX 4090 at ≤ `VASTAI_MAX_PRICE_PER_HOUR`, default $0.50/h; a run is typically 2–6 h wall-clock, so $1.50–3.00).

**Rule from past incidents:** do not destroy the Vast.ai instance until you have either verified the adapter generates sensible captions or fully diagnosed why it did not. Trigger with `auto_destroy=false` and destroy by hand afterwards — a run can complete all eight steps and still yield an adapter that produces nothing usable, and the instance is the only place the evidence lives.

Commands assume you are on the VM in the compose checkout (`http://localhost:8000`); substitute `http://<vm-ip>:8000` when remote.

---

## What the task does

`POST /training/trigger/{niche}` dispatches `trigger_cloud_training` ([tasks/training_tasks.py](../tasks/training_tasks.py)) on the `maintenance` queue (runs in `celery-worker-scraper`). Eight steps, each logged as `Step N/8`:

| Step | What happens | Typical time |
|---|---|---|
| 1 | Export LLM-refined captions for the niche to `/tmp/training_data_<niche>.jsonl` (in the worker container). Aborts below **100** rows. | 1–2 min |
| 2 | Search Vast.ai offers (`RTX_4090`, ≥ 100 GB disk, ≤ max price, reliability ≥ 0.9), create the cheapest | 2–5 min |
| 3 | `wait_for_ready` — poll until SSH works (timeout 1200 s) | 3–10 min |
| 4 | Upload the JSONL and `training/train_vastai.py` to `/root/training/` on the instance | 1–2 min |
| 5 | Run QLoRA on `mistralai/Mistral-Small-24B-Instruct-2501` (rank 32, alpha 64, lr 2e-4, 2 epochs, batch 1 × grad-accum 8). Timeout `VASTAI_MAX_TRAINING_HOURS` (12 h) | 2–4 h |
| 6 | `scp` `adapter_model.safetensors` + `adapter_config.json` to `/data/lora_adapters/<niche>/` | < 1 min |
| 7 | Convert PEFT → GGUF **on the LLM host over SSH**: `cd $LLM_HOST_LLAMA_CPP_DIR && python convert_lora_to_gguf.py /data/lora_adapters/<niche> --outfile /data/lora_adapters/<niche>/adapter.gguf` | ~5 min |
| 8 | Restart llama-server with `--lora /data/lora_adapters/<niche>/adapter.gguf` (over SSH), record a `trained_models` row, destroy the instance unless `auto_destroy=false` | < 1 min |

Steps 7 and 8 fail soft: the task still reports success with a warning in the logs. Always check the adapter afterwards (below).

---

## Pre-flight checklist

### 1. Corpus size

```bash
curl -s "http://localhost:8000/training/data/count/motivation?min_upvotes=100"
```

`ready_for_training` flips at 100; aim for 500+ for an adapter that actually learns the niche's voice. The export takes `scraped_captions` rows whose `source_subreddit` is in the niche's `NicheConfig.subreddits`, with non-empty `llm_refined_text`, ≥ 100 upvotes, ≥ 100 characters, and `training_status != 'rejected'`. If the count is unexpectedly low, check that the niche's caption subreddits are actually registered with the scraper (`GET /subreddits/`) and that LLM refinement is keeping up (`GET /dashboard/captions/stats`).

Per-niche SQL, if you prefer:

```bash
docker compose exec -T postgres psql -U captionsuser -d captions -c "
SELECT source_subreddit, COUNT(*) FILTER (WHERE llm_refined_text <> '') AS refined
FROM scraped_captions GROUP BY 1 ORDER BY 2 DESC;"
```

### 2. `VASTAI_API_KEY`

The key rotates. Verify the one the containers actually loaded, not the one in your shell:

```bash
docker compose exec -T api python -c "from config.settings import settings; print(settings.VASTAI_API_KEY[:8])"
curl -s -H "Authorization: Bearer $VASTAI_API_KEY" https://console.vast.ai/api/v0/users/current/ | jq '{email, credit}'
```

If `.env` on the VM is stale, fix it and `docker compose up -d --force-recreate --no-deps api celery-worker-scraper` (a plain restart does not re-read `.env`). If the deploy workflow writes `.env` from GitHub secrets, update the secret too or the next deploy reverts it.

### 3. Billing credit

Keep at least $5 of credit; a slow instance can double the cost, and a run that hits zero credit dies mid-training with no useful error.

### 4. Vast.ai SSH key

The instance is driven over SSH with `/root/.ssh/vastai_key`. `utils/vastai_client.py` generates the key on first use and registers the public half with your Vast.ai account. The shipped `docker-compose.yml` persists it through the `vastai_ssh` volume on the `api` service only; the task itself runs in `celery-worker-scraper`, which does not mount that volume, so a rebuild of that container produces a fresh key (and one more key registered on the account). Add `vastai_ssh:/root/.ssh` to the scraper worker's volumes if you want it stable. Either way, confirm the key exists in the container that will run the task:

```bash
docker compose exec -T celery-worker-scraper ls -la /root/.ssh/
```

Expect `vastai_key` and `vastai_key.pub`. If they are missing the first run creates them; if SSH to the instance still fails, check that the public key shows up under *Account → SSH keys* in the Vast.ai console.

### 5. Host SSH for steps 7–8

The GGUF conversion and the llama-server restart run on the **host** through `utils/llama_server_control.run_host_ssh`, which needs `LLM_HOST_IP`, `LLM_HOST_SSH_USER`, and `LLM_HOST_SSH_KEY_PATH` (preferred) or `LLM_HOST_SSH_PASSWORD` in `.env`. The host must have llama.cpp at `LLM_HOST_LLAMA_CPP_DIR` (default `/opt/llama.cpp`, with `convert_lora_to_gguf.py` and the Python deps installed) and the models at `LLM_HOST_MODELS_DIR` (default `/opt/models`). It must also see the adapters at the same path the containers use — `/data/lora_adapters` by default (`LLM_HOST_LORA_BASE_PATH`); bind-mount or symlink the `lora_adapters` volume there.

```bash
docker compose exec -T api python -c "
from utils.llama_server_control import run_host_ssh
print(run_host_ssh('ls /opt/llama.cpp/convert_lora_to_gguf.py /opt/models && ls /data/lora_adapters'))"
```

A `HostSSHNotConfigured` error here means steps 7–8 will silently warn-and-skip.

### 6. Disk

```bash
df -h /
docker compose exec -T api df -h /data /tmp
```

### 7. Training-script guards

Two past silent failures were fixed inside `training/train_vastai.py`; confirm the guards are still there before spending money:

```bash
grep -n "PYTHONUNBUFFERED" training/train_vastai.py      # progress must reach stdout unbuffered
grep -n "eval_loss" training/train_vastai.py              # must be guarded with "'eval_loss' in eval_results"
```

---

## Trigger

```bash
NICHE=motivation
TASK_ID=$(curl -s -X POST "http://localhost:8000/training/trigger/$NICHE?auto_destroy=false" | jq -r .task_id)
echo "$TASK_ID"
```

`auto_destroy=false` is deliberate. The default (`true`) tears the instance down at the end of step 8 whether or not the adapter is any good.

---

## Monitor

```bash
watch -n 30 "curl -s http://localhost:8000/training/vastai/status/$TASK_ID | jq ."
docker compose logs -f celery-worker-scraper | grep -E 'Step [0-9]/8|Instance ready|Exported|adapter'
```

To watch training itself, pull the instance address from the `Instance ready: <host>:<port>` log line and SSH in from inside the container that holds the key:

```bash
docker compose exec -T celery-worker-scraper ssh -i /root/.ssh/vastai_key -o StrictHostKeyChecking=no -p <ssh_port> root@<ssh_host> \
  "nvidia-smi --query-gpu=memory.used --format=csv; ls -la /root/training/output; tail -50 /root/training/output/training_log.json 2>/dev/null"
```

The training script's stdout is captured by the task and printed to the worker log when step 5 ends; `training_log.json` in `/root/training/output/` records the final metrics and hyper-parameters.

---

## Verify the adapter (do this before destroying anything)

1. Files present:

```bash
curl -s http://localhost:8000/training/adapters/$NICHE | jq '{files: (.files|keys), ready_for_inference}'
```

`ready_for_inference` must be `true` (`adapter.gguf` exists). If only `adapter_model.safetensors` is there, step 7 failed — see the playbook.

2. llama-server is up with the adapter:

```bash
curl -s http://localhost:8000/llama-server/status | jq .
curl -s http://localhost:1234/v1/models | jq .     # on the host
```

3. Generate a handful of captions and read them:

```bash
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:8000/llama-server/generate -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"Write a short motivational caption (40-80 words). Second person.\n\nCaption:\", \"max_tokens\": 160, \"temperature\": 0.85, \"niche\": \"$NICHE\"}" | jq -r .text
  echo ---
done
```

Or trigger a real BG-first cycle (`POST /pipeline/trigger/generate/$NICHE`) and inspect the results in the Generation tab.

You are looking for:
- non-empty output on every call (an adapter that loads but was trained on garbage tends to emit blank or single-word completions);
- the niche's register — a motivation adapter should produce a blunt truth plus a next action, not a generic greeting-card line;
- variety across the five — different openers, not the same first sentence;
- no leaked boilerplate from the OCR corpus (`@handle`, `www.`, `follow for more`, `link in bio`).

If any of those fail, keep the instance and go to the playbook.

---

## Destroy the instance

```bash
INSTANCE_ID=$(curl -s http://localhost:8000/training/vastai/status/$TASK_ID | jq -r .result.instance_id)
docker compose exec -T api vastai destroy instance $INSTANCE_ID
# or: curl -X DELETE -H "Authorization: Bearer $VASTAI_API_KEY" https://console.vast.ai/api/v0/instances/$INSTANCE_ID/
```

Check the Vast.ai console afterwards; a forgotten instance bills by the hour.

---

## Failure playbook

### The task reports success but the adapter generates nothing usable

Seen on the first `motivation` run. Check in this order:

1. **Training crashed after saving nothing, but the script exited 0.** Read the captured stdout in the worker log and `training_log.json` on the instance. Known cause: an unguarded `eval_results['eval_loss']` access when evaluation was disabled (pre-flight #7). If `training_log.json` is missing, the adapter files were never written and step 6 copied stale files.
2. **stdout buffering hid the error** (pre-flight #7).
3. **GGUF conversion failed** (`adapter.gguf` missing). Re-run it on the host, or from the container through the same SSH helper:

   ```bash
   docker compose exec -T api python -c "
   from tasks.training_tasks import _convert_peft_to_gguf; print(_convert_peft_to_gguf('$NICHE'))"
   ```

   Common causes: `LLM_HOST_*` not set, llama.cpp's Python requirements not installed on the host, or the host cannot see `/data/lora_adapters` (pre-flight #5). `training/convert_peft_to_gguf.py --adapter-path /data/lora_adapters/$NICHE --output /data/lora_adapters/$NICHE/adapter.gguf` does the same thing locally if llama.cpp is checked out where it looks (`~/llama.cpp`, `/opt/llama.cpp`, `/workspace/llama.cpp`).
4. **llama-server did not restart with the adapter.** `curl -X POST http://localhost:8000/llama-server/restart -H 'Content-Type: application/json' -d "{\"niche\": \"$NICHE\"}"`, then on the host `systemctl status llama-server` and `tail /var/log/llama-server.log`. The systemd unit (`scripts/llama-server.service`) starts the base model without `--lora`; the pipeline's SSH-driven `start_server(niche=…)` is what adds it.

### Step 3 times out (`did not become ready within 1200s`)

Almost always the SSH key (pre-flight #4). Otherwise the instance is genuinely slow to provision: check it in the Vast.ai console, SSH in by hand, and either wait for the next attempt or destroy it and re-trigger (a different offer will be picked).

### "Not enough training data"

Fewer than 100 exportable captions (pre-flight #1). Wait for scraping, or check the subreddit lists.

### Vast.ai API key rejected / no offers found

Pre-flight #2. `No RTX_4090 instances available under $0.50/hr` means raise `VASTAI_MAX_PRICE_PER_HOUR` in `.env` or try later.

### Out of memory during training

Mistral-Small-24B in 4-bit with rank-32 LoRA needs the full 24 GB; the search filters for ≥ 20 GB VRAM. If the log shows CUDA OOM, the instance likely came with a smaller card than advertised — destroy it and re-trigger.

---

## Lessons log

Append a bullet per run. The point is that the next person does not rediscover the same failure.

- **motivation (first run):** unguarded `eval_loss` read crashed after the adapter step, stdout buffering hid it, the run looked green and generated nothing. Fixed in `train_vastai.py`; keep pre-flight #7.
- **motivation (first run):** SSH key lost on container rebuild → step 3 hang. Fixed by the `vastai_ssh` volume.
- **fitness / cooking / travel:** _fill in after the run._

---

## Related code

- `api/training.py` — `trigger_vastai_training`, `get_vastai_training_status`, adapter listing
- `tasks/training_tasks.py` — `trigger_cloud_training`, `_convert_peft_to_gguf`, `_restart_llama_server_with_lora`
- `training/export_training_data.py` — corpus export and cleaning
- `training/train_vastai.py` — the script that runs on the instance
- `training/convert_peft_to_gguf.py` — standalone converter
- `utils/vastai_client.py` — Vast.ai CLI wrapper (offers, launch, SSH, scp)
- `utils/llama_server_control.py` — host SSH helper, llama-server profiles, `start_server(niche=…)`
- `scripts/llama-server.service` — systemd unit for the host
