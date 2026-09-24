"""
Training API endpoints

Includes:
- Local LoRA training (legacy)
- Vast.ai cloud training for Mistral-Small-24B
- Training data export
- Adapter management
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from tasks.training_tasks import train_lora_model, export_training_data, trigger_cloud_training
from tasks.celery_app import celery_app
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


class TrainingRequest(BaseModel):
    num_epochs: int = 3


class TrainingStatusResponse(BaseModel):
    task_id: str
    status: str
    stage: Optional[str] = None
    progress: Optional[int] = None
    result: Optional[dict] = None


@router.post("/start")
async def start_training(request: TrainingRequest = TrainingRequest()):
    """
    Start LoRA training on Mistral-7B.

    This is a long-running task (2-6 hours) that:
    1. Exports training data from database
    2. Creates train/validation split
    3. Runs LoRA fine-tuning

    Returns task_id to track progress.
    """
    task = train_lora_model.delay(num_epochs=request.num_epochs)

    return {
        "task_id": task.id,
        "status": "started",
        "message": f"Training started with {request.num_epochs} epochs. This may take 2-6 hours."
    }


@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """Get status of a training task."""
    result = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": result.status
    }

    if result.state == 'PROGRESS':
        response["stage"] = result.info.get('stage', 'unknown')
        response["progress"] = result.info.get('progress', 0)
    elif result.ready():
        response["result"] = result.result

    return response


@router.post("/export-data")
async def export_data():
    """
    Export training data from database without starting training.
    Useful for checking data quality before a training run.
    """
    task = export_training_data.delay()

    return {
        "task_id": task.id,
        "status": "started",
        "message": "Exporting training data from database"
    }


@router.get("/model/status")
async def get_model_status(niche: str):
    """Check if a trained model exists for the given niche."""
    model_path = f"/data/training_models/mistral-{niche}-lora"

    if os.path.exists(model_path):
        files = os.listdir(model_path)
        has_adapter = any(f.endswith('.bin') or f.endswith('.safetensors') for f in files)
        has_config = 'adapter_config.json' in files

        return {
            "niche": niche,
            "model_path": model_path,
            "model_exists": True,
            "has_adapter_weights": has_adapter,
            "has_config": has_config,
            "files": files
        }
    else:
        return {
            "niche": niche,
            "model_path": model_path,
            "model_exists": False,
            "message": f"No trained model found for {niche}. Run /training/trigger/{niche} to train."
        }


@router.get("/data/status")
async def get_data_status():
    """Check training data status."""
    data_dir = "/app/training/data"

    result = {
        "data_exists": False,
        "train_file": None,
        "val_file": None
    }

    if os.path.exists(data_dir):
        train_file = os.path.join(data_dir, "captions_train.jsonl")
        val_file = os.path.join(data_dir, "captions_val.jsonl")

        if os.path.exists(train_file):
            result["train_file"] = {
                "exists": True,
                "size_kb": os.path.getsize(train_file) // 1024
            }
            result["data_exists"] = True

        if os.path.exists(val_file):
            result["val_file"] = {
                "exists": True,
                "size_kb": os.path.getsize(val_file) // 1024
            }

    return result


# ============================================================================
# Vast.ai Cloud Training Endpoints
# ============================================================================

class VastaiTrainingRequest(BaseModel):
    """Request model for triggering Vast.ai training."""
    niche: str
    provider: str = "vastai"


@router.post("/trigger/{niche}")
async def trigger_vastai_training(niche: str, auto_destroy: bool = True):
    """
    Trigger Vast.ai training for a specific niche category.

    This is a long-running task (2-6 hours) that:
    1. Exports training data for the niche
    2. Launches a Vast.ai RTX 4090 instance
    3. Uploads data and runs QLoRA training on Mistral-Small-24B
    4. Downloads the trained adapter
    5. Converts PEFT to GGUF format
    6. Restarts llama-server with the new adapter

    Args:
        niche: Niche category (motivation, fitness, cooking, travel)
        auto_destroy: If False, leave the Vast.ai instance running after
            training completes (or fails) so a human can SSH in and inspect
            before destroying. Default True preserves the original behavior.
            See docs/training_runbook.md for the diagnose-before-destroy flow.

    Returns:
        task_id to track progress
    """
    # Validate niche
    from config.automation_config import get_automation_config
    config = get_automation_config()

    if niche not in config.niches:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown niche: {niche}. Available: {list(config.niches.keys())}"
        )

    # Check if VASTAI_API_KEY is configured
    from config.settings import settings
    if not settings.VASTAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="VASTAI_API_KEY not configured. Set it in .env file."
        )

    task = trigger_cloud_training.delay(niche=niche, provider="vastai", auto_destroy=auto_destroy)

    return {
        "task_id": task.id,
        "status": "started",
        "niche": niche,
        "message": f"Vast.ai training started for {niche}. This may take 2-6 hours.",
        "monitor_url": f"/training/vastai/status/{task.id}"
    }


@router.get("/vastai/status/{task_id}")
async def get_vastai_training_status(task_id: str):
    """
    Get status of a Vast.ai training task.

    Returns detailed progress including:
    - Current step (exporting, launching, training, downloading, etc.)
    - Elapsed time
    - Estimated cost
    """
    result = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready()
    }

    if result.state == 'PROGRESS':
        response["progress"] = result.info
    elif result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)

    return response


@router.get("/adapters")
async def list_trained_adapters():
    """
    List all trained LoRA adapters.

    Returns adapters stored at /data/lora_adapters/{niche}/
    """
    adapters_dir = Path("/data/lora_adapters")

    if not adapters_dir.exists():
        return {"adapters": [], "message": "No adapters directory found"}

    adapters = []
    for niche_dir in adapters_dir.iterdir():
        if niche_dir.is_dir():
            adapter_info = {
                "niche": niche_dir.name,
                "path": str(niche_dir),
                "files": [],
                "has_peft": False,
                "has_gguf": False
            }

            for file in niche_dir.iterdir():
                adapter_info["files"].append(file.name)
                if file.name == "adapter_model.safetensors":
                    adapter_info["has_peft"] = True
                    adapter_info["peft_size_mb"] = file.stat().st_size / (1024 * 1024)
                if file.name == "adapter.gguf":
                    adapter_info["has_gguf"] = True
                    adapter_info["gguf_size_mb"] = file.stat().st_size / (1024 * 1024)

            adapters.append(adapter_info)

    return {
        "adapters": adapters,
        "count": len(adapters)
    }


@router.get("/adapters/{niche}")
async def get_adapter_details(niche: str):
    """
    Get details of a specific adapter.
    """
    adapter_dir = Path(f"/data/lora_adapters/{niche}")

    if not adapter_dir.exists():
        raise HTTPException(status_code=404, detail=f"No adapter found for {niche}")

    files = {}
    for file in adapter_dir.iterdir():
        files[file.name] = {
            "size_mb": file.stat().st_size / (1024 * 1024),
            "modified": file.stat().st_mtime
        }

    # Check for adapter_config.json for LoRA details
    config_file = adapter_dir / "adapter_config.json"
    lora_config = None
    if config_file.exists():
        import json
        with open(config_file) as f:
            lora_config = json.load(f)

    return {
        "niche": niche,
        "path": str(adapter_dir),
        "files": files,
        "lora_config": lora_config,
        "ready_for_inference": "adapter.gguf" in files
    }


@router.get("/data/count/{niche}")
async def get_training_data_count(niche: str, min_upvotes: int = 100):
    """
    Get count of available training captions for a niche.

    Args:
        niche: Niche category
        min_upvotes: Minimum upvote threshold
    """
    from training.export_training_data import get_niche_caption_count

    try:
        count = get_niche_caption_count(niche, min_upvotes=min_upvotes)

        # Get subreddits for context
        from config.automation_config import get_automation_config
        config = get_automation_config()
        niche_config = config.get_niche(niche)

        return {
            "niche": niche,
            "subreddits": niche_config.subreddits if niche_config else [],
            "min_upvotes": min_upvotes,
            "available_captions": count,
            "ready_for_training": count >= 100
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
