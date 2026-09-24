"""
GPU Control API - Manage GPU resources for training

Provides endpoints to:
- Enable/disable extraction processing
- Enable/disable OCR on specific GPUs
- Enable/disable LLM batch processing
- Prepare GPUs for training (unload all models)
- Get comprehensive GPU status
"""
import redis
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from database.db import get_db
from database.models import TrainedModel, TrainingJob
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-manager/gpu-control", tags=["gpu-control"])

# Redis keys for GPU/extraction control
EXTRACTION_ENABLED_KEY = "extraction:enabled"
OCR_GPU0_ENABLED_KEY = "ocr:gpu0:enabled"
OCR_GPU1_ENABLED_KEY = "ocr:gpu1:enabled"
LLM_BATCH_ENABLED_KEY = "llm:batch:enabled"


def get_redis_client():
    return redis.from_url(settings.celery_broker_url)


# ============================================================================
# Pydantic Models
# ============================================================================

class GPUStatus(BaseModel):
    gpu_id: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    ocr_enabled: bool
    models_loaded: List[str]
    available_for_training: bool


class ExtractionStatus(BaseModel):
    extraction_enabled: bool
    ocr_gpu0_enabled: bool
    ocr_gpu1_enabled: bool
    llm_batch_enabled: bool
    queue_size: int
    active_extractions: int


class GPUControlStatus(BaseModel):
    extraction: ExtractionStatus
    gpus: List[GPUStatus]
    active_training_jobs: List[Dict[str, Any]]
    recommendations: List[str]


# ============================================================================
# Status Endpoints
# ============================================================================

@router.get("/status")
async def get_gpu_control_status(db: Session = Depends(get_db)):
    """
    Get comprehensive status of GPU control settings.

    Returns extraction status, GPU status, and training recommendations.
    """
    try:
        r = get_redis_client()

        # Get extraction control flags
        extraction_enabled = r.get(EXTRACTION_ENABLED_KEY) != b"0"
        ocr_gpu0_enabled = r.get(OCR_GPU0_ENABLED_KEY) != b"0"
        ocr_gpu1_enabled = r.get(OCR_GPU1_ENABLED_KEY) != b"0"
        llm_batch_enabled = r.get(LLM_BATCH_ENABLED_KEY) != b"0"

        # Get extraction queue size
        queue_size = r.llen("extraction:pending") or 0

        # Get active training jobs
        active_jobs = db.query(TrainingJob).filter(
            TrainingJob.status.in_(["queued", "preparing_data", "training", "evaluating", "saving"])
        ).all()

        active_training = [
            {
                "id": j.id,
                "job_name": j.job_name,
                "status": j.status,
                "target_gpu": getattr(j, 'target_gpu', 0),
                "progress_percent": j.progress_percent
            }
            for j in active_jobs
        ]

        # Get loaded models
        loaded_models = db.query(TrainedModel).filter(TrainedModel.is_loaded == True).all()

        # Build recommendations
        recommendations = []

        if active_training:
            recommendations.append("Training in progress - consider disabling extraction to free GPU memory")

        if extraction_enabled and not (ocr_gpu0_enabled or ocr_gpu1_enabled):
            recommendations.append("Extraction enabled but all OCR workers disabled - extraction won't process")

        if loaded_models:
            model_names = [m.name for m in loaded_models]
            recommendations.append(f"Loaded models ({', '.join(model_names)}) using GPU memory - unload before training")

        if queue_size > 100:
            recommendations.append(f"Large extraction queue ({queue_size} items) - consider enabling more GPU workers")

        return {
            "extraction": {
                "extraction_enabled": extraction_enabled,
                "ocr_gpu0_enabled": ocr_gpu0_enabled,
                "ocr_gpu1_enabled": ocr_gpu1_enabled,
                "llm_batch_enabled": llm_batch_enabled,
                "queue_size": queue_size,
                "active_extractions": 0  # Would need to check Celery for this
            },
            "loaded_models": [
                {"id": m.id, "name": m.name, "target_gpu": getattr(m, 'target_gpu', 0)}
                for m in loaded_models
            ],
            "active_training_jobs": active_training,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error getting GPU control status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpus")
async def get_all_gpus_status():
    """
    Get detailed status of all GPUs from Redis cache.

    GPU status is cached periodically by workers (every 30 seconds).
    This avoids timeouts when workers are busy with long-running tasks.
    """
    import json
    from datetime import datetime

    GPU_STATUS_CACHE_KEY_PREFIX = "gpu:status:cache:"

    try:
        r = get_redis_client()
        ocr_gpu0_enabled = r.get(OCR_GPU0_ENABLED_KEY) != b"0"
        ocr_gpu1_enabled = r.get(OCR_GPU1_ENABLED_KEY) != b"0"

        gpus = []
        errors = []
        cache_info = {
            "source": "redis_cache",
            "gpu_0_cached": False,
            "gpu_1_cached": False,
        }

        # Read GPU 0 status from cache
        try:
            cached_0 = r.get(f"{GPU_STATUS_CACHE_KEY_PREFIX}0")
            if cached_0:
                result0 = json.loads(cached_0)
                result0["ocr_enabled"] = ocr_gpu0_enabled
                gpus.append(result0)
                cache_info["gpu_0_cached"] = True
                cache_info["gpu_0_cached_at"] = result0.get("cached_at")
            else:
                errors.append("GPU 0: no cached status (worker may be starting)")
        except Exception as e:
            logger.warning(f"Could not read GPU 0 cache: {e}")
            errors.append(f"GPU 0: cache read error - {str(e)}")

        # Read GPU 1 status from cache
        try:
            cached_1 = r.get(f"{GPU_STATUS_CACHE_KEY_PREFIX}1")
            if cached_1:
                result1 = json.loads(cached_1)
                result1["ocr_enabled"] = ocr_gpu1_enabled
                gpus.append(result1)
                cache_info["gpu_1_cached"] = True
                cache_info["gpu_1_cached_at"] = result1.get("cached_at")
            else:
                errors.append("GPU 1: no cached status (worker may be starting)")
        except Exception as e:
            logger.warning(f"Could not read GPU 1 cache: {e}")
            errors.append(f"GPU 1: cache read error - {str(e)}")

        # Sort by GPU index
        gpus.sort(key=lambda x: x.get("index", 0))

        result = {
            "gpus": gpus,
            "total_memory_used_mb": sum(g.get("memory_used_mb", 0) for g in gpus),
            "total_memory_mb": sum(g.get("memory_total_mb", 0) for g in gpus),
            "cache_info": cache_info,
        }

        if errors:
            result["errors"] = errors
            if not gpus:
                result["status"] = "no_cached_data"
                result["message"] = "No cached GPU status available. Workers may be starting up."

        return result

    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gpus/refresh")
async def force_refresh_gpu_status(timeout_seconds: int = 10):
    """
    Force refresh GPU status by triggering cache update tasks on both workers.

    Triggers the cache update tasks and waits for them to complete,
    then returns the fresh data. Falls back to cached data if workers
    don't respond within the timeout.

    Args:
        timeout_seconds: Max time to wait for workers (default 10s)
    """
    import json
    from datetime import datetime

    GPU_STATUS_CACHE_KEY_PREFIX = "gpu:status:cache:"

    try:
        from tasks.maintenance_tasks import cache_gpu_0_status, cache_gpu_1_status

        # Trigger cache updates on both GPU workers
        task0 = cache_gpu_0_status.delay()
        task1 = cache_gpu_1_status.delay()

        results = {
            "gpu_0": {"status": "pending"},
            "gpu_1": {"status": "pending"},
        }

        # Wait for both tasks with timeout
        try:
            result0 = task0.get(timeout=timeout_seconds)
            results["gpu_0"] = {"status": "success", "result": result0}
        except Exception as e:
            results["gpu_0"] = {"status": "timeout", "error": str(e)}

        try:
            result1 = task1.get(timeout=timeout_seconds)
            results["gpu_1"] = {"status": "success", "result": result1}
        except Exception as e:
            results["gpu_1"] = {"status": "timeout", "error": str(e)}

        # Now read the freshly updated cache
        r = get_redis_client()
        ocr_gpu0_enabled = r.get(OCR_GPU0_ENABLED_KEY) != b"0"
        ocr_gpu1_enabled = r.get(OCR_GPU1_ENABLED_KEY) != b"0"

        gpus = []
        cache_info = {
            "source": "force_refresh",
            "gpu_0_cached": False,
            "gpu_1_cached": False,
            "refresh_results": results,
        }

        # Read GPU 0 from cache
        cached_0 = r.get(f"{GPU_STATUS_CACHE_KEY_PREFIX}0")
        if cached_0:
            result0 = json.loads(cached_0)
            result0["ocr_enabled"] = ocr_gpu0_enabled
            gpus.append(result0)
            cache_info["gpu_0_cached"] = True
            cache_info["gpu_0_cached_at"] = result0.get("cached_at")

        # Read GPU 1 from cache
        cached_1 = r.get(f"{GPU_STATUS_CACHE_KEY_PREFIX}1")
        if cached_1:
            result1 = json.loads(cached_1)
            result1["ocr_enabled"] = ocr_gpu1_enabled
            gpus.append(result1)
            cache_info["gpu_1_cached"] = True
            cache_info["gpu_1_cached_at"] = result1.get("cached_at")

        gpus.sort(key=lambda x: x.get("index", 0))

        return {
            "gpus": gpus,
            "total_memory_used_mb": sum(g.get("memory_used_mb", 0) for g in gpus),
            "total_memory_mb": sum(g.get("memory_total_mb", 0) for g in gpus),
            "cache_info": cache_info,
        }

    except Exception as e:
        logger.error(f"Error force refreshing GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Extraction Control
# ============================================================================

@router.post("/extraction/enable")
async def enable_extraction():
    """Enable extraction processing (GPU workers will process extraction queue)."""
    r = get_redis_client()
    r.set(EXTRACTION_ENABLED_KEY, "1")
    logger.info("Extraction processing enabled")
    return {"extraction_enabled": True, "message": "Extraction processing enabled"}


@router.post("/extraction/disable")
async def disable_extraction(unload_models: bool = True):
    """
    Disable extraction processing and optionally unload GPU models.

    GPU workers will stop picking up new extraction tasks.
    Running tasks will complete.

    Args:
        unload_models: If True (default), unload OCR models from both GPUs to free memory
    """
    r = get_redis_client()
    r.set(EXTRACTION_ENABLED_KEY, "0")
    logger.info("Extraction processing disabled")

    unload_results = {}
    if unload_models:
        try:
            from tasks.maintenance_tasks import unload_gpu_0_models, unload_gpu_1_models

            # Trigger model unload on both GPU workers
            task0 = unload_gpu_0_models.delay()
            task1 = unload_gpu_1_models.delay()

            # Wait briefly for results (non-blocking if they take too long)
            try:
                result0 = task0.get(timeout=10)
                unload_results['gpu_0'] = result0
            except Exception as e:
                unload_results['gpu_0'] = {'status': 'pending', 'note': str(e)}

            try:
                result1 = task1.get(timeout=10)
                unload_results['gpu_1'] = result1
            except Exception as e:
                unload_results['gpu_1'] = {'status': 'pending', 'note': str(e)}

            logger.info(f"Model unload results: {unload_results}")
        except Exception as e:
            logger.warning(f"Could not trigger model unload: {e}")
            unload_results['error'] = str(e)

    return {
        "extraction_enabled": False,
        "message": "Extraction processing disabled. Running tasks will complete.",
        "models_unloaded": unload_results if unload_models else "skipped"
    }


# ============================================================================
# OCR Control (Per GPU)
# ============================================================================

@router.post("/ocr/{gpu_id}/enable")
async def enable_ocr(gpu_id: int):
    """Enable OCR processing on a specific GPU."""
    if gpu_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="GPU ID must be 0 or 1")

    r = get_redis_client()
    key = OCR_GPU0_ENABLED_KEY if gpu_id == 0 else OCR_GPU1_ENABLED_KEY
    r.set(key, "1")
    logger.info(f"OCR enabled on GPU {gpu_id}")
    return {"gpu_id": gpu_id, "ocr_enabled": True, "message": f"OCR enabled on GPU {gpu_id}"}


@router.post("/ocr/{gpu_id}/disable")
async def disable_ocr(gpu_id: int):
    """
    Disable OCR processing on a specific GPU.

    This frees the GPU for training by preventing new OCR tasks from being assigned.
    """
    if gpu_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="GPU ID must be 0 or 1")

    r = get_redis_client()
    key = OCR_GPU0_ENABLED_KEY if gpu_id == 0 else OCR_GPU1_ENABLED_KEY
    r.set(key, "0")
    logger.info(f"OCR disabled on GPU {gpu_id}")
    return {"gpu_id": gpu_id, "ocr_enabled": False, "message": f"OCR disabled on GPU {gpu_id}"}


# ============================================================================
# LLM Batch Control
# ============================================================================

@router.post("/llm/enable")
async def enable_llm_batch():
    """Enable LLM batch processing (refines OCR output with Mistral-7B)."""
    r = get_redis_client()
    r.set(LLM_BATCH_ENABLED_KEY, "1")
    logger.info("LLM batch processing enabled")
    return {"llm_enabled": True, "message": "LLM batch processing enabled"}


@router.post("/llm/disable")
async def disable_llm_batch():
    """
    Disable LLM batch processing.

    OCR extraction will continue but LLM refinement will be skipped.
    Frees ~3-4GB GPU memory on GPU 0.
    """
    r = get_redis_client()
    r.set(LLM_BATCH_ENABLED_KEY, "0")
    logger.info("LLM batch processing disabled")
    return {"llm_enabled": False, "message": "LLM batch processing disabled"}


# ============================================================================
# GPU Preparation for Training
# ============================================================================

def _get_model_loaded_gpu(model_id: int) -> int:
    """
    Get which GPU a model is loaded on from Redis tracking.

    Returns GPU index (0 or 1), or None if not tracked.
    """
    try:
        r = get_redis_client()
        gpu = r.get(f"model:{model_id}:loaded_on_gpu")
        if gpu:
            return int(gpu.decode('utf-8'))
    except Exception as e:
        logger.debug(f"Could not get model GPU from Redis: {e}")
    return None


@router.post("/{gpu_id}/prepare-for-training")
async def prepare_gpu_for_training(gpu_id: int, db: Session = Depends(get_db)):
    """
    Prepare a GPU for training by:
    1. Disabling OCR on that GPU
    2. Unloading any loaded models (queues GPU-specific unload tasks)
    3. Clearing GPU memory cache

    Use this before starting a training job to maximize available memory.
    """
    if gpu_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="GPU ID must be 0 or 1")

    try:
        r = get_redis_client()
        actions_taken = []

        # 1. Disable OCR on this GPU
        key = OCR_GPU0_ENABLED_KEY if gpu_id == 0 else OCR_GPU1_ENABLED_KEY
        r.set(key, "0")
        actions_taken.append(f"Disabled OCR on GPU {gpu_id}")

        # 2. If GPU 0, also disable LLM (since LLM only runs on GPU 0)
        if gpu_id == 0:
            r.set(LLM_BATCH_ENABLED_KEY, "0")
            actions_taken.append("Disabled LLM batch processing")

        # 3. Unload any loaded models - queue GPU-specific unload tasks
        loaded_models = db.query(TrainedModel).filter(
            TrainedModel.is_loaded == True
        ).all()

        for model in loaded_models:
            # Check which GPU the model is on (from Redis tracking)
            model_gpu = _get_model_loaded_gpu(model.id)
            if model_gpu is None:
                model_gpu = getattr(model, 'target_gpu', 0)

            if model_gpu == gpu_id:
                # Queue unload task to the correct GPU worker
                try:
                    if gpu_id == 0:
                        from tasks.training_tasks import unload_model_task_gpu_0
                        unload_model_task_gpu_0.delay(model.id)
                    else:
                        from tasks.training_tasks import unload_model_task_gpu_1
                        unload_model_task_gpu_1.delay(model.id)
                    actions_taken.append(f"Queued unload for model '{model.name}' on GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Could not queue unload for model {model.id}: {e}")
                    # Fallback: update DB directly
                    model.is_loaded = False
                    model.status = "ready"
                    actions_taken.append(f"Marked model '{model.name}' for unload (DB only)")

        db.commit()

        # 4. Queue deep clean for this GPU
        try:
            if gpu_id == 0:
                from tasks.maintenance_tasks import deep_clean_gpu_0
                deep_clean_gpu_0.delay()
            else:
                from tasks.maintenance_tasks import deep_clean_gpu_1
                deep_clean_gpu_1.delay()
            actions_taken.append(f"Queued deep clean for GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Could not queue deep clean: {e}")

        logger.info(f"Prepared GPU {gpu_id} for training: {actions_taken}")

        return {
            "gpu_id": gpu_id,
            "status": "prepared",
            "actions_taken": actions_taken,
            "message": f"GPU {gpu_id} prepared for training. Wait a few seconds for memory to clear."
        }

    except Exception as e:
        logger.error(f"Error preparing GPU {gpu_id} for training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prepare-all-for-training")
async def prepare_all_gpus_for_training(db: Session = Depends(get_db)):
    """
    Prepare all GPUs for training by:
    1. Disabling all extraction
    2. Disabling all OCR
    3. Disabling LLM batch
    4. Unloading all models (queues GPU-specific unload tasks)
    5. Clearing GPU memory

    Use this for maximum GPU memory availability.
    """
    try:
        r = get_redis_client()
        actions_taken = []

        # Disable everything
        r.set(EXTRACTION_ENABLED_KEY, "0")
        actions_taken.append("Disabled extraction processing")

        r.set(OCR_GPU0_ENABLED_KEY, "0")
        r.set(OCR_GPU1_ENABLED_KEY, "0")
        actions_taken.append("Disabled OCR on all GPUs")

        r.set(LLM_BATCH_ENABLED_KEY, "0")
        actions_taken.append("Disabled LLM batch processing")

        # Unload all models - queue GPU-specific unload tasks
        loaded_models = db.query(TrainedModel).filter(
            TrainedModel.is_loaded == True
        ).all()

        for model in loaded_models:
            # Check which GPU the model is on (from Redis tracking)
            model_gpu = _get_model_loaded_gpu(model.id)
            if model_gpu is None:
                model_gpu = getattr(model, 'target_gpu', 0)

            try:
                if model_gpu == 0:
                    from tasks.training_tasks import unload_model_task_gpu_0
                    unload_model_task_gpu_0.delay(model.id)
                else:
                    from tasks.training_tasks import unload_model_task_gpu_1
                    unload_model_task_gpu_1.delay(model.id)
                actions_taken.append(f"Queued unload for model '{model.name}' on GPU {model_gpu}")
            except Exception as e:
                logger.warning(f"Could not queue unload for model {model.id}: {e}")
                # Fallback: update DB directly
                model.is_loaded = False
                model.status = "ready"
                # Clear Redis tracking
                try:
                    r.delete(f"model:{model.id}:loaded_on_gpu")
                except:
                    pass
                actions_taken.append(f"Marked model '{model.name}' for unload (DB only)")

        db.commit()

        # Queue deep clean for both GPUs
        try:
            from tasks.maintenance_tasks import deep_clean_gpu_0, deep_clean_gpu_1
            deep_clean_gpu_0.delay()
            deep_clean_gpu_1.delay()
            actions_taken.append("Queued deep clean for all GPUs")
        except Exception as e:
            logger.warning(f"Could not queue deep clean: {e}")

        logger.info(f"Prepared all GPUs for training: {actions_taken}")

        return {
            "status": "prepared",
            "actions_taken": actions_taken,
            "message": "All GPUs prepared for training. Wait a few seconds for memory to clear."
        }

    except Exception as e:
        logger.error(f"Error preparing GPUs for training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Restore After Training
# ============================================================================

@router.post("/restore-extraction")
async def restore_extraction_after_training():
    """
    Restore normal extraction processing after training completes.

    Re-enables:
    - Extraction processing
    - OCR on both GPUs
    - LLM batch processing
    """
    r = get_redis_client()

    r.set(EXTRACTION_ENABLED_KEY, "1")
    r.set(OCR_GPU0_ENABLED_KEY, "1")
    r.set(OCR_GPU1_ENABLED_KEY, "1")
    r.set(LLM_BATCH_ENABLED_KEY, "1")

    logger.info("Restored extraction processing after training")

    return {
        "status": "restored",
        "extraction_enabled": True,
        "ocr_gpu0_enabled": True,
        "ocr_gpu1_enabled": True,
        "llm_enabled": True,
        "message": "Extraction processing restored. GPU workers will resume processing."
    }


# ============================================================================
# Check Training Readiness
# ============================================================================

@router.get("/{gpu_id}/training-readiness")
async def check_training_readiness(gpu_id: int, db: Session = Depends(get_db)):
    """
    Check if a GPU is ready for training.

    Returns status and any blockers that need to be resolved.
    """
    if gpu_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="GPU ID must be 0 or 1")

    try:
        r = get_redis_client()
        blockers = []
        warnings = []

        # Check OCR status
        key = OCR_GPU0_ENABLED_KEY if gpu_id == 0 else OCR_GPU1_ENABLED_KEY
        if r.get(key) != b"0":
            blockers.append(f"OCR is still enabled on GPU {gpu_id}")

        # Check LLM status (only matters for GPU 0)
        if gpu_id == 0 and r.get(LLM_BATCH_ENABLED_KEY) != b"0":
            blockers.append("LLM batch processing is still enabled (GPU 0)")

        # Check for loaded models on this GPU
        loaded_models = db.query(TrainedModel).filter(
            TrainedModel.is_loaded == True
        ).all()

        for model in loaded_models:
            model_gpu = getattr(model, 'target_gpu', 0)
            if model_gpu == gpu_id:
                blockers.append(f"Model '{model.name}' is loaded on GPU {gpu_id}")

        # Check for active training on this GPU
        active_jobs = db.query(TrainingJob).filter(
            TrainingJob.status.in_(["queued", "preparing_data", "training", "evaluating", "saving"])
        ).all()

        for job in active_jobs:
            job_gpu = getattr(job, 'target_gpu', 0)
            if job_gpu == gpu_id:
                blockers.append(f"Training job '{job.job_name}' is already running on GPU {gpu_id}")

        # Check extraction queue
        queue_size = r.llen("extraction:pending") or 0
        if queue_size > 0 and r.get(EXTRACTION_ENABLED_KEY) != b"0":
            warnings.append(f"Extraction queue has {queue_size} items - they will wait until extraction is re-enabled")

        is_ready = len(blockers) == 0

        return {
            "gpu_id": gpu_id,
            "is_ready": is_ready,
            "blockers": blockers,
            "warnings": warnings,
            "recommendation": "GPU is ready for training" if is_ready else "Use /prepare-for-training to resolve blockers"
        }

    except Exception as e:
        logger.error(f"Error checking training readiness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Force Unload Models
# ============================================================================

@router.post("/force-unload-all")
async def force_unload_all_models(db: Session = Depends(get_db)):
    """
    Force unload ALL models from ALL GPUs immediately.

    This is more aggressive than prepare-for-training:
    - Unloads OCR models on both GPUs
    - Unloads LLM model on GPU 0
    - Unloads trained models via GPU-specific tasks
    - Clears GPU memory cache

    Does NOT disable extraction (use /extraction/disable for that).
    """
    try:
        from tasks.maintenance_tasks import unload_gpu_0_models, unload_gpu_1_models
        from tasks.training_tasks import unload_model_task_gpu_0, unload_model_task_gpu_1

        r = get_redis_client()
        results = {
            "gpu_0": None,
            "gpu_1": None,
            "trained_models_unloaded": [],
            "deep_clean_queued": False,
        }

        # Unload OCR/LLM models on GPU 0
        try:
            task0 = unload_gpu_0_models.delay()
            result0 = task0.get(timeout=15)
            results["gpu_0"] = result0
        except Exception as e:
            results["gpu_0"] = {"error": str(e)}

        # Unload OCR models on GPU 1
        try:
            task1 = unload_gpu_1_models.delay()
            result1 = task1.get(timeout=15)
            results["gpu_1"] = result1
        except Exception as e:
            results["gpu_1"] = {"error": str(e)}

        # Unload all trained models via GPU-specific tasks
        loaded_models = db.query(TrainedModel).filter(
            TrainedModel.is_loaded == True
        ).all()

        for model in loaded_models:
            # Check which GPU the model is on
            model_gpu = _get_model_loaded_gpu(model.id)
            if model_gpu is None:
                model_gpu = getattr(model, 'target_gpu', 0)

            try:
                if model_gpu == 0:
                    unload_model_task_gpu_0.delay(model.id)
                else:
                    unload_model_task_gpu_1.delay(model.id)
                results["trained_models_unloaded"].append({
                    "model": model.name,
                    "gpu": model_gpu
                })
            except Exception as e:
                logger.warning(f"Could not queue unload for model {model.id}: {e}")
                # Fallback: update DB directly
                model.is_loaded = False
                model.status = "ready"
                try:
                    r.delete(f"model:{model.id}:loaded_on_gpu")
                except:
                    pass

        db.commit()

        # Queue deep clean for both GPUs
        try:
            from tasks.maintenance_tasks import deep_clean_gpu_0, deep_clean_gpu_1
            deep_clean_gpu_0.delay()
            deep_clean_gpu_1.delay()
            results["deep_clean_queued"] = True
        except Exception as e:
            logger.warning(f"Could not queue deep clean: {e}")

        logger.info(f"Force unloaded all models: {results}")

        return {
            "status": "success",
            "message": "All GPU models force unloaded",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error force unloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep-clean-all")
async def deep_clean_all_gpus():
    """
    Deep clean ALL GPU memory on both workers.

    Aggressively clears memory by:
    - Unloading all models (even if not tracked as loaded)
    - Forcing garbage collection
    - Emptying CUDA cache
    - Resetting memory stats

    Returns before/after memory usage for both GPUs.
    """
    try:
        from tasks.maintenance_tasks import deep_clean_gpu_0, deep_clean_gpu_1

        results = {
            "gpu_0": None,
            "gpu_1": None,
            "total_freed_gb": 0,
        }

        # Deep clean GPU 0
        try:
            task0 = deep_clean_gpu_0.delay()
            result0 = task0.get(timeout=30)
            results["gpu_0"] = result0
            results["total_freed_gb"] += result0.get("memory_freed_gb", 0)
        except Exception as e:
            results["gpu_0"] = {"error": str(e)}

        # Deep clean GPU 1
        try:
            task1 = deep_clean_gpu_1.delay()
            result1 = task1.get(timeout=30)
            results["gpu_1"] = result1
            results["total_freed_gb"] += result1.get("memory_freed_gb", 0)
        except Exception as e:
            results["gpu_1"] = {"error": str(e)}

        logger.info(f"Deep cleaned all GPUs: freed {results['total_freed_gb']:.2f}GB total")

        return {
            "status": "success",
            "message": f"Deep cleaned all GPUs (freed {results['total_freed_gb']:.2f}GB)",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error deep cleaning GPUs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Queue Purge (Emergency Training Recovery)
# ============================================================================

@router.post("/purge-gpu-queue")
async def purge_gpu_queue():
    """
    Selectively revoke extraction tasks from GPU queues.

    Use this when:
    - Training is stuck waiting behind extraction tasks
    - OCR was disabled but tasks were already dispatched
    - Need to immediately free GPU for training

    This will:
    1. Revoke pending extraction tasks (preserves training/generation tasks)
    2. Revoke any active extraction tasks
    3. Clear the processing markers in Redis

    WARNING: This will lose any in-progress extraction work.
    """
    try:
        from tasks.celery_app import celery_app

        r = get_redis_client()
        results = {
            "reserved_extraction_revoked": 0,
            "active_extraction_revoked": 0,
            "processing_markers_cleared": 0,
            "preserved_tasks": [],
        }

        # 1. Get inspection of all tasks (reserved = queued, active = running)
        i = celery_app.control.inspect()

        # 2. Revoke RESERVED (queued) extraction tasks only
        reserved = i.reserved() or {}
        for worker, tasks in reserved.items():
            for task in tasks:
                task_name = task.get('name', '')
                task_id = task.get('id')
                # Only revoke extraction-related tasks
                if any(keyword in task_name.lower() for keyword in ['extract', 'ocr', 'llm_refine', 'watermark']):
                    if task_id:
                        celery_app.control.revoke(task_id)
                        results["reserved_extraction_revoked"] += 1
                        logger.debug(f"Revoked reserved task: {task_name} ({task_id})")
                else:
                    # Preserve training, generation, composition tasks
                    results["preserved_tasks"].append(task_name)

        # 3. Revoke ACTIVE (running) extraction tasks
        active = i.active() or {}
        for worker, tasks in active.items():
            for task in tasks:
                task_name = task.get('name', '')
                task_id = task.get('id')
                if any(keyword in task_name.lower() for keyword in ['extract', 'ocr', 'watermark']):
                    if task_id:
                        celery_app.control.revoke(task_id, terminate=True)
                        results["active_extraction_revoked"] += 1
                        logger.debug(f"Revoked active task: {task_name} ({task_id})")

        # 4. Clear processing markers from Redis extraction queue
        processing_key = "captions:extraction_processing"
        processing_count = r.hlen(processing_key) or 0
        r.delete(processing_key)
        results["processing_markers_cleared"] = processing_count

        # 5. Also clear the Redis extraction queue itself to prevent re-dispatch
        extraction_queue_key = "captions:extraction_queue"
        queue_length = r.llen(extraction_queue_key) or 0
        r.delete(extraction_queue_key)
        results["extraction_queue_cleared"] = queue_length

        # 6. CRITICAL: Clear the Celery broker queue for 'gpu'
        # The broker stores tasks in Redis keys named after the queue
        # This clears tasks that were dispatched but not yet picked up by workers
        # WARNING: This will also clear training tasks, so we re-queue them after
        gpu_broker_queue = r.llen("gpu") or 0
        if gpu_broker_queue > 0:
            r.delete("gpu")
            results["celery_gpu_queue_cleared"] = gpu_broker_queue
            logger.info(f"Cleared {gpu_broker_queue} tasks from Celery gpu broker queue")

            # 7. Re-queue any active training jobs
            from database.db import get_db_context
            from database.models import TrainingJob
            with get_db_context() as db:
                active_training = db.query(TrainingJob).filter(
                    TrainingJob.status.in_(["queued", "preparing_data", "training"])
                ).all()

                for job in active_training:
                    from tasks.training_tasks import run_training_job
                    result = run_training_job.apply_async(args=[job.id], queue="gpu")
                    job.celery_task_id = result.id
                    db.commit()
                    results["training_requeued"] = results.get("training_requeued", []) + [job.id]
                    logger.info(f"Re-queued training job {job.id}")

        logger.warning(f"Selectively purged extraction tasks: {results}")

        return {
            "status": "success",
            "message": f"Revoked {results['reserved_extraction_revoked']} queued + {results['active_extraction_revoked']} active extraction tasks",
            "results": results,
            "next_steps": [
                "Wait 10-30 seconds for workers to finish",
                "Training/generation tasks are preserved and will run next",
                "Monitor at /training-manager/jobs/{job_id}"
            ]
        }

    except Exception as e:
        logger.error(f"Error purging extraction tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
