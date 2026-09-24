"""
Maintenance tasks for cleanup and database management
"""
from tasks.celery_app import celery_app
from database.db import get_db_context
from database.models import Video
from scrapers.video_downloader import VideoDownloader
from config.settings import settings
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


@celery_app.task
def cleanup_old_videos():
    """
    Delete videos older than configured max age

    Scheduled to run weekly via Celery Beat
    """
    try:
        max_age_days = settings.video_cache_max_age_days
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

        logger.info(f"Cleaning up videos older than {max_age_days} days (before {cutoff_date})")

        with get_db_context() as db:
            # Find old videos
            old_videos = db.query(Video).filter(
                Video.download_date < cutoff_date
            ).all()

            if not old_videos:
                logger.info("No old videos to clean up")
                return {'status': 'success', 'deleted': 0}

            deleted_count = 0
            downloader = VideoDownloader()

            for video in old_videos:
                try:
                    # Delete file from disk
                    if os.path.exists(video.storage_path):
                        downloader.delete_video(video.storage_path)

                    # Delete from database
                    db.delete(video)
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error deleting video {video.id}: {e}")
                    continue

            db.commit()

            logger.info(f"Cleaned up {deleted_count} old videos")

            return {
                'status': 'success',
                'deleted': deleted_count,
                'max_age_days': max_age_days
            }

    except Exception as e:
        logger.error(f"Error in cleanup_old_videos: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def database_maintenance():
    """
    Perform database maintenance tasks

    - Vacuum database
    - Update statistics
    - Remove orphaned records

    Scheduled to run daily via Celery Beat
    """
    try:
        logger.info("Starting database maintenance")

        with get_db_context() as db:
            # Remove videos with missing files
            videos = db.query(Video).all()
            orphaned = []

            for video in videos:
                if not os.path.exists(video.storage_path):
                    orphaned.append(video.id)
                    db.delete(video)

            if orphaned:
                db.commit()
                logger.info(f"Removed {len(orphaned)} orphaned video records")

            # Get database stats
            total_videos = db.query(Video).count()
            total_size = db.query(func.sum(Video.file_size_bytes)).scalar() or 0

        logger.info(
            f"Database maintenance complete. "
            f"Videos: {total_videos}, "
            f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB"
        )

        return {
            'status': 'success',
            'orphaned_removed': len(orphaned) if orphaned else 0,
            'total_videos': total_videos,
            'total_size_gb': total_size / 1024 / 1024 / 1024
        }

    except Exception as e:
        logger.error(f"Error in database_maintenance: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def cleanup_failed_downloads():
    """Remove videos with failed download status"""
    try:
        with get_db_context() as db:
            failed = db.query(Video).filter(
                Video.processing_status == 'failed'
            ).all()

            if not failed:
                return {'status': 'success', 'deleted': 0}

            downloader = VideoDownloader()
            deleted_count = 0

            for video in failed:
                try:
                    if os.path.exists(video.storage_path):
                        downloader.delete_video(video.storage_path)

                    db.delete(video)
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error deleting failed video {video.id}: {e}")

            db.commit()

            logger.info(f"Cleaned up {deleted_count} failed downloads")

            return {'status': 'success', 'deleted': deleted_count}

    except Exception as e:
        logger.error(f"Error in cleanup_failed_downloads: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def cleanup_idle_gpu_models(idle_timeout_minutes: int = 5):
    """
    Unload GPU models that have been idle for too long.

    This frees up VRAM when workers are not actively processing.
    Runs on the GPU queue so it executes on GPU workers.

    Args:
        idle_timeout_minutes: Minutes of inactivity before unloading (default: 5)

    Returns:
        Dict with cleanup results
    """
    try:
        results = {
            'ocr_unloaded': False,
            'llm_unloaded': False,
            'ocr_status': None,
            'llm_status': None,
        }

        # Check and unload OCR model (Qwen2-VL-2B)
        try:
            from scrapers.caption_extractor_qwen2vl import (
                unload_if_idle as unload_ocr_if_idle,
                get_model_status as get_ocr_status
            )
            results['ocr_status'] = get_ocr_status()
            results['ocr_unloaded'] = unload_ocr_if_idle(idle_timeout_minutes)
        except Exception as e:
            logger.warning(f"Error checking OCR model: {e}")

        # Check and unload LLM (Mistral-7B)
        try:
            from scrapers.caption_postprocessor import (
                unload_llm_if_idle,
                get_llm_status
            )
            results['llm_status'] = get_llm_status()
            results['llm_unloaded'] = unload_llm_if_idle(idle_timeout_minutes)
        except Exception as e:
            logger.warning(f"Error checking LLM model: {e}")

        # Log cleanup results
        if results['ocr_unloaded'] or results['llm_unloaded']:
            logger.info(
                f"GPU cleanup: OCR unloaded={results['ocr_unloaded']}, "
                f"LLM unloaded={results['llm_unloaded']}"
            )
        else:
            logger.debug("GPU cleanup: no idle models to unload")

        return {'status': 'success', **results}

    except Exception as e:
        logger.error(f"Error in cleanup_idle_gpu_models: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task
def force_unload_gpu_models():
    """
    Force unload all GPU models immediately.

    Use this before running training or when you need to free GPU memory.

    Returns:
        Dict with unload results
    """
    try:
        results = {
            'ocr_unloaded': False,
            'llm_unloaded': False,
        }

        # Force unload OCR model
        try:
            from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_ocr
            cleanup_ocr()
            results['ocr_unloaded'] = True
            logger.info("Force unloaded OCR model (Qwen2-VL-2B)")
        except Exception as e:
            logger.warning(f"Error unloading OCR model: {e}")

        # Force unload LLM
        try:
            from scrapers.caption_postprocessor import cleanup_llm
            cleanup_llm()
            results['llm_unloaded'] = True
            logger.info("Force unloaded LLM model (Mistral-7B)")
        except Exception as e:
            logger.warning(f"Error unloading LLM model: {e}")

        return {'status': 'success', **results}

    except Exception as e:
        logger.error(f"Error in force_unload_gpu_models: {e}")
        return {'status': 'error', 'error': str(e)}


# GPU-specific unload tasks - each routed to a dedicated queue that only one worker listens to
# This allows the API to unload models on BOTH GPUs by sending tasks to each worker's dedicated queue

@celery_app.task(queue='gpu_status_0')
def unload_gpu_0_models():
    """
    Force unload models on GPU 0 worker.

    Routed to gpu_status_0 queue which only worker 0 listens to.
    GPU 0 runs both OCR (Qwen2-VL) and LLM (Mistral-7B).
    """
    import os
    gpu_id = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
    results = {'status': 'success', 'gpu': gpu_id, 'ocr_unloaded': False, 'llm_unloaded': False}

    # Unload OCR model (Qwen2-VL-2B)
    try:
        from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_ocr
        cleanup_ocr()
        results['ocr_unloaded'] = True
        logger.info(f"Unloaded OCR model on GPU {gpu_id} worker")
    except Exception as e:
        logger.warning(f"Error unloading OCR model on GPU {gpu_id}: {e}")
        results['ocr_error'] = str(e)

    # Unload LLM model (Mistral-7B) - only runs on GPU 0
    try:
        from scrapers.caption_postprocessor import cleanup_llm
        cleanup_llm()
        results['llm_unloaded'] = True
        logger.info(f"Unloaded LLM model on GPU {gpu_id} worker")
    except Exception as e:
        logger.warning(f"Error unloading LLM model on GPU {gpu_id}: {e}")
        results['llm_error'] = str(e)

    return results


@celery_app.task(queue='gpu_status_1')
def unload_gpu_1_models():
    """
    Force unload models on GPU 1 worker.

    Routed to gpu_status_1 queue which only worker 1 listens to.
    GPU 1 only runs OCR (Qwen2-VL) - LLM is disabled due to 8GB VRAM limit.
    """
    import os
    gpu_id = os.environ.get('NVIDIA_VISIBLE_DEVICES', '1')
    results = {'status': 'success', 'gpu': gpu_id, 'ocr_unloaded': False}

    # Unload OCR model (Qwen2-VL-2B)
    try:
        from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_ocr
        cleanup_ocr()
        results['ocr_unloaded'] = True
        logger.info(f"Unloaded OCR model on GPU {gpu_id} worker")
    except Exception as e:
        logger.warning(f"Error unloading OCR model on GPU {gpu_id}: {e}")
        results['ocr_error'] = str(e)

    return results


# Deep clean tasks - aggressively clear GPU memory on specific workers

def _get_nvidia_smi_memory_mb():
    """Get GPU memory usage from nvidia-smi (more accurate than torch.cuda.memory_allocated)."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return 0


@celery_app.task(queue='gpu_status_0')
def deep_clean_gpu_0():
    """
    Deep clean GPU 0 memory - unloads all models and clears CUDA cache.

    Use when GPU shows high memory but no models tracked as loaded.
    Reports nvidia-smi memory (same as GPU Control panel) for accurate comparison.
    """
    import torch
    import gc
    import os
    import time

    gpu_id = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
    results = {
        'status': 'success',
        'gpu': gpu_id,
        'memory_before_mb': 0,
        'memory_after_mb': 0,
        'memory_before_gb': 0,
        'memory_after_gb': 0,
        'models_cleared': [],
    }

    # Get memory before (using nvidia-smi for consistency with GPU Control panel)
    results['memory_before_mb'] = _get_nvidia_smi_memory_mb()
    results['memory_before_gb'] = round(results['memory_before_mb'] / 1024, 2)

    # Unload OCR
    try:
        from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_ocr
        cleanup_ocr()
        results['models_cleared'].append('Qwen2-VL-2B (OCR)')
    except Exception as e:
        logger.debug(f"OCR cleanup: {e}")

    # Unload LLM (GPU 0 only)
    try:
        from scrapers.caption_postprocessor import cleanup_llm
        cleanup_llm()
        results['models_cleared'].append('Mistral-7B (LLM)')
    except Exception as e:
        logger.debug(f"LLM cleanup: {e}")

    # Aggressive cleanup
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        gc.collect()

    # Wait briefly for memory to be released
    time.sleep(1)

    # Get memory after (using nvidia-smi)
    results['memory_after_mb'] = _get_nvidia_smi_memory_mb()
    results['memory_after_gb'] = round(results['memory_after_mb'] / 1024, 2)
    results['memory_freed_mb'] = results['memory_before_mb'] - results['memory_after_mb']
    results['memory_freed_gb'] = round(results['memory_freed_mb'] / 1024, 2)

    logger.info(f"Deep cleaned GPU {gpu_id}: {results['memory_before_mb']}MB -> {results['memory_after_mb']}MB (freed {results['memory_freed_mb']}MB)")

    return results


@celery_app.task(queue='gpu_status_1')
def deep_clean_gpu_1():
    """
    Deep clean GPU 1 memory - unloads all models and clears CUDA cache.

    Use when GPU shows high memory but no models tracked as loaded.
    Reports nvidia-smi memory (same as GPU Control panel) for accurate comparison.
    """
    import torch
    import gc
    import os
    import time

    gpu_id = os.environ.get('NVIDIA_VISIBLE_DEVICES', '1')
    results = {
        'status': 'success',
        'gpu': gpu_id,
        'memory_before_mb': 0,
        'memory_after_mb': 0,
        'memory_before_gb': 0,
        'memory_after_gb': 0,
        'models_cleared': [],
    }

    # Get memory before (using nvidia-smi for consistency with GPU Control panel)
    results['memory_before_mb'] = _get_nvidia_smi_memory_mb()
    results['memory_before_gb'] = round(results['memory_before_mb'] / 1024, 2)

    # Unload OCR (GPU 1 only has OCR, no LLM)
    try:
        from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_ocr
        cleanup_ocr()
        results['models_cleared'].append('Qwen2-VL-2B (OCR)')
    except Exception as e:
        logger.debug(f"OCR cleanup: {e}")

    # Aggressive cleanup
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        gc.collect()

    # Wait briefly for memory to be released
    time.sleep(1)

    # Get memory after (using nvidia-smi)
    results['memory_after_mb'] = _get_nvidia_smi_memory_mb()
    results['memory_after_gb'] = round(results['memory_after_mb'] / 1024, 2)
    results['memory_freed_mb'] = results['memory_before_mb'] - results['memory_after_mb']
    results['memory_freed_gb'] = round(results['memory_freed_mb'] / 1024, 2)

    logger.info(f"Deep cleaned GPU {gpu_id}: {results['memory_before_mb']}MB -> {results['memory_after_mb']}MB (freed {results['memory_freed_mb']}MB)")

    return results


# ============================================================================
# GPU Status Caching Tasks
# ============================================================================
# These tasks run periodically on each GPU worker to cache their status in Redis.
# The API reads from cache instead of querying workers directly, avoiding timeouts
# when workers are busy with long-running tasks.

GPU_STATUS_CACHE_KEY_PREFIX = "gpu:status:cache:"
GPU_STATUS_CACHE_TTL = 120  # Cache expires after 2 minutes (stale data protection)


def _get_worker_gpu_status_for_cache():
    """Get GPU status from the current worker for caching."""
    import subprocess
    from datetime import datetime

    try:
        # Get GPU info via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {"error": "nvidia-smi failed"}

        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return {"error": "No GPU data"}

        # Parse first GPU (worker only sees its assigned GPU)
        parts = [p.strip() for p in lines[0].split(',')]
        if len(parts) < 6:
            return {"error": f"Unexpected nvidia-smi output: {lines[0]}"}

        gpu_data = {
            "name": parts[1],
            "memory_used_mb": int(parts[2]),
            "memory_total_mb": int(parts[3]),
            "memory_percent": round(int(parts[2]) / int(parts[3]) * 100, 1) if int(parts[3]) > 0 else 0,
            "utilization_percent": int(parts[4]) if parts[4].isdigit() else 0,
            "temperature_c": int(parts[5]) if parts[5].isdigit() else 0,
            "loaded_models": [],
            "cached_at": datetime.utcnow().isoformat(),
        }

        # Check for loaded OCR model
        try:
            from scrapers.caption_extractor_qwen2vl import get_model_status
            ocr_status = get_model_status()
            if ocr_status.get("loaded"):
                gpu_data["loaded_models"].append({
                    "name": "Qwen2-VL-2B-Instruct",
                    "type": "ocr",
                    "vram_estimate_gb": 5.0,
                    "idle_seconds": ocr_status.get("idle_seconds", 0)
                })
        except Exception:
            pass

        # Check for loaded LLM model (GPU 0 only)
        try:
            from scrapers.caption_postprocessor import get_llm_status
            llm_status = get_llm_status()
            if llm_status.get("loaded"):
                gpu_data["loaded_models"].append({
                    "name": "Mistral-7B-Instruct",
                    "type": "llm",
                    "vram_estimate_gb": 4.0,
                    "idle_seconds": llm_status.get("idle_seconds", 0)
                })
        except Exception:
            pass

        # Check for loaded trained model
        try:
            from tasks.training_tasks import _loaded_model_id
            from database.db import get_db_context
            from database.models import TrainedModel
            if _loaded_model_id:
                with get_db_context() as db:
                    model = db.query(TrainedModel).filter_by(id=_loaded_model_id).first()
                    if model:
                        gpu_data["loaded_models"].append({
                            "name": model.name,
                            "type": "trained_lora",
                            "vram_estimate_gb": 4.0,
                            "model_id": model.id
                        })
        except Exception:
            pass

        return gpu_data

    except Exception as e:
        return {"error": str(e)}


@celery_app.task(queue='gpu_status_0')
def cache_gpu_0_status():
    """
    Cache GPU 0 status in Redis.

    Runs periodically on GPU 0 worker to update cached status.
    The API reads from this cache instead of querying directly.
    """
    import redis
    import json
    from config.settings import settings

    try:
        status = _get_worker_gpu_status_for_cache()
        status["index"] = 0
        status["worker_gpu"] = 0

        r = redis.from_url(settings.celery_broker_url)
        r.setex(
            f"{GPU_STATUS_CACHE_KEY_PREFIX}0",
            GPU_STATUS_CACHE_TTL,
            json.dumps(status)
        )
        logger.debug(f"Cached GPU 0 status: {status.get('memory_used_mb', '?')}MB used")
        return {"status": "cached", "gpu": 0}
    except Exception as e:
        logger.warning(f"Failed to cache GPU 0 status: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(queue='gpu_status_1')
def cache_gpu_1_status():
    """
    Cache GPU 1 status in Redis.

    Runs periodically on GPU 1 worker to update cached status.
    The API reads from this cache instead of querying directly.
    """
    import redis
    import json
    from config.settings import settings

    try:
        status = _get_worker_gpu_status_for_cache()
        status["index"] = 1
        status["worker_gpu"] = 1

        r = redis.from_url(settings.celery_broker_url)
        r.setex(
            f"{GPU_STATUS_CACHE_KEY_PREFIX}1",
            GPU_STATUS_CACHE_TTL,
            json.dumps(status)
        )
        logger.debug(f"Cached GPU 1 status: {status.get('memory_used_mb', '?')}MB used")
        return {"status": "cached", "gpu": 1}
    except Exception as e:
        logger.warning(f"Failed to cache GPU 1 status: {e}")
        return {"status": "error", "error": str(e)}
