"""
Celery tasks for Training Management System

Includes:
- Training job execution with progress reporting
- Model loading/unloading
- Caption generation
- GPU memory management
"""
import os
import time
import traceback
import logging
from datetime import datetime
from celery import current_task
from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Global model cache (for loaded models)
_loaded_model = None
_loaded_model_id = None
_loaded_tokenizer = None


def cleanup_generation_model():
    """
    Cleanup the loaded generation model to free GPU memory.

    This should be called after generation jobs complete to prevent
    memory leaks between pipeline phases.
    """
    global _loaded_model, _loaded_model_id, _loaded_tokenizer
    import gc
    import torch

    if _loaded_model is not None:
        try:
            # Move model to CPU first to release VRAM
            _loaded_model.to('cpu')
        except Exception as e:
            logger.debug(f"Could not move generation model to CPU: {e}")
        del _loaded_model
        _loaded_model = None

    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None

    _loaded_model_id = None

    # Force garbage collection
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Generation model unloaded and GPU memory cleared")


# ============================================================================
# Training Job Tasks
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.run_training_job', time_limit=43200, soft_time_limit=42000)
def run_training_job(self, job_id: int):
    """
    Run a complete training job with progress reporting.

    This task:
    1. Prepares training data from selected subreddits
    2. Runs LoRA fine-tuning with live progress updates
    3. Saves the trained model
    4. Creates a TrainedModel record

    Args:
        job_id: TrainingJob database ID
    """
    from database.db import get_db_context
    from database.models import TrainingJob, TrainedModel, ScrapedCaption
    import torch

    start_time = time.time()

    with get_db_context() as db:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Training job {job_id} not found")
            return {"status": "error", "message": f"Job {job_id} not found"}

        # Guard against duplicate execution (e.g., from Redis visibility timeout redelivery)
        # If job already completed or has results, skip re-execution
        if job.status == "completed" or (job.final_train_loss is not None and job.final_train_loss > 0):
            logger.warning(f"Training job {job_id} already completed (status={job.status}, final_train_loss={job.final_train_loss}). Skipping duplicate execution.")
            return {"status": "skipped", "message": f"Job {job_id} already completed", "reason": "duplicate_execution"}

        if job.status == "cancelled":
            logger.info(f"Training job {job_id} was cancelled. Skipping execution.")
            return {"status": "skipped", "message": f"Job {job_id} was cancelled"}

        try:
            # Get target GPU from job configuration (used for queue routing only)
            target_gpu = getattr(job, 'target_gpu', 0)

            # IMPORTANT: Inside Docker containers, the GPU routing is handled by docker-compose
            # device_ids mapping. Each worker container only sees its assigned GPU as GPU 0.
            # So we always use CUDA_VISIBLE_DEVICES=0, regardless of target_gpu.
            # The target_gpu value is used for Celery queue routing (gpu_training_0 vs gpu_training_1)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            logger.info(f"Set CUDA_VISIBLE_DEVICES=0 for training job {job_id} (target_gpu={target_gpu} used for queue routing)")

            # Update status to preparing
            job.status = "preparing_data"
            job.started_at = datetime.utcnow()
            db.commit()

            logger.info(f"Starting training job {job_id}: {job.job_name} on GPU {target_gpu}")

            # ================================================================
            # Step 1: Prepare Training Data
            # ================================================================
            logger.info("Preparing training data...")

            # Query captions based on job configuration
            captions = db.query(ScrapedCaption).filter(
                ScrapedCaption.source_subreddit.in_(job.source_subreddits),
                ScrapedCaption.llm_refined_text.isnot(None),
                ScrapedCaption.llm_refined_text != "",
                ScrapedCaption.upvotes >= job.min_upvotes
            ).all()

            # Filter by length
            filtered_captions = [
                c for c in captions
                if job.min_caption_length <= len(c.llm_refined_text) <= job.max_caption_length
            ]

            if len(filtered_captions) < 10:
                raise ValueError(f"Not enough training data: {len(filtered_captions)} samples (minimum 10)")

            job.total_samples = len(filtered_captions)

            # Create train/val split (90/10)
            import random
            random.shuffle(filtered_captions)
            split_idx = int(len(filtered_captions) * 0.9)
            train_captions = filtered_captions[:split_idx]
            val_captions = filtered_captions[split_idx:]

            job.train_samples = len(train_captions)
            job.val_samples = len(val_captions)
            job.progress_percent = 5.0
            db.commit()

            logger.info(f"Data prepared: {len(train_captions)} train, {len(val_captions)} val")

            # ================================================================
            # Step 2: Setup Model and Training
            # ================================================================
            job.status = "training"
            job.progress_percent = 10.0
            db.commit()

            # First, clear Qwen2-VL caption extractor to free GPU memory for training
            try:
                from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_qwen2vl
                cleanup_qwen2vl()
                logger.info("Cleared Qwen2-VL model before starting training")
            except Exception as e:
                logger.debug(f"Qwen2-VL cleanup: {e}")

            logger.info("Loading base model and tokenizer...")

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(job.base_model)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )

            # Load base model
            # Note: attn_implementation="eager" required for Turing GPUs (RTX 2080 Ti)
            # FlashAttention only supports Ampere (RTX 30xx) or newer
            model = AutoModelForCausalLM.from_pretrained(
                job.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Disable FlashAttention for Turing GPU compatibility
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1

            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)

            # LoRA config
            lora_config = LoraConfig(
                r=job.lora_rank,
                lora_alpha=job.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)

            job.progress_percent = 15.0
            db.commit()

            logger.info("Model loaded, preparing dataset...")

            # ================================================================
            # Step 3: Prepare Dataset
            # ================================================================
            from datasets import Dataset

            def format_caption(caption):
                return f"<s>[INST] Generate a caption: [/INST]\n{caption.llm_refined_text}</s>"

            train_data = Dataset.from_dict({
                "text": [format_caption(c) for c in train_captions]
            })
            val_data = Dataset.from_dict({
                "text": [format_caption(c) for c in val_captions]
            })

            # Calculate total steps
            total_steps = (len(train_data) // job.batch_size // job.gradient_accumulation_steps) * job.num_epochs
            job.total_steps = total_steps
            job.progress_percent = 20.0
            db.commit()

            # ================================================================
            # Step 4: Setup Trainer with Custom Callback
            # ================================================================
            # Create output directory
            model_name = job.job_name.lower().replace(" ", "-").replace("/", "-")
            output_dir = f"/data/training_models/{model_name}"
            os.makedirs(output_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=job.num_epochs,
                per_device_train_batch_size=job.batch_size,
                gradient_accumulation_steps=job.gradient_accumulation_steps,
                learning_rate=job.learning_rate,
                warmup_ratio=job.warmup_ratio,
                fp16=True,
                logging_steps=1,
                save_strategy="epoch",
                eval_strategy="epoch",
                load_best_model_at_end=False,
                report_to="none",
                gradient_checkpointing=True,
                max_grad_norm=0.3,
                optim="paged_adamw_32bit",
            )

            # Custom callback for progress reporting
            from transformers import TrainerCallback

            class ProgressCallback(TrainerCallback):
                def __init__(self, job_id, db_session):
                    self.job_id = job_id
                    self.last_update = time.time()

                def on_log(self, args, state, control, logs=None, **kwargs):
                    # Update progress every 10 seconds max
                    if time.time() - self.last_update < 10:
                        return

                    from database.db import get_db_context
                    with get_db_context() as db:
                        job = db.query(TrainingJob).filter_by(id=self.job_id).first()
                        if job:
                            # Calculate progress (20% for data prep, 80% for training)
                            if state.max_steps > 0:
                                train_progress = (state.global_step / state.max_steps) * 80
                            else:
                                train_progress = 0
                            job.progress_percent = 20 + train_progress
                            job.current_step = state.global_step
                            job.current_epoch = int(state.epoch) if state.epoch else 0

                            if logs:
                                job.current_loss = logs.get("loss")
                                if job.best_loss is None or (job.current_loss and job.current_loss < job.best_loss):
                                    job.best_loss = job.current_loss

                            db.commit()
                            self.last_update = time.time()

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                dataset_text_field="text",
                max_seq_length=job.max_seq_length,
                callbacks=[ProgressCallback(job_id, db)],
            )

            # ================================================================
            # Step 5: Run Training
            # ================================================================
            logger.info("Starting training...")
            train_result = trainer.train()

            # ================================================================
            # Step 6: Evaluate and Save
            # ================================================================
            job.status = "evaluating"
            job.progress_percent = 95.0
            db.commit()

            logger.info("Evaluating model...")
            eval_result = trainer.evaluate()

            job.status = "saving"
            db.commit()

            logger.info("Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)

            # ================================================================
            # Step 7: Create TrainedModel Record
            # ================================================================
            training_duration = int(time.time() - start_time)

            trained_model = TrainedModel(
                name=job.job_name,
                description=f"Trained on {', '.join(job.source_subreddits)} with {job.total_samples} samples (GPU {target_gpu})",
                base_model=job.base_model,
                adapter_path=output_dir,
                source_subreddits=job.source_subreddits,
                niche=getattr(job, 'niche', None),  # Per-niche model tracking
                training_samples=job.total_samples,
                min_upvotes_filter=job.min_upvotes,
                target_gpu=target_gpu,
                hyperparameters={
                    "lora_rank": job.lora_rank,
                    "lora_alpha": job.lora_alpha,
                    "learning_rate": job.learning_rate,
                    "num_epochs": job.num_epochs,
                    "batch_size": job.batch_size,
                    "gradient_accumulation_steps": job.gradient_accumulation_steps,
                    "max_seq_length": job.max_seq_length,
                    "warmup_ratio": job.warmup_ratio,
                    "target_gpu": target_gpu
                },
                final_loss=train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                validation_loss=eval_result.get("eval_loss"),
                training_duration_seconds=training_duration,
                status="ready"
            )

            db.add(trained_model)
            db.flush()

            # Update job with model reference
            job.model_id = trained_model.id
            job.status = "completed"
            job.progress_percent = 100.0
            job.completed_at = datetime.utcnow()
            job.final_train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
            job.final_val_loss = eval_result.get("eval_loss")
            job.training_duration_seconds = training_duration
            db.commit()

            # Cleanup GPU memory
            del model
            del trainer
            torch.cuda.empty_cache()

            logger.info(f"Training job {job_id} completed successfully!")
            return {
                "status": "completed",
                "model_id": trained_model.id,
                "model_name": trained_model.name,
                "final_loss": job.final_train_loss,
                "val_loss": job.final_val_loss,
                "duration_seconds": training_duration
            }

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            logger.error(traceback.format_exc())

            job.status = "failed"
            job.error_message = str(e)
            job.error_traceback = traceback.format_exc()
            job.completed_at = datetime.utcnow()
            db.commit()

            # Cleanup GPU memory on error
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

            return {
                "status": "failed",
                "error": str(e)
            }


# ============================================================================
# Model Loading/Unloading Tasks
# ============================================================================

def _load_model_impl(model_id: int, target_gpu: int = None):
    """
    Internal implementation for loading a trained model into GPU memory.

    Args:
        model_id: ID of the model to load
        target_gpu: GPU index (0 or 1) to track where the model is loaded.
                   If None, attempts to detect from NVIDIA_VISIBLE_DEVICES.
    """
    global _loaded_model, _loaded_model_id, _loaded_tokenizer

    from database.db import get_db_context
    from database.models import TrainedModel
    import torch
    import redis
    import os
    from config.settings import settings

    # Determine which GPU we're on
    if target_gpu is None:
        # Try to detect from environment
        visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
        target_gpu = int(visible_devices.split(',')[0]) if visible_devices else 0

    with get_db_context() as db:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            return {"status": "error", "message": f"Model {model_id} not found"}

        try:
            # First, unload the Qwen2-VL caption extractor to free GPU memory
            # This is important because Qwen2-VL uses ~5GB and stays loaded indefinitely
            try:
                from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_qwen2vl
                cleanup_qwen2vl()
                logger.info("Cleared Qwen2-VL model before loading trained model")
            except Exception as e:
                logger.debug(f"Qwen2-VL cleanup: {e}")

            # Unload any currently loaded trained model
            if _loaded_model is not None:
                del _loaded_model
                del _loaded_tokenizer
                torch.cuda.empty_cache()
                _loaded_model = None
                _loaded_tokenizer = None

                # Update old model status and clear Redis tracking
                if _loaded_model_id:
                    old_model = db.query(TrainedModel).filter_by(id=_loaded_model_id).first()
                    if old_model:
                        old_model.is_loaded = False
                        db.commit()
                    try:
                        r = redis.from_url(settings.celery_broker_url)
                        r.delete(f"model:{_loaded_model_id}:loaded_on_gpu")
                    except:
                        pass

            logger.info(f"Loading model {model_id}: {model.name} on GPU {target_gpu}")
            model.status = "loading"
            db.commit()

            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Load base model
            # Note: attn_implementation="eager" required for Turing GPUs (RTX 2080 Ti)
            base_model = AutoModelForCausalLM.from_pretrained(
                model.base_model,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Disable FlashAttention for Turing GPU
            )

            # Load LoRA adapter
            _loaded_model = PeftModel.from_pretrained(base_model, model.adapter_path)
            _loaded_tokenizer = AutoTokenizer.from_pretrained(model.base_model)
            _loaded_tokenizer.pad_token = _loaded_tokenizer.eos_token
            _loaded_model_id = model_id

            # Update model status
            model.status = "loaded"
            model.is_loaded = True
            model.last_loaded_at = datetime.utcnow()
            db.commit()

            # Track which GPU the model is loaded on in Redis
            try:
                r = redis.from_url(settings.celery_broker_url)
                r.set(f"model:{model_id}:loaded_on_gpu", str(target_gpu))
                logger.info(f"Tracked model {model_id} as loaded on GPU {target_gpu}")
            except Exception as e:
                logger.warning(f"Failed to track model GPU in Redis: {e}")

            logger.info(f"Model {model_id} loaded successfully on GPU {target_gpu}")
            return {"status": "success", "model_id": model_id, "model_name": model.name, "gpu": target_gpu}

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            model.status = "error"
            db.commit()
            return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.load_model_task', queue='gpu')
def load_model_task(self, model_id: int):
    """Load a trained model into GPU memory (legacy - routes to generic gpu queue)."""
    return _load_model_impl(model_id)


# GPU-specific load tasks for targeted GPU loading
@celery_app.task(bind=True, name='tasks.training_tasks.load_model_task_gpu_0', queue='gpu_status_0')
def load_model_task_gpu_0(self, model_id: int):
    """Load a trained model into GPU 0 memory. Routed to gpu_status_0 queue."""
    return _load_model_impl(model_id, target_gpu=0)


@celery_app.task(bind=True, name='tasks.training_tasks.load_model_task_gpu_1', queue='gpu_status_1')
def load_model_task_gpu_1(self, model_id: int):
    """Load a trained model into GPU 1 memory. Routed to gpu_status_1 queue."""
    return _load_model_impl(model_id, target_gpu=1)


def _unload_model_impl(model_id: int):
    """
    Internal implementation for unloading a model from GPU memory.

    Called by GPU-specific unload tasks to ensure the unload happens
    on the same worker that loaded the model.
    """
    global _loaded_model, _loaded_model_id, _loaded_tokenizer

    from database.db import get_db_context
    from database.models import TrainedModel
    import torch
    import redis
    from config.settings import settings

    with get_db_context() as db:
        model = db.query(TrainedModel).filter_by(id=model_id).first()
        if not model:
            return {"status": "error", "message": f"Model {model_id} not found"}

        if _loaded_model_id != model_id:
            # Model not in memory on this worker - fix stale DB state if needed
            if model.is_loaded:
                logger.info(f"Fixing stale DB state for model {model_id} (was marked loaded but not in memory on this worker)")
                model.is_loaded = False
                model.status = "ready"
                db.commit()
                # Clear Redis tracking
                try:
                    r = redis.from_url(settings.celery_broker_url)
                    r.delete(f"model:{model_id}:loaded_on_gpu")
                except:
                    pass
                return {"status": "success", "message": f"Fixed stale state - model {model_id} was not actually loaded on this worker"}
            return {"status": "success", "message": f"Model {model_id} is not loaded"}

        try:
            import gc
            import time

            logger.info(f"Unloading model {model_id}")

            # Delete model and tokenizer references
            del _loaded_model
            del _loaded_tokenizer

            _loaded_model = None
            _loaded_tokenizer = None
            _loaded_model_id = None

            # Aggressive GPU memory cleanup (same as deep_clean)
            gc.collect()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()

            # Brief pause to allow memory release
            time.sleep(0.5)

            model.status = "ready"
            model.is_loaded = False
            db.commit()

            # Clear Redis tracking
            try:
                r = redis.from_url(settings.celery_broker_url)
                r.delete(f"model:{model_id}:loaded_on_gpu")
            except:
                pass

            logger.info(f"Model {model_id} unloaded successfully with aggressive cleanup")
            return {"status": "success", "model_id": model_id}

        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.unload_model_task', queue='gpu')
def unload_model_task(self, model_id: int):
    """
    Unload a model from GPU memory (legacy - routes to generic gpu queue).

    WARNING: This may run on the wrong GPU worker! Use unload_model_task_gpu_0
    or unload_model_task_gpu_1 instead for reliable unloading.
    """
    return _unload_model_impl(model_id)


# GPU-specific unload tasks - ensures unload runs on the same worker that loaded the model
@celery_app.task(bind=True, name='tasks.training_tasks.unload_model_task_gpu_0', queue='gpu_status_0')
def unload_model_task_gpu_0(self, model_id: int):
    """Unload a model from GPU 0 memory. Routed to gpu_status_0 queue."""
    return _unload_model_impl(model_id)


@celery_app.task(bind=True, name='tasks.training_tasks.unload_model_task_gpu_1', queue='gpu_status_1')
def unload_model_task_gpu_1(self, model_id: int):
    """Unload a model from GPU 1 memory. Routed to gpu_status_1 queue."""
    return _unload_model_impl(model_id)


# ============================================================================
# Caption Generation Task
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.generate_caption_task', queue='gpu')
def generate_caption_task(
    self,
    model_id: int,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15
):
    """Generate a caption using a loaded model."""
    global _loaded_model, _loaded_model_id, _loaded_tokenizer

    import torch

    if _loaded_model is None or _loaded_model_id != model_id:
        return {"status": "error", "message": f"Model {model_id} is not loaded"}

    try:
        start_time = time.time()

        # Format prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]\n"

        # Tokenize
        inputs = _loaded_tokenizer(formatted_prompt, return_tensors="pt").to(_loaded_model.device)

        # Generate
        with torch.no_grad():
            outputs = _loaded_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=_loaded_tokenizer.eos_token_id,
            )

        # Decode
        generated_text = _loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated caption
        caption = generated_text.split("[/INST]")[-1].strip()

        # Post-process to clean up artifacts and spam.
        # Look up the model's niche so the postprocessor's niche-aware
        # rewrites (e.g. per-niche spelling normalisation) fire on test gens.
        # Cheap one-shot DB hit; this task is not in the hot path.
        niche = None
        try:
            from database.db import get_db_context
            from database.models import TrainedModel
            with get_db_context() as db:
                m = db.query(TrainedModel).filter_by(id=model_id).first()
                if m:
                    niche = getattr(m, 'niche', None)
        except Exception:
            pass  # niche-aware rewrites are belt-and-braces; OK to skip
        from utils.generation_postprocessor import clean_generated_caption, get_final_score
        caption = clean_generated_caption(caption, aggressive=True, niche=niche)

        # Use strict scoring (without LLM for speed in test generation)
        # This applies rule-based deductions for quick feedback
        quality_score = get_final_score(caption, model=_loaded_model, tokenizer=_loaded_tokenizer)

        generation_time = time.time() - start_time

        return {
            "status": "success",
            "text": caption,
            "quality_score": quality_score,
            "time": round(generation_time, 2)
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# GPU Memory Management
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.get_gpu_status_task', queue='gpu')
def get_gpu_status_task(self):
    """
    Get GPU status from the celery-worker (which has GPU access).

    Returns GPU memory usage and utilization for the GPU this worker has access to.
    Note: Each worker only sees its assigned GPU due to NVIDIA_VISIBLE_DEVICES.
    """
    import subprocess
    import torch
    import os

    # Get the physical GPU index from environment (set by docker-compose)
    physical_gpu = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
    try:
        physical_gpu_id = int(physical_gpu.split(',')[0])
    except ValueError:
        physical_gpu_id = 0

    try:
        # Try nvidia-smi first for detailed info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        # Use physical GPU index from environment, not container-local index
                        "index": physical_gpu_id,
                        "name": parts[1],
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                        "memory_percent": round(int(parts[2]) / int(parts[3]) * 100, 1),
                        "utilization_percent": int(parts[4]),
                        "temperature_c": int(parts[5])
                    })

            return {
                "gpus": gpus,
                "total_memory_used_mb": sum(g["memory_used_mb"] for g in gpus),
                "total_memory_mb": sum(g["memory_total_mb"] for g in gpus),
                "worker_gpu": physical_gpu_id
            }
        else:
            # Fallback to PyTorch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i)
                    total = props.total_memory
                    gpus.append({
                        "index": i,
                        "name": props.name,
                        "memory_used_mb": int(allocated / 1024**2),
                        "memory_total_mb": int(total / 1024**2),
                        "memory_percent": round(allocated / total * 100, 1),
                        "utilization_percent": 0,  # Can't get from PyTorch
                        "temperature_c": 0  # Can't get from PyTorch
                    })
                return {
                    "gpus": gpus,
                    "total_memory_used_mb": sum(g["memory_used_mb"] for g in gpus),
                    "total_memory_mb": sum(g["memory_total_mb"] for g in gpus)
                }
            else:
                return {"error": "No GPU available"}

    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout"}
    except FileNotFoundError:
        # nvidia-smi not found, try PyTorch
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                total = props.total_memory
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory_used_mb": int(allocated / 1024**2),
                    "memory_total_mb": int(total / 1024**2),
                    "memory_percent": round(allocated / total * 100, 1),
                    "utilization_percent": 0,
                    "temperature_c": 0
                })
            return {
                "gpus": gpus,
                "total_memory_used_mb": sum(g["memory_used_mb"] for g in gpus),
                "total_memory_mb": sum(g["memory_total_mb"] for g in gpus)
            }
        return {"error": "nvidia-smi not found and no CUDA available"}
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {"error": str(e)}


# GPU-specific status tasks - each routed to a dedicated queue that only one worker listens to
# This allows the API to query both GPUs by sending tasks to each worker's dedicated queue

@celery_app.task(bind=True, name='tasks.training_tasks.get_gpu_0_status_task', queue='gpu_status_0')
def get_gpu_0_status_task(self):
    """
    Get GPU status from worker 0 (GPU 0).

    This task is routed to gpu_status_0 queue which only worker 0 listens to.
    Returns GPU memory usage and utilization for GPU 0.
    """
    return _get_worker_gpu_status()


@celery_app.task(bind=True, name='tasks.training_tasks.get_gpu_1_status_task', queue='gpu_status_1')
def get_gpu_1_status_task(self):
    """
    Get GPU status from worker 1 (GPU 1).

    This task is routed to gpu_status_1 queue which only worker 1 listens to.
    Returns GPU memory usage and utilization for GPU 1.
    """
    return _get_worker_gpu_status()


def _get_loaded_models_info():
    """
    Get info about models currently loaded in GPU memory on this worker.

    Returns:
        List of dictionaries with model info.
    """
    global _loaded_model, _loaded_model_id

    models = []

    # Check if OCR model (Qwen2-VL) is loaded
    try:
        from scrapers.caption_extractor_qwen2vl import get_qwen2vl_model_info
        qwen_info = get_qwen2vl_model_info()
        if qwen_info:
            models.append(qwen_info)
    except Exception as e:
        logger.debug(f"Could not check Qwen2-VL status: {e}")

    # Check if LLM (Mistral-7B) is loaded
    try:
        from scrapers.caption_postprocessor import get_llm_status
        llm_status = get_llm_status()
        if llm_status and llm_status.get("loaded"):
            models.append({
                "name": llm_status.get("model_name", "Mistral-7B-Instruct"),
                "type": "llm",
                "vram_estimate_gb": 3.5,
                "idle_seconds": llm_status.get("idle_seconds"),
            })
    except Exception as e:
        logger.debug(f"Could not check LLM status: {e}")

    # Check if a trained LoRA model is loaded
    if _loaded_model is not None and _loaded_model_id is not None:
        try:
            from database.db import get_db_context
            from database.models import TrainedModel

            with get_db_context() as db:
                model = db.query(TrainedModel).filter_by(id=_loaded_model_id).first()
                if model:
                    models.append({
                        "name": model.name,
                        "type": "trained_lora",
                        "vram_estimate_gb": 4.0,
                        "model_id": model.id,
                    })
        except Exception as e:
            logger.debug(f"Could not get trained model info: {e}")
            models.append({
                "name": f"Trained Model #{_loaded_model_id}",
                "type": "trained_lora",
                "vram_estimate_gb": 4.0,
                "model_id": _loaded_model_id,
            })

    return models


def _get_worker_gpu_status():
    """
    Internal function to get GPU status from the current worker.

    Returns the physical GPU ID based on NVIDIA_VISIBLE_DEVICES environment variable.
    Also includes info about models currently loaded in memory.
    """
    import subprocess
    import torch
    import os

    # Get the physical GPU index from environment (set by docker-compose)
    physical_gpu = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
    try:
        physical_gpu_id = int(physical_gpu.split(',')[0])
    except ValueError:
        physical_gpu_id = 0

    # Get loaded models info
    loaded_models = _get_loaded_models_info()

    try:
        # Try nvidia-smi first for detailed info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    return {
                        "index": physical_gpu_id,
                        "name": parts[1],
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                        "memory_percent": round(int(parts[2]) / int(parts[3]) * 100, 1),
                        "utilization_percent": int(parts[4]),
                        "temperature_c": int(parts[5]),
                        "worker_gpu": physical_gpu_id,
                        "loaded_models": loaded_models,
                    }

        # Fallback to PyTorch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)  # Container sees only one GPU as index 0
            allocated = torch.cuda.memory_allocated(0)
            total = props.total_memory
            return {
                "index": physical_gpu_id,
                "name": props.name,
                "memory_used_mb": int(allocated / 1024**2),
                "memory_total_mb": int(total / 1024**2),
                "memory_percent": round(allocated / total * 100, 1),
                "utilization_percent": 0,
                "temperature_c": 0,
                "worker_gpu": physical_gpu_id,
                "loaded_models": loaded_models,
            }

        return {"error": "No GPU available", "worker_gpu": physical_gpu_id, "loaded_models": loaded_models}

    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout", "worker_gpu": physical_gpu_id, "loaded_models": loaded_models}
    except FileNotFoundError:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            total = props.total_memory
            return {
                "index": physical_gpu_id,
                "name": props.name,
                "memory_used_mb": int(allocated / 1024**2),
                "memory_total_mb": int(total / 1024**2),
                "memory_percent": round(allocated / total * 100, 1),
                "utilization_percent": 0,
                "temperature_c": 0,
                "worker_gpu": physical_gpu_id,
                "loaded_models": loaded_models,
            }
        return {"error": "nvidia-smi not found and no CUDA available", "worker_gpu": physical_gpu_id, "loaded_models": loaded_models}
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {"error": str(e), "worker_gpu": physical_gpu_id, "loaded_models": loaded_models}


@celery_app.task(bind=True, name='tasks.training_tasks.clear_gpu_memory_task', queue='gpu')
def clear_gpu_memory_task(self):
    """
    Deep clean GPU memory - aggressively clears ALL GPU memory.

    This is more aggressive than normal cleanup:
    1. Unloads all models (OCR, LLM, trained) regardless of tracking state
    2. Forces garbage collection multiple times
    3. Synchronizes CUDA and empties all caches
    4. Reports before/after memory usage

    Use this when GPU shows high memory usage but no models appear loaded.
    """
    global _loaded_model, _loaded_model_id, _loaded_tokenizer

    import torch
    import gc

    try:
        cleared_models = []

        # Get memory BEFORE cleanup
        memory_before = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_before[i] = {
                    'allocated_gb': round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    'reserved_gb': round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                }

        # 1. Clear loaded trained model if any
        if _loaded_model is not None:
            from database.db import get_db_context
            from database.models import TrainedModel

            with get_db_context() as db:
                if _loaded_model_id:
                    model = db.query(TrainedModel).filter_by(id=_loaded_model_id).first()
                    if model:
                        model.is_loaded = False
                        model.status = "ready"
                        db.commit()
                        cleared_models.append(f"trained model '{model.name}'")

            try:
                _loaded_model.to('cpu')
            except:
                pass
            del _loaded_model
            if _loaded_tokenizer is not None:
                del _loaded_tokenizer
            _loaded_model = None
            _loaded_tokenizer = None
            _loaded_model_id = None

        # 2. Clear the Qwen2-VL caption extractor model (OCR)
        try:
            from scrapers.caption_extractor_qwen2vl import cleanup_model as cleanup_qwen2vl
            cleanup_qwen2vl()
            cleared_models.append("Qwen2-VL-2B (OCR)")
            logger.info("Cleared Qwen2-VL caption extractor model")
        except Exception as e:
            logger.debug(f"Could not clear Qwen2-VL model: {e}")

        # 3. Clear the LLM (Mistral-7B) model
        try:
            from scrapers.caption_postprocessor import cleanup_llm
            cleanup_llm()
            cleared_models.append("Mistral-7B (LLM)")
            logger.info("Cleared Mistral-7B LLM model")
        except Exception as e:
            logger.debug(f"Could not clear LLM model: {e}")

        # 4. Aggressive garbage collection
        gc.collect()
        gc.collect()

        # 5. CUDA cleanup
        if torch.cuda.is_available():
            # Synchronize all streams
            torch.cuda.synchronize()

            # Empty cache
            torch.cuda.empty_cache()

            # Reset peak memory stats
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            # Final garbage collection
            gc.collect()

            # Get memory AFTER cleanup
            memory_after = {}
            for i in range(torch.cuda.device_count()):
                memory_after[i] = {
                    'allocated_gb': round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    'reserved_gb': round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                }
                freed = memory_before.get(i, {}).get('allocated_gb', 0) - memory_after[i]['allocated_gb']
                logger.info(
                    f"GPU {i}: {memory_after[i]['allocated_gb']:.2f}GB allocated "
                    f"(freed {freed:.2f}GB), {memory_after[i]['reserved_gb']:.2f}GB reserved"
                )

        message = f"GPU memory deep cleaned. Models unloaded: {', '.join(cleared_models) if cleared_models else 'none'}"
        return {
            "status": "success",
            "message": message,
            "models_cleared": cleared_models,
            "memory_before": memory_before,
            "memory_after": memory_after if torch.cuda.is_available() else {},
        }

    except Exception as e:
        logger.error(f"Failed to clear GPU memory: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Legacy Tasks (kept for backwards compatibility)
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.train_lora_model', time_limit=21600, soft_time_limit=21000)
def train_lora_model(self, num_epochs: int = 3):
    """Legacy training task - use run_training_job instead."""
    import subprocess

    self.update_state(state='PROGRESS', meta={'stage': 'exporting_data', 'progress': 0})

    try:
        result = subprocess.run(
            ['python', 'training/export_training_data.py'],
            capture_output=True, text=True, cwd='/app', timeout=300
        )
        if result.returncode != 0:
            return {'status': 'failed', 'stage': 'export', 'error': result.stderr}
    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'stage': 'export', 'error': 'Export timed out'}

    self.update_state(state='PROGRESS', meta={'stage': 'creating_split', 'progress': 10})

    try:
        result = subprocess.run(
            ['python', 'training/create_split.py'],
            capture_output=True, text=True, cwd='/app', timeout=60
        )
        if result.returncode != 0:
            return {'status': 'failed', 'stage': 'split', 'error': result.stderr}
    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'stage': 'split', 'error': 'Split creation timed out'}

    self.update_state(state='PROGRESS', meta={'stage': 'training', 'progress': 15})

    try:
        result = subprocess.run(
            ['python', 'training/train_lora.py'],
            capture_output=True, text=True, cwd='/app', timeout=21000
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            return {'status': 'failed', 'stage': 'training', 'error': output[-2000:]}

        return {
            'status': 'completed',
            'model_path': 'training/models/mistral-motivation-lora/',
            'epochs': num_epochs
        }

    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'stage': 'training', 'error': 'Training timed out'}


@celery_app.task(bind=True, name='tasks.training_tasks.export_training_data')
def export_training_data(self):
    """Export training data from database."""
    import subprocess

    try:
        result = subprocess.run(
            ['python', 'training/export_training_data.py'],
            capture_output=True, text=True, cwd='/app', timeout=300
        )

        if result.returncode != 0:
            return {'status': 'failed', 'error': result.stderr}

        return {
            'status': 'completed',
            'output_file': 'training/data/captions_all.jsonl'
        }

    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'error': 'Export timed out'}


# ============================================================================
# Batch Generation Tasks
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.run_generation_job', queue='maintenance', time_limit=7200, soft_time_limit=7000)
def run_generation_job(self, job_id: int):
    """
    Run a batch caption generation job using llama-server API.

    This task uses the external llama-server (with LoRA adapter) running on the host
    machine, NOT the in-container PEFT loading. This is much more efficient because:
    1. llama-server already has the model loaded with tensor parallelism
    2. The LoRA adapter is already loaded in GGUF format
    3. No GPU memory needed in the container

    Args:
        job_id: GenerationJob database ID
    """
    from database.db import get_db_context
    from database.models import GenerationJob, GeneratedCaption, TrainedModel
    from utils.llama_server_control import is_server_running, generate as llama_generate, LLAMA_SERVER_URL

    start_time = time.time()

    with get_db_context() as db:
        job = db.query(GenerationJob).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Generation job {job_id} not found")
            return {"status": "error", "message": f"Job {job_id} not found"}

        try:
            # Update status to running
            job.status = "running"
            job.started_at = datetime.utcnow()
            db.commit()

            logger.info(f"Starting generation job {job_id}: {job.job_name} ({job.num_captions} captions)")

            # Get the model info (optional, for metadata)
            model = None
            model_name = "llama-server"
            niche = None
            if job.model_id:
                model = db.query(TrainedModel).filter_by(id=job.model_id).first()
                if model:
                    model_name = model.name
                    niche = getattr(model, 'niche', None)

            # Verify llama-server is running
            if not is_server_running():
                raise RuntimeError(f"llama-server is not running at {LLAMA_SERVER_URL}. Start it with the LoRA adapter first.")

            logger.info(f"Using llama-server at {LLAMA_SERVER_URL} for generation")

            # Generate captions one at a time
            for i in range(job.num_captions):
                # Check if job was cancelled
                db.refresh(job)
                if job.status == "cancelled":
                    logger.info(f"Generation job {job_id} was cancelled")
                    return {"status": "cancelled", "captions_generated": job.captions_generated}

                # Format prompt for Mistral instruction format
                formatted_prompt = f"<s>[INST] {job.prompt} [/INST]\n"

                # Generate using llama-server API
                generated_text = llama_generate(
                    prompt=formatted_prompt,
                    max_tokens=job.max_new_tokens,
                    temperature=job.temperature,
                    repetition_penalty=job.repetition_penalty,
                    top_p=job.top_p
                )

                if not generated_text:
                    logger.warning(f"Generation {i + 1} returned empty, skipping")
                    continue

                # Clean up the output
                caption_text = generated_text.strip()

                # Post-process to clean up artifacts and spam
                from utils.generation_postprocessor import clean_generated_caption, extract_tags, should_reject_caption, apply_rule_based_deductions
                caption_text = clean_generated_caption(caption_text, aggressive=True, niche=niche)

                # Quality scoring: word-count base + pre-rejection + rule-based deductions
                word_count = len(caption_text.split())

                # Step 1: Check for pre-rejection (gibberish, extreme repetition, etc.)
                rejected, reject_reason = should_reject_caption(caption_text)
                if rejected:
                    quality_score = 0.0
                    logger.debug(f"Caption {i + 1} rejected: {reject_reason}")
                else:
                    # Step 2: Word-count base score
                    if word_count < 20:
                        base_score = 20.0
                    elif 40 <= word_count <= 100:
                        base_score = 80.0
                    elif 20 <= word_count < 40:
                        base_score = 60.0
                    elif 100 < word_count <= 140:
                        base_score = 70.0
                    else:
                        base_score = 50.0

                    # Step 3: Apply rule-based deductions (punctuation, caps, spam, contamination, etc.)
                    quality_score, deductions = apply_rule_based_deductions(caption_text, base_score)
                    if deductions:
                        logger.debug(f"Caption {i + 1} score {base_score} -> {quality_score}: {deductions}")

                # Extract activity tags for background video matching
                caption_tags = extract_tags(caption_text)

                # Generate title using llama-server
                title_prompt = f"<s>[INST] Generate a short, catchy Reddit post title (under 100 characters) for this caption. Only output the title, nothing else:\n\n{caption_text[:300]} [/INST]\n"
                generated_title = llama_generate(
                    prompt=title_prompt,
                    max_tokens=50,
                    temperature=0.7
                )
                if generated_title:
                    generated_title = generated_title.strip()[:100]
                    logger.debug(f"Generated title: {generated_title[:50]}...")

                # Store the generated caption
                generated_caption = GeneratedCaption(
                    generation_job_id=job_id,
                    caption_text=caption_text,
                    generated_title=generated_title,
                    llm_model=model_name,
                    generation_prompt=job.prompt,
                    quality_score=quality_score,
                    status="pending_review",
                    temperature=job.temperature,
                    top_p=job.top_p,
                    max_tokens=job.max_new_tokens,
                    tags=caption_tags,
                    niche=niche,
                    judge_status="pending",
                )
                db.add(generated_caption)
                db.flush()  # populate generated_caption.id for log line below

                # Stage 3: LLM judge. Skip if we already pre-rejected the caption
                # — no point burning tokens on gibberish. llama-server is already
                # warm with the same Mistral that just generated, so this adds
                # ~1-2 s per caption.
                judge_result = None
                if quality_score > 0:
                    try:
                        from tasks.caption_judge import judge_caption, apply_judge_result
                        judge_result = judge_caption(caption_text)
                        apply_judge_result(generated_caption, judge_result)
                    except Exception as judge_err:
                        logger.warning(f"Judge failed for caption {generated_caption.id}: {judge_err}")
                        generated_caption.judge_status = "failed"
                else:
                    # Pre-rejected caption gets a default-fail judge to keep
                    # selection logic uniform.
                    generated_caption.judge_status = "judged"
                    generated_caption.judge_pass = False
                    generated_caption.judge_scores = {
                        "grammar": 1, "flow": 1, "bg_consistency": 1,
                        "appeal": 1, "overall": 1,
                    }
                    generated_caption.judge_issues = ["pre-rejected by rule-based filter"]

                # Update progress
                job.captions_generated = i + 1
                job.progress_percent = ((i + 1) / job.num_captions) * 100
                db.commit()

                judge_summary = ""
                if judge_result is not None:
                    s = judge_result["scores"]
                    judge_summary = (
                        f", judge: {'PASS' if judge_result['passed'] else 'fail'} "
                        f"(g{s['grammar']}/f{s['flow']}/bg{s['bg_consistency']}/a{s['appeal']}/o{s['overall']})"
                    )
                logger.info(
                    f"Generated caption {i + 1}/{job.num_captions} for job {job_id} "
                    f"(score: {quality_score:.0f}, title: {'yes' if generated_title else 'no'}{judge_summary})"
                )

            # Mark job as completed
            job.status = "completed"
            job.progress_percent = 100.0
            job.completed_at = datetime.utcnow()
            db.commit()

            duration = int(time.time() - start_time)
            logger.info(f"Generation job {job_id} completed: {job.num_captions} captions in {duration}s")

            return {
                "status": "completed",
                "job_id": job_id,
                "captions_generated": job.num_captions,
                "duration_seconds": duration
            }

        except Exception as e:
            logger.error(f"Generation job {job_id} failed: {e}")
            logger.error(traceback.format_exc())

            job.status = "failed"
            job.error_message = str(e)
            job.error_traceback = traceback.format_exc()
            job.completed_at = datetime.utcnow()
            db.commit()

            return {
                "status": "failed",
                "error": str(e)
            }


# ============================================================================
# LLM-BASED TAG EXTRACTION
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.extract_tags_llm_batch', queue='gpu_llm')
def extract_tags_llm_batch(self, limit: int = 50):
    """
    Extract tags from generated captions using LLM (Mistral-7B).

    More accurate than regex-based extraction because it understands context
    and can identify nuanced references to activities and settings.

    Args:
        limit: Maximum number of captions to process

    Returns:
        Dict with status and processing results
    """
    import time
    from database.db import get_db_context
    from database.models import GeneratedCaption
    from sqlalchemy import cast, String

    logger.info(f"Starting LLM-based tag extraction for up to {limit} captions")
    start_time = time.time()

    try:
        # Load LLM model (will be reused if already loaded)
        from scrapers.caption_postprocessor import _load_llm
        model, tokenizer = _load_llm(force=True)

        if model is None:
            return {
                "status": "error",
                "message": "Failed to load LLM model"
            }

        # Import the LLM extraction function
        from utils.generation_postprocessor import extract_tags_llm

        with get_db_context() as db:
            # Find captions needing tag extraction
            captions = db.query(GeneratedCaption).filter(
                (GeneratedCaption.tags == None) |
                (cast(GeneratedCaption.tags, String) == '[]')
            ).limit(limit).all()

            if not captions:
                return {
                    "status": "success",
                    "message": "No captions need tag extraction",
                    "processed": 0
                }

            total = len(captions)
            processed = 0
            tags_found = 0

            for i, caption in enumerate(captions):
                try:
                    if caption.caption_text:
                        # Use LLM to extract tags
                        tags = extract_tags_llm(caption.caption_text, model=model, tokenizer=tokenizer)
                        caption.tags = tags

                        if tags:
                            tags_found += len(tags)

                        processed += 1

                        # Update progress
                        if (i + 1) % 5 == 0:
                            self.update_state(
                                state='PROGRESS',
                                meta={
                                    'current': i + 1,
                                    'total': total,
                                    'tags_extracted': tags_found
                                }
                            )
                            db.commit()  # Commit periodically

                except Exception as e:
                    logger.error(f"Error extracting tags for caption {caption.id}: {e}")
                    continue

            db.commit()

            # Count remaining
            remaining = db.query(GeneratedCaption).filter(
                (GeneratedCaption.tags == None) |
                (cast(GeneratedCaption.tags, String) == '[]')
            ).count()

            duration = int(time.time() - start_time)
            logger.info(f"LLM tag extraction completed: {processed} captions, {tags_found} total tags in {duration}s")

            return {
                "status": "success",
                "method": "llm",
                "processed": processed,
                "tags_extracted": tags_found,
                "remaining": remaining,
                "duration_seconds": duration
            }

    except Exception as e:
        logger.error(f"LLM tag extraction failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================================
# LLM-BASED QUALITY SCORING
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.score_captions_llm_batch', queue='gpu_llm')
def score_captions_llm_batch(self, limit: int = 50):
    """
    Score generated captions using LLM quality evaluation.

    Evaluates captions on:
    - Grammar & Writing Quality (0-30 points)
    - Engagement & Appeal (0-40 points)
    - Story Clarity (0-30 points)

    Only processes captions with quality_score = None or 0.

    Args:
        limit: Maximum number of captions to process

    Returns:
        Dict with status and processing results
    """
    start_time = time.time()

    from database.db import get_db_context
    from database.models import GeneratedCaption
    from utils.generation_postprocessor import score_caption_llm

    try:
        # Load LLM once for batch processing
        from scrapers.caption_postprocessor import _load_llm
        model, tokenizer = _load_llm(force=True)
        if model is None:
            return {"status": "error", "message": "LLM not available for scoring"}

        with get_db_context() as db:
            # Find captions without LLM scores (null or 0)
            captions = db.query(GeneratedCaption).filter(
                (GeneratedCaption.quality_score == None) |
                (GeneratedCaption.quality_score == 0)
            ).limit(limit).all()

            if not captions:
                return {
                    "status": "success",
                    "message": "No captions need scoring",
                    "processed": 0,
                    "remaining": 0
                }

            total = len(captions)
            processed = 0
            scored = 0

            logger.info(f"Starting LLM scoring for {total} captions")

            for i, caption in enumerate(captions):
                try:
                    if caption.caption_text:
                        # Score the caption using LLM
                        score = score_caption_llm(caption.caption_text, model=model, tokenizer=tokenizer)

                        if score is not None:
                            caption.quality_score = score
                            scored += 1
                            logger.debug(f"Scored caption {caption.id}: {score}/100")

                        processed += 1

                        # Update progress
                        if (i + 1) % 5 == 0:
                            self.update_state(
                                state='PROGRESS',
                                meta={
                                    'current': i + 1,
                                    'total': total,
                                    'scored': scored
                                }
                            )
                            db.commit()  # Commit periodically

                except Exception as e:
                    logger.error(f"Error scoring caption {caption.id}: {e}")
                    continue

            db.commit()

            # Count remaining
            remaining = db.query(GeneratedCaption).filter(
                (GeneratedCaption.quality_score == None) |
                (GeneratedCaption.quality_score == 0)
            ).count()

            duration = int(time.time() - start_time)
            logger.info(f"LLM scoring completed: {scored}/{processed} captions scored in {duration}s")

            return {
                "status": "success",
                "method": "llm",
                "processed": processed,
                "scored": scored,
                "remaining": remaining,
                "duration_seconds": duration
            }

    except Exception as e:
        logger.error(f"LLM scoring batch failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================================
# CLOUD TRAINING HELPER FUNCTIONS (called directly, not as subtasks)
# ============================================================================

def _convert_peft_to_gguf(niche: str, peft_path: str = None) -> dict:
    """
    Convert a PEFT LoRA adapter to GGUF format for llama.cpp.
    This is a direct function call, not a Celery task.
    """
    from utils.llama_server_control import HOST_LLAMA_CPP_DIR, run_host_ssh

    if peft_path is None:
        peft_path = f"/data/lora_adapters/{niche}"

    output_path = f"/data/lora_adapters/{niche}/adapter.gguf"

    logger.info(f"Converting PEFT adapter to GGUF: {peft_path} -> {output_path}")

    try:
        # Run conversion on the LLM host via SSH (llama.cpp lives there).
        # Host address/credentials come from settings.LLM_HOST_*.
        ssh_cmd = (
            f"cd {HOST_LLAMA_CPP_DIR} && "
            f"python convert_lora_to_gguf.py {peft_path} --outfile {output_path}"
        )

        returncode, _stdout, stderr = run_host_ssh(ssh_cmd, timeout=300)

        if returncode != 0:
            logger.error(f"Conversion failed: {stderr}")
            return {"status": "error", "message": stderr}

        logger.info(f"Conversion successful: {output_path}")
        return {"status": "success", "gguf_path": output_path, "niche": niche}

    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}")
        return {"status": "error", "message": str(e)}


def _restart_llama_server_with_lora(niche: str) -> dict:
    """
    Restart llama-server with a LoRA adapter.
    This is a direct function call, not a Celery task.
    """
    try:
        from utils.llama_server_control import restart_server, get_server_status

        success = restart_server(niche=niche)

        if success:
            status = get_server_status()
            return {
                "status": "success",
                "server_status": status,
                "niche": niche,
                "lora_loaded": True
            }
        else:
            return {"status": "error", "message": "Failed to restart llama-server"}

    except Exception as e:
        logger.error(f"llama-server restart failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# CLOUD TRAINING TASKS (Vast.ai / AWS)
# ============================================================================

@celery_app.task(bind=True, name='tasks.training_tasks.trigger_cloud_training', queue='maintenance', time_limit=57600, soft_time_limit=54000)
def trigger_cloud_training(self, niche: str, provider: str = "vastai", job_id: int = None, auto_destroy: bool = True):
    """
    Trigger cloud training for a niche category on Vast.ai.

    Complete workflow:
    1. Exports training data to a JSONL file
    2. Launches Vast.ai RTX 4090 instance
    3. Uploads training data + script
    4. Executes training (monitors progress)
    5. Downloads the resulting PEFT adapter
    6. Converts PEFT to GGUF format
    7. Restarts llama-server with the new adapter
    8. Destroys the Vast.ai instance

    Args:
        niche: Niche category to train (e.g., "motivation")
        provider: Cloud provider ("vastai" only for now)
        job_id: Optional TrainingJob ID to update with status (for pipeline orchestrator)

    Returns:
        Dict with status and training results
    """
    from database.db import get_db_context
    from database.models import TrainedModel, TrainingJob
    from pathlib import Path
    from config.settings import settings
    from config.automation_config import get_automation_config
    from training.export_training_data import export_niche_training_data

    def update_job_status(status: str, message: str = None, **kwargs):
        """Helper to update TrainingJob status if job_id provided."""
        if not job_id:
            return
        try:
            with get_db_context() as db:
                job = db.query(TrainingJob).filter_by(id=job_id).first()
                if job:
                    job.status = status
                    if message:
                        job.error_message = message
                    for key, value in kwargs.items():
                        if hasattr(job, key):
                            setattr(job, key, value)
                    db.commit()
        except Exception as e:
            logger.warning(f"Failed to update job status: {e}")

    logger.info(f"Starting cloud training for niche: {niche} on {provider}")
    start_time = time.time()
    instance_id = None

    # Update job status to started
    update_job_status("training", started_at=datetime.utcnow())

    # Validate provider
    if provider != "vastai":
        update_job_status("failed", f"Provider {provider} not implemented")
        return {"status": "error", "message": f"Provider {provider} not implemented. Use 'vastai'."}

    # Check for API key
    api_key = settings.VASTAI_API_KEY
    if not api_key:
        update_job_status("failed", "VASTAI_API_KEY not configured")
        return {"status": "error", "message": "VASTAI_API_KEY not configured. Set it in .env file."}

    # Get automation config for this niche
    config = get_automation_config()
    niche_config = config.get_niche(niche)

    if not niche_config:
        update_job_status("failed", f"Unknown niche: {niche}")
        return {"status": "error", "message": f"Unknown niche: {niche}. Available: {list(config.niches.keys())}"}

    try:
        from utils.vastai_client import VastaiClient

        # ================================================================
        # Step 1: Export training data
        # ================================================================
        logger.info("Step 1/8: Exporting training data...")

        export_path = f"/tmp/training_data_{niche}.jsonl"
        export_stats = export_niche_training_data(
            niche=niche,
            output_path=export_path,
            min_upvotes=100,
            min_length=25
        )

        if export_stats["exported"] < 100:
            update_job_status("failed", f"Not enough training data: {export_stats['exported']} captions")
            return {
                "status": "error",
                "message": f"Not enough training data: {export_stats['exported']} captions (need 100+)",
                "export_stats": export_stats
            }

        logger.info(f"Exported {export_stats['exported']} captions to {export_path}")

        # ================================================================
        # Step 2: Initialize Vast.ai client and launch instance
        # ================================================================
        logger.info("Step 2/8: Launching Vast.ai instance...")

        client = VastaiClient(api_key=api_key)

        # Find available GPU
        max_price = settings.VASTAI_MAX_PRICE_PER_HOUR
        gpu_name = settings.VASTAI_PREFERRED_GPU

        offers = client.search_gpu_offers(
            gpu_name=gpu_name,
            max_price=max_price,
            min_vram_gb=24,
            min_disk_gb=settings.VASTAI_MIN_DISK_GB  # 100GB needed for Mistral-Small-24B
        )

        if not offers:
            return {
                "status": "error",
                "message": f"No {gpu_name} instances available under ${max_price}/hr"
            }

        logger.info(f"Found {len(offers)} offers, cheapest at ${offers[0]['price_per_hour']}/hr")

        # Launch instance
        launch_result = client.launch_training_instance(
            gpu_name=gpu_name,
            max_price=max_price,
            disk_gb=settings.VASTAI_MIN_DISK_GB  # 100GB needed for Mistral-Small-24B
        )
        instance_id = launch_result["id"]
        logger.info(f"Launched instance {instance_id}")

        # ================================================================
        # Step 3: Wait for instance to be ready
        # ================================================================
        logger.info("Step 3/8: Waiting for instance to be ready...")

        status = client.wait_for_ready(instance_id, timeout_seconds=1200)
        logger.info(f"Instance ready: {status.get('ssh_host')}:{status.get('ssh_port')}")

        # ================================================================
        # Step 4: Upload training data and script
        # ================================================================
        logger.info("Step 4/8: Uploading training data and script...")

        # Use /root/ instead of /workspace/ - more reliable on Vast.ai
        # /workspace may be overwritten or not properly mounted initially
        work_dir = "/root/training"

        # Upload training data
        client.upload_file(instance_id, export_path, f"{work_dir}/training_data.jsonl")

        # Upload training script
        training_script = Path(__file__).parent.parent / "training" / "train_vastai.py"
        client.upload_file(instance_id, str(training_script), f"{work_dir}/train_vastai.py")

        # ================================================================
        # Step 5: Execute training
        # ================================================================
        logger.info("Step 5/8: Starting training (this may take 2-4 hours)...")

        train_cmd = (
            f"mkdir -p {work_dir}/output && cd {work_dir} && "
            f"pip install unsloth transformers datasets peft accelerate bitsandbytes safetensors tqdm && "
            f"python train_vastai.py --niche {niche} --data {work_dir}/training_data.jsonl --output {work_dir}/output"
        )

        # Use polling-based execution that waits for output file
        result = client.execute_training_and_wait(
            instance_id,
            train_cmd,
            output_marker=f"{work_dir}/output/adapter/adapter_model.safetensors",
            timeout_seconds=settings.VASTAI_MAX_TRAINING_HOURS * 3600,
            poll_interval=60  # Check every minute
        )

        if result["exit_code"] != 0:
            logger.error(f"Training failed: {result['stderr']}")
            logger.error(f"Training log:\n{result.get('training_log', 'No log available')[:2000]}")
            return {
                "status": "error",
                "message": f"Training failed: {result['stderr']}",
                "instance_id": instance_id,
                "stdout": result["stdout"],
                "training_log": result.get("training_log", "")[:5000]
            }

        logger.info("Training completed successfully")
        if result.get("training_log"):
            logger.info(f"Training log tail:\n{result['training_log'][-1000:]}")

        # ================================================================
        # Step 6: Download adapter
        # ================================================================
        logger.info("Step 6/8: Downloading trained adapter...")

        adapter_dir = Path(f"/data/lora_adapters/{niche}")
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Download PEFT adapter files from the work_dir we used
        # Training script saves to output/adapter/ subdirectory
        client.download_file(
            instance_id,
            f"{work_dir}/output/adapter/adapter_model.safetensors",
            str(adapter_dir / "adapter_model.safetensors")
        )
        client.download_file(
            instance_id,
            f"{work_dir}/output/adapter/adapter_config.json",
            str(adapter_dir / "adapter_config.json")
        )

        logger.info(f"Adapter downloaded to {adapter_dir}")

        # ================================================================
        # Step 7: Convert PEFT to GGUF
        # ================================================================
        logger.info("Step 7/8: Converting PEFT adapter to GGUF format...")

        # Run conversion directly (not as subtask to avoid Celery issues)
        gguf_result = _convert_peft_to_gguf(niche, str(adapter_dir))

        if gguf_result.get("status") != "success":
            logger.warning(f"GGUF conversion failed: {gguf_result.get('message')}")
            # Continue anyway - PEFT adapter is still useful for some use cases
        else:
            logger.info(f"GGUF adapter created: {gguf_result.get('gguf_path')}")

        # ================================================================
        # Step 8: Restart llama-server with new adapter
        # ================================================================
        logger.info("Step 8/8: Restarting llama-server with new adapter...")

        # Call restart directly (not as subtask)
        restart_result = _restart_llama_server_with_lora(niche)

        if restart_result.get("status") != "success":
            logger.warning(f"llama-server restart failed: {restart_result.get('message')}")
        else:
            logger.info("llama-server restarted with new LoRA adapter")

        # Calculate costs
        elapsed_hours = (time.time() - start_time) / 3600
        estimated_cost = elapsed_hours * offers[0]["price_per_hour"]

        # Create TrainedModel record
        # Get subreddits for this niche from config
        from config.automation_config import get_automation_config
        config = get_automation_config()
        subreddits = config.get_subreddits_for_niche(niche)

        with get_db_context() as db:
            trained_model = TrainedModel(
                name=f"{niche}_vastai_{datetime.now().strftime('%Y%m%d_%H%M')}",
                base_model="mistralai/Mistral-Small-24B-Instruct-2501",
                adapter_path=str(adapter_dir),
                source_subreddits=subreddits,  # Required JSON field
                training_samples=export_stats["exported"],
                training_duration_seconds=int(time.time() - start_time),
                hyperparameters={  # Store LoRA config in JSON field
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "learning_rate": 2e-4,
                    "epochs": 3,
                    "quantization": "4bit"
                },
                niche=niche,
            )
            db.add(trained_model)
            db.commit()

            # Update TrainingJob as completed
            update_job_status(
                "completed",
                completed_at=datetime.utcnow(),
                total_samples=export_stats["exported"],
                final_train_loss=0.0,  # Placeholder - could parse from Vast.ai output
                progress_percent=100.0
            )

            return {
                "status": "success",
                "niche": niche,
                "provider": provider,
                "instance_id": instance_id,
                "training_samples": export_stats["exported"],
                "training_duration_hours": elapsed_hours,
                "estimated_cost": f"${estimated_cost:.2f}",
                "adapter_path": str(adapter_dir),
                "gguf_conversion": gguf_result.get("status"),
                "llama_server_restart": restart_result.get("status"),
                "trained_model_id": trained_model.id
            }

    except Exception as e:
        logger.error(f"Cloud training failed: {e}")
        traceback.print_exc()
        update_job_status("failed", str(e), completed_at=datetime.utcnow())
        return {
            "status": "error",
            "message": str(e),
            "instance_id": instance_id,
            "elapsed_seconds": int(time.time() - start_time)
        }

    finally:
        # Cleanup instance unless caller asked us to leave it running.
        # The "leave it" path exists so a human can SSH in and diagnose
        # silent training failures (e.g. zero-captions-generated mode we
        # hit on the motivation run) before the instance is destroyed and
        # the evidence is gone. Pass auto_destroy=False from the Celery
        # task call site, then manually destroy via vastai CLI / API
        # once you've either pulled the adapter or are sure it's a
        # write-off. See docs/training_runbook.md.
        if instance_id and auto_destroy:
            try:
                logger.info(f"Cleaning up Vast.ai instance {instance_id}...")
                client.destroy_instance(instance_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup instance {instance_id}: {e}")
        elif instance_id and not auto_destroy:
            logger.warning(
                f"auto_destroy=False — leaving Vast.ai instance {instance_id} running. "
                f"Destroy manually with: vastai destroy instance {instance_id}"
            )


@celery_app.task(bind=True, name='tasks.training_tasks.download_cloud_adapter', queue='maintenance')
def download_cloud_adapter(self, niche: str, adapter_url: str):
    """
    Download a trained LoRA adapter from cloud storage.

    Args:
        niche: Niche category
        adapter_url: URL to download the PEFT adapter from

    Returns:
        Dict with status and local adapter path
    """
    import subprocess
    from pathlib import Path

    logger.info(f"Downloading adapter for {niche} from {adapter_url}")

    try:
        # Create adapter directory
        adapter_dir = Path(f"/data/lora_adapters/{niche}")
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Download the adapter files
        result = subprocess.run(
            ["curl", "-L", "-o", str(adapter_dir / "adapter.tar.gz"), adapter_url],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            return {"status": "error", "message": f"Download failed: {result.stderr}"}

        # Extract the archive
        result = subprocess.run(
            ["tar", "-xzf", str(adapter_dir / "adapter.tar.gz"), "-C", str(adapter_dir)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {"status": "error", "message": f"Extraction failed: {result.stderr}"}

        # Clean up tar file
        (adapter_dir / "adapter.tar.gz").unlink()

        return {
            "status": "success",
            "adapter_path": str(adapter_dir),
            "niche": niche
        }

    except Exception as e:
        logger.error(f"Adapter download failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.convert_adapter_to_gguf', queue='maintenance')
def convert_adapter_to_gguf(self, niche: str, peft_path: str = None):
    """
    Convert a PEFT LoRA adapter to GGUF format for llama.cpp.

    This runs on the host machine via SSH since llama.cpp is installed there.

    Args:
        niche: Niche category
        peft_path: Path to PEFT adapter (defaults to /data/lora_adapters/{niche})

    Returns:
        Dict with status and GGUF adapter path
    """
    from utils.llama_server_control import HOST_LLAMA_CPP_DIR, run_host_ssh

    if peft_path is None:
        peft_path = f"/data/lora_adapters/{niche}"

    output_path = f"/data/lora_adapters/{niche}/adapter.gguf"

    logger.info(f"Converting PEFT adapter to GGUF: {peft_path} -> {output_path}")

    try:
        # Run conversion on the LLM host via SSH. The convert_lora_to_gguf.py
        # script is part of llama.cpp; host address/credentials come from
        # settings.LLM_HOST_*.
        ssh_cmd = (
            f"cd {HOST_LLAMA_CPP_DIR} && "
            f"python convert_lora_to_gguf.py {peft_path} --outfile {output_path}"
        )

        returncode, _stdout, stderr = run_host_ssh(ssh_cmd, timeout=300)

        if returncode != 0:
            logger.error(f"Conversion failed: {stderr}")
            return {"status": "error", "message": stderr}

        logger.info(f"Conversion successful: {output_path}")
        return {
            "status": "success",
            "gguf_path": output_path,
            "niche": niche
        }

    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.restart_llama_server', queue='maintenance')
def restart_llama_server(self, niche: str = None, lora_path: str = None):
    """
    Restart llama-server with an optional LoRA adapter.

    This manages the llama-server process on the host machine.

    Args:
        niche: Niche category (used to find adapter at /data/lora_adapters/{niche}/adapter.gguf)
        lora_path: Direct path to LoRA adapter GGUF file (overrides niche)

    Returns:
        Dict with status and server info
    """
    try:
        from utils.llama_server_control import restart_server, get_server_status

        success = restart_server(lora_adapter=lora_path, niche=niche)

        if success:
            status = get_server_status()
            return {
                "status": "success",
                "server_status": status,
                "niche": niche,
                "lora_loaded": niche is not None or lora_path is not None
            }
        else:
            return {
                "status": "error",
                "message": "Failed to restart llama-server"
            }

    except Exception as e:
        logger.error(f"llama-server restart failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.generate_with_llama_server', queue='maintenance')
def generate_with_llama_server(
    self,
    prompt: str,
    niche: str = None,
    max_tokens: int = 512,
    temperature: float = 0.8
):
    """
    Generate captions using llama-server (Mistral-Small-24B with optional LoRA).

    This uses the external llama-server running on the host machine.

    Args:
        prompt: Generation prompt
        niche: Niche category (for context, not for model switching)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dict with generated text and metadata
    """
    try:
        from utils.llama_server_control import generate, is_server_running

        if not is_server_running():
            return {
                "status": "error",
                "message": "llama-server is not running"
            }

        # Format prompt for Mistral instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]\n"

        result = generate(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if result:
            return {
                "status": "success",
                "text": result,
                "niche": niche,
                "model": "Mistral-Small-24B-Instruct"
            }
        else:
            return {
                "status": "error",
                "message": "Generation failed"
            }

    except Exception as e:
        logger.error(f"llama-server generation failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, name='tasks.training_tasks.rescore_captions_strict_batch', queue='gpu_llm')
def rescore_captions_strict_batch(
    self,
    limit: int = 100,
    min_current_score: float = None,
    max_current_score: float = None
):
    """
    Rescore existing captions using the new strict scoring system.

    This applies:
    1. Pre-rejection filter for obviously low-quality captions
    2. Stricter LLM scoring prompt with explicit penalties
    3. Rule-based deductions for specific issues

    Args:
        limit: Maximum number of captions to rescore
        min_current_score: Only rescore captions with current score >= this value
        max_current_score: Only rescore captions with current score <= this value

    Returns:
        Dict with status, score changes, and processing results
    """
    start_time = time.time()

    from database.db import get_db_context
    from database.models import GeneratedCaption
    from utils.generation_postprocessor import score_caption_strict

    try:
        # Load LLM once for batch processing
        from scrapers.caption_postprocessor import _load_llm
        model, tokenizer = _load_llm(force=True)
        if model is None:
            return {"status": "error", "message": "LLM not available for scoring"}

        with get_db_context() as db:
            # Build query with optional filters
            query = db.query(GeneratedCaption).filter(
                GeneratedCaption.caption_text != None,
                GeneratedCaption.caption_text != ''
            )

            if min_current_score is not None:
                query = query.filter(GeneratedCaption.quality_score >= min_current_score)
            if max_current_score is not None:
                query = query.filter(GeneratedCaption.quality_score <= max_current_score)

            captions = query.limit(limit).all()

            if not captions:
                return {
                    "status": "success",
                    "message": "No captions found matching criteria",
                    "processed": 0
                }

            total = len(captions)
            processed = 0
            rejected = 0
            score_changes = []

            logger.info(f"Starting strict rescoring for {total} captions")

            for i, caption in enumerate(captions):
                try:
                    old_score = caption.quality_score or 0

                    # Use the new strict scoring
                    result = score_caption_strict(
                        caption.caption_text,
                        model=model,
                        tokenizer=tokenizer
                    )

                    if result['rejected']:
                        # Caption failed pre-rejection filter
                        new_score = 0
                        rejected += 1
                        change_reason = f"rejected: {result['reject_reason']}"
                    else:
                        new_score = result['final_score']
                        change_reason = f"deductions: {result.get('deductions', 'none')}"

                    # Update score
                    caption.quality_score = new_score

                    # Track significant changes
                    score_diff = new_score - old_score
                    if abs(score_diff) >= 5:  # Track changes of 5+ points
                        score_changes.append({
                            "caption_id": caption.id,
                            "old_score": round(old_score, 1),
                            "new_score": round(new_score, 1),
                            "change": round(score_diff, 1),
                            "reason": change_reason
                        })

                    processed += 1

                    # Update progress
                    if (i + 1) % 10 == 0:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'current': i + 1,
                                'total': total,
                                'rejected': rejected,
                                'score_changes': len(score_changes)
                            }
                        )
                        db.commit()  # Commit periodically

                except Exception as e:
                    logger.error(f"Error rescoring caption {caption.id}: {e}")
                    continue

            db.commit()

            duration = int(time.time() - start_time)
            logger.info(f"Strict rescoring completed: {processed} captions, {rejected} rejected, in {duration}s")

            # Calculate score distribution stats
            avg_change = sum(c['change'] for c in score_changes) / len(score_changes) if score_changes else 0
            dropped_below_80 = len([c for c in score_changes if c['old_score'] >= 80 and c['new_score'] < 80])
            dropped_below_50 = len([c for c in score_changes if c['old_score'] >= 50 and c['new_score'] < 50])

            return {
                "status": "success",
                "processed": processed,
                "rejected": rejected,
                "significant_changes": len(score_changes),
                "average_score_change": round(avg_change, 1),
                "dropped_below_80": dropped_below_80,
                "dropped_below_50": dropped_below_50,
                "duration_seconds": duration,
                "sample_changes": score_changes[:20]  # Return first 20 examples
            }

    except Exception as e:
        logger.error(f"Strict rescoring batch failed: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }
