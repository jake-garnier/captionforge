"""
Vast.ai Training Script for Mistral-Small-24B QLoRA Fine-tuning

This script runs on a Vast.ai RTX 4090 (24GB) instance to train a LoRA adapter.
Uses Unsloth for 2x faster training and 70% less VRAM.

Usage (on Vast.ai instance):
    python train_vastai.py --niche motivation --data-path /workspace/data/captions.jsonl

Output:
    - PEFT adapter saved to /workspace/output/adapter/
    - Training metrics logged to /workspace/output/training_log.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Check for Unsloth (preferred) or fall back to standard transformers
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("Using Unsloth for optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not found, using standard transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset


# Model configuration
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # Auto-detect (float16 for A10G/4090)
LOAD_IN_4BIT = True

# LoRA configuration
LORA_R = 32  # Rank
LORA_ALPHA = 64  # Alpha (2x rank is common)
LORA_DROPOUT = 0.05
# Target all linear layers for best quality
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training configuration
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01


def format_prompt(text: str, niche: str) -> str:
    """Format training example with Mistral instruction template.

    The instruction is intentionally minimal — the LoRA learns the niche's
    voice from the examples themselves, and generation-time prompts
    (tasks/bg_first_generation.py) supply the scene grounding.
    """
    # Mistral-Small uses standard chat template
    return f"<s>[INST] Generate a {niche} caption. [/INST]\n{text}</s>"


def load_training_data(data_path: str, niche: str, tokenizer):
    """Load and tokenize training data from JSONL file."""
    print(f"Loading training data from {data_path}")

    # Load JSONL dataset
    dataset = load_dataset("json", data_files={"train": data_path})["train"]
    print(f"Loaded {len(dataset)} training examples")

    # Format prompts
    def format_examples(examples):
        texts = [format_prompt(text, niche) for text in examples["text"]]
        return {"text": texts}

    dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Split for validation (5%)
    split = tokenized.train_test_split(test_size=0.05, seed=42)

    return split["train"], split["test"]


def load_model_unsloth():
    """Load model with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=42,
    )

    return model, tokenizer


def load_model_standard():
    """Load model with standard transformers + PEFT."""
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer


def train(args):
    """Main training function."""
    print("=" * 80)
    print(f"VAST.AI TRAINING - Mistral-Small-24B QLoRA")
    print(f"Niche: {args.niche}")
    print(f"Data: {args.data_path}")
    print("=" * 80)

    start_time = time.time()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/5] Loading model...")
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth()
    else:
        model, tokenizer = load_model_standard()

    tokenizer.pad_token = tokenizer.eos_token
    model.print_trainable_parameters()

    # Load data
    print("\n[2/5] Loading and tokenizing data...")
    train_dataset, eval_dataset = load_training_data(
        args.data_path, args.niche, tokenizer
    )
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Training arguments
    print("\n[3/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer
    print("\n[4/5] Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    train_result = trainer.train()

    # Save final model
    print("\n[5/5] Saving adapter...")
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    end_time = time.time()
    duration = end_time - start_time

    # Log training results
    training_log = {
        "niche": args.niche,
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "final_train_loss": train_result.training_loss,
        "completed_at": datetime.utcnow().isoformat(),
        "adapter_path": str(adapter_path),
    }

    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"Final train loss: {train_result.training_loss:.4f}")
    print(f"Adapter saved to: {adapter_path}")
    print(f"Log saved to: {log_path}")

    return training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter on Vast.ai")
    parser.add_argument("--niche", required=True, help="Niche category (e.g., motivation)")
    parser.add_argument("--data-path", required=True, help="Path to JSONL training data")
    parser.add_argument("--output-dir", default="/workspace/output", help="Output directory")

    args = parser.parse_args()

    train(args)
