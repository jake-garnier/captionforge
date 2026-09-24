"""
Train LoRA adapter for Mistral-7B on motivation caption dataset
Simplified version using standard Trainer

GPU Assignment: Uses BOTH GPUs for training (disable scraper during training!)
- Container Device 0 = RTX 4060 Ti (8GB)
- Container Device 1 = RTX 2080 Ti (11GB)
Total: ~19GB VRAM available for training
"""
import os
# Use both GPUs for training - model will be distributed across them
# IMPORTANT: Disable scraper during training to avoid GPU conflicts!

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import time

def main():
    print("="*80)
    print("LORA TRAINING - Mistral-7B Motivation Caption Generation")
    print("="*80)

    # Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4-bit quantization
    print("\n[2/7] Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False  # Required for gradient checkpointing
    model.config.pretraining_tp = 1

    # Prepare model for k-bit training (enables input gradients for quantized models)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("\n[3/7] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\n[4/7] Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": "training/data/captions_train.jsonl",
        "validation": "training/data/captions_val.jsonl",
    })

    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")

    # Tokenize dataset
    print("\n[5/7] Tokenizing dataset...")
    def tokenize(examples):
        prompts = [
            f"<s>[INST] Generate a motivation caption: [/INST]\n{text}</s>"
            for text in examples['text']
        ]
        return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)

    # Training arguments
    print("\n[6/7] Setting up training...")
    training_args = TrainingArguments(
        output_dir="training/models/mistral-motivation-lora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
    )

    # Train
    print("\n[7/7] Starting training...")
    print("="*80)
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    duration = end_time - start_time

    # Save final model
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    trainer.save_model()
    print(f"\n✓ Model saved to: training/models/mistral-motivation-lora/")
    print(f"✓ Training duration: {duration:.4f} seconds ({duration/60:.2f} minutes)")
    print(f"✓ Final training loss: {trainer.state.log_history[-2]['loss']}")

    # Get validation loss
    eval_results = trainer.evaluate()
    print(f"✓ Validation loss: {eval_results['eval_loss']}")

if __name__ == "__main__":
    main()
