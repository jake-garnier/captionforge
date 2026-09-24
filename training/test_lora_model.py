"""
Test the trained LoRA model by generating sample captions
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("Loading base model...")

    # Load base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "training/models/mistral-motivation-lora")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "Generate a motivation caption about transformation:",
        "Generate a motivation caption about perseverance:",
        "Generate a motivation caption about finishing what you started:",
    ]

    print("\n" + "="*80)
    print("GENERATING CAPTIONS")
    print("="*80 + "\n")

    for i, prompt_text in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"CAPTION {i}")
        print(f"{'='*80}\n")

        # Format prompt
        prompt = f"<s>[INST] {prompt_text} [/INST]\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        print(f"Prompt: {prompt_text}")
        print("\nGenerating...\n")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated caption (remove prompt)
        caption = generated_text.split("[/INST]")[-1].strip()

        print(f"Generated Caption ({len(caption)} chars):")
        print("-" * 80)
        print(caption)
        print("-" * 80)

        # Count slides if multi-slide
        slide_count = caption.count(" *|* ") + 1
        if slide_count > 1:
            print(f"\n(Multi-slide caption: {slide_count} slides)")

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
