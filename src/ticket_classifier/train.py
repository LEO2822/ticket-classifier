"""Local SFT training on Apple MPS (M-series GPU)."""

import os

# MPS memory management - set before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import (
    BASE_MODEL,
    ADAPTER_DIR,
    LOGS_DIR,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
    WARMUP_RATIO,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    LOGGING_STEPS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)
from .data_prep import load_and_prepare_dataset


def get_device() -> str:
    """Detect best available device."""
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
        return "mps"
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("WARNING: No GPU detected, falling back to CPU (will be very slow)")
        return "cpu"


def train():
    """Run SFT training with LoRA on local hardware."""
    device = get_device()

    # Load and prepare dataset
    dataset_split, _intents = load_and_prepare_dataset()
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"\nLoading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32 if device == "mps" else torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # Training configuration
    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        max_length=MAX_SEQ_LENGTH,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        # Checkpointing
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        # Logging
        logging_dir=str(LOGS_DIR),
        logging_steps=LOGGING_STEPS,
        # Optimization
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        # MPS compatibility
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        # Misc
        seed=42,
        report_to="none",
    )

    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    # Save final adapter
    print(f"\nSaving LoRA adapter to {ADAPTER_DIR}...")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print("Adapter saved.")

    return trainer


if __name__ == "__main__":
    train()
