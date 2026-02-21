"""Local SFT training on Apple MPS (M-series GPU)."""

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ticket_classifier.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LOGS_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    WARMUP_RATIO,
    get_device,
    get_logger,
)
from ticket_classifier.data.prep import load_and_prepare_dataset

logger = get_logger(__name__)


def train() -> SFTTrainer:
    """Run SFT training with LoRA on local hardware."""
    device = get_device()
    logger.info("Using device: %s", device)

    # Load and prepare dataset
    dataset_split, _intents = load_and_prepare_dataset()
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    logger.info("Loading tokenizer for %s...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s...", BASE_MODEL)
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

    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Training complete!")
    logger.info("  Total steps: %d", train_result.global_step)
    logger.info("  Training loss: %.4f", train_result.training_loss)

    # Save final adapter
    logger.info("Saving LoRA adapter to %s...", ADAPTER_DIR)
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    logger.info("Adapter saved.")

    return trainer


if __name__ == "__main__":
    train()
