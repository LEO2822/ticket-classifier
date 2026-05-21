"""Merge LoRA adapter with base model and save locally."""

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from ticket_classifier.config import BASE_MODEL, ADAPTER_DIR, MERGED_DIR, get_logger

logger = get_logger(__name__)


def merge_and_save():
    """Load base model + LoRA adapter, merge, and save full model."""
    logger.info("Loading base model: %s...", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # Find the latest checkpoint directory inside ADAPTER_DIR
    checkpoints = sorted(
        ADAPTER_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {ADAPTER_DIR}")
    adapter_path = checkpoints[-1]

    logger.info("Loading LoRA adapter from: %s...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s...", MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    size_gb = sum(f.stat().st_size for f in MERGED_DIR.glob("*")) / 1e9
    logger.info("Merged model saved to %s (%.2f GB)", MERGED_DIR, size_gb)


if __name__ == "__main__":
    merge_and_save()
