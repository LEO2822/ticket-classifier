"""Merge LoRA adapter with base model and save locally."""

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import BASE_MODEL, ADAPTER_DIR, MERGED_DIR


def merge_and_save():
    """Load base model + LoRA adapter, merge, and save full model."""
    print(f"Loading base model: {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # Find the latest checkpoint directory inside ADAPTER_DIR
    checkpoints = sorted(ADAPTER_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {ADAPTER_DIR}")
    adapter_path = checkpoints[-1]

    print(f"Loading LoRA adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to {MERGED_DIR}")
    print(f"  Model size: {sum(f.stat().st_size for f in MERGED_DIR.glob('*')) / 1e9:.2f} GB")


if __name__ == "__main__":
    merge_and_save()
