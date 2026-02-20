"""Load the fine-tuned model and classify customer support tickets."""

import json
import os
import sys
from .utils import strip_think_tags

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MERGED_DIR, SYSTEM_PROMPT


# Cached model/tokenizer
_model = None
_tokenizer = None
_device = None


def load_model():
    """Load the merged fine-tuned model (cached after first call)."""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model from {MERGED_DIR}...")
    _tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR), trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_DIR),
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(_device)
    _model.eval()  # noqa: torch inference mode, not python eval()

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print(f"Model loaded on {_device}")
    return _model, _tokenizer, _device


def classify(ticket_text: str, intents: list[str] | None = None) -> dict:
    """Classify a customer support ticket.

    Args:
        ticket_text: The customer's message/ticket.
        intents: List of allowed intents. If None, model uses its trained intents.

    Returns:
        Dict with keys: intent, confidence, reason
    """
    model, tokenizer, device = load_model()

    intents_str = ", ".join(sorted(intents)) if intents else "all trained intents"
    system_msg = SYSTEM_PROMPT.format(intents=intents_str)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ticket_text},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=448).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_output = strip_think_tags(raw_output)

    try:
        result = json.loads(raw_output)
        return result
    except json.JSONDecodeError:
        return {"raw_output": raw_output, "error": "Failed to parse JSON"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run -m src.ticket_classifier.inference 'your ticket text here'")
        sys.exit(1)

    ticket = " ".join(sys.argv[1:])
    print(f"Ticket: {ticket}\n")
    result = classify(ticket)
    print(json.dumps(result, indent=2))
