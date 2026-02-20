"""Load the GGUF model via llama-cpp-python and classify customer support tickets."""

import json
import sys

from llama_cpp import Llama

from .config import GGUF_MODEL, SYSTEM_PROMPT
from .utils import strip_think_tags

_llm = None


def load_model():
    """Load the GGUF model (cached after first call)."""
    global _llm

    if _llm is not None:
        return _llm

    if not GGUF_MODEL.exists():
        print(f"ERROR: GGUF model not found at {GGUF_MODEL}")
        print("Run the GGUF conversion first.")
        sys.exit(1)

    print(f"Loading GGUF model from {GGUF_MODEL}...")
    _llm = Llama(
        model_path=str(GGUF_MODEL),
        n_ctx=40960,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Model loaded.")
    return _llm


def classify(ticket_text: str, intents: list[str] | None = None) -> dict:
    """Classify a customer support ticket.

    Args:
        ticket_text: The customer's message/ticket.
        intents: List of allowed intents. If None, model uses its trained intents.

    Returns:
        Dict with keys: intent, confidence, reason
    """
    llm = load_model()

    intents_str = ", ".join(sorted(intents)) if intents else "all trained intents"
    system_msg = SYSTEM_PROMPT.format(intents=intents_str)

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": ticket_text},
        ],
        max_tokens=128,
        temperature=0.0,
    )

    raw_output = response["choices"][0]["message"]["content"].strip()
    raw_output = strip_think_tags(raw_output)

    try:
        return json.loads(raw_output)
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
