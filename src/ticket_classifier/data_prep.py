"""Dataset loading, confidence heuristic, and chat-style formatting."""

import json
import re
from collections import Counter

from datasets import load_dataset, DatasetDict

from .config import (
    DATASET_NAME,
    DATASET_SPLIT,
    SYSTEM_PROMPT,
    TEST_SIZE,
    SEED,
)


def compute_confidence(instruction: str, intent: str) -> float:
    """Heuristic confidence based on instruction clarity and intent alignment."""
    text = instruction.lower()
    words = text.split()
    length = len(words)

    conf = 0.90

    # Explicit intent keywords boost confidence
    intent_keywords = intent.replace("_", " ").split()
    keyword_hits = sum(1 for kw in intent_keywords if kw in text)
    if keyword_hits >= 2:
        conf += 0.06
    elif keyword_hits >= 1:
        conf += 0.03

    # Longer, more detailed → higher confidence
    if length > 15:
        conf += 0.03
    elif length < 5:
        conf -= 0.10

    # Hedging / uncertainty language
    hedging = ["maybe", "i think", "possibly", "not sure", "might", "perhaps",
               "i guess", "kind of", "sort of"]
    if any(h in text for h in hedging):
        conf -= 0.08

    # Typos or very informal language (simple proxy: high ratio of short words)
    if length > 3:
        short_ratio = sum(1 for w in words if len(w) <= 2) / length
        if short_ratio > 0.5:
            conf -= 0.04

    # Questions tend to be slightly less explicit
    if text.strip().endswith("?"):
        conf -= 0.02

    return round(min(max(conf, 0.70), 0.99), 2)


def generate_reason(instruction: str, intent: str) -> str:
    """Generate a short reason string for the classification."""
    readable_intent = intent.replace("_", " ")
    # Clean template placeholders for the reason
    clean_text = re.sub(r"\{\{.*?\}\}", "mentioned entity", instruction)
    if len(clean_text) > 60:
        clean_text = clean_text[:57] + "..."
    return f"Customer request relates to {readable_intent} based on: {clean_text}"


def format_example(example: dict, all_intents: list[str]) -> dict:
    """Convert a raw dataset row into chat-style messages for SFT."""
    instruction = example["instruction"]
    intent = example["intent"]

    confidence = compute_confidence(instruction, intent)
    reason = generate_reason(instruction, intent)

    assistant_response = json.dumps(
        {"intent": intent, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )

    intents_str = ", ".join(sorted(all_intents))
    system_msg = SYSTEM_PROMPT.format(intents=intents_str)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant_response},
    ]

    return {"messages": messages}


def load_and_prepare_dataset() -> tuple[DatasetDict, list[str]]:
    """Load dataset, discover intents, format, and split."""
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Loaded {len(ds)} rows")

    # Discover all intents
    intent_counts = Counter(ds["intent"])
    all_intents = sorted(intent_counts.keys())
    print(f"Found {len(all_intents)} intents:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count}")

    # Format into chat messages
    print("\nFormatting dataset into chat-style messages...")
    ds_formatted = ds.map(
        lambda ex: format_example(ex, all_intents),
        remove_columns=ds.column_names,
        desc="Formatting",
    )

    # Train/eval split
    print(f"Splitting into train ({1 - TEST_SIZE:.0%}) / eval ({TEST_SIZE:.0%})...")
    split = ds_formatted.train_test_split(test_size=TEST_SIZE, seed=SEED)
    print(f"  Train: {len(split['train'])} examples")
    print(f"  Eval:  {len(split['test'])} examples")

    return split, all_intents


if __name__ == "__main__":
    split, intents = load_and_prepare_dataset()
    # Preview one example
    sample = split["train"][0]
    print("\n── Sample training example ──")
    for msg in sample["messages"]:
        print(f"[{msg['role'].upper()}]")
        print(msg["content"][:200])
        print()
