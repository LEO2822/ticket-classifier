"""Dataset loading, confidence heuristic, and chat-style formatting."""

import json
import re
from collections import Counter

from datasets import load_dataset, DatasetDict

from ticket_classifier.config import (
    CONFIDENCE_BASE,
    CONFIDENCE_HEDGING_PENALTY,
    CONFIDENCE_LONG_TEXT_BOOST,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    CONFIDENCE_MULTI_KEYWORD_BOOST,
    CONFIDENCE_QUESTION_PENALTY,
    CONFIDENCE_SHORT_TEXT_PENALTY,
    CONFIDENCE_SHORT_WORD_PENALTY,
    CONFIDENCE_SINGLE_KEYWORD_BOOST,
    DATASET_NAME,
    DATASET_SPLIT,
    LONG_TEXT_THRESHOLD,
    REASON_TEXT_TRUNCATE_LENGTH,
    SEED,
    SHORT_TEXT_THRESHOLD,
    SHORT_WORD_RATIO_THRESHOLD,
    SYSTEM_PROMPT,
    TEST_SIZE,
    get_logger,
)

logger = get_logger(__name__)


def compute_confidence(instruction: str, intent: str) -> float:
    """Heuristic confidence based on instruction clarity and intent alignment."""
    text = instruction.lower()
    words = text.split()
    length = len(words)

    conf = CONFIDENCE_BASE

    # Explicit intent keywords boost confidence
    intent_keywords = intent.replace("_", " ").split()
    keyword_hits = sum(1 for kw in intent_keywords if kw in text)
    if keyword_hits >= 2:
        conf += CONFIDENCE_MULTI_KEYWORD_BOOST
    elif keyword_hits >= 1:
        conf += CONFIDENCE_SINGLE_KEYWORD_BOOST

    # Longer, more detailed -> higher confidence
    if length > LONG_TEXT_THRESHOLD:
        conf += CONFIDENCE_LONG_TEXT_BOOST
    elif length < SHORT_TEXT_THRESHOLD:
        conf -= CONFIDENCE_SHORT_TEXT_PENALTY

    # Hedging / uncertainty language
    hedging = [
        "maybe",
        "i think",
        "possibly",
        "not sure",
        "might",
        "perhaps",
        "i guess",
        "kind of",
        "sort of",
    ]
    if any(h in text for h in hedging):
        conf -= CONFIDENCE_HEDGING_PENALTY

    # Typos or very informal language (simple proxy: high ratio of short words)
    if length > 3:
        short_ratio = sum(1 for w in words if len(w) <= 2) / length
        if short_ratio > SHORT_WORD_RATIO_THRESHOLD:
            conf -= CONFIDENCE_SHORT_WORD_PENALTY

    # Questions tend to be slightly less explicit
    if text.strip().endswith("?"):
        conf -= CONFIDENCE_QUESTION_PENALTY

    return round(min(max(conf, CONFIDENCE_MIN), CONFIDENCE_MAX), 2)


def generate_reason(instruction: str, intent: str) -> str:
    """Generate a short reason string for the classification."""
    readable_intent = intent.replace("_", " ")
    # Clean template placeholders for the reason
    clean_text = re.sub(r"\{\{.*?\}\}", "mentioned entity", instruction)
    if len(clean_text) > REASON_TEXT_TRUNCATE_LENGTH + 3:
        clean_text = clean_text[:REASON_TEXT_TRUNCATE_LENGTH] + "..."
    return f"Customer request relates to {readable_intent} based on: {clean_text}"


def format_example(example: dict[str, str], all_intents: list[str]) -> dict:
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
    logger.info("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Loaded %d rows", len(ds))

    # Discover all intents
    intent_counts = Counter(ds["intent"])
    all_intents = sorted(intent_counts.keys())
    logger.info("Found %d intents:", len(all_intents))
    for intent, count in intent_counts.most_common():
        logger.info("  %s: %d", intent, count)

    # Format into chat messages
    logger.info("Formatting dataset into chat-style messages...")
    ds_formatted = ds.map(
        lambda ex: format_example(ex, all_intents),
        remove_columns=ds.column_names,
        desc="Formatting",
    )

    # Train/eval split
    logger.info(
        "Splitting into train (%.0f%%) / eval (%.0f%%)...",
        (1 - TEST_SIZE) * 100,
        TEST_SIZE * 100,
    )
    split = ds_formatted.train_test_split(test_size=TEST_SIZE, seed=SEED)
    logger.info("  Train: %d examples", len(split["train"]))
    logger.info("  Eval:  %d examples", len(split["test"]))

    return split, all_intents


if __name__ == "__main__":
    split, intents = load_and_prepare_dataset()
    # Preview one example
    sample = split["train"][0]
    logger.info("── Sample training example ──")
    for msg in sample["messages"]:
        logger.info("[%s]", msg["role"].upper())
        logger.info("%s", msg["content"][:200])
