"""Shared utilities for ticket classifier."""

import re


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (Qwen3 thinking tags)."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
