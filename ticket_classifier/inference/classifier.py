"""Load the GGUF model via llama-cpp-python and classify customer support tickets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse

from ticket_classifier.config import (
    GGUF_MODEL,
    INFERENCE_CONTEXT_LENGTH,
    INFERENCE_MAX_TOKENS,
    SYSTEM_PROMPT,
    get_logger,
)
from ticket_classifier.utils import strip_think_tags

logger = get_logger(__name__)


class TicketClassifier:
    """Classify customer support tickets using a GGUF model."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = Path(model_path) if model_path else GGUF_MODEL
        self._llm: Llama | None = None

    def load(self) -> None:
        """Load the GGUF model into memory.

        Raises:
            FileNotFoundError: If the GGUF model file does not exist.
        """
        if self._llm is not None:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {self._model_path}. "
                "Run the GGUF conversion first."
            )

        logger.info("Loading GGUF model from %s ...", self._model_path)
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=INFERENCE_CONTEXT_LENGTH,
            n_gpu_layers=-1,
            verbose=False,
        )
        logger.info("Model loaded.")

    def _ensure_loaded(self) -> Llama:
        """Lazy-load the model if it has not been loaded yet."""
        if self._llm is None:
            self.load()
        assert self._llm is not None  # noqa: S101
        return self._llm

    def classify(
        self, ticket_text: str, intents: list[str] | None = None
    ) -> dict[str, Any]:
        """Classify a customer support ticket.

        Args:
            ticket_text: The customer's message/ticket.
            intents: List of allowed intents. If None, the model uses its
                trained intents.

        Returns:
            Dict with keys: intent, confidence, reason.

        Raises:
            ValueError: If *ticket_text* is empty or whitespace-only.
        """
        if not ticket_text or not ticket_text.strip():
            raise ValueError("ticket_text must not be empty or whitespace-only")

        llm = self._ensure_loaded()

        intents_str = ", ".join(sorted(intents)) if intents else "all trained intents"
        system_msg = SYSTEM_PROMPT.format(intents=intents_str)

        response = cast(
            CreateChatCompletionResponse,
            llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": ticket_text},
                ],
                max_tokens=INFERENCE_MAX_TOKENS,
                temperature=0.0,
                stream=False,
            ),
        )

        try:
            content = response["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("Model returned empty content")
            raw_output = content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("Unexpected model response structure: %s", exc)
            return {
                "raw_output": str(response),
                "error": "Unexpected response structure",
            }

        raw_output = strip_think_tags(raw_output)

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"raw_output": raw_output, "error": "Failed to parse JSON"}

    def unload(self) -> None:
        """Release the loaded model from memory."""
        self._llm = None


# ── Backward-compatible module-level API ────────────────────────────────

_default_classifier = TicketClassifier()


def classify(ticket_text: str, intents: list[str] | None = None) -> dict[str, Any]:
    """Module-level convenience wrapper around :class:`TicketClassifier`."""
    return _default_classifier.classify(ticket_text, intents=intents)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run -m ticket_classifier.inference.classifier 'your ticket text'"
        )
        sys.exit(1)

    ticket = " ".join(sys.argv[1:])
    print(f"Ticket: {ticket}\n")
    result = classify(ticket)
    print(json.dumps(result, indent=2))
