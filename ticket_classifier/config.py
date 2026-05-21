"""Training configuration, constants, and shared utilities."""

import logging
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "adapter"
MERGED_DIR = MODELS_DIR / "merged"
GGUF_DIR = MODELS_DIR / "gguf"
GGUF_MODEL = GGUF_DIR / "ticket-router-f16.gguf"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ── Model & Dataset ───────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-0.6B"
DATASET_NAME = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
DATASET_SPLIT = "train"

# ── Training Hyperparameters ──────────────────────────────────────────
MAX_SEQ_LENGTH = 512
MAX_STEPS = 800
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_STEPS = 100
SAVE_STEPS = 200
LOGGING_STEPS = 10

# ── LoRA ──────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Data Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.10
SEED = 42

# ── Confidence Heuristic ─────────────────────────────────────────────
CONFIDENCE_BASE = 0.90
CONFIDENCE_MULTI_KEYWORD_BOOST = 0.06
CONFIDENCE_SINGLE_KEYWORD_BOOST = 0.03
CONFIDENCE_LONG_TEXT_BOOST = 0.03
CONFIDENCE_SHORT_TEXT_PENALTY = 0.10
CONFIDENCE_HEDGING_PENALTY = 0.08
CONFIDENCE_SHORT_WORD_PENALTY = 0.04
CONFIDENCE_QUESTION_PENALTY = 0.02
CONFIDENCE_MIN = 0.70
CONFIDENCE_MAX = 0.99

LONG_TEXT_THRESHOLD = 15
SHORT_TEXT_THRESHOLD = 5
SHORT_WORD_RATIO_THRESHOLD = 0.5

# ── Reason Generation ────────────────────────────────────────────────
REASON_TEXT_TRUNCATE_LENGTH = 57

# ── Inference / Evaluation ───────────────────────────────────────────
INFERENCE_CONTEXT_LENGTH = 40960
INFERENCE_MAX_TOKENS = 128
EVAL_MAX_INPUT_LENGTH = 448

# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a customer support ticket classifier for a retail banking service. \
Classify the given ticket into exactly one intent.

Respond with a JSON object containing exactly these three keys:
- "intent": one of the allowed intents listed below (string)
- "confidence": your confidence score between 0.0 and 1.0 (float)
- "reason": a brief one-sentence explanation (string)

Output ONLY the JSON object. No extra text, no markdown fences.

Allowed intents:
{intents}"""


# ── Shared Utilities ─────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def configure_mps_environment() -> None:
    """Set MPS memory management env vars. Must be called before importing torch."""
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# Eagerly configure MPS so any module importing config before torch is safe.
configure_mps_environment()


def get_device() -> str:
    """Detect best available compute device (MPS > CUDA > CPU)."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
