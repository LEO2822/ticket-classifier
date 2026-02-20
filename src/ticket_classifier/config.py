"""Training configuration and constants."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "adapter"
MERGED_DIR = MODELS_DIR / "merged"
GGUF_DIR = MODELS_DIR / "gguf"
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
