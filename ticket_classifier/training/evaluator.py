"""Evaluate the fine-tuned model on held-out data.

Metrics:
  - accuracy: intent matches ground truth (extracted from any format)
  - valid_json_rate: output parses as valid JSON
  - schema_pass_rate: JSON has all 3 keys (intent, confidence, reason)
  - accuracy_on_schema: correct intent among schema-valid outputs
"""

import json
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ticket_classifier.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    EVAL_MAX_INPUT_LENGTH,
    INFERENCE_MAX_TOKENS,
    MERGED_DIR,
    OUTPUTS_DIR,
    get_device,
    get_logger,
)
from ticket_classifier.data.prep import load_and_prepare_dataset
from ticket_classifier.utils import strip_think_tags

logger = get_logger(__name__)

REQUIRED_SCHEMA_KEYS = {"intent", "confidence", "reason"}


def load_model(
    use_merged: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load the fine-tuned model (merged or adapter)."""
    device = get_device()
    logger.info("Using device: %s", device)

    if use_merged and MERGED_DIR.exists() and any(MERGED_DIR.iterdir()):
        logger.info("Loading merged model from %s...", MERGED_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_DIR),
            dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(MERGED_DIR), trust_remote_code=True
        )
    else:
        logger.info("Loading base model + LoRA adapter...")
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, device


def extract_intent_from_text(text: str) -> str | None:
    """Try to extract intent from raw text even if JSON is malformed."""
    text = text.strip()
    # Try JSON parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "intent" in data:
            return data["intent"]
    except json.JSONDecodeError:
        pass

    # Fallback: look for "intent": "..." pattern
    match = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)

    return None


def generate_prediction(model, tokenizer, messages: list[dict], device: str) -> str:
    """Generate model prediction for a set of messages."""
    # Build prompt from system + user messages only (no assistant)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=EVAL_MAX_INPUT_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=INFERENCE_MAX_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return strip_think_tags(raw)


def run_evaluation(max_samples: int | None = None):
    """Run evaluation and report metrics."""
    model, tokenizer, device = load_model()
    dataset_split, _intents = load_and_prepare_dataset()
    held_out = dataset_split["test"]

    if max_samples:
        held_out = held_out.select(range(min(max_samples, len(held_out))))

    logger.info("Running inference on %d held-out samples...", len(held_out))

    total = 0
    correct_any = 0
    valid_json = 0
    schema_pass = 0
    correct_on_schema = 0

    results = []

    for i, example in enumerate(tqdm(held_out, desc="Inference")):
        messages = example["messages"]

        # Ground truth from assistant message
        gt_text = messages[-1]["content"]
        gt_data = json.loads(gt_text)
        gt_intent = gt_data["intent"]

        # Generate prediction
        prediction = generate_prediction(model, tokenizer, messages, device)
        total += 1

        # Metric 1: accuracy (extract intent from any format)
        pred_intent = extract_intent_from_text(prediction)
        if pred_intent == gt_intent:
            correct_any += 1

        # Metric 2: valid_json_rate
        is_valid_json = False
        parsed = None
        try:
            parsed = json.loads(prediction)
            is_valid_json = True
            valid_json += 1
        except json.JSONDecodeError:
            pass

        # Metric 3: schema_pass_rate
        is_schema_valid = False
        if is_valid_json and isinstance(parsed, dict):
            if REQUIRED_SCHEMA_KEYS.issubset(parsed.keys()):
                is_schema_valid = True
                schema_pass += 1

                # Metric 4: accuracy_on_schema
                if parsed["intent"] == gt_intent:
                    correct_on_schema += 1

        results.append(
            {
                "index": i,
                "ground_truth": gt_intent,
                "prediction": prediction,
                "pred_intent": pred_intent,
                "valid_json": is_valid_json,
                "schema_valid": is_schema_valid,
                "correct": pred_intent == gt_intent,
            }
        )

    # Compute final metrics
    metrics = {
        "total_samples": total,
        "accuracy": round(correct_any / total, 4) if total else 0,
        "valid_json_rate": round(valid_json / total, 4) if total else 0,
        "schema_pass_rate": round(schema_pass / total, 4) if total else 0,
        "accuracy_on_schema": round(correct_on_schema / schema_pass, 4)
        if schema_pass
        else 0,
    }

    # Log results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.2f%%", f"{k:25s}", v * 100)
        else:
            logger.info("  %s: %s", f"{k:25s}", v)
    logger.info("=" * 50)

    # Save to file
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"metrics": metrics, "samples": results[:50]}
    report_path = OUTPUTS_DIR / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Full report saved to %s", report_path)

    return metrics


if __name__ == "__main__":
    import sys

    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_evaluation(max_samples=max_samples)
