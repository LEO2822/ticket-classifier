# Ticket Classifier — Fine-tuned Qwen3-0.6B for Retail Banking Intent Classification

A production-style, end-to-end **LLM fine-tuning project** that turns a 0.6B-parameter base model into a domain-specific customer-support ticket router. The fine-tuned model classifies incoming retail-banking support tickets into one of ~27 intents and returns a **strict JSON object** (`intent`, `confidence`, `reason`) suitable for downstream routing systems.

> **Held-out evaluation: 100% accuracy and 100% JSON-schema validity on 50 unseen tickets.**
> See [`outputs/eval_results.json`](outputs/eval_results.json) for the full per-sample report.

---

## Highlights

- **Parameter-efficient fine-tuning** of `Qwen/Qwen3-0.6B` using **LoRA** (PEFT) — only ~0.5% of weights trained.
- **Supervised fine-tuning** via Hugging Face **TRL `SFTTrainer`** on the public [Bitext retail-banking dataset](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset).
- **Trained locally on Apple Silicon (M-series GPU / MPS backend)** — no cloud GPU required.
- **Structured JSON output** with a heuristic confidence score and a one-line reason, ready for deterministic downstream parsing.
- **GGUF (f16) export** of the merged model for fast local inference via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).
- Clean, modular Python package layout (`data/`, `training/`, `inference/`, `conversion/`) — every stage is independently runnable.

---

## Architecture

```
┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│  Bitext Dataset  │───▶│  Chat-style SFT   │───▶│  LoRA Adapter      │
│  (retail banking)│    │  (TRL SFTTrainer) │    │  (PEFT, r=16)      │
└──────────────────┘    └───────────────────┘    └─────────┬──────────┘
                                                           │
                                                           ▼
                                                 ┌────────────────────┐
                                                 │  Merge into base   │
                                                 │  Qwen3-0.6B        │
                                                 └─────────┬──────────┘
                                                           │
                                                           ▼
                                                 ┌────────────────────┐
                                                 │  GGUF (f16) export │
                                                 │  → llama-cpp-python│
                                                 └─────────┬──────────┘
                                                           │
                                                           ▼
                                                 ┌────────────────────┐
                                                 │  classify(ticket)  │
                                                 │  → JSON response   │
                                                 └────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Fine-tuning | `transformers`, `trl` (SFTTrainer), `peft` (LoRA), `accelerate` |
| Dataset | `datasets` — Bitext retail-banking corpus |
| Training hardware | Apple Silicon GPU via PyTorch MPS backend |
| Inference | `llama-cpp-python` over a GGUF (f16) checkpoint |
| Tooling | `uv` (project/dep manager), `ruff` (lint + format), Python 3.13 |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Method | Supervised Fine-Tuning (SFT) with LoRA |
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | 512 |
| Max steps | 800 |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.03 |
| Batch size × grad accumulation | 4 × 4 (effective 16) |
| Optimizer | AdamW (torch) |
| Eval / save interval | every 100 / 200 steps |
| Train/eval split | 90 / 10, seed 42 |

---

## Evaluation Results

Computed on a 50-sample held-out slice with the GGUF-quantized model running locally through `llama-cpp-python`.

| Metric | Value |
|---|---|
| Accuracy | **100%** (50 / 50) |
| Valid JSON rate | **100%** |
| Schema-valid rate (correct keys + types) | **100%** |
| Accuracy on schema-valid predictions | **100%** |

Full per-sample predictions: [`outputs/eval_results.json`](outputs/eval_results.json).

---

## Example Output

```python
from ticket_classifier import classify

ticket = "I forgot my password and I can't log in to my account. Can you help me reset it?"
print(classify(ticket))
```

```json
{
  "intent": "get_password",
  "confidence": 0.94,
  "reason": "Customer request relates to get password based on: I forgot my password and I can't log in to my account..."
}
```

---

## Project Layout

```
ticket_classifier/
├── config.py                 # paths, hyperparameters, system prompt
├── utils.py                  # think-tag stripping for Qwen3 reasoning output
├── data/
│   └── prep.py               # dataset load + chat-style formatting + confidence heuristic
├── training/
│   ├── trainer.py            # SFT + LoRA training loop (TRL)
│   └── evaluator.py          # held-out eval with JSON-schema validation
├── conversion/
│   ├── merge.py              # merge LoRA adapter into base model
│   └── gguf.py               # convert merged model → GGUF (f16)
└── inference/
    └── classifier.py         # llama-cpp-python inference → structured JSON
main.py                       # one-line demo
outputs/eval_results.json     # held-out evaluation report
```

---

## Reproducing the Pipeline

> Trained weights (`models/`) are **not** included in the repo — they total ~4.3 GB. Re-train locally with the commands below.

```bash
# 1. Install deps (uv handles the lockfile)
uv sync

# 2. Train the LoRA adapter (Apple Silicon recommended)
uv run ticket_classifier/training/trainer.py

# 3. Merge adapter into base model
uv run ticket_classifier/conversion/merge.py

# 4. Export merged model to GGUF f16
uv run ticket_classifier/conversion/gguf.py

# 5. Evaluate on held-out split
uv run ticket_classifier/training/evaluator.py

# 6. Try a single classification
uv run main.py
```

---

## Notes

- The confidence score is a transparent heuristic computed at dataset-build time from instruction length, intent-keyword overlap, hedging language, typo density, and question form — making confidence reproducible and explainable rather than a raw model logit.
- The system prompt enforces JSON-only output with a fixed schema; combined with the small model size, this yields fast, deterministic, structured predictions.
- Qwen3 emits `<think>...</think>` reasoning blocks that are stripped post-generation (see `utils.strip_think_tags`).

---

## License

MIT.
