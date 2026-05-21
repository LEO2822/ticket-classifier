"""Convert merged HuggingFace model to GGUF (f16) format for Ollama/llama.cpp."""

import subprocess
import sys

from ticket_classifier.config import MERGED_DIR, GGUF_DIR, PROJECT_ROOT, get_logger

logger = get_logger(__name__)

LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
GGUF_OUTPUT = GGUF_DIR / "ticket-router-f16.gguf"


def ensure_llama_cpp():
    """Clone llama.cpp if not present."""
    if LLAMA_CPP_DIR.exists():
        logger.info("llama.cpp already present at %s", LLAMA_CPP_DIR)
        return

    logger.info("Cloning llama.cpp (needed for GGUF conversion)...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(LLAMA_CPP_DIR),
        ],
        check=True,
    )
    # gguf and sentencepiece are managed as dev deps in pyproject.toml;
    # skip installing llama.cpp's requirements.txt to avoid version conflicts.


def convert_to_gguf():
    """Convert merged model to f16 GGUF."""
    ensure_llama_cpp()

    if not MERGED_DIR.exists() or not any(MERGED_DIR.iterdir()):
        raise FileNotFoundError(
            f"No merged model found at {MERGED_DIR}. Run merge first."
        )

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}.")

    logger.info("Converting %s to GGUF (f16)...", MERGED_DIR)
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(MERGED_DIR),
            "--outfile",
            str(GGUF_OUTPUT),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    size = GGUF_OUTPUT.stat().st_size / 1e9
    logger.info("F16 GGUF saved: %s (%.2f GB)", GGUF_OUTPUT, size)

    logger.info("To use with Ollama:")
    logger.info("  1. Create a Modelfile with: FROM %s", GGUF_OUTPUT)
    logger.info("  2. ollama create ticket-router -f Modelfile")
    logger.info("  3. ollama run ticket-router")


if __name__ == "__main__":
    convert_to_gguf()
