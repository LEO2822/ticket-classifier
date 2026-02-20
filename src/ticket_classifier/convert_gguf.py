"""Convert merged HuggingFace model to GGUF format for Ollama/llama.cpp."""

import subprocess
import sys

from .config import MERGED_DIR, GGUF_DIR, PROJECT_ROOT

LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
GGUF_OUTPUT = GGUF_DIR / "ticket-router-q8_0.gguf"


def ensure_llama_cpp():
    """Clone llama.cpp if not present."""
    if LLAMA_CPP_DIR.exists():
        print(f"llama.cpp already present at {LLAMA_CPP_DIR}")
        return

    print("Cloning llama.cpp (needed for GGUF conversion)...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git",
         str(LLAMA_CPP_DIR)],
        check=True,
    )
    print("Installing llama.cpp Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         str(LLAMA_CPP_DIR / "requirements.txt")],
        check=True,
    )


def convert_to_gguf(quantize: str = "q8_0"):
    """Convert merged model to GGUF with optional quantization.

    Quantization options: f16, q8_0, q4_0, q4_k_m, q5_k_m
    """
    ensure_llama_cpp()

    if not MERGED_DIR.exists() or not any(MERGED_DIR.iterdir()):
        print(f"ERROR: No merged model found at {MERGED_DIR}")
        print("Run merge_model.py first.")
        sys.exit(1)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: convert_hf_to_gguf.py not found at {convert_script}")
        sys.exit(1)

    fp16_path = GGUF_DIR / "ticket-router-f16.gguf"

    # Step 1: Convert to f16 GGUF
    print(f"Converting {MERGED_DIR} to GGUF (f16)...")
    subprocess.run(
        [sys.executable, str(convert_script),
         str(MERGED_DIR),
         "--outfile", str(fp16_path),
         "--outtype", "f16"],
        check=True,
    )
    print(f"F16 GGUF saved: {fp16_path}")

    if quantize != "f16":
        # Step 2: Quantize
        # Build llama-quantize if needed
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            print("Building llama.cpp quantize tool...")
            build_dir = LLAMA_CPP_DIR / "build"
            build_dir.mkdir(exist_ok=True)
            subprocess.run(
                ["cmake", "-B", str(build_dir), "-S", str(LLAMA_CPP_DIR),
                 "-DGGML_METAL=ON"],
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", "llama-quantize",
                 "-j", "8"],
                check=True,
            )

        quantized_path = GGUF_DIR / f"ticket-router-{quantize}.gguf"
        print(f"Quantizing to {quantize}...")
        subprocess.run(
            [str(quantize_bin), str(fp16_path), str(quantized_path), quantize],
            check=True,
        )
        print(f"Quantized GGUF saved: {quantized_path}")

        # Optionally remove the large f16 file
        size_f16 = fp16_path.stat().st_size / 1e9
        size_q = quantized_path.stat().st_size / 1e9
        print(f"\nFile sizes:")
        print(f"  F16:         {size_f16:.2f} GB")
        print(f"  {quantize}:  {size_q:.2f} GB")

    print("\nTo use with Ollama:")
    print(f'  1. Create a Modelfile with: FROM {GGUF_DIR / f"ticket-router-{quantize}.gguf"}')
    print(f'  2. ollama create ticket-router -f Modelfile')
    print(f'  3. ollama run ticket-router')


if __name__ == "__main__":
    quant = sys.argv[1] if len(sys.argv) > 1 else "q8_0"
    convert_to_gguf(quantize=quant)
