"""
Convert PEFT LoRA adapter to GGUF format for llama.cpp

This script converts a Hugging Face PEFT adapter to GGUF format
that can be loaded by llama-server with the --lora flag.

Usage:
    python convert_peft_to_gguf.py --adapter-path /path/to/adapter --output adapter.gguf

Requirements:
    - llama.cpp repo cloned at ~/llama.cpp (for convert_lora_to_gguf.py)
    - Python packages: safetensors, numpy
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_llama_cpp() -> Path:
    """Find llama.cpp installation."""
    # Check common locations
    possible_paths = [
        Path.home() / "llama.cpp",
        Path("/workspace/llama.cpp"),  # Vast.ai
        Path("/opt/llama.cpp"),
        Path.cwd() / "llama.cpp",
    ]

    for path in possible_paths:
        if (path / "convert_lora_to_gguf.py").exists():
            return path

    raise FileNotFoundError(
        "Could not find llama.cpp. Please clone it: "
        "git clone https://github.com/ggerganov/llama.cpp.git"
    )


def convert_peft_to_gguf(adapter_path: str, output_path: str, base_model: str = None) -> bool:
    """
    Convert PEFT adapter to GGUF format.

    Args:
        adapter_path: Path to PEFT adapter directory (contains adapter_model.safetensors)
        output_path: Output path for GGUF file
        base_model: Base model name (for weight scaling info). If None, auto-detected.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    # Validate adapter exists
    safetensors_file = adapter_path / "adapter_model.safetensors"
    if not safetensors_file.exists():
        # Try bin format
        safetensors_file = adapter_path / "adapter_model.bin"
        if not safetensors_file.exists():
            print(f"Error: No adapter found at {adapter_path}")
            return False

    # Find llama.cpp converter
    llama_cpp_path = find_llama_cpp()
    converter = llama_cpp_path / "convert_lora_to_gguf.py"

    print(f"Using converter: {converter}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")

    # Build command
    cmd = [
        sys.executable,
        str(converter),
        str(adapter_path),
        "--outfile", str(output_path),
    ]

    if base_model:
        cmd.extend(["--base", base_model])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"Conversion successful!")
            print(f"Output: {output_path}")
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"Size: {size_mb:.1f} MB")
            return True
        else:
            print(f"Conversion failed with code {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Conversion timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT adapter to GGUF")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to PEFT adapter directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name (optional, for metadata)"
    )

    args = parser.parse_args()

    success = convert_peft_to_gguf(
        args.adapter_path,
        args.output,
        args.base_model
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
