"""
llama-server Control Utility

Manages the llama-server process on the host machine for LLM inference.
Supports starting/stopping the server with different LoRA adapters.

The server runs on the host machine (not in Docker) and is accessible at:
- From containers: http://host.docker.internal:1234
- Externally: http://<llm-host-ip>:1234

Process control (start/stop/restart) is done over SSH to the host. The
host address and credentials come from config/settings.py:
  LLM_HOST_IP, LLM_HOST_SSH_USER, and either LLM_HOST_SSH_KEY_PATH
  (preferred, `ssh -i`) or LLM_HOST_SSH_PASSWORD (via sshpass).
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Optional
import requests

from config.settings import settings

# Configuration
LLAMA_SERVER_HOST = os.environ.get("LLAMA_SERVER_HOST", "host.docker.internal")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "1234"))
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

# Paths on the host machine. Override via env if your llama.cpp checkout or
# model directory lives elsewhere (must match scripts/llama-server.service).
HOST_LLAMA_CPP_DIR = os.environ.get("LLM_HOST_LLAMA_CPP_DIR", "/opt/llama.cpp")
HOST_MODELS_DIR = os.environ.get("LLM_HOST_MODELS_DIR", "/opt/models")
HOST_LORA_BASE_PATH = os.environ.get("LLM_HOST_LORA_BASE_PATH", "/data/lora_adapters")
HOST_LLAMA_SERVER_BIN = os.environ.get(
    "LLM_HOST_LLAMA_SERVER_BIN", f"{HOST_LLAMA_CPP_DIR}/build/bin/llama-server"
)

# GPU split ratio (RTX 2080 Ti 11GB : RTX 4060 Ti 8GB).
# Note: 0.58/0.42 was previously chosen to balance the model across both
# cards, but after a VLM→LLM profile swap GPU 0 sometimes can't allocate
# the KV cache (CUDA fragmentation in the llama.cpp allocator). Sliding
# more onto GPU 1 leaves headroom on GPU 0 for the post-load allocations.
TENSOR_SPLIT = "0.50,0.50"
N_GPU_LAYERS = 99  # Offload all layers

# Server profiles — two different models (LLM for text, VLM for vision)
# that are mutually exclusive on the same GPU pool. We stop-and-start
# when switching between tagging (stage 1) and generation (stage 2/3).
PROFILES = {
    "llm": {
        "model": f"{HOST_MODELS_DIR}/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        "mmproj": None,
        "ctx_size": 2048,
        "default_port": 1234,
    },
    "vlm": {
        # Qwen2.5-VL-7B Q4_K_M — fast, fits with massive headroom, the
        # established default for production tagging.
        "model": f"{HOST_MODELS_DIR}/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj": f"{HOST_MODELS_DIR}/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
        "ctx_size": 8192,  # room for multi-image input
        "default_port": 1234,  # same port; profiles are mutually exclusive
    },
    "vlm-internvl3": {
        # InternVL3-14B Q6_K (~12 GB weights, 0.7 GB mmproj). Different
        # model family from Qwen, ~2x params at near-FP16 precision.
        # Targeted upgrade vs Qwen2.5-VL-7B-Q4 for fine-grained
        # disambiguation of subjects, activities and settings.
        # Rejected the "32B" path because the Qwen2.5-VL-32B variants
        # OOM the vision encoder workspace on this dual-GPU pool. InternVL3-14B-Q6 leaves ~5 GB inference
        # headroom (vs ~0.1 GB for the failed 32B-Q3 config).
        #
        # Required llama.cpp upgrade: tag 7458 had mtmd_helper_eval
        # bugs that hit ~50% of multimodal calls. Built fresh from
        # a newer commit the failure rate dropped to 0.
        #
        # ctx=16384 + 0.45/0.55 split: shifting weights toward CUDA1
        # (2080 Ti, 11 GB) frees CUDA0 (4060 Ti, 8 GB) which carries the
        # mmproj. With 50/50 ctx=8192 the 2080 Ti was the bottleneck
        # (94% used, 637 MiB free). Tilting to 0.45/0.55 should give
        # CUDA0 more headroom for the doubled KV cache at ctx=16384.
        # InternVL3 emits ~768 vision tokens per frame, so ctx=16384
        # comfortably fits 6 frames worth of visual context plus prompt.
        "model": f"{HOST_MODELS_DIR}/InternVL3-14B-Instruct-Q6_K.gguf",
        "mmproj": f"{HOST_MODELS_DIR}/mmproj-InternVL3-14B-f16.gguf",
        "ctx_size": 16384,
        "tensor_split": "0.45,0.55",
        "default_port": 1234,
    },
}

# Backward-compat aliases — existing code reads HOST_MODEL_PATH / CTX_SIZE
HOST_MODEL_PATH = PROFILES["llm"]["model"]
CTX_SIZE = PROFILES["llm"]["ctx_size"]


def is_server_running() -> bool:
    """Check if llama-server is responding."""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_loaded_model() -> Optional[dict]:
    """Get info about currently loaded model."""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                return data["data"][0]
        return None
    except requests.exceptions.RequestException:
        return None


def wait_for_server(timeout: int = 60) -> bool:
    """Wait for server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        if is_server_running():
            return True
        time.sleep(2)
    return False


class HostSSHNotConfigured(RuntimeError):
    """Raised when no SSH credential for the LLM host is configured."""


def build_host_ssh_command(remote_cmd: str) -> tuple[list[str], dict]:
    """
    Build the argv (and environment) for running ``remote_cmd`` on the LLM host.

    Authentication, in order of preference:
      1. ``settings.LLM_HOST_SSH_KEY_PATH`` -> ``ssh -i <key>`` (non-interactive)
      2. ``settings.LLM_HOST_SSH_PASSWORD`` -> ``sshpass -e ssh ...`` (password is
         passed through the SSHPASS env var, never on the command line)

    Raises:
        HostSSHNotConfigured: if neither credential is set.
    """
    target = f"{settings.LLM_HOST_SSH_USER}@{settings.LLM_HOST_IP}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    env = os.environ.copy()

    if settings.LLM_HOST_SSH_KEY_PATH:
        argv = [
            "ssh", "-i", settings.LLM_HOST_SSH_KEY_PATH,
            "-o", "BatchMode=yes", *ssh_opts,
            target, remote_cmd,
        ]
    elif settings.LLM_HOST_SSH_PASSWORD:
        env["SSHPASS"] = settings.LLM_HOST_SSH_PASSWORD
        argv = [
            "sshpass", "-e",
            "ssh", "-o", "PreferredAuthentications=password", *ssh_opts,
            target, remote_cmd,
        ]
    else:
        raise HostSSHNotConfigured(
            "No SSH credential for the LLM host: set LLM_HOST_SSH_KEY_PATH "
            "(preferred) or LLM_HOST_SSH_PASSWORD in the environment/.env "
            f"(host={settings.LLM_HOST_IP}, user={settings.LLM_HOST_SSH_USER})."
        )
    return argv, env


def run_host_ssh(remote_cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """
    Execute a command on the LLM host via SSH.

    Returns:
        (returncode, stdout, stderr). Timeouts and transport errors are
        reported as returncode -1 with the message in stderr.

    Raises:
        HostSSHNotConfigured: if no SSH credential is configured (this is a
        deployment error, so it is raised rather than swallowed).
    """
    argv, env = build_host_ssh_command(remote_cmd)
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"SSH command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def start_server(
    lora_adapter: Optional[str] = None,
    niche: Optional[str] = None,
    profile: str = "llm",
) -> bool:
    """
    Start llama-server on the host machine with the requested profile.

    Args:
        lora_adapter: Path to LoRA adapter GGUF file (on host). Only
            applicable to the "llm" profile.
        niche: Niche name to load adapter for (e.g., "motivation").
            If provided, looks for adapter at /data/lora_adapters/{niche}/adapter.gguf.
            Only applicable to the "llm" profile.
        profile: "llm" (Mistral-Small for generation/judge) or "vlm"
            (Qwen2.5-VL-7B for visual tagging/review). Profiles are
            mutually exclusive — you must stop_server() before starting
            a different profile.

    Returns:
        True if server started successfully
    """
    if profile not in PROFILES:
        print(f"Unknown profile: {profile}. Valid: {list(PROFILES)}")
        return False

    cfg = PROFILES[profile]

    # Per-profile tensor_split override for cases where the default
    # 50/50 split doesn't balance well — e.g. VLM profiles where the
    # mmproj loads disproportionately on CUDA0, so we want fewer
    # weight-layers there to leave room for vision-encoder workspace.
    tensor_split = cfg.get("tensor_split") or TENSOR_SPLIT

    # Build the command
    cmd_parts = [
        HOST_LLAMA_SERVER_BIN,
        "--model", cfg["model"],
        "--host", "0.0.0.0",
        "--port", str(LLAMA_SERVER_PORT),
        "--split-mode", "layer",
        "--tensor-split", tensor_split,
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--ctx-size", str(cfg["ctx_size"]),
    ]

    # Multimodal projector for VLM profiles
    if cfg.get("mmproj"):
        cmd_parts.extend(["--mmproj", cfg["mmproj"]])

    # LoRA only makes sense for the llm profile
    if profile == "llm":
        if niche and not lora_adapter:
            lora_adapter = f"{HOST_LORA_BASE_PATH}/{niche}/adapter.gguf"
        if lora_adapter:
            cmd_parts.extend(["--lora", lora_adapter])

    cmd = " ".join(cmd_parts)

    # Start server in background using nohup
    start_cmd = f"nohup {cmd} > /tmp/llama-server.log 2>&1 &"

    print(f"Starting llama-server (profile={profile})...")
    print(f"Command: {cmd}")

    try:
        returncode, stdout, stderr = run_host_ssh(start_cmd)
    except HostSSHNotConfigured as e:
        print(f"Failed to start server: {e}")
        return False

    if returncode != 0:
        print(f"Failed to start server: {stderr}")
        return False

    # Wait for server to be ready. Both profiles can take 2-3 min when
    # tensor-split across both GPUs is loading from cold cache; LoRA also
    # adds startup time. Be generous.
    wait_timeout = 300 if profile == "vlm" else 240
    print("Waiting for server to start...")
    if wait_for_server(timeout=wait_timeout):
        print("llama-server started successfully!")
        return True
    else:
        print("Server failed to start within timeout")
        return False


def stop_server() -> bool:
    """Stop llama-server on the host machine."""
    print("Stopping llama-server...")

    # Kill the process
    kill_cmd = "pkill -f 'llama-server' || true"
    try:
        run_host_ssh(kill_cmd)
    except HostSSHNotConfigured as e:
        print(f"Failed to stop server: {e}")
        return False

    # Wait a moment
    time.sleep(2)

    # Verify it's stopped
    if not is_server_running():
        print("llama-server stopped successfully")
        return True
    else:
        print("Warning: Server may still be running")
        return False


def restart_server(
    lora_adapter: Optional[str] = None,
    niche: Optional[str] = None,
    profile: str = "llm",
) -> bool:
    """Stop and start the server with optional new LoRA adapter / profile."""
    stop_server()
    time.sleep(2)
    return start_server(lora_adapter=lora_adapter, niche=niche, profile=profile)


def get_server_status() -> dict:
    """Get comprehensive server status."""
    status = {
        "running": False,
        "url": LLAMA_SERVER_URL,
        "model": None,
        "error": None
    }

    try:
        if is_server_running():
            status["running"] = True
            model_info = get_loaded_model()
            if model_info:
                status["model"] = model_info.get("id", "unknown")
        else:
            status["error"] = "Server not responding"
    except Exception as e:
        status["error"] = str(e)

    return status


def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    repetition_penalty: float = 1.1,
    top_p: float = 0.95,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
) -> Optional[str]:
    """
    Generate text using the llama-server API.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeated tokens (1.0 = no penalty, >1.0 = less repetition)
        top_p: Nucleus sampling threshold
        frequency_penalty: OpenAI-style frequency penalty
        presence_penalty: OpenAI-style presence penalty
        seed: RNG seed. None => random per call (the safe default for variety;
            without an explicit seed llama.cpp may use a fixed default and
            produce the same output for the same prompt).

    Returns:
        Generated text or None if error
    """
    if not is_server_running():
        print("Error: llama-server is not running")
        return None

    if seed is None:
        # Force a fresh seed per call so identical prompts don't collapse to
        # identical outputs. llama.cpp accepts -1 to mean "use a random seed",
        # but being explicit with our own random int is more portable across
        # server versions.
        import random as _random
        seed = _random.randint(1, 2**31 - 1)

    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat_penalty": repetition_penalty,  # llama.cpp uses repeat_penalty
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stop": ["</s>", "[/INST]"],
            },
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["text"].strip()
        else:
            print(f"Generation failed: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Generation error: {e}")
        return None


def chat_completion(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.8
) -> Optional[str]:
    """
    Chat completion using OpenAI-compatible API.

    Args:
        messages: List of message dicts with "role" and "content"
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Assistant's response or None if error
    """
    if not is_server_running():
        print("Error: llama-server is not running")
        return None

    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"Chat completion failed: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Chat completion error: {e}")
        return None


# For use as a module
__all__ = [
    "is_server_running",
    "get_loaded_model",
    "wait_for_server",
    "start_server",
    "stop_server",
    "restart_server",
    "get_server_status",
    "generate",
    "chat_completion",
    "build_host_ssh_command",
    "run_host_ssh",
    "HostSSHNotConfigured",
    "LLAMA_SERVER_URL",
    "HOST_LLAMA_CPP_DIR",
    "HOST_MODELS_DIR",
    "HOST_LORA_BASE_PATH",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llama_server_control.py [status|start|stop|restart|test]")
        sys.exit(1)

    action = sys.argv[1]

    if action == "status":
        status = get_server_status()
        print(f"Running: {status['running']}")
        print(f"URL: {status['url']}")
        if status['model']:
            print(f"Model: {status['model']}")
        if status['error']:
            print(f"Error: {status['error']}")

    elif action == "start":
        niche = sys.argv[2] if len(sys.argv) > 2 else None
        success = start_server(niche=niche)
        sys.exit(0 if success else 1)

    elif action == "stop":
        success = stop_server()
        sys.exit(0 if success else 1)

    elif action == "restart":
        niche = sys.argv[2] if len(sys.argv) > 2 else None
        success = restart_server(niche=niche)
        sys.exit(0 if success else 1)

    elif action == "test":
        if not is_server_running():
            print("Server not running!")
            sys.exit(1)

        result = generate("Hello, how are you?", max_tokens=50)
        if result:
            print(f"Response: {result}")
        else:
            print("Generation failed")
            sys.exit(1)

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
