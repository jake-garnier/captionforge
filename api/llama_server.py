"""
llama-server Management API

Endpoints for controlling the llama-server process on the host machine.
Used for generation with Mistral-Small-24B and LoRA adapters.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llama-server", tags=["llama-server"])


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    niche: Optional[str] = None


class StartServerRequest(BaseModel):
    niche: Optional[str] = None
    lora_path: Optional[str] = None


@router.get("/status")
def get_llama_server_status():
    """Get the current status of llama-server."""
    try:
        from utils.llama_server_control import get_server_status, LLAMA_SERVER_URL

        status = get_server_status()
        status["url"] = LLAMA_SERVER_URL

        return status
    except Exception as e:
        logger.error(f"Error getting llama-server status: {e}")
        return {
            "running": False,
            "error": str(e)
        }


@router.post("/start")
def start_llama_server(request: StartServerRequest = None):
    """
    Start llama-server on the host machine.

    Args:
        niche: Optional niche to load LoRA adapter for
        lora_path: Optional direct path to LoRA adapter
    """
    try:
        from utils.llama_server_control import start_server, is_server_running

        if is_server_running():
            return {
                "status": "already_running",
                "message": "llama-server is already running"
            }

        niche = request.niche if request else None
        lora_path = request.lora_path if request else None

        success = start_server(lora_adapter=lora_path, niche=niche)

        if success:
            return {
                "status": "started",
                "niche": niche,
                "lora_loaded": niche is not None or lora_path is not None
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start llama-server")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting llama-server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
def stop_llama_server():
    """Stop llama-server on the host machine."""
    try:
        from utils.llama_server_control import stop_server, is_server_running

        if not is_server_running():
            return {
                "status": "already_stopped",
                "message": "llama-server is not running"
            }

        success = stop_server()

        if success:
            return {"status": "stopped"}
        else:
            raise HTTPException(status_code=500, detail="Failed to stop llama-server")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping llama-server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
def restart_llama_server(request: StartServerRequest = None):
    """
    Restart llama-server, optionally with a different LoRA adapter.

    Args:
        niche: Optional niche to load LoRA adapter for
        lora_path: Optional direct path to LoRA adapter
    """
    try:
        from utils.llama_server_control import restart_server, get_server_status

        niche = request.niche if request else None
        lora_path = request.lora_path if request else None

        success = restart_server(lora_adapter=lora_path, niche=niche)

        if success:
            status = get_server_status()
            return {
                "status": "restarted",
                "server_status": status,
                "niche": niche,
                "lora_loaded": niche is not None or lora_path is not None
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to restart llama-server")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting llama-server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
def generate_text(request: GenerateRequest):
    """
    Generate text using llama-server.

    Args:
        prompt: The prompt to generate from
        max_tokens: Maximum tokens to generate (default 512)
        temperature: Sampling temperature (default 0.8)
        niche: Optional niche context (for logging only)
    """
    try:
        from utils.llama_server_control import generate, is_server_running

        if not is_server_running():
            raise HTTPException(status_code=503, detail="llama-server is not running")

        result = generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        if result:
            return {
                "status": "success",
                "text": result,
                "niche": request.niche,
                "model": "Mistral-Small-24B-Instruct"
            }
        else:
            raise HTTPException(status_code=500, detail="Generation failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def llama_server_health():
    """Quick health check for llama-server."""
    try:
        from utils.llama_server_control import is_server_running

        running = is_server_running()
        return {
            "status": "healthy" if running else "unhealthy",
            "running": running
        }
    except Exception as e:
        return {
            "status": "error",
            "running": False,
            "error": str(e)
        }
