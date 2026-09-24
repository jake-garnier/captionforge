"""
Vast.ai Client Wrapper

Uses the vastai CLI tool for reliable GPU instance management.

Usage:
    client = VastaiClient(api_key="your_api_key")
    instance = client.launch_training_instance()
    client.upload_file(instance["id"], "/local/data.jsonl", "/workspace/data.jsonl")
    client.execute_command(instance["id"], "python train.py")
    client.download_file(instance["id"], "/workspace/output", "/local/adapters/motivation")
    client.destroy_instance(instance["id"])
"""

import os
import subprocess
import time
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Default training image with CUDA and PyTorch
DEFAULT_IMAGE = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"

# SSH key path for Vast.ai (persistent across container restarts)
SSH_KEY_PATH = Path("/root/.ssh/vastai_key")


class VastaiClient:
    """
    High-level client for Vast.ai GPU instance management.

    Uses the vastai CLI tool for reliable operation.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vast.ai client.

        Args:
            api_key: Vast.ai API key. If not provided, uses VASTAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            raise ValueError("Vast.ai API key required. Set VASTAI_API_KEY or pass api_key.")

        # Ensure SSH key is set up for Vast.ai access
        self._ensure_ssh_key()

    def _ensure_ssh_key(self) -> bool:
        """
        Ensure SSH key exists and is registered with Vast.ai.

        Creates a new SSH key if none exists and registers it.
        """
        private_key = SSH_KEY_PATH
        public_key = SSH_KEY_PATH.with_suffix(".pub")

        # Create SSH key if it doesn't exist
        if not private_key.exists():
            logger.info("Creating SSH key for Vast.ai access...")
            private_key.parent.mkdir(parents=True, exist_ok=True)

            try:
                result = subprocess.run(
                    [
                        "ssh-keygen", "-t", "ed25519",
                        "-f", str(private_key),
                        "-N", "",  # No passphrase
                        "-C", "vastai-training"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.error(f"Failed to create SSH key: {result.stderr}")
                    return False

                # Fix permissions
                os.chmod(private_key, 0o600)
                logger.info(f"SSH key created: {private_key}")

            except Exception as e:
                logger.error(f"Error creating SSH key: {e}")
                return False

        # Check if key is registered with Vast.ai
        if not self._ensure_vastai_installed():
            return False

        # Check existing keys
        result = self._run_vastai(["show", "ssh-keys", "--raw"])
        try:
            keys = json.loads(result["stdout"]) if result["stdout"] else []
        except json.JSONDecodeError:
            keys = []

        # Look for our key
        pub_key_content = public_key.read_text().strip() if public_key.exists() else ""
        key_registered = any(
            pub_key_content in str(k.get("ssh_key", ""))
            for k in keys
        )

        if not key_registered and pub_key_content:
            logger.info("Registering SSH key with Vast.ai...")

            # Create SSH key in Vast.ai (positional argument, not flag)
            result = self._run_vastai([
                "create", "ssh-key",
                pub_key_content,  # Public key as positional arg
                "-y"  # Auto-confirm
            ])

            if result["returncode"] == 0:
                logger.info("SSH key registered with Vast.ai")
            else:
                logger.warning(f"Failed to register SSH key: {result['stderr']}")
                # Continue anyway - key might already exist with different name

        return True

    def _ssh_command(
        self,
        ssh_host: str,
        ssh_port: int,
        command: str,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute command on instance via SSH.

        Args:
            ssh_host: SSH host
            ssh_port: SSH port
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            Dict with stdout, stderr, exit_code
        """
        ssh_cmd = [
            "ssh",
            "-i", str(SSH_KEY_PATH),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=10",
            "-p", str(ssh_port),
            f"root@{ssh_host}",
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"SSH command timed out after {timeout}s",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }

    def _ensure_vastai_installed(self) -> bool:
        """Ensure vastai CLI is installed, install if missing."""
        try:
            result = subprocess.run(
                ["vastai", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.info("vastai CLI not found, installing...")
            try:
                # Install vastai with --no-deps to avoid dependency conflicts
                result = subprocess.run(
                    ["pip", "install", "--no-deps", "vastai>=0.5.0"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    logger.info("vastai CLI installed successfully")
                    return True
                else:
                    logger.error(f"Failed to install vastai: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"Error installing vastai: {e}")
                return False
        except Exception:
            return False

    def _run_vastai(self, args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        Run a vastai CLI command.

        Args:
            args: Command arguments (e.g., ["search", "offers", "gpu_name=RTX_4090"])
            timeout: Command timeout in seconds

        Returns:
            Dict with 'stdout', 'stderr', 'returncode'
        """
        # Ensure vastai is installed
        if not self._ensure_vastai_installed():
            return {
                "stdout": "",
                "stderr": "vastai CLI not installed and could not be installed",
                "returncode": -1
            }

        env = os.environ.copy()
        env["VAST_API_KEY"] = self.api_key

        cmd = ["vastai"] + args
        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def search_gpu_offers(
        self,
        gpu_name: str = "RTX_4090",
        max_price: float = 0.50,
        min_vram_gb: int = 24,
        min_disk_gb: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for available GPU offers.

        Args:
            gpu_name: GPU model name (e.g., "RTX_4090", "RTX_3090", "A100")
            max_price: Maximum hourly price in USD
            min_vram_gb: Minimum VRAM in GB
            min_disk_gb: Minimum disk space in GB

        Returns:
            List of available offers sorted by price
        """
        query = f"gpu_name={gpu_name} rented=false rentable=true"
        result = self._run_vastai(["search", "offers", query, "-o", "dph_total", "--raw"])

        if result["returncode"] != 0:
            logger.error(f"Search failed: {result['stderr']}")
            return []

        try:
            offers = json.loads(result["stdout"])

            # Filter by price and requirements
            filtered = []
            for offer in offers:
                try:
                    price = float(offer.get("dph_total", 999))
                    vram = float(offer.get("gpu_ram", 0)) / 1024  # Convert MB to GB
                    disk = float(offer.get("disk_space", 0))

                    reliability = float(offer.get("reliability2", 0))
                    inet_down = float(offer.get("inet_down", 0))

                    if price <= max_price and vram >= min_vram_gb and disk >= min_disk_gb and reliability >= 0.90:
                        filtered.append({
                            "id": offer.get("id"),
                            "gpu_name": offer.get("gpu_name"),
                            "num_gpus": offer.get("num_gpus", 1),
                            "gpu_ram_gb": vram,
                            "price_per_hour": price,
                            "disk_gb": disk,
                            "cuda_version": offer.get("cuda_max_good"),
                            "reliability": reliability,
                            "inet_down": inet_down,
                            "location": offer.get("geolocation", "Unknown"),
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping offer due to parse error: {e}")
                    continue

            # Sort by price
            filtered.sort(key=lambda x: x["price_per_hour"])

            logger.info(f"Found {len(filtered)} offers matching criteria (max ${max_price}/hr)")
            return filtered

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse offers: {e}")
            return []

    def launch_training_instance(
        self,
        gpu_name: str = "RTX_4090",
        image: str = DEFAULT_IMAGE,
        disk_gb: int = 100,
        max_price: float = 0.50
    ) -> Dict[str, Any]:
        """
        Launch a new training instance.

        Args:
            gpu_name: GPU model to use
            image: Docker image for the instance
            disk_gb: Disk space in GB
            max_price: Maximum hourly price

        Returns:
            Instance details dict with 'id', etc.
        """
        # First find cheapest offer
        offers = self.search_gpu_offers(gpu_name=gpu_name, max_price=max_price)
        if not offers:
            raise RuntimeError(f"No {gpu_name} instances available under ${max_price}/hr")

        offer_id = offers[0]["id"]
        logger.info(f"Creating instance from offer {offer_id} at ${offers[0]['price_per_hour']}/hr")

        # Create instance using CLI
        result = self._run_vastai([
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--ssh",
            "--direct",
            "--raw"
        ], timeout=120)

        if result["returncode"] != 0:
            raise RuntimeError(f"Failed to create instance: {result['stderr']}")

        # Parse result to get instance ID
        try:
            data = json.loads(result["stdout"])
            instance_id = data.get("new_contract") or data.get("instance_id") or data.get("id")
            if not instance_id:
                # Try to extract from text output
                match = re.search(r'(?:new_contract|instance)[:\s]+(\d+)', result["stdout"])
                if match:
                    instance_id = int(match.group(1))
                else:
                    raise RuntimeError(f"Could not find instance ID in response: {result['stdout'][:200]}")

            logger.info(f"Launched instance {instance_id}")
            return {"id": instance_id, "offer": offers[0]}

        except json.JSONDecodeError:
            # Try regex on raw output
            match = re.search(r'(?:new_contract|instance_id|started)[:\s]+(\d+)', result["stdout"])
            if match:
                instance_id = int(match.group(1))
                logger.info(f"Launched instance {instance_id}")
                return {"id": instance_id, "offer": offers[0]}
            raise RuntimeError(f"Failed to parse instance ID: {result['stdout'][:200]}")

    def get_instance_status(self, instance_id: int) -> Dict[str, Any]:
        """
        Get status of a running instance.

        Args:
            instance_id: Vast.ai instance ID

        Returns:
            Instance status dict with 'status', 'ssh_host', 'ssh_port', etc.
        """
        result = self._run_vastai(["show", "instance", str(instance_id), "--raw"])

        if result["returncode"] != 0:
            return {"status": "error", "message": result["stderr"]}

        try:
            data = json.loads(result["stdout"])
            return {
                "id": instance_id,
                "status": data.get("actual_status", "unknown"),
                "ssh_host": data.get("ssh_host"),
                "ssh_port": data.get("ssh_port"),
                "public_ipaddr": data.get("public_ipaddr"),
                "jupyter_url": data.get("jupyter_url"),
                "raw": data
            }
        except json.JSONDecodeError:
            return {"status": "unknown", "raw": result["stdout"]}

    def wait_for_ready(self, instance_id: int, timeout_seconds: int = 600) -> Dict[str, Any]:
        """
        Wait for instance to be ready for SSH.

        Args:
            instance_id: Vast.ai instance ID
            timeout_seconds: Maximum time to wait

        Returns:
            Instance status when ready

        Raises:
            TimeoutError: If instance doesn't become ready
        """
        start_time = time.time()
        poll_interval = 10

        while time.time() - start_time < timeout_seconds:
            status = self.get_instance_status(instance_id)

            actual_status = status.get("status", "")
            ssh_host = status.get("ssh_host")

            logger.info(f"Instance {instance_id} status: {actual_status}")

            if actual_status == "running" and ssh_host:
                # Wait for SSH to actually accept connections
                ssh_port = status.get("ssh_port")
                if ssh_port:
                    logger.info(f"Instance running, waiting for SSH to be ready at {ssh_host}:{ssh_port}...")
                    ssh_ready = False
                    for attempt in range(12):  # Up to 2 minutes of SSH retries
                        time.sleep(10)
                        test = self._ssh_command(ssh_host, ssh_port, "echo ready", timeout=15)
                        if test.get("exit_code") == 0 and "ready" in test.get("stdout", ""):
                            logger.info(f"SSH ready after {(attempt + 1) * 10}s")
                            ssh_ready = True
                            break
                        logger.info(f"SSH not ready yet (attempt {attempt + 1}/12): {test.get('stderr', '')[:100]}")
                    if not ssh_ready:
                        logger.warning("SSH did not become ready after 2 minutes, returning status anyway")
                return status

            if actual_status in ("error", "terminated", "exited"):
                raise RuntimeError(f"Instance failed with status: {actual_status}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Instance {instance_id} did not become ready within {timeout_seconds}s")

    def upload_file(self, instance_id: int, local_path: str, remote_path: str) -> bool:
        """
        Upload a file to the instance using SCP.

        Args:
            instance_id: Vast.ai instance ID
            local_path: Local file path
            remote_path: Remote destination path

        Returns:
            True if successful
        """
        # Get instance SSH info
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")

        if not ssh_host or not ssh_port:
            logger.error(f"Instance {instance_id} not ready for SSH")
            return False

        # Create remote directory first
        remote_dir = str(Path(remote_path).parent)
        mkdir_result = self._ssh_command(ssh_host, ssh_port, f"mkdir -p {remote_dir}", timeout=30)
        if mkdir_result["exit_code"] != 0:
            logger.error(f"Failed to create directory {remote_dir}: {mkdir_result['stderr']}")
            return False

        # Use SCP to upload file
        scp_cmd = [
            "scp",
            "-i", str(SSH_KEY_PATH),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
            local_path,
            f"root@{ssh_host}:{remote_path}"
        ]

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"SCP upload failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"SCP upload timed out")
            return False

        logger.info(f"Uploaded {local_path} to {instance_id}:{remote_path}")
        return True

    def download_file(self, instance_id: int, remote_path: str, local_path: str) -> bool:
        """
        Download a file from the instance using SCP.

        Args:
            instance_id: Vast.ai instance ID
            remote_path: Remote file path
            local_path: Local destination path

        Returns:
            True if successful
        """
        # Get instance SSH info
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")

        if not ssh_host or not ssh_port:
            logger.error(f"Instance {instance_id} not ready for SSH")
            return False

        # Create local directory
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)

        # Use SCP to download file (or directory with -r)
        scp_cmd = [
            "scp",
            "-r",  # Recursive for directories
            "-i", str(SSH_KEY_PATH),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
            f"root@{ssh_host}:{remote_path}",
            local_path
        ]

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"SCP download failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"SCP download timed out")
            return False

        logger.info(f"Downloaded {instance_id}:{remote_path} to {local_path}")
        return True

    def execute_command(
        self,
        instance_id: int,
        command: str,
        timeout_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute a command on the instance via vastai CLI.

        Note: vastai execute is non-blocking for long commands.
        For training, use execute_training_and_wait() instead.

        Args:
            instance_id: Vast.ai instance ID
            command: Command to execute
            timeout_seconds: Command timeout

        Returns:
            Dict with 'stdout', 'stderr', 'exit_code'
        """
        result = self._run_vastai([
            "execute", str(instance_id), command
        ], timeout=timeout_seconds)

        return {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["returncode"]
        }

    def execute_training_and_wait(
        self,
        instance_id: int,
        train_command: str,
        output_marker: str = "/workspace/output/adapter_model.safetensors",
        timeout_seconds: int = 14400,  # 4 hours default
        poll_interval: int = 60  # Check every minute
    ) -> Dict[str, Any]:
        """
        Execute training command via SSH and poll until output file exists.

        Uses SSH for running commands on the instance since vastai execute
        only works on stopped instances.

        Args:
            instance_id: Vast.ai instance ID
            train_command: Training command to execute
            output_marker: File path that indicates training completion
            timeout_seconds: Maximum time to wait for training
            poll_interval: Seconds between status checks

        Returns:
            Dict with 'stdout', 'stderr', 'exit_code', 'training_log'
        """
        # Get SSH connection details
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")

        if not ssh_host or not ssh_port:
            return {
                "stdout": "",
                "stderr": f"Could not get SSH details for instance {instance_id}",
                "exit_code": -1,
                "training_log": ""
            }

        logger.info(f"Starting training on instance {instance_id} ({ssh_host}:{ssh_port})")
        logger.info(f"Command: {train_command}")

        # Start training in background with output logging
        # Use /root/ for log file - more reliable than /workspace on Vast.ai
        log_file = "/root/training.log"
        bg_command = f"nohup bash -c '{train_command}' > {log_file} 2>&1 &"

        start_result = self._ssh_command(ssh_host, ssh_port, bg_command, timeout=60)

        if start_result["exit_code"] != 0:
            return {
                "stdout": start_result["stdout"],
                "stderr": f"Failed to start training: {start_result['stderr']}",
                "exit_code": -1,
                "training_log": ""
            }

        logger.info("Training started, polling for completion...")

        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            time.sleep(poll_interval)

            elapsed = int(time.time() - start_time)
            logger.info(f"Checking training status... ({elapsed}s elapsed)")

            # Check if output file exists
            check_cmd = f"test -f {output_marker} && echo 'COMPLETE' || echo 'RUNNING'"
            check_result = self._ssh_command(ssh_host, ssh_port, check_cmd, timeout=30)

            if "COMPLETE" in check_result["stdout"]:
                logger.info("Training completed - output file found!")

                # Get final training log
                log_result = self._ssh_command(
                    ssh_host, ssh_port, "cat /root/training.log", timeout=60
                )

                return {
                    "stdout": "Training completed successfully",
                    "stderr": "",
                    "exit_code": 0,
                    "training_log": log_result["stdout"]
                }

            # Check if training process is still running
            proc_check = self._ssh_command(
                ssh_host, ssh_port,
                "pgrep -f 'python.*train_vastai' > /dev/null && echo 'ALIVE' || echo 'DEAD'",
                timeout=30
            )

            if "DEAD" in proc_check["stdout"]:
                # Process died — check exit status and errors
                error_check = self._ssh_command(
                    ssh_host, ssh_port,
                    "tail -20 /root/training.log 2>/dev/null | grep -iE 'error:|exception:|traceback|cuda out of memory|RuntimeError|ImportError|ModuleNotFoundError|SyntaxError|killed' | grep -v 'Requirement already satisfied'",
                    timeout=30
                )

                if error_check["stdout"].strip():
                    log_result = self._ssh_command(
                        ssh_host, ssh_port, "tail -100 /root/training.log", timeout=60
                    )
                    return {
                        "stdout": "",
                        "stderr": f"Training error detected: {error_check['stdout']}",
                        "exit_code": 1,
                        "training_log": log_result["stdout"]
                    }

                # Process died but no errors in last 20 lines — might have finished
                # Give it one more cycle to check for output file
                logger.warning("Training process ended, checking for output on next cycle...")

            # Show progress from log (last few lines)
            progress_result = self._ssh_command(
                ssh_host, ssh_port,
                "tail -3 /root/training.log 2>/dev/null",
                timeout=30
            )

            if progress_result["stdout"].strip():
                for line in progress_result["stdout"].strip().split('\n'):
                    if line.strip():
                        logger.info(f"  Training: {line.strip()[:100]}")

        # Timeout reached
        log_result = self._ssh_command(
            ssh_host, ssh_port, "cat /root/training.log", timeout=60
        )

        return {
            "stdout": "",
            "stderr": f"Training timed out after {timeout_seconds}s",
            "exit_code": -1,
            "training_log": log_result["stdout"]
        }

    def destroy_instance(self, instance_id: int) -> bool:
        """
        Destroy an instance.

        Args:
            instance_id: Vast.ai instance ID

        Returns:
            True if successful
        """
        result = self._run_vastai(["destroy", "instance", str(instance_id)])

        if result["returncode"] != 0:
            logger.error(f"Destroy failed: {result['stderr']}")
            return False

        logger.info(f"Destroyed instance {instance_id}")
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        """
        List all active instances.

        Returns:
            List of instance dicts
        """
        result = self._run_vastai(["show", "instances", "--raw"])

        if result["returncode"] != 0:
            logger.error(f"List instances failed: {result['stderr']}")
            return []

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            return []
