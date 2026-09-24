"""
VNC Manager for interactive browser sessions.

Provides a virtual display (Xvfb) with VNC access via noVNC web interface.
This allows users to see and interact with Playwright browsers through their web browser.
"""
import os
import subprocess
import time
import logging
import signal
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# VNC configuration
VNC_DISPLAY = ":99"
VNC_PORT = 5900
NOVNC_PORT = 6080
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1100
SCREEN_DEPTH = 24


class VNCManager:
    """
    Manages Xvfb, x11vnc, and websockify for browser display.

    Usage:
        vnc = VNCManager()
        vnc.start()  # Start display server
        # ... run browser with DISPLAY=:99 ...
        vnc.stop()   # Clean up
    """

    def __init__(self):
        self.xvfb_process: Optional[subprocess.Popen] = None
        self.vnc_process: Optional[subprocess.Popen] = None
        self.websockify_process: Optional[subprocess.Popen] = None
        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if VNC session is running."""
        return self._active and self._check_processes()

    @property
    def display(self) -> str:
        """Get the X display to use for browsers."""
        return VNC_DISPLAY

    @property
    def novnc_url(self) -> str:
        """Get the noVNC web URL."""
        return f"http://localhost:{NOVNC_PORT}/vnc.html?autoconnect=true"

    def _check_processes(self) -> bool:
        """Check if all processes are still running."""
        if not self.xvfb_process or self.xvfb_process.poll() is not None:
            return False
        if not self.vnc_process or self.vnc_process.poll() is not None:
            return False
        if not self.websockify_process or self.websockify_process.poll() is not None:
            return False
        return True

    def start(self) -> Dict[str, Any]:
        """
        Start the VNC display session.

        Returns:
            dict with status and connection info
        """
        if self.is_active:
            return {
                "status": "already_running",
                "display": VNC_DISPLAY,
                "novnc_port": NOVNC_PORT,
                "novnc_url": self.novnc_url
            }

        try:
            # Clean up any existing processes
            self._cleanup()

            # 1. Start Xvfb (virtual framebuffer)
            logger.info(f"Starting Xvfb on display {VNC_DISPLAY}...")
            xvfb_cmd = [
                "Xvfb", VNC_DISPLAY,
                "-screen", "0", f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}x{SCREEN_DEPTH}",
                "-ac",  # Disable access control
                "-nolisten", "tcp"
            ]
            self.xvfb_process = subprocess.Popen(
                xvfb_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)  # Give Xvfb time to start

            if self.xvfb_process.poll() is not None:
                raise RuntimeError("Xvfb failed to start")

            # 2. Start x11vnc (VNC server)
            logger.info(f"Starting x11vnc on port {VNC_PORT}...")
            vnc_cmd = [
                "x11vnc",
                "-display", VNC_DISPLAY,
                "-forever",  # Don't exit after first client disconnects
                "-shared",   # Allow multiple clients
                "-nopw",     # No password (internal use only)
                "-rfbport", str(VNC_PORT),
                "-quiet"
            ]
            self.vnc_process = subprocess.Popen(
                vnc_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)  # Give x11vnc time to start

            if self.vnc_process.poll() is not None:
                raise RuntimeError("x11vnc failed to start")

            # 3. Start websockify (noVNC proxy)
            logger.info(f"Starting websockify on port {NOVNC_PORT}...")
            # Find noVNC web files
            novnc_web = "/usr/share/novnc"
            if not os.path.exists(novnc_web):
                novnc_web = "/usr/share/javascript/novnc"

            websockify_cmd = [
                "websockify",
                "--web", novnc_web,
                str(NOVNC_PORT),
                f"localhost:{VNC_PORT}"
            ]
            self.websockify_process = subprocess.Popen(
                websockify_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)  # Give websockify time to start

            if self.websockify_process.poll() is not None:
                raise RuntimeError("websockify failed to start")

            self._active = True
            logger.info(f"VNC session started successfully. noVNC available at port {NOVNC_PORT}")

            return {
                "status": "started",
                "display": VNC_DISPLAY,
                "vnc_port": VNC_PORT,
                "novnc_port": NOVNC_PORT,
                "novnc_url": self.novnc_url,
                "screen_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}"
            }

        except Exception as e:
            logger.error(f"Failed to start VNC session: {e}")
            self._cleanup()
            return {
                "status": "error",
                "error": str(e)
            }

    def stop(self) -> Dict[str, Any]:
        """Stop the VNC display session."""
        if not self._active:
            return {"status": "not_running"}

        self._cleanup()
        return {"status": "stopped"}

    def _cleanup(self):
        """Clean up all processes."""
        for name, proc in [
            ("websockify", self.websockify_process),
            ("x11vnc", self.vnc_process),
            ("Xvfb", self.xvfb_process)
        ]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception as e:
                    logger.debug(f"Error stopping {name}: {e}")

        self.xvfb_process = None
        self.vnc_process = None
        self.websockify_process = None
        self._active = False

        # Also kill any orphaned processes
        try:
            subprocess.run(["pkill", "-f", f"Xvfb {VNC_DISPLAY}"], capture_output=True)
            subprocess.run(["pkill", "-f", f"x11vnc.*{VNC_DISPLAY}"], capture_output=True)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Get current VNC session status."""
        if not self._active:
            return {
                "active": False,
                "display": None,
                "novnc_port": NOVNC_PORT
            }

        return {
            "active": self.is_active,
            "display": VNC_DISPLAY,
            "vnc_port": VNC_PORT,
            "novnc_port": NOVNC_PORT,
            "novnc_url": self.novnc_url,
            "screen_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            "processes": {
                "xvfb": self.xvfb_process.poll() is None if self.xvfb_process else False,
                "x11vnc": self.vnc_process.poll() is None if self.vnc_process else False,
                "websockify": self.websockify_process.poll() is None if self.websockify_process else False
            }
        }


# Global singleton
vnc_manager = VNCManager()
