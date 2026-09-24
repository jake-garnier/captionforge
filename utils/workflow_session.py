"""
Generic Workflow Session Manager for browser automation.

Manages a persistent browser session with VNC display for recording and
replaying browser automation workflows. Can be used for any website
(Reddit, Patreon, etc.) without hardcoded URLs.

Usage:
    from utils.workflow_session import workflow_session_manager

    # Start session
    result = await workflow_session_manager.start_session("https://reddit.com")
    # Connect to result["novnc_url"] to see the browser

    # Record actions
    await workflow_session_manager.start_recording()
    # ... user interacts with browser via VNC ...
    actions = await workflow_session_manager.stop_recording()

    # Execute workflow with variables
    async for step in workflow_session_manager.execute_workflow(actions, {"{{title}}": "My Post"}):
        print(f"Executing step {step['index']}: {step['action']['type']}")
"""

import asyncio
import base64
import os
import random
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime

logger = logging.getLogger(__name__)


# Human-like behavior utilities
async def human_delay(min_sec: float = 0.5, max_sec: float = 2.0):
    """Random delay to simulate human thinking/reaction time."""
    await asyncio.sleep(random.uniform(min_sec, max_sec))


async def human_short_delay():
    """Short delay for quick actions (0.1-0.4s)."""
    await asyncio.sleep(random.uniform(0.1, 0.4))


async def human_type(page, text: str, min_delay: int = 30, max_delay: int = 120):
    """Type text with human-like variable speed."""
    for char in text:
        if random.random() < 0.05:  # 5% chance of pause
            await asyncio.sleep(random.uniform(0.3, 0.8))
        delay = random.randint(min_delay, max_delay)
        if char in '.,!?;:':
            delay = random.randint(80, 200)
        await page.keyboard.type(char, delay=delay)
        if char == ' ' and random.random() < 0.15:
            await asyncio.sleep(random.uniform(0.1, 0.3))


class WorkflowSessionManager:
    """
    Generic browser session manager with VNC for workflow development.

    Supports:
    - Starting/stopping browser sessions with VNC display
    - Recording user actions via injected JavaScript
    - Executing recorded workflows with variable substitution
    - Real-time execution progress streaming
    """

    SCREENSHOT_DIR = "/data/workflow_screenshots"

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._lock = asyncio.Lock()
        self._started_at: Optional[datetime] = None
        self._start_url: Optional[str] = None
        self._latest_screenshot: Optional[str] = None
        self._last_activity: Optional[str] = None
        self._vnc_active = False

        # Recording state
        self._recording_active = False
        self._recorded_actions: List[Dict] = []

        # Execution state (for live highlighting)
        self._executing = False
        self._execution_step = -1
        self._execution_total = 0
        self._execution_status = "idle"  # idle, running, completed, failed, cancelled
        self._execution_error: Optional[str] = None

        # Subscribers for execution updates (WebSocket connections)
        self._execution_subscribers: List[asyncio.Queue] = []

        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        vnc_info = {}
        if self._vnc_active:
            try:
                from utils.vnc_manager import vnc_manager
                vnc_status = vnc_manager.get_status()
                vnc_info = {
                    "vnc_active": vnc_status.get("active", False),
                    "novnc_port": vnc_status.get("novnc_port"),
                    "novnc_url": vnc_status.get("novnc_url"),
                }
            except Exception:
                vnc_info = {"vnc_active": False}

        return {
            "active": self.page is not None,
            "start_url": self._start_url,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_activity": self._last_activity,
            "has_screenshot": self._latest_screenshot is not None,
            "recording_active": self._recording_active,
            "recorded_action_count": len(self._recorded_actions),
            "executing": self._executing,
            "execution_step": self._execution_step,
            "execution_total": self._execution_total,
            "execution_status": self._execution_status,
            **vnc_info,
        }

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status for frontend highlighting."""
        return {
            "executing": self._executing,
            "current_step": self._execution_step,
            "total_steps": self._execution_total,
            "status": self._execution_status,
            "error": self._execution_error,
        }

    async def _take_screenshot(self, activity: str) -> Optional[str]:
        """Take a screenshot and return the path."""
        if not self.page:
            return None

        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_{timestamp}.png"
            filepath = os.path.join(self.SCREENSHOT_DIR, filename)

            await self.page.screenshot(path=filepath)
            self._latest_screenshot = filepath
            self._last_activity = activity

            logger.info(f"Screenshot saved: {filepath} ({activity})")
            return filepath

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def get_latest_screenshot(self) -> Dict[str, Any]:
        """Get the latest screenshot as base64."""
        if not self._latest_screenshot or not os.path.exists(self._latest_screenshot):
            return {"available": False}

        try:
            with open(self._latest_screenshot, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            return {
                "available": True,
                "path": self._latest_screenshot,
                "activity": self._last_activity,
                "data": f"data:image/png;base64,{image_data}"
            }
        except Exception as e:
            logger.error(f"Failed to read screenshot: {e}")
            return {"available": False, "error": str(e)}

    # =========================================================================
    # Session Management
    # =========================================================================

    async def start_session(self, start_url: str = "about:blank") -> Dict[str, Any]:
        """
        Start a browser session with VNC display.

        Args:
            start_url: Initial URL to navigate to

        Returns:
            Dict with session info including noVNC URL
        """
        logger.info(f"[WORKFLOW] start_session called with url: {start_url}")

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            return {"status": "error", "error": "Session busy, try again"}

        try:
            if self.page is not None:
                # Session already running
                from utils.vnc_manager import vnc_manager
                vnc_status = vnc_manager.get_status()
                return {
                    "status": "already_running",
                    "start_url": self._start_url,
                    "novnc_port": vnc_status.get("novnc_port"),
                    "novnc_url": vnc_status.get("novnc_url"),
                }

            # Start VNC display
            logger.info("[WORKFLOW] Starting VNC display...")
            from utils.vnc_manager import vnc_manager
            vnc_result = vnc_manager.start()
            if vnc_result.get("status") == "error":
                return {"status": "error", "error": f"VNC failed: {vnc_result.get('error')}"}

            self._vnc_active = True
            display = vnc_result.get("display", ":99")
            novnc_port = vnc_result.get("novnc_port", 6080)
            novnc_url = vnc_result.get("novnc_url")
            logger.info(f"[WORKFLOW] VNC started on display {display}")

            # Set DISPLAY for browser
            os.environ["DISPLAY"] = display

            # Start Playwright
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()

            # Create browser context (visible on VNC)
            user_data_dir = "/tmp/workflow_browser"
            os.makedirs(user_data_dir, exist_ok=True)

            viewport_width = random.randint(1350, 1400)
            viewport_height = random.randint(950, 1050)

            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=False,  # Visible on VNC
                viewport={'width': viewport_width, 'height': viewport_height},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-gpu',
                    '--start-maximized',
                    '--disable-dev-shm-usage',
                    '--disable-infobars',
                ],
            )

            # Get or create page
            if self.context.pages:
                self.page = self.context.pages[0]
            else:
                self.page = await self.context.new_page()

            # Navigate to start URL
            if start_url and start_url != "about:blank":
                logger.info(f"[WORKFLOW] Navigating to {start_url}")
                await self.page.goto(start_url, wait_until="domcontentloaded", timeout=30000)

            self._started_at = datetime.utcnow()
            self._start_url = start_url

            # Take initial screenshot
            await self._take_screenshot("session_started")

            logger.info("[WORKFLOW] Session started successfully")
            return {
                "status": "started",
                "start_url": start_url,
                "novnc_port": novnc_port,
                "novnc_url": novnc_url,
            }

        except Exception as e:
            logger.error(f"[WORKFLOW] Failed to start session: {e}")
            await self._stop_session_internal()
            return {"status": "error", "error": str(e)}
        finally:
            self._lock.release()

    async def _stop_session_internal(self):
        """Internal cleanup without lock."""
        try:
            if self.context:
                await self.context.close()
        except Exception as e:
            logger.warning(f"Error closing context: {e}")

        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")

        try:
            if self._vnc_active:
                from utils.vnc_manager import vnc_manager
                vnc_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping VNC: {e}")

        self.playwright = None
        self.context = None
        self.page = None
        self._vnc_active = False
        self._started_at = None
        self._start_url = None
        self._recording_active = False
        self._recorded_actions = []

    async def stop_session(self) -> Dict[str, Any]:
        """Stop the browser session."""
        async with self._lock:
            if self.page is None:
                return {"status": "not_running"}

            await self._stop_session_internal()
            logger.info("[WORKFLOW] Session stopped")
            return {"status": "stopped"}

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        if not self.page:
            return {"status": "error", "error": "No active session"}

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self._take_screenshot(f"navigate_{url[:30]}")
            return {"status": "success", "url": url}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # Recording
    # =========================================================================

    async def start_recording(self) -> Dict[str, Any]:
        """Start recording user actions."""
        if not self.page:
            return {"status": "error", "error": "No active session"}

        if self._recording_active:
            return {"status": "already_recording"}

        try:
            # Inject action recorder JavaScript
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()
            await recorder.inject_recorder(self.page)
            result = await recorder.start_recording(self.page)

            self._recording_active = True
            self._recorded_actions = []

            logger.info("[WORKFLOW] Recording started")
            return {"status": "started", **result}

        except Exception as e:
            logger.error(f"[WORKFLOW] Failed to start recording: {e}")
            return {"status": "error", "error": str(e)}

    async def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return captured actions."""
        if not self.page:
            return {"status": "error", "error": "No active session"}

        if not self._recording_active:
            return {"status": "not_recording", "actions": []}

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()
            result = await recorder.stop_recording(self.page)

            self._recording_active = False
            self._recorded_actions = result.get("actions", [])

            logger.info(f"[WORKFLOW] Recording stopped: {len(self._recorded_actions)} actions")
            return {
                "status": "stopped",
                "action_count": len(self._recorded_actions),
                "actions": self._recorded_actions,
            }

        except Exception as e:
            logger.error(f"[WORKFLOW] Failed to stop recording: {e}")
            self._recording_active = False
            return {"status": "error", "error": str(e), "actions": []}

    async def get_recorded_actions(self, clear: bool = False) -> List[Dict]:
        """Get recorded actions (poll during active recording)."""
        if not self.page or not self._recording_active:
            return self._recorded_actions

        try:
            from utils.action_recorder import get_action_recorder
            recorder = get_action_recorder()
            actions = await recorder.poll_actions(self.page, clear=False)
            self._recorded_actions = actions
            return actions
        except Exception as e:
            logger.error(f"[WORKFLOW] Failed to get actions: {e}")
            return self._recorded_actions

    # =========================================================================
    # Execution
    # =========================================================================

    def subscribe_to_execution(self) -> asyncio.Queue:
        """Subscribe to execution updates (for WebSocket)."""
        queue = asyncio.Queue()
        self._execution_subscribers.append(queue)
        return queue

    def unsubscribe_from_execution(self, queue: asyncio.Queue):
        """Unsubscribe from execution updates."""
        if queue in self._execution_subscribers:
            self._execution_subscribers.remove(queue)

    async def _notify_execution_update(self):
        """Notify all subscribers of execution state change."""
        update = self.get_execution_status()
        for queue in self._execution_subscribers:
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                pass

    async def execute_workflow(
        self,
        actions: List[Dict],
        variables: Optional[Dict[str, str]] = None,
        speed_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute a workflow with live progress updates.

        Args:
            actions: List of recorded actions
            variables: Variable substitutions (e.g., {"{{title}}": "My Post"})
            speed_multiplier: Speed up (>1) or slow down (<1) execution

        Returns:
            Execution results
        """
        if not self.page:
            return {"status": "error", "error": "No active session"}

        if self._executing:
            return {"status": "error", "error": "Already executing"}

        variables = variables or {}
        results = []
        last_timestamp = 0

        self._executing = True
        self._execution_step = -1
        self._execution_total = len(actions)
        self._execution_status = "running"
        self._execution_error = None
        await self._notify_execution_update()

        logger.info(f"[WORKFLOW] Executing {len(actions)} actions")

        try:
            for i, action in enumerate(actions):
                self._execution_step = i
                await self._notify_execution_update()

                action_type = action.get("type")
                timestamp = action.get("timestamp", 0)

                # Calculate delay with variance
                delay_ms = (timestamp - last_timestamp) / speed_multiplier
                if delay_ms > 100:
                    delay_ms = delay_ms * random.uniform(0.8, 1.2)
                    await asyncio.sleep(delay_ms / 1000)
                last_timestamp = timestamp

                try:
                    result = await self._execute_action(action, variables)
                    results.append({"action": i, "type": action_type, **result})
                except Exception as e:
                    logger.error(f"[WORKFLOW] Action {i} ({action_type}) failed: {e}")
                    results.append({"action": i, "type": action_type, "status": "error", "error": str(e)})

            # Execution complete
            success_count = sum(1 for r in results if r.get("status") == "success")
            failed_count = sum(1 for r in results if r.get("status") in ("failed", "error"))

            self._execution_status = "completed"
            self._executing = False
            await self._notify_execution_update()

            logger.info(f"[WORKFLOW] Execution complete: {success_count} succeeded, {failed_count} failed")

            return {
                "status": "completed",
                "total_actions": len(actions),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results,
            }

        except Exception as e:
            self._execution_status = "failed"
            self._execution_error = str(e)
            self._executing = False
            await self._notify_execution_update()
            logger.error(f"[WORKFLOW] Execution failed: {e}")
            return {"status": "error", "error": str(e), "results": results}

    async def _execute_action(self, action: Dict, variables: Dict[str, str]) -> Dict[str, Any]:
        """Execute a single action."""
        action_type = action.get("type")

        if action_type == "navigate":
            url = action.get("url", "")
            for placeholder, value in variables.items():
                url = url.replace(placeholder, value)
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return {"status": "success"}

        elif action_type == "click":
            selector = action.get("selector")
            coords = action.get("coordinates")

            # Try selector first
            if selector:
                try:
                    element = self.page.locator(selector).first
                    if await element.is_visible(timeout=2000):
                        await element.click()
                        await human_short_delay()
                        return {"status": "success", "method": "selector"}
                except Exception:
                    pass

            # Fall back to coordinates
            if coords:
                await self.page.mouse.click(coords["x"], coords["y"])
                await human_short_delay()
                return {"status": "success", "method": "coordinates"}

            return {"status": "failed", "error": "No selector or coordinates"}

        elif action_type == "type":
            selector = action.get("selector")
            text = action.get("text", "")

            # Apply variable substitutions
            for placeholder, value in variables.items():
                text = text.replace(placeholder, value)

            if selector:
                try:
                    element = self.page.locator(selector).first
                    if action.get("is_contenteditable"):
                        await element.click()
                        await element.fill("")
                        await human_type(self.page, text)
                    else:
                        await element.fill(text)
                    return {"status": "success"}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}

            return {"status": "failed", "error": "No selector"}

        elif action_type == "keypress":
            key = action.get("key")
            if key:
                await self.page.keyboard.press(key)
                await human_short_delay()
                return {"status": "success"}
            return {"status": "failed", "error": "No key"}

        elif action_type == "scroll":
            scroll_x = action.get("scroll_x", 0)
            scroll_y = action.get("scroll_y", 0)
            await self.page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
            return {"status": "success"}

        elif action_type == "file_upload":
            selector = action.get("selector")
            file_path = variables.get("{{file_path}}") or variables.get("{{video_path}}")
            if selector and file_path:
                try:
                    await self.page.locator(selector).set_input_files(file_path)
                    return {"status": "success"}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
            return {"status": "skipped", "reason": "No file path provided"}

        else:
            return {"status": "skipped", "reason": f"Unknown action type: {action_type}"}

    async def cancel_execution(self) -> Dict[str, Any]:
        """Cancel ongoing execution."""
        if not self._executing:
            return {"status": "not_executing"}

        self._execution_status = "cancelled"
        self._executing = False
        await self._notify_execution_update()
        return {"status": "cancelled"}


# Global singleton instance
workflow_session_manager = WorkflowSessionManager()
