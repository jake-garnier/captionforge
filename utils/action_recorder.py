"""
Patreon Action Recorder - Python Manager

Manages recording and replay of user actions captured via injected JavaScript.
"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Storage directory for recordings
RECORDINGS_DIR = Path("/data/patreon_recordings")


@dataclass
class RecordedAction:
    """A single recorded action"""
    type: str
    timestamp: int  # ms since recording start
    selector: Optional[str] = None
    coordinates: Optional[Dict[str, int]] = None
    text: Optional[str] = None
    element_text: Optional[str] = None
    element_tag: Optional[str] = None
    url: Optional[str] = None
    key: Optional[str] = None
    scroll_x: Optional[int] = None
    scroll_y: Optional[int] = None
    is_contenteditable: Optional[bool] = None
    with_shift: Optional[bool] = None
    with_ctrl: Optional[bool] = None
    with_alt: Optional[bool] = None


@dataclass
class Recording:
    """A complete recording with metadata"""
    name: str
    niche: str
    recorded_at: str
    version: int = 1
    actions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class ActionRecorder:
    """Manages action recording and replay"""

    def __init__(self):
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """Ensure recordings directory exists"""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_recording_path(self, name: str) -> Path:
        """Get path for a recording file"""
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return RECORDINGS_DIR / f"{safe_name}.json"

    def _get_js_path(self) -> Path:
        """Get path to the JavaScript recorder"""
        return Path(__file__).parent / "action_recorder.js"

    def get_recorder_js(self) -> str:
        """Read the JavaScript recorder code"""
        js_path = self._get_js_path()
        if not js_path.exists():
            raise FileNotFoundError(f"Recorder JS not found at {js_path}")
        return js_path.read_text()

    async def inject_recorder(self, page) -> bool:
        """Inject the recorder JavaScript into a page"""
        try:
            js_code = self.get_recorder_js()
            await page.evaluate(js_code)
            logger.info("Action recorder injected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to inject recorder: {e}")
            return False

    async def start_recording(self, page) -> Dict[str, Any]:
        """Start recording actions on the page"""
        try:
            # Ensure recorder is injected
            is_initialized = await page.evaluate("() => window.__actionRecorderInitialized || false")
            if not is_initialized:
                await self.inject_recorder(page)

            # Start recording
            result = await page.evaluate("() => window.__startRecording()")
            logger.info(f"Recording started: {result}")
            return {"status": "started", "result": result}
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return {"status": "error", "error": str(e)}

    async def stop_recording(self, page) -> Dict[str, Any]:
        """Stop recording and return captured actions"""
        try:
            result = await page.evaluate("() => window.__stopRecording()")
            logger.info(f"Recording stopped: {result.get('action_count', 0)} actions captured")
            return result
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return {"status": "error", "error": str(e), "actions": []}

    async def poll_actions(self, page, clear: bool = True) -> List[Dict[str, Any]]:
        """Poll for recorded actions (useful during active recording)"""
        try:
            actions = await page.evaluate(f"() => window.__getRecordedActions({str(clear).lower()})")
            return actions or []
        except Exception as e:
            logger.error(f"Failed to poll actions: {e}")
            return []

    async def is_recording(self, page) -> bool:
        """Check if recording is active"""
        try:
            return await page.evaluate("() => window.__isRecording()")
        except Exception:
            return False

    def save_recording(
        self,
        name: str,
        actions: List[Dict[str, Any]],
        niche: str = "unknown"
    ) -> Dict[str, Any]:
        """Save a recording to disk"""
        try:
            recording = Recording(
                name=name,
                niche=niche,
                recorded_at=datetime.utcnow().isoformat() + "Z",
                actions=actions
            )

            path = self._get_recording_path(name)
            with open(path, "w") as f:
                json.dump(asdict(recording), f, indent=2)

            logger.info(f"Recording saved: {path} ({len(actions)} actions)")
            return {
                "status": "saved",
                "path": str(path),
                "name": name,
                "action_count": len(actions)
            }
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return {"status": "error", "error": str(e)}

    def load_recording(self, name: str) -> Optional[Recording]:
        """Load a recording from disk"""
        path = self._get_recording_path(name)
        if not path.exists():
            logger.warning(f"Recording not found: {path}")
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return Recording(**data)
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return None

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all saved recordings"""
        recordings = []
        for path in RECORDINGS_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                recordings.append({
                    "name": data.get("name", path.stem),
                    "niche": data.get("niche", "unknown"),
                    "recorded_at": data.get("recorded_at"),
                    "action_count": len(data.get("actions", [])),
                    "path": str(path)
                })
            except Exception as e:
                logger.warning(f"Failed to read recording {path}: {e}")
        return sorted(recordings, key=lambda r: r.get("recorded_at", ""), reverse=True)

    def delete_recording(self, name: str) -> bool:
        """Delete a recording"""
        path = self._get_recording_path(name)
        if path.exists():
            path.unlink()
            logger.info(f"Recording deleted: {path}")
            return True
        return False

    async def replay_recording(
        self,
        page,
        recording: Recording,
        variables: Optional[Dict[str, str]] = None,
        speed_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Replay a recording on the page

        Args:
            page: Playwright page object
            recording: Recording to replay
            variables: Dict of placeholder replacements (e.g., {"{{title}}": "My Video"})
            speed_multiplier: Speed up (>1) or slow down (<1) replay

        Returns:
            Dict with replay results
        """
        variables = variables or {}
        results = []
        last_timestamp = 0

        logger.info(f"Replaying recording: {recording.name} ({len(recording.actions)} actions)")

        for i, action in enumerate(recording.actions):
            action_type = action.get("type")
            timestamp = action.get("timestamp", 0)

            # Calculate delay (with variance)
            delay_ms = (timestamp - last_timestamp) / speed_multiplier
            if delay_ms > 100:
                # Add 20% variance
                import random
                delay_ms = delay_ms * random.uniform(0.8, 1.2)
                await asyncio.sleep(delay_ms / 1000)

            last_timestamp = timestamp

            try:
                if action_type == "navigate":
                    url = action.get("url")
                    if url:
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                        results.append({"action": i, "type": "navigate", "status": "success"})

                elif action_type == "click":
                    selector = action.get("selector")
                    coords = action.get("coordinates")

                    # Try selector first
                    if selector:
                        try:
                            element = page.locator(selector).first
                            if await element.is_visible(timeout=2000):
                                await element.click()
                                results.append({"action": i, "type": "click", "status": "success", "method": "selector"})
                                continue
                        except Exception:
                            pass

                    # Fall back to coordinates
                    if coords:
                        await page.mouse.click(coords["x"], coords["y"])
                        results.append({"action": i, "type": "click", "status": "success", "method": "coordinates"})
                    else:
                        results.append({"action": i, "type": "click", "status": "failed", "error": "No selector or coordinates"})

                elif action_type == "type":
                    selector = action.get("selector")
                    text = action.get("text", "")

                    # Replace placeholders
                    for placeholder, value in variables.items():
                        text = text.replace(placeholder, value)

                    if selector:
                        try:
                            element = page.locator(selector).first
                            if action.get("is_contenteditable"):
                                await element.click()
                                await element.fill("")
                                # Type with slight delay to appear human-like
                                await page.keyboard.type(text, delay=50)
                            else:
                                await element.fill(text)
                            results.append({"action": i, "type": "type", "status": "success"})
                        except Exception as e:
                            results.append({"action": i, "type": "type", "status": "failed", "error": str(e)})
                    else:
                        results.append({"action": i, "type": "type", "status": "failed", "error": "No selector"})

                elif action_type == "keypress":
                    key = action.get("key")
                    if key:
                        await page.keyboard.press(key)
                        results.append({"action": i, "type": "keypress", "status": "success"})

                elif action_type == "scroll":
                    scroll_x = action.get("scroll_x", 0)
                    scroll_y = action.get("scroll_y", 0)
                    await page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
                    results.append({"action": i, "type": "scroll", "status": "success"})

                elif action_type == "file_upload":
                    selector = action.get("selector")
                    file_path = variables.get("{{video_path}}")
                    if selector and file_path:
                        try:
                            await page.locator(selector).set_input_files(file_path)
                            results.append({"action": i, "type": "file_upload", "status": "success"})
                        except Exception as e:
                            results.append({"action": i, "type": "file_upload", "status": "failed", "error": str(e)})
                    else:
                        results.append({"action": i, "type": "file_upload", "status": "skipped", "reason": "No file path provided"})

                else:
                    results.append({"action": i, "type": action_type, "status": "skipped", "reason": "Unknown action type"})

            except Exception as e:
                logger.error(f"Action {i} ({action_type}) failed: {e}")
                results.append({"action": i, "type": action_type, "status": "error", "error": str(e)})

        # Summary
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = sum(1 for r in results if r.get("status") in ("failed", "error"))

        logger.info(f"Replay complete: {success_count} succeeded, {failed_count} failed")

        return {
            "status": "completed",
            "total_actions": len(recording.actions),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }


# Global instance
_recorder = None


def get_action_recorder() -> ActionRecorder:
    """Get or create the global ActionRecorder instance"""
    global _recorder
    if _recorder is None:
        _recorder = ActionRecorder()
    return _recorder
