"""
Telegram Publisher - Post videos to Telegram channels.

Features:
- Post videos with captions to channels
- Verify bot tokens
- Get channel info and updates
- Support for discussion group comments (automatic if channel has linked group)
"""

import httpx
import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> dict:
    """Get video width, height, and duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            return {}

        return {
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'duration': int(float(data.get('format', {}).get('duration', 0))),
        }
    except Exception as e:
        logger.warning(f"Failed to get video metadata: {e}")
        return {}


class TelegramPublisher:
    """Publish videos to Telegram channels using the Bot API."""

    # Telegram bot API limits
    MAX_FILE_SIZE_MB = 50  # Bot API limit (premium bots can do 2GB)
    MAX_CAPTION_LENGTH = 1024

    def __init__(self, bot_token: str):
        """
        Initialize publisher with bot token.

        Args:
            bot_token: Telegram Bot API token from @BotFather
        """
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    async def verify_bot(self) -> dict:
        """
        Verify the bot token is valid and get bot info.

        Returns:
            Bot info dict with username, name, etc.

        Raises:
            Exception if token is invalid
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/getMe")
            result = response.json()

        if not result.get("ok"):
            raise Exception(f"Invalid bot token: {result.get('description')}")

        return result["result"]

    async def get_updates(self, limit: int = 10) -> list:
        """
        Get recent updates (messages) to find channel IDs.

        The bot must receive at least one message/post in the channel
        to discover its ID.

        Args:
            limit: Maximum number of updates to return

        Returns:
            List of update objects
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/getUpdates",
                params={"limit": limit}
            )
            result = response.json()

        if not result.get("ok"):
            raise Exception(f"Failed to get updates: {result.get('description')}")

        return result.get("result", [])

    async def get_chat_info(self, chat_id: str) -> dict:
        """
        Get information about a chat/channel.

        Args:
            chat_id: Channel ID or @username

        Returns:
            Chat info dict including linked_chat_id if discussion group exists
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/getChat",
                data={"chat_id": chat_id}
            )
            result = response.json()

        if not result.get("ok"):
            raise Exception(f"Failed to get chat info: {result.get('description')}")

        return result["result"]

    async def send_message(self, chat_id: str, text: str) -> dict:
        """
        Send a text message to a chat/channel.

        Args:
            chat_id: Channel ID or @username
            text: Message text

        Returns:
            Message result dict
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/sendMessage",
                data={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                }
            )
            result = response.json()

        if not result.get("ok"):
            raise Exception(f"Failed to send message: {result.get('description')}")

        return result["result"]

    async def post_video(
        self,
        chat_id: str,
        video_path: str,
        caption: str = "",
        thumbnail_path: Optional[str] = None,
    ) -> dict:
        """
        Post a video to a channel.

        Comments are automatically enabled if the channel has a linked
        discussion group configured in Telegram settings.

        Args:
            chat_id: Channel ID (e.g., -1001234567890) or @username
            video_path: Path to the video file
            caption: Optional caption (max 1024 chars) - pass empty string for no caption
            thumbnail_path: Optional thumbnail image path

        Returns:
            Message result dict with message_id, chat info, etc.

        Raises:
            ValueError: If video is too large
            Exception: If upload fails
        """
        file_path = Path(video_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Video too large: {file_size_mb:.1f}MB "
                f"(max {self.MAX_FILE_SIZE_MB}MB for bot API)"
            )

        # Get video metadata for proper display
        metadata = get_video_metadata(video_path)
        width = metadata.get('width')
        height = metadata.get('height')
        duration = metadata.get('duration')

        logger.info(f"Uploading video to Telegram: {file_path.name} ({file_size_mb:.1f}MB, {width}x{height}, {duration}s)")

        # Increase timeout for larger files (2 min base + 1 min per 10MB)
        timeout = 120 + int(file_size_mb / 10) * 60

        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {"video": open(video_path, "rb")}
            if thumbnail_path and Path(thumbnail_path).exists():
                files["thumbnail"] = open(thumbnail_path, "rb")

            # Build request data
            data = {
                "chat_id": chat_id,
                "supports_streaming": "true",
            }

            # Only add caption if provided
            if caption:
                if len(caption) > self.MAX_CAPTION_LENGTH:
                    caption = caption[:self.MAX_CAPTION_LENGTH - 3] + "..."
                data["caption"] = caption
                data["parse_mode"] = "HTML"

            # Add video metadata for proper display/autoplay
            if width:
                data["width"] = str(width)
            if height:
                data["height"] = str(height)
            if duration:
                data["duration"] = str(duration)

            try:
                response = await client.post(
                    f"{self.base_url}/sendVideo",
                    data=data,
                    files=files
                )
            finally:
                # Close file handles
                for f in files.values():
                    f.close()

            result = response.json()

        if not result.get("ok"):
            raise Exception(f"Failed to post video: {result.get('description')}")

        logger.info(f"Video posted successfully. Message ID: {result['result']['message_id']}")
        return result["result"]

    def get_post_url(self, channel_username: str, message_id: int) -> str:
        """
        Generate a public URL for a channel post.

        Args:
            channel_username: Channel username (with or without @)
            message_id: Message ID

        Returns:
            Public URL like https://t.me/ChannelName/123
        """
        username = channel_username.lstrip("@")
        return f"https://t.me/{username}/{message_id}"

    def get_post_url_private(self, channel_id: str, message_id: int) -> str:
        """
        Generate a URL for a private channel post.

        Args:
            channel_id: Channel ID (numeric, including -100 prefix)
            message_id: Message ID

        Returns:
            URL like https://t.me/c/1234567890/123
        """
        # Remove -100 prefix if present
        numeric_id = str(channel_id).replace("-100", "")
        return f"https://t.me/c/{numeric_id}/{message_id}"

    async def discover_channel_id(self) -> list[dict]:
        """
        Discover channel IDs from recent bot updates.

        Returns:
            List of discovered channels with id, title, username
        """
        updates = await self.get_updates(limit=50)
        channels = []
        seen_ids = set()

        for update in updates:
            # Check channel posts
            if "channel_post" in update:
                chat = update["channel_post"]["chat"]
                if chat["id"] not in seen_ids:
                    seen_ids.add(chat["id"])
                    channels.append({
                        "id": chat["id"],
                        "title": chat.get("title"),
                        "username": chat.get("username"),
                        "type": chat.get("type"),
                    })

            # Check messages (for groups/private chats)
            if "message" in update:
                chat = update["message"]["chat"]
                if chat["id"] not in seen_ids:
                    seen_ids.add(chat["id"])
                    channels.append({
                        "id": chat["id"],
                        "title": chat.get("title") or chat.get("first_name"),
                        "username": chat.get("username"),
                        "type": chat.get("type"),
                    })

        return channels


def publish_video_sync(
    bot_token: str,
    channel_id: str,
    video_path: str,
    caption: str = ""
) -> dict:
    """
    Synchronous wrapper for posting a video.

    For use in Celery tasks or other sync contexts.

    Args:
        bot_token: Telegram Bot API token
        channel_id: Channel ID or @username
        video_path: Path to video file
        caption: Optional caption

    Returns:
        Message result dict
    """
    publisher = TelegramPublisher(bot_token)
    return asyncio.run(publisher.post_video(channel_id, video_path, caption))
