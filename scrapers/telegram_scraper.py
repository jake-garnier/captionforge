"""
Telegram Channel Scraper - Download videos from private Telegram channels.

Uses Telethon to authenticate as a user account, giving access to all channels
the user has joined (including private ones).

Features:
- One-time phone verification, then session persists
- Download videos/images/GIFs from any channel
- Incremental scraping with message ID tracking
- Metadata extraction (views, forwards, date)
"""

import asyncio
import logging
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    DocumentAttributeVideo,
    DocumentAttributeAnimated,
    DocumentAttributeFilename,
)
from telethon.errors import SessionPasswordNeededError

from config.settings import settings

logger = logging.getLogger(__name__)


class TelegramScraperAuth:
    """Handle Telegram authentication flow."""

    def __init__(self):
        self.api_id = settings.TELEGRAM_API_ID
        self.api_hash = settings.TELEGRAM_API_HASH
        self.session_path = settings.TELEGRAM_SESSION_PATH
        self.client: Optional[TelegramClient] = None
        self._phone_code_hash: Optional[str] = None

    async def _get_client(self) -> TelegramClient:
        """Get or create the Telegram client."""
        if self.client is None:
            self.client = TelegramClient(
                self.session_path,
                self.api_id,
                self.api_hash
            )
        return self.client

    async def is_authenticated(self) -> bool:
        """Check if we have a valid session."""
        try:
            client = await self._get_client()
            await client.connect()
            return await client.is_user_authorized()
        except Exception as e:
            logger.warning(f"Auth check failed: {e}")
            return False
        finally:
            if self.client:
                await self.client.disconnect()

    async def start_auth(self, phone: str) -> Dict[str, Any]:
        """
        Start the authentication flow by sending a verification code.

        Args:
            phone: Phone number in international format (+1234567890)

        Returns:
            Dict with status and next steps
        """
        try:
            client = await self._get_client()
            await client.connect()

            if await client.is_user_authorized():
                return {
                    "status": "already_authenticated",
                    "message": "Already logged in with valid session"
                }

            # Request verification code
            result = await client.send_code_request(phone)
            self._phone_code_hash = result.phone_code_hash

            return {
                "status": "code_sent",
                "message": "Verification code sent to your phone/Telegram app",
                "phone_code_hash": result.phone_code_hash
            }

        except Exception as e:
            logger.error(f"Failed to start auth: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            if self.client:
                await self.client.disconnect()

    async def verify_code(
        self,
        phone: str,
        code: str,
        phone_code_hash: str,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete authentication by verifying the code.

        Args:
            phone: Phone number used in start_auth
            code: Verification code received
            phone_code_hash: Hash from start_auth response
            password: 2FA password if enabled

        Returns:
            Dict with status and user info
        """
        try:
            client = await self._get_client()
            await client.connect()

            try:
                # Try to sign in with code
                await client.sign_in(phone, code, phone_code_hash=phone_code_hash)
            except SessionPasswordNeededError:
                # 2FA is enabled
                if not password:
                    return {
                        "status": "2fa_required",
                        "message": "Two-factor authentication is enabled. Please provide password."
                    }
                await client.sign_in(password=password)

            # Get user info
            me = await client.get_me()

            return {
                "status": "authenticated",
                "message": "Successfully authenticated",
                "user": {
                    "id": me.id,
                    "username": me.username,
                    "first_name": me.first_name,
                    "phone": me.phone
                }
            }

        except Exception as e:
            logger.error(f"Failed to verify code: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            if self.client:
                await self.client.disconnect()


class TelegramScraper:
    """Scrape videos from Telegram channels using user account."""

    def __init__(self):
        self.api_id = settings.TELEGRAM_API_ID
        self.api_hash = settings.TELEGRAM_API_HASH
        self.session_path = settings.TELEGRAM_SESSION_PATH
        self.client: Optional[TelegramClient] = None
        self.video_dir = Path(settings.VIDEO_STORAGE_PATH)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Context manager entry - connect and verify auth."""
        self.client = TelegramClient(
            self.session_path,
            self.api_id,
            self.api_hash
        )
        await self.client.connect()

        if not await self.client.is_user_authorized():
            raise Exception("Not authenticated. Please complete authentication first.")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect."""
        if self.client:
            await self.client.disconnect()

    async def get_dialogs(self) -> List[Dict[str, Any]]:
        """
        Get all dialogs (chats/channels) the user has access to.
        Useful for discovering channel IDs.

        Returns:
            List of dialog info dicts
        """
        dialogs = await self.client.get_dialogs()
        result = []

        for dialog in dialogs:
            entity = dialog.entity
            dialog_type = "unknown"

            if hasattr(entity, 'megagroup') and entity.megagroup:
                dialog_type = "supergroup"
            elif hasattr(entity, 'broadcast') and entity.broadcast:
                dialog_type = "channel"
            elif hasattr(entity, 'gigagroup') and entity.gigagroup:
                dialog_type = "gigagroup"
            elif hasattr(entity, 'bot') and entity.bot:
                dialog_type = "bot"
            elif hasattr(entity, 'first_name'):
                dialog_type = "user"
            else:
                dialog_type = "group"

            result.append({
                "id": entity.id,
                "name": dialog.name,
                "username": getattr(entity, 'username', None),
                "type": dialog_type,
                "unread_count": dialog.unread_count,
            })

        return result

    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """
        Get information about a specific channel.

        Args:
            channel_id: Channel ID or @username

        Returns:
            Channel info dict
        """
        try:
            entity = await self.client.get_entity(channel_id)

            return {
                "id": entity.id,
                "title": getattr(entity, 'title', None),
                "username": getattr(entity, 'username', None),
                "participants_count": getattr(entity, 'participants_count', None),
                "broadcast": getattr(entity, 'broadcast', False),
                "megagroup": getattr(entity, 'megagroup', False),
            }
        except Exception as e:
            logger.error(f"Failed to get channel info for {channel_id}: {e}")
            raise

    async def get_media_messages(
        self,
        channel_id: str,
        limit: int = 25,
        offset_id: int = 0,
        min_id: int = 0
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Fetch video/image messages from a channel.

        Args:
            channel_id: Channel ID or @username
            limit: Maximum messages to fetch
            offset_id: Start from this message ID (0 = latest)
            min_id: Don't fetch messages older than this ID

        Returns:
            Tuple of (messages list, next offset_id for pagination)
        """
        entity = await self.client.get_entity(channel_id)
        messages = await self.client.get_messages(
            entity,
            limit=limit,
            offset_id=offset_id,
            min_id=min_id,
        )

        result = []
        next_offset_id = None

        for msg in messages:
            if not msg.media:
                continue

            media_type = self._get_media_type(msg.media)
            if not media_type:
                continue

            result.append({
                "id": msg.id,
                "date": msg.date.isoformat() if msg.date else None,
                "text": msg.text or "",
                "views": msg.views or 0,
                "forwards": msg.forwards or 0,
                "media_type": media_type,
                "channel_id": str(entity.id),
                "channel_name": getattr(entity, 'title', None),
            })

            # Track last message ID for pagination
            next_offset_id = msg.id

        return result, next_offset_id

    def _get_media_type(self, media) -> Optional[str]:
        """Determine the media type from a message's media object."""
        if isinstance(media, MessageMediaPhoto):
            return "image"

        if isinstance(media, MessageMediaDocument):
            doc = media.document
            if not doc:
                return None

            is_video = False
            is_gif = False

            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeVideo):
                    is_video = True
                if isinstance(attr, DocumentAttributeAnimated):
                    is_gif = True

            if is_gif:
                return "gif"
            if is_video:
                return "video"

            # Check mime type as fallback
            mime = doc.mime_type or ""
            if mime.startswith("video/"):
                return "video"
            if mime.startswith("image/"):
                return "image"

        return None

    async def download_media(
        self,
        channel_id: str,
        message_id: int,
        output_dir: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download media from a specific message.

        Args:
            channel_id: Channel ID or @username
            message_id: Message ID to download from
            output_dir: Directory to save file (defaults to VIDEO_STORAGE_PATH)

        Returns:
            Dict with file info or None if failed
        """
        if output_dir is None:
            output_dir = self.video_dir

        try:
            entity = await self.client.get_entity(channel_id)
            messages = await self.client.get_messages(entity, ids=message_id)

            if not messages or not messages[0]:
                logger.warning(f"Message {message_id} not found in {channel_id}")
                return None

            msg = messages[0]
            if not msg.media:
                logger.warning(f"Message {message_id} has no media")
                return None

            media_type = self._get_media_type(msg.media)
            if not media_type:
                logger.warning(f"Message {message_id} has unsupported media type")
                return None

            # Determine file extension
            ext = self._get_file_extension(msg.media, media_type)

            # Generate unique filename
            filename = f"tg_{channel_id}_{message_id}{ext}"
            output_path = output_dir / filename

            # Download the file
            logger.info(f"Downloading {media_type} from message {message_id}...")
            await self.client.download_media(
                msg.media,
                file=str(output_path)
            )

            if not output_path.exists():
                logger.error(f"Download failed - file not created: {output_path}")
                return None

            # Calculate file hash
            file_hash = self._calculate_file_hash(output_path)

            # Get file info
            stat = output_path.stat()

            return {
                "path": str(output_path),
                "filename": filename,
                "file_hash": file_hash,
                "file_size": stat.st_size,
                "media_type": media_type,
                "message_id": message_id,
                "channel_id": str(entity.id),
                "date": msg.date.isoformat() if msg.date else None,
                "text": msg.text or "",
                "views": msg.views or 0,
            }

        except Exception as e:
            logger.error(f"Failed to download message {message_id}: {e}")
            return None

    def _get_file_extension(self, media, media_type: str) -> str:
        """Get appropriate file extension for media."""
        if isinstance(media, MessageMediaPhoto):
            return ".jpg"

        if isinstance(media, MessageMediaDocument):
            doc = media.document
            if doc:
                # Check for filename attribute
                for attr in doc.attributes:
                    if isinstance(attr, DocumentAttributeFilename):
                        name = attr.file_name
                        if "." in name:
                            return "." + name.rsplit(".", 1)[1].lower()

                # Fall back to mime type
                mime = doc.mime_type or ""
                if mime == "video/mp4":
                    return ".mp4"
                if mime == "video/webm":
                    return ".webm"
                if mime == "image/gif":
                    return ".gif"
                if mime.startswith("image/"):
                    return ".jpg"

        # Default based on media type
        if media_type == "video":
            return ".mp4"
        if media_type == "gif":
            return ".gif"
        return ".jpg"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# Synchronous wrapper functions for Celery tasks


def is_authenticated_sync() -> bool:
    """Check if Telegram is authenticated (sync wrapper)."""
    auth = TelegramScraperAuth()
    return asyncio.run(auth.is_authenticated())


def get_dialogs_sync() -> List[Dict[str, Any]]:
    """Get all dialogs (sync wrapper)."""
    async def _run():
        async with TelegramScraper() as scraper:
            return await scraper.get_dialogs()
    return asyncio.run(_run())


def get_channel_info_sync(channel_id: str) -> Dict[str, Any]:
    """Get channel info (sync wrapper)."""
    async def _run():
        async with TelegramScraper() as scraper:
            return await scraper.get_channel_info(channel_id)
    return asyncio.run(_run())


def get_media_messages_sync(
    channel_id: str,
    limit: int = 25,
    offset_id: int = 0,
    min_id: int = 0
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Get media messages (sync wrapper)."""
    async def _run():
        async with TelegramScraper() as scraper:
            return await scraper.get_media_messages(
                channel_id, limit, offset_id, min_id
            )
    return asyncio.run(_run())


def download_media_sync(
    channel_id: str,
    message_id: int,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Download media (sync wrapper)."""
    async def _run():
        async with TelegramScraper() as scraper:
            return await scraper.download_media(channel_id, message_id, output_dir)
    return asyncio.run(_run())
