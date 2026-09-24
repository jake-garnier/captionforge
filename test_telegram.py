#!/usr/bin/env python3
"""
Ad-hoc script: verify a Telegram bot token and test video uploading.

Usage:
    export TELEGRAM_BOT_TOKEN=123456:ABC...   # from @BotFather
    python test_telegram.py                    # verify token, list recent chats
    python test_telegram.py <chat_id> [video]  # send a test message / video
"""

import httpx
import asyncio
import os
import sys
from pathlib import Path

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    sys.exit("Set TELEGRAM_BOT_TOKEN in the environment (get one from @BotFather).")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


async def get_bot_info():
    """Verify bot token is valid."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/getMe")
        return response.json()


async def get_updates():
    """Get recent messages/updates to find chat IDs."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/getUpdates")
        return response.json()


async def send_message(chat_id: str, text: str):
    """Send a test message."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/sendMessage",
            data={"chat_id": chat_id, "text": text}
        )
        return response.json()


async def send_video(chat_id: str, video_path: str, caption: str = "Test video"):
    """Send a video to a chat/channel."""
    file_path = Path(video_path)
    if not file_path.exists():
        return {"ok": False, "description": f"File not found: {video_path}"}

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {file_path.name} ({file_size_mb:.1f} MB)...")

    async with httpx.AsyncClient(timeout=120) as client:
        with open(video_path, "rb") as f:
            response = await client.post(
                f"{BASE_URL}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": caption,
                    "supports_streaming": "true",
                },
                files={"video": f}
            )
    return response.json()


async def main():
    # Step 1: Verify bot
    print("=" * 50)
    print("Step 1: Verifying bot token...")
    print("=" * 50)
    bot_info = await get_bot_info()
    if bot_info.get("ok"):
        bot = bot_info["result"]
        print(f"✓ Bot verified: @{bot['username']} ({bot['first_name']})")
    else:
        print(f"✗ Bot verification failed: {bot_info}")
        return

    # Step 2: Check for updates (to find chat IDs)
    print("\n" + "=" * 50)
    print("Step 2: Checking for recent messages...")
    print("=" * 50)
    updates = await get_updates()
    if updates.get("ok") and updates.get("result"):
        print(f"Found {len(updates['result'])} updates:")
        for update in updates["result"][-5:]:  # Last 5
            if "message" in update:
                msg = update["message"]
                chat = msg.get("chat", {})
                print(f"  - Chat ID: {chat.get('id')} | Type: {chat.get('type')} | Title/Name: {chat.get('title') or chat.get('first_name')}")
            if "channel_post" in update:
                post = update["channel_post"]
                chat = post.get("chat", {})
                print(f"  - Channel ID: {chat.get('id')} | Title: {chat.get('title')}")
    else:
        print("No recent updates found.")
        print("\nTo get your chat/channel ID:")
        print("1. Add the bot to your channel as admin")
        print("2. Send a message in the channel")
        print("3. Run this script again")
        print("\nOr message the bot directly to get your personal chat ID")

    # Step 3: If chat_id provided, test sending
    if len(sys.argv) >= 2:
        chat_id = sys.argv[1]
        video_path = sys.argv[2] if len(sys.argv) >= 3 else None

        print("\n" + "=" * 50)
        print(f"Step 3: Testing send to chat_id: {chat_id}")
        print("=" * 50)

        # Test message first
        print("Sending test message...")
        msg_result = await send_message(chat_id, "🎬 Test message from captions bot!")
        if msg_result.get("ok"):
            print(f"✓ Message sent! Message ID: {msg_result['result']['message_id']}")
        else:
            print(f"✗ Message failed: {msg_result.get('description')}")
            return

        # Test video if path provided
        if video_path:
            print(f"\nSending video: {video_path}")
            video_result = await send_video(chat_id, video_path, "🎬 Test video upload!")
            if video_result.get("ok"):
                msg_id = video_result['result']['message_id']
                print(f"✓ Video sent! Message ID: {msg_id}")

                # Try to construct URL
                chat_info = video_result['result']['chat']
                if chat_info.get('username'):
                    print(f"✓ Post URL: https://t.me/{chat_info['username']}/{msg_id}")
            else:
                print(f"✗ Video failed: {video_result.get('description')}")
        else:
            print("\nNo video path provided. Usage:")
            print(f"  python {sys.argv[0]} <chat_id> <video_path>")
    else:
        print("\n" + "=" * 50)
        print("Next steps:")
        print("=" * 50)
        print("1. Add your bot to your channel as admin")
        print("2. Post something in the channel (or message the bot directly)")
        print("3. Run: python test_telegram.py")
        print("4. Copy the chat/channel ID from the output")
        print("5. Run: python test_telegram.py <chat_id> /path/to/video.mp4")


if __name__ == "__main__":
    asyncio.run(main())
