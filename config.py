import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROJECTS_DIR = Path(os.getenv("PROJECTS_DIR", Path.home() / "Projects"))
GENERAL_TOPIC_ID = 0

# Authorized chat IDs - only these group chats can use the bot
# Set via ALLOWED_CHATS env var as comma-separated Telegram chat IDs
# Example: ALLOWED_CHATS=-1001234567890,-1009876543210
_allowed_chats_str = os.getenv("ALLOWED_CHATS", "")
ALLOWED_CHATS: set[int] = {
    int(cid.strip()) for cid in _allowed_chats_str.split(",") if cid.strip()
}
