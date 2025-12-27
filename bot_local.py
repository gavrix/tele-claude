#!/usr/bin/env python3
"""
Local project bot entry point.

Launches a bot instance anchored to the CURRENT WORKING DIRECTORY.
No folder picker - every new topic auto-starts a session in CWD.

Usage:
  cd /path/to/your/project
  python /path/to/tele-bot/bot_local.py

Requires .env.telebot in CWD with:
  BOT_TOKEN=your-telegram-bot-token
  ALLOWED_CHATS=-100xxxxx  # optional
"""
import os
import sys
from pathlib import Path

# Get CWD BEFORE any imports that might change it
LOCAL_CWD = Path.cwd().resolve()

# Load .env.telebot from CWD
from dotenv import load_dotenv
env_file = LOCAL_CWD / ".env.telebot"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    print(f"Error: {env_file} not found", file=sys.stderr)
    print(f"Create .env.telebot in {LOCAL_CWD} with BOT_TOKEN=...", file=sys.stderr)
    sys.exit(1)

# Now import everything else (config.py will run but we've already loaded our env)
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters

from handlers import handle_callback, handle_message, handle_photo, handle_help, is_authorized_chat
from session import start_session_local, sessions
from logger import setup_logging

# Setup logging with prefix for this instance
setup_logging()
logger = logging.getLogger("tele-claude.bot_local")


async def handle_topic_created_local(update: Update, context) -> None:
    """Handle new topic creation - auto-start session with CWD folder."""
    message = update.message
    if message is None or message.forum_topic_created is None:
        return

    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized topic creation from chat {message.chat_id}")
        return

    thread_id = message.message_thread_id
    chat_id = message.chat_id

    if thread_id is None:
        return

    # Get local project folder from app context
    local_dir = context.application.bot_data.get("local_project_dir")
    local_name = context.application.bot_data.get("local_project_name")

    logger.info(f"New topic {thread_id} - starting session in {local_dir}")

    success = await start_session_local(chat_id, thread_id, local_dir, context.bot)
    if not success:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=f"❌ Failed to start session in {local_name}"
        )


def main() -> None:
    bot_token = os.getenv("BOT_TOKEN")

    if not bot_token:
        print("Error: BOT_TOKEN not found in .env.telebot", file=sys.stderr)
        sys.exit(1)

    if not LOCAL_CWD.exists():
        print(f"Error: Directory does not exist: {LOCAL_CWD}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting bot anchored to: {LOCAL_CWD}")
    print(f"Project name: {LOCAL_CWD.name}")
    logger.info(f"Bot starting for project: {LOCAL_CWD}")

    app = Application.builder().token(bot_token).build()

    # Store local project path in bot_data
    app.bot_data["local_project_dir"] = str(LOCAL_CWD)
    app.bot_data["local_project_name"] = LOCAL_CWD.name

    # Handle new forum topic created - auto-start session
    app.add_handler(MessageHandler(
        filters.StatusUpdate.FORUM_TOPIC_CREATED & filters.ChatType.SUPERGROUP,
        handle_topic_created_local
    ))

    # Handle /help command
    app.add_handler(CommandHandler(
        "help",
        handle_help,
        filters=filters.ChatType.SUPERGROUP
    ))

    # Handle inline keyboard button clicks
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Handle text messages
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.SUPERGROUP,
        handle_message
    ))

    # Handle photos
    app.add_handler(MessageHandler(
        filters.PHOTO & filters.ChatType.SUPERGROUP,
        handle_photo
    ))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
