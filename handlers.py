"""
Telegram bot handlers for Claude Code bridge.

Handles commands, callbacks, and message forwarding to Claude sessions.
"""
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import ContextTypes

from config import GENERAL_TOPIC_ID
from utils import get_project_folders
from session import sessions, start_session, send_to_claude


async def handle_new_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new command to create a topic and show folder picker."""
    message = update.message
    if message is None:
        return

    # Only allow in General topic
    if message.message_thread_id not in (None, GENERAL_TOPIC_ID):
        return

    # Get topic name from command arguments
    if not context.args:
        await message.reply_text("Usage: /new topic-name")
        return

    topic_name = " ".join(context.args)

    try:
        # Create the new forum topic
        forum_topic = await context.bot.create_forum_topic(
            chat_id=message.chat_id,
            name=topic_name
        )
        await message.reply_text(f"Created topic: {topic_name}")

        # Get project folders and build keyboard
        folders = get_project_folders()
        if not folders:
            await context.bot.send_message(
                chat_id=message.chat_id,
                message_thread_id=forum_topic.message_thread_id,
                text="No project folders found in ~/Projects"
            )
            return

        # Build folder picker keyboard
        keyboard = []
        for folder in folders[:20]:  # Limit to 20 folders
            keyboard.append([InlineKeyboardButton(
                folder,
                callback_data=f"folder:{forum_topic.message_thread_id}:{folder}"
            )])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=message.chat_id,
            message_thread_id=forum_topic.message_thread_id,
            text="Select a project folder to start Claude session:",
            reply_markup=reply_markup
        )
    except Exception as e:
        await message.reply_text(f"Failed to create topic: {e}")


async def handle_topic_created(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle automatic detection of new forum topic creation."""
    message = update.message
    if message is None or message.forum_topic_created is None:
        return

    thread_id = message.message_thread_id
    chat_id = message.chat_id

    # Get project folders and build keyboard
    folders = get_project_folders()
    if not folders:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text="No project folders found in ~/Projects"
        )
        return

    # Build folder picker keyboard
    keyboard = []
    for folder in folders[:20]:  # Limit to 20 folders
        keyboard.append([InlineKeyboardButton(
            folder,
            callback_data=f"folder:{thread_id}:{folder}"
        )])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=thread_id,
        text="Select a project folder to start Claude session:",
        reply_markup=reply_markup
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button clicks."""
    query = update.callback_query
    if query is None:
        return

    await query.answer()  # Dismiss the loading state

    data = query.data
    if data is None:
        return

    # Handle folder selection: "folder:<thread_id>:<folder_name>"
    if data.startswith("folder:"):
        parts = data.split(":", 2)
        if len(parts) == 3:
            _, thread_id_str, folder_name = parts
            thread_id = int(thread_id_str)

            callback_message = query.message
            if callback_message is None or not isinstance(callback_message, Message):
                return

            chat_id = callback_message.chat.id

            # Update message to show selection
            await callback_message.edit_text(f"Starting Claude session in <code>{folder_name}</code>...", parse_mode="HTML")

            # Start the Claude session
            success = await start_session(chat_id, thread_id, folder_name, context.bot)
            if not success:
                await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=f"Failed to start session: folder '{folder_name}' not found"
                )
            return

    # Unknown callback - ignore


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages and forward to Claude session if active."""
    message = update.message
    if message is None:
        return

    thread_id = message.message_thread_id
    text = message.text
    if text is None:
        return

    # Check if this thread has an active Claude session
    if thread_id and thread_id in sessions:
        await send_to_claude(thread_id, text, context.bot)
        return

    # Ignore messages in General topic (no echo needed)
    if thread_id in (None, GENERAL_TOPIC_ID):
        return
