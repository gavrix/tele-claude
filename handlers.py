"""
Telegram bot handlers for Claude Code bridge.

Handles commands, callbacks, and message forwarding to Claude sessions.
"""
import asyncio
import os
import tempfile
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import ContextTypes

import logging

# Use standard logging - logger.py's setup_logging() will configure the handlers
logger = logging.getLogger("tele-claude.handlers")

from config import GENERAL_TOPIC_ID, ALLOWED_CHATS
from utils import get_project_folders
from session import sessions, start_session, send_to_claude, resolve_permission, interrupt_session
from commands import get_command_prompt, get_help_message


def is_authorized_chat(chat_id: int | None) -> bool:
    """Check if a chat is authorized to use the bot.

    Returns True if:
    - ALLOWED_CHATS is empty (no restrictions configured)
    - chat_id is in the ALLOWED_CHATS set
    """
    if not ALLOWED_CHATS:
        # No restrictions configured - allow all (backwards compatible)
        return True
    if chat_id is None:
        return False
    return chat_id in ALLOWED_CHATS


async def handle_new_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new command to create a topic and show folder picker."""
    message = update.message
    if message is None:
        return

    # Authorization check
    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized /new attempt from chat {message.chat_id}")
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
        logger.error(f"Failed to create topic '{topic_name}': {e}", exc_info=True)
        await message.reply_text(f"Failed to create topic: {e}")


async def handle_topic_created(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle automatic detection of new forum topic creation."""
    message = update.message
    if message is None or message.forum_topic_created is None:
        return

    # Authorization check
    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized topic creation from chat {message.chat_id}")
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

    # Authorization check
    callback_message = query.message
    chat_id = callback_message.chat.id if callback_message else None
    if not is_authorized_chat(chat_id):
        logger.warning(f"Unauthorized callback from chat {chat_id}")
        await query.answer("Unauthorized", show_alert=True)
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

    # Handle permission responses: "perm:<action>:<request_id>:<tool_name>"
    if data.startswith("perm:"):
        parts = data.split(":", 3)
        if len(parts) == 4:
            _, action, request_id, tool_name = parts

            callback_message = query.message
            if callback_message is None or not isinstance(callback_message, Message):
                return

            # Try to get logger from session for this thread
            perm_thread_id: Optional[int] = callback_message.message_thread_id
            session = sessions.get(perm_thread_id) if perm_thread_id else None
            if session and session.logger:
                session.logger.log_permission_callback(request_id, action, tool_name)
                session.logger.log_debug("callback", f"Calling resolve_permission",
                    request_id=request_id, action=action, sessions_keys=list(sessions.keys()))

            if action == "allow":
                success = await resolve_permission(request_id, allowed=True, always=False, tool_name=tool_name)
                if session and session.logger:
                    session.logger.log_debug("callback", f"resolve_permission returned {success}", request_id=request_id)
                if success:
                    await callback_message.edit_text(
                        f"✅ Allowed <code>{tool_name}</code> (one-time)",
                        parse_mode="HTML"
                    )
                else:
                    await callback_message.edit_text("⚠️ Permission request expired")

            elif action == "deny":
                success = await resolve_permission(request_id, allowed=False, always=False, tool_name=tool_name)
                if session and session.logger:
                    session.logger.log_debug("callback", f"resolve_permission (deny) returned {success}", request_id=request_id)
                if success:
                    await callback_message.edit_text(
                        f"❌ Denied <code>{tool_name}</code>",
                        parse_mode="HTML"
                    )
                else:
                    await callback_message.edit_text("⚠️ Permission request expired")

            elif action == "always":
                success = await resolve_permission(request_id, allowed=True, always=True, tool_name=tool_name)
                if session and session.logger:
                    session.logger.log_debug("callback", f"resolve_permission (always) returned {success}", request_id=request_id)
                if success:
                    await callback_message.edit_text(
                        f"✅ Always allowed <code>{tool_name}</code>",
                        parse_mode="HTML"
                    )
                else:
                    await callback_message.edit_text("⚠️ Permission request expired")

            return

    # Unknown callback - ignore


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command - show all available commands."""
    message = update.message
    if message is None:
        return

    # Authorization check
    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized /help attempt from chat {message.chat_id}")
        return

    thread_id = message.message_thread_id
    chat_id = message.chat_id

    # Get contextual commands if in an active session
    contextual_commands: list = []
    if thread_id and thread_id in sessions:
        contextual_commands = sessions[thread_id].contextual_commands

    help_text = get_help_message(contextual_commands)
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=thread_id,
        text=help_text,
        parse_mode="HTML"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages - save to temp dir and send to Claude."""
    message = update.message
    if message is None or not message.photo:
        return

    # Authorization check
    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized photo from chat {message.chat_id}")
        return

    thread_id = message.message_thread_id
    if not thread_id or thread_id not in sessions:
        return

    session = sessions[thread_id]

    # Get largest photo (last in array)
    photo = message.photo[-1]

    # Download photo to system temp directory
    file = await context.bot.get_file(photo.file_id)
    photo_path = os.path.join(tempfile.gettempdir(), f"telegram_photo_{photo.file_unique_id}.jpg")
    await file.download_to_drive(photo_path)

    caption = message.caption
    if caption:
        # Image with caption - send immediately
        prompt = f"{photo_path}\n\n{caption}"

        # Interrupt any ongoing query first
        was_interrupted = await interrupt_session(thread_id)
        if was_interrupted:
            await asyncio.sleep(0.1)

        asyncio.create_task(send_to_claude(thread_id, prompt, context.bot))
    else:
        # Image without caption - buffer silently, wait for next text message
        session.pending_image_path = photo_path


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages and forward to Claude session if active."""
    message = update.message
    if message is None:
        return

    # Authorization check
    if not is_authorized_chat(message.chat_id):
        logger.warning(f"Unauthorized message from chat {message.chat_id}")
        return

    thread_id = message.message_thread_id
    text = message.text
    if text is None:
        return

    # Check if this thread has an active Claude session
    if thread_id and thread_id in sessions:
        session = sessions[thread_id]

        # Check for slash commands
        prompt = text
        if text.startswith("/"):
            # Extract command name (handle /cmd or /cmd@botname)
            command_part = text.split()[0]
            command_name = command_part.lstrip("/").split("@")[0]

            # Look up the command
            cmd_prompt = get_command_prompt(command_name, session.contextual_commands)
            if cmd_prompt is not None:
                # Use command's prompt (silently, no echo)
                prompt = cmd_prompt
                logger.debug(f"Executing slash command /{command_name}")
            else:
                # Unknown command - pass through to Claude as-is
                logger.debug(f"Unknown command /{command_name}, passing to Claude")

        # Check for pending image
        pending_image = session.pending_image_path
        if pending_image:
            session.pending_image_path = None  # Clear it
            prompt = f"{pending_image}\n\n{prompt}"

        # Interrupt any ongoing query first
        was_interrupted = await interrupt_session(thread_id)
        if was_interrupted:
            # Small delay to let interrupt complete
            await asyncio.sleep(0.1)

        # Run as background task - don't await!
        # Awaiting would block callback processing (deadlock for permission buttons)
        asyncio.create_task(send_to_claude(thread_id, prompt, context.bot))
        return

    # Ignore messages in General topic (no echo needed)
    if thread_id in (None, GENERAL_TOPIC_ID):
        return
