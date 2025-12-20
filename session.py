"""
Claude Code SDK session management for Telegram bot.

Manages conversations with Claude through the Code SDK,
streaming responses to Telegram messages.
"""
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Any, Union

import mistune
from telegram import Bot, Message, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ProcessError, PermissionResultAllow, PermissionResultDeny, HookMatcher, HookContext
from claude_agent_sdk.types import (
    SystemPromptPreset, ToolPermissionContext, HookInput, HookJSONOutput,
    UserMessage, AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
)

from config import PROJECTS_DIR
from logger import SessionLogger
from diff_image import edit_to_image
from commands import load_contextual_commands, register_commands_for_chat
from mcp_tools import create_telegram_mcp_server

# Module logger (named _log to avoid collision with SessionLogger variables named 'logger')
_log = logging.getLogger("tele-claude.session")


# Context window sizes by model (tokens)
MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-5-20251101": 200000,
    "claude-sonnet-4-5-20251101": 200000,
    "claude-sonnet-4-20250514": 200000,
    "default": 200000,
}

# Warn user when context remaining drops below this percentage
CONTEXT_WARNING_THRESHOLD = 15

# Tools that are always allowed without prompting
DEFAULT_ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebSearch",
    "mcp__telegram-tools__send_to_telegram",  # Custom tool for sending files to chat
]

# Persistent allowlist file
ALLOWLIST_FILE = Path(__file__).parent / "tool_allowlist.json"

# Pending permission requests: request_id -> (Future, SessionLogger)
pending_permissions: dict[str, tuple[asyncio.Future, Optional["SessionLogger"]]] = {}


def load_allowlist() -> set[str]:
    """Load the persistent tool allowlist."""
    if ALLOWLIST_FILE.exists():
        try:
            with open(ALLOWLIST_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_allowlist(tools: set[str]) -> None:
    """Save the persistent tool allowlist."""
    try:
        with open(ALLOWLIST_FILE, "w") as f:
            json.dump(list(tools), f)
    except Exception:
        pass


def add_to_allowlist(tool_name: str) -> None:
    """Add a tool to the persistent allowlist."""
    tools = load_allowlist()
    tools.add(tool_name)
    save_allowlist(tools)


def is_tool_allowed(tool_name: str) -> bool:
    """Check if a tool is in the default or persistent allowlist."""
    if tool_name in DEFAULT_ALLOWED_TOOLS:
        return True
    return tool_name in load_allowlist()


async def resolve_permission(request_id: str, allowed: bool, always: bool = False, tool_name: Optional[str] = None) -> bool:
    """Resolve a pending permission request."""
    entry = pending_permissions.pop(request_id, None)
    if entry is None:
        _log.warning(f"Permission future not found for request_id={request_id}, pending_keys={list(pending_permissions.keys())}")
        return False

    future, logger = entry

    if logger:
        logger.log_permission_resolved(request_id, allowed, found=True)

    if always and tool_name:
        add_to_allowlist(tool_name)

    future.set_result(allowed)
    return True


@dataclass
class ClaudeSession:
    """Represents an active Claude session for a Telegram thread."""
    chat_id: int
    thread_id: int
    cwd: str
    bot: Optional[Bot] = None  # Reference for permission prompts
    logger: Optional[SessionLogger] = None
    last_send: float = field(default_factory=time.time)
    send_interval: float = 1.0
    last_typing_action: float = 0.0
    active: bool = True
    session_id: Optional[str] = None  # For multi-turn conversation
    client: Optional[ClaudeSDKClient] = None  # Active SDK client for interrupt support
    last_context_percent: Optional[float] = None  # Last known context remaining %
    pending_image_path: Optional[str] = None  # Buffered image waiting for prompt
    contextual_commands: list = field(default_factory=list)  # Project-specific slash commands


async def interrupt_session(thread_id: int) -> bool:
    """Interrupt the active Claude response for a session.

    Returns True if an active query was interrupted, False otherwise.
    """
    session = sessions.get(thread_id)
    if not session or not session.client:
        return False

    await session.client.interrupt()
    return True


async def request_tool_permission(
    session: ClaudeSession,
    tool_name: str,
    tool_input: dict
) -> bool:
    """Send permission request to Telegram and wait for user response."""
    if session.bot is None:
        if session.logger:
            session.logger.log_error("request_tool_permission", Exception("No bot reference available"))
        return False

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]

    # Format tool input for display
    input_preview = []
    for k, v in list(tool_input.items())[:3]:  # Show first 3 args
        v_str = str(v)
        if len(v_str) > 100:
            v_str = v_str[:100] + "..."
        v_str = escape_html(v_str)
        input_preview.append(f"  <code>{k}</code>: {v_str}")
    input_text = "\n".join(input_preview) if input_preview else "  (no arguments)"

    # Build message with inline keyboard
    message_text = (
        f"🔐 <b>Permission Request</b>\n\n"
        f"Tool: <code>{tool_name}</code>\n"
        f"Arguments:\n{input_text}"
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Allow", callback_data=f"perm:allow:{request_id}:{tool_name}"),
            InlineKeyboardButton("❌ Deny", callback_data=f"perm:deny:{request_id}:{tool_name}"),
        ],
        [
            InlineKeyboardButton("✅ Always Allow", callback_data=f"perm:always:{request_id}:{tool_name}"),
        ]
    ])

    # Send permission request message
    await session.bot.send_message(
        chat_id=session.chat_id,
        message_thread_id=session.thread_id,
        text=message_text,
        parse_mode="HTML",
        reply_markup=keyboard
    )

    # Log the permission request
    if session.logger:
        session.logger.log_permission_request(request_id, tool_name, tool_input)

    # Create future and wait for response
    # Use get_running_loop() - get_event_loop() is deprecated and may return wrong loop
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending_permissions[request_id] = (future, session.logger)

    if session.logger:
        session.logger.log_debug("permission", f"Waiting for user response", request_id=request_id, pending_keys=list(pending_permissions.keys()))

    try:
        # Wait for user response (timeout after 5 minutes)
        allowed = await asyncio.wait_for(future, timeout=300.0)
        if session.logger:
            session.logger.log_debug("permission", f"Got user response: {allowed}", request_id=request_id)
        return allowed
    except asyncio.TimeoutError:
        pending_permissions.pop(request_id, None)
        if session.logger:
            session.logger.log_debug("permission", "Request timed out", request_id=request_id)
        await session.bot.send_message(
            chat_id=session.chat_id,
            message_thread_id=session.thread_id,
            text="⏰ Permission request timed out (denied)"
        )
        return False


def create_permission_handler(session: ClaudeSession):
    """Create a can_use_tool callback for the given session."""
    async def handle_permission(
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext
    ) -> Union[PermissionResultAllow, PermissionResultDeny]:
        try:
            in_allowlist = is_tool_allowed(tool_name)

            if session.logger:
                session.logger.log_permission_check(tool_name, in_allowlist)

            # Check if tool is in allowlist
            if in_allowlist:
                if session.logger:
                    session.logger.log_debug("permission_handler", f"Returning PermissionResultAllow (allowlist)")
                return PermissionResultAllow(updated_input=tool_input)

            # Request permission from user
            allowed = await request_tool_permission(session, tool_name, tool_input)

            if allowed:
                if session.logger:
                    session.logger.log_debug("permission_handler", f"Returning PermissionResultAllow (user allowed)")
                return PermissionResultAllow(updated_input=tool_input)
            else:
                if session.logger:
                    session.logger.log_debug("permission_handler", f"Returning PermissionResultDeny (user denied)")
                return PermissionResultDeny(message=f"User denied permission for {tool_name}")
        except Exception as e:
            if session.logger:
                session.logger.log_error("permission_handler", e)
            # Return deny on error
            return PermissionResultDeny(message=f"Permission error: {str(e)}")

    return handle_permission


def create_pre_compact_hook(session: ClaudeSession):
    """Create a PreCompact hook to log and notify user when context is being compacted."""
    async def handle_pre_compact(
        input_data: HookInput,
        tool_use_id: Optional[str],
        context: HookContext
    ) -> HookJSONOutput:
        """Called before the SDK compacts conversation history."""
        try:
            # Log the compaction event with full input data for analysis
            if session.logger:
                # Cast to dict for logging since HookInput is a TypedDict
                session.logger.log_compact_event(dict(input_data))  # type: ignore[arg-type]

            # Notify user in chat
            if session.bot:
                await session.bot.send_message(
                    chat_id=session.chat_id,
                    message_thread_id=session.thread_id,
                    text="📦 <b>Context compacting...</b>\nConversation history is being summarized to free up space.",
                    parse_mode="HTML"
                )
        except Exception as e:
            if session.logger:
                session.logger.log_error("pre_compact_hook", e)

        # Return async hook output to allow compaction to proceed
        return {"async_": True}

    return handle_pre_compact


# Active sessions: thread_id -> ClaudeSession
sessions: dict[int, ClaudeSession] = {}

# Minimum seconds between sends to avoid flood control
MIN_SEND_INTERVAL = 1.0

# Typing action expires after ~5s, resend every 4s
TYPING_ACTION_INTERVAL = 4.0


def calculate_context_remaining(usage: Optional[dict[str, Any]], model: str = "default") -> Optional[float]:
    """Calculate percentage of context window remaining from usage data.

    Returns percentage remaining (0-100), or None if usage data insufficient.

    Note: We intentionally exclude cache_read_input_tokens from the calculation.
    The cache_read tokens appear to include accumulated reads from server-side
    tools (like web_search, web_fetch) that don't represent actual conversation
    context. The official docs confirm this issue:
    https://platform.claude.com/docs/en/build-with-claude/context-editing#client-side-compaction-sdk

    "When using server-side tools, the SDK may incorrectly calculate token usage...
    the cache_read_input_tokens value includes accumulated reads from multiple
    internal API calls made by the server-side tool, not your actual conversation
    context."

    We'll revisit this calculation when we observe an actual PreCompact event
    and can correlate the token counts with real context exhaustion.
    """
    if not usage:
        return None

    # Only count non-cached tokens: input + output
    # Exclude cache_read_input_tokens (seems to include shared system cache)
    # Exclude cache_creation_input_tokens (represents what's being cached, not consumed)
    total_tokens = (
        usage.get("input_tokens", 0) +
        usage.get("output_tokens", 0)
    )

    if total_tokens == 0:
        return None

    # Get context window for model
    context_window = MODEL_CONTEXT_WINDOWS.get(model, MODEL_CONTEXT_WINDOWS["default"])

    # Calculate remaining percentage
    used_percent = (total_tokens / context_window) * 100
    remaining_percent = 100 - used_percent

    return max(0, remaining_percent)


async def start_session(chat_id: int, thread_id: int, folder_name: str, bot: Bot) -> bool:
    """Start a new Claude session for a Telegram thread."""
    cwd = PROJECTS_DIR / folder_name
    if not cwd.exists():
        return False

    # Create logger
    logger = SessionLogger(thread_id, chat_id, str(cwd))

    # Load contextual commands from project's commands/ directory
    contextual_commands = load_contextual_commands(str(cwd))

    # Store session with bot reference
    sessions[thread_id] = ClaudeSession(
        chat_id=chat_id,
        thread_id=thread_id,
        cwd=str(cwd),
        bot=bot,
        logger=logger,
        contextual_commands=contextual_commands,
    )

    # Register commands with Telegram for autocompletion
    await register_commands_for_chat(bot, chat_id, contextual_commands)

    # Send welcome message
    await bot.send_message(
        chat_id=chat_id,
        message_thread_id=thread_id,
        text=f"Claude session started in <code>{folder_name}</code>",
        parse_mode="HTML"
    )

    return True


async def stop_session(thread_id: int) -> bool:
    """Stop and clean up a Claude session."""
    session = sessions.get(thread_id)
    if not session:
        return False

    session.active = False

    # Close logger
    if session.logger:
        session.logger.log_session_end("stopped")
        session.logger.close()

    del sessions[thread_id]
    return True


async def send_to_claude(thread_id: int, prompt: str, bot: Bot) -> None:
    """Send a message to Claude and stream the response to Telegram."""
    session = sessions.get(thread_id)
    if not session or not session.active:
        return

    # Log user input
    if session.logger:
        session.logger.log_user_input(prompt)

    # Send typing indicator
    await send_typing_action(session, bot)

    # Track current response message for streaming edits
    response_msg: Optional[Message] = None
    response_text = ""
    response_msg_text_len = 0  # Length of text in current response_msg

    # Buffer for batching consecutive tool calls of same type
    tool_buffer: list[tuple[str, dict]] = []  # [(tool_name, input), ...]
    tool_buffer_name: Optional[str] = None  # Current tool type being buffered
    tool_buffer_msg: Optional[Message] = None  # Message being edited for batch

    # Buffer for diff images to send as media group at end
    diff_images: list[tuple[BytesIO, str]] = []  # [(image_buffer, filename), ...]

    # Track model and usage for context calculation
    current_model: Optional[str] = None
    last_usage: Optional[dict] = None

    def format_current_tool_buffer() -> str:
        """Format current tool buffer contents."""
        if len(tool_buffer) == 1:
            name, input_dict = tool_buffer[0]
            return format_tool_call(name, input_dict)
        else:
            return format_tool_calls_batch(tool_buffer_name or "Tool", tool_buffer)

    async def update_tool_buffer_message():
        """Send or edit the tool buffer message."""
        nonlocal tool_buffer_msg
        text = format_current_tool_buffer()

        if tool_buffer_msg:
            # Edit existing message
            try:
                await bot.edit_message_text(
                    chat_id=session.chat_id,
                    message_id=tool_buffer_msg.message_id,
                    text=text,
                    parse_mode="HTML"
                )
            except Exception as e:
                if session.logger and "message is not modified" not in str(e).lower():
                    session.logger.log_error("update_tool_buffer_message", e)
        else:
            # Send new message
            tool_buffer_msg = await send_message(session, bot, text, parse_mode="HTML")

    async def flush_tool_buffer():
        """Clear tool buffer state (message already sent/edited)."""
        nonlocal tool_buffer, tool_buffer_name, tool_buffer_msg
        tool_buffer = []
        tool_buffer_name = None
        tool_buffer_msg = None

    try:
        # Check if AGENTS.md exists and pre-load its content into system prompt
        agents_md_path = Path(session.cwd) / "AGENTS.md"
        system_prompt: Optional[SystemPromptPreset] = None
        if agents_md_path.exists():
            try:
                agents_content = agents_md_path.read_text()
                system_prompt = SystemPromptPreset(
                    type="preset",
                    preset="claude_code",
                    append=f"# Project Context (from AGENTS.md)\n\n{agents_content}"
                )
            except Exception:
                # If we can't read the file, fall back to instruction-based approach
                system_prompt = SystemPromptPreset(
                    type="preset",
                    preset="claude_code",
                    append="IMPORTANT: This project has an AGENTS.md file in the root directory. "
                           "Read it at the start of the session to understand project context and instructions."
                )

        # Create telegram MCP server bound to this session
        telegram_mcp = create_telegram_mcp_server(session)

        # Configure options - use permission handler for interactive tool approval
        options = ClaudeAgentOptions(
            allowed_tools=[],  # Empty - let can_use_tool handle all permissions
            can_use_tool=create_permission_handler(session),
            permission_mode="acceptEdits",
            cwd=session.cwd,
            resume=session.session_id,  # Resume previous conversation if exists
            system_prompt=system_prompt,
            mcp_servers={
                "telegram-tools": telegram_mcp,
            },
            hooks={
                "PreCompact": [
                    HookMatcher(hooks=[create_pre_compact_hook(session)])
                ]
            }
        )

        # Query Claude using ClaudeSDKClient (required for can_use_tool support)
        # can_use_tool requires streaming mode - wrap prompt in async generator
        # Format from SDK examples/streaming_mode.py
        async def prompt_stream():
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": prompt
                },
                "parent_tool_use_id": None,
                "session_id": session.session_id or "default"
            }

        async with ClaudeSDKClient(options=options) as client:
            # Store client reference for interrupt support
            session.client = client

            await client.query(prompt_stream())
            async for message in client.receive_response():
                # Refresh typing indicator on each message
                await send_typing_action(session, bot)

                # Log SDK message
                if session.logger:
                    session.logger.log_sdk_message(message)

                # Handle different message types based on their class
                if isinstance(message, (UserMessage, AssistantMessage)):
                    content = message.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, TextBlock):
                                # Text content - flush tools first, then accumulate text
                                await flush_tool_buffer()

                                response_text += block.text
                                response_msg, response_msg_text_len = await send_or_edit_response(
                                    session, bot, response_msg, response_text, response_msg_text_len
                                )

                            elif isinstance(block, ToolUseBlock):
                                # Tool use block - buffer it
                                if response_text.strip():
                                    response_msg, response_msg_text_len = await send_or_edit_response(
                                        session, bot, response_msg, response_text, response_msg_text_len
                                    )
                                    response_msg = None
                                    response_text = ""
                                    response_msg_text_len = 0

                                tool_name = block.name
                                tool_input = block.input

                                # If different tool type, flush buffer first
                                if tool_buffer_name and tool_buffer_name != tool_name:
                                    await flush_tool_buffer()

                                # Add to buffer and update message immediately
                                tool_buffer.append((tool_name, tool_input))
                                tool_buffer_name = tool_name
                                await update_tool_buffer_message()

                                # Buffer diff image for Edit tool
                                if tool_name == "Edit" and "old_string" in tool_input and "new_string" in tool_input:
                                    file_path = tool_input.get("file_path", "file")
                                    img_buffer = edit_to_image(
                                        file_path=file_path,
                                        old_string=tool_input["old_string"],
                                        new_string=tool_input["new_string"]
                                    )
                                    if img_buffer:
                                        filename = file_path.split("/")[-1] if "/" in file_path else file_path
                                        diff_images.append((img_buffer, filename))

                    elif isinstance(content, str) and content:
                        # Tool result - flush tool buffer first
                        await flush_tool_buffer()

                        output = format_tool_output(content)
                        if output:
                            # Escape HTML entities in output
                            safe_output = escape_html(output)
                            await send_message(session, bot, f"<pre>{safe_output}</pre>", parse_mode="HTML")

                        # Refresh typing - more content likely coming after tool result
                        await send_typing_action(session, bot)

                # Capture model from AssistantMessage
                if isinstance(message, AssistantMessage):
                    current_model = message.model

                # Capture session_id and usage from ResultMessage
                if isinstance(message, ResultMessage):
                    if message.session_id:
                        session.session_id = message.session_id
                    if message.usage:
                        last_usage = message.usage
                    # Log stats (but don't show cost to user - it's included in subscription)
                    cost = message.total_cost_usd
                    duration = message.duration_ms
                    if session.logger and cost is not None:
                        session.logger.log_session_stats(cost, duration, last_usage or {})

            # Flush any remaining buffers (inside async with, after loop)
            await flush_tool_buffer()
            if response_text.strip() and response_msg is None:
                await send_message(session, bot, response_text)

            # Send diff images as media group (gallery)
            if diff_images:
                await send_diff_images_gallery(session, bot, diff_images)

            # Calculate and store context remaining
            if last_usage:
                context_remaining = calculate_context_remaining(last_usage, current_model or "default")
                if context_remaining is not None:
                    session.last_context_percent = context_remaining

                    # Warn user if context is running low - append to last text response
                    if context_remaining < CONTEXT_WARNING_THRESHOLD:
                        if session.logger:
                            session.logger._write_log(f"CONTEXT WARNING: {context_remaining:.1f}% remaining")

                        warning = f"\n\n⚠️ {context_remaining:.0f}% context remaining"

                        # Only append to text response message (not tool messages)
                        if response_msg and response_msg_text_len > 0:
                            # Check if warning fits in current message
                            if response_msg_text_len + len(warning) <= 4000:
                                # Get the text currently in the message and append warning
                                # Use the portion of response_text that's in this message
                                current_msg_text = response_text[-response_msg_text_len:] if len(response_text) > response_msg_text_len else response_text
                                warning_text = current_msg_text + warning
                                try:
                                    html_text = markdown_to_html(warning_text)
                                    await bot.edit_message_text(
                                        chat_id=session.chat_id,
                                        message_id=response_msg.message_id,
                                        text=html_text,
                                        parse_mode="HTML"
                                    )
                                except Exception:
                                    # Edit failed, send as separate message
                                    await send_message(session, bot, f"⚠️ {context_remaining:.0f}% context remaining")
                            else:
                                # Warning doesn't fit, send separately
                                await send_message(session, bot, f"⚠️ {context_remaining:.0f}% context remaining")

            # Clear client reference when done
            session.client = None

    except ProcessError as e:
        # Log stderr from CLI process
        if session.logger:
            session.logger.log_error("send_to_claude", e)
            if e.stderr:
                session.logger.log_stderr(e.stderr)
        error_msg = f"❌ Error: {str(e)}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr[:500]}"
        await send_message(session, bot, error_msg)
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        await send_message(session, bot, error_msg)
        if session.logger:
            session.logger.log_error("send_to_claude", e)
    finally:
        # Always clear client reference
        session.client = None


async def send_or_edit_response(
    session: ClaudeSession,
    bot: Bot,
    existing_msg: Optional[Message],
    text: str,
    msg_text_len: int = 0
) -> tuple[Optional[Message], int]:
    """Send a new response message or edit an existing one.

    Handles overflow by starting new messages when text exceeds 4000 chars.
    Splits long responses into multiple messages to avoid truncation.

    Args:
        session: The Claude session
        bot: Telegram bot instance
        existing_msg: Existing message to edit, or None to send new
        text: Full accumulated text to display
        msg_text_len: Length of text already in existing_msg (for overflow detection)

    Returns:
        Tuple of (last message sent, length of text in that message)
    """
    if not text.strip():
        return existing_msg, msg_text_len

    # Check if we need to overflow to new messages
    if len(text) > 4000 and existing_msg and msg_text_len > 0:
        # Current message is full, send overflow text as new message(s)
        overflow_text = text[msg_text_len:]

        # Split overflow into chunks and send each as a new message
        chunks = split_text(overflow_text, 4000)
        last_msg: Optional[Message] = None
        last_len = 0

        for chunk in chunks:
            new_msg = await _send_with_fallback(session, bot, chunk, existing_msg=None)
            if new_msg:
                last_msg = new_msg
                last_len = len(chunk)

        return last_msg if last_msg else existing_msg, last_len if last_msg else msg_text_len

    # Text fits in one message - edit existing or send new
    display_text = text
    if len(display_text) > 4000:
        display_text = display_text[:3990] + "\n..."

    result_msg = await _send_with_fallback(session, bot, display_text, existing_msg=existing_msg)
    if result_msg:
        return result_msg, len(display_text)
    return existing_msg, msg_text_len


async def _send_with_fallback(
    session: ClaudeSession,
    bot: Bot,
    text: str,
    existing_msg: Optional[Message] = None,
    max_retries: int = 3
) -> Optional[Message]:
    """Send or edit a message with multiple fallback strategies.

    Fallback order:
    1. Try HTML formatted message
    2. If HTML fails, try plain text (stripped of HTML tags)
    3. If still failing (e.g., flood control), retry with exponential backoff

    Args:
        session: The Claude session
        bot: Telegram bot instance
        text: Text to send (markdown format)
        existing_msg: Existing message to edit, or None to send new
        max_retries: Maximum retry attempts for transient errors

    Returns:
        Message object if successful, None otherwise
    """
    # Strategy 1: Try with HTML formatting
    html_text = markdown_to_html(text)

    for attempt in range(max_retries):
        try:
            if existing_msg:
                await bot.edit_message_text(
                    chat_id=session.chat_id,
                    message_id=existing_msg.message_id,
                    text=html_text,
                    parse_mode="HTML"
                )
                return existing_msg
            else:
                return await bot.send_message(
                    chat_id=session.chat_id,
                    message_thread_id=session.thread_id,
                    text=html_text,
                    parse_mode="HTML"
                )
        except Exception as e:
            error_str = str(e).lower()

            # Skip logging for "message not modified" - not an error
            if "message is not modified" in error_str:
                return existing_msg

            if session.logger:
                session.logger.log_error("send_with_fallback_html", e)

            # Strategy 2: If HTML parsing failed, try plain text
            if "parse entities" in error_str or "can't parse" in error_str:
                plain_text = strip_html_tags(html_text)
                try:
                    if existing_msg:
                        await bot.edit_message_text(
                            chat_id=session.chat_id,
                            message_id=existing_msg.message_id,
                            text=plain_text
                        )
                        return existing_msg
                    else:
                        return await bot.send_message(
                            chat_id=session.chat_id,
                            message_thread_id=session.thread_id,
                            text=plain_text
                        )
                except Exception as plain_err:
                    if session.logger:
                        session.logger.log_error("send_with_fallback_plain", plain_err)
                    error_str = str(plain_err).lower()

            # Strategy 3: Retry with backoff for transient errors
            if "flood control" in error_str or "retry" in error_str or "timed out" in error_str:
                # Extract retry time if available, otherwise use exponential backoff
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                if "retry in" in error_str:
                    try:
                        # Try to extract the retry time from error message
                        import re
                        match = re.search(r'retry in (\d+)', error_str)
                        if match:
                            wait_time = min(int(match.group(1)), 30)  # Cap at 30 seconds
                    except Exception:
                        pass

                if attempt < max_retries - 1:
                    if session.logger:
                        session.logger.log_debug("send_with_fallback", f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue

            # Non-retryable error or max retries exceeded
            break

    return None


async def send_message(
    session: ClaudeSession,
    bot: Bot,
    text: str,
    parse_mode: Optional[str] = None
) -> Optional[Message]:
    """Send a new Telegram message with rate limiting."""
    if not text.strip():
        return None

    # Rate limiting
    now = time.time()
    elapsed = now - session.last_send
    if elapsed < session.send_interval:
        await asyncio.sleep(session.send_interval - elapsed)

    # Split if too long
    chunks = split_text(text, 4000)

    msg = None
    for chunk in chunks:
        try:
            msg = await bot.send_message(
                chat_id=session.chat_id,
                message_thread_id=session.thread_id,
                text=chunk,
                parse_mode=parse_mode
            )
            # Reset interval on success
            session.send_interval = MIN_SEND_INTERVAL
            session.last_send = time.time()
        except Exception as e:
            if "flood control" in str(e).lower():
                # Back off on rate limit
                session.send_interval = min(session.send_interval * 2, 30.0)
            if session.logger:
                session.logger.log_error("send_message", e)

    return msg


async def send_typing_action(session: ClaudeSession, bot: Bot) -> None:
    """Send typing indicator if enough time has passed."""
    now = time.time()
    if now - session.last_typing_action < TYPING_ACTION_INTERVAL:
        return

    try:
        await bot.send_chat_action(
            chat_id=session.chat_id,
            message_thread_id=session.thread_id,
            action=ChatAction.TYPING
        )
        session.last_typing_action = now
    except Exception as e:
        # Only log if it's not a rate limit (those are expected)
        if session.logger and "flood" not in str(e).lower():
            session.logger.log_debug("send_typing_action", f"Failed: {e}")


async def send_diff_images_gallery(
    session: ClaudeSession,
    bot: Bot,
    images: list[tuple[BytesIO, str]]
) -> None:
    """Send diff images as a media group (gallery)."""
    if not images:
        return

    try:
        if len(images) == 1:
            # Single image - send as photo with caption
            img_buffer, filename = images[0]
            await bot.send_photo(
                chat_id=session.chat_id,
                message_thread_id=session.thread_id,
                photo=img_buffer,
                caption=f"📝 {filename}"
            )
        else:
            # Multiple images - send as media group
            media = [
                InputMediaPhoto(
                    media=img_buffer,
                    caption=f"📝 {filename}"
                )
                for img_buffer, filename in images
            ]
            await bot.send_media_group(
                chat_id=session.chat_id,
                message_thread_id=session.thread_id,
                media=media
            )
    except Exception as e:
        if session.logger:
            session.logger.log_error("send_diff_images_gallery", e)


def escape_html(text: str) -> str:
    """Escape HTML entities for Telegram."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def strip_html_tags(text: str) -> str:
    """Remove all HTML tags from text, leaving only content."""
    return re.sub(r'<[^>]+>', '', text)


class TelegramHTMLRenderer(mistune.HTMLRenderer):
    """Custom mistune renderer that outputs Telegram-compatible HTML.

    Telegram only supports a subset of HTML tags:
    <b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>,
    <span class="tg-spoiler">, <a href="">, <code>, <pre>
    """

    def text(self, text: str) -> str:
        """Escape HTML entities in plain text."""
        return escape_html(text)

    def emphasis(self, text: str) -> str:
        """Render *italic* text."""
        return f'<i>{text}</i>'

    def strong(self, text: str) -> str:
        """Render **bold** text."""
        return f'<b>{text}</b>'

    def codespan(self, text: str) -> str:
        """Render `inline code`."""
        return f'<code>{escape_html(text)}</code>'

    def block_code(self, code: str, info: Optional[str] = None) -> str:
        """Render ```code blocks```."""
        return f'<pre>{escape_html(code)}</pre>\n'

    def link(self, text: str, url: str, title: Optional[str] = None) -> str:
        """Render [text](url) links."""
        return f'<a href="{escape_html(url)}">{text}</a>'

    def strikethrough(self, text: str) -> str:
        """Render ~~strikethrough~~ text."""
        return f'<s>{text}</s>'

    def heading(self, text: str, level: int, **attrs: Any) -> str:
        """Render headings as bold text (Telegram doesn't support h1-h6)."""
        prefix = '#' * level
        return f'<b>{prefix} {text}</b>\n'

    def paragraph(self, text: str) -> str:
        """Render paragraphs with newlines."""
        return f'{text}\n\n'

    def linebreak(self) -> str:
        """Render line breaks."""
        return '\n'

    def softbreak(self) -> str:
        """Render soft breaks (single newlines in source)."""
        return '\n'

    def blank_line(self) -> str:
        """Render blank lines."""
        return '\n'

    def thematic_break(self) -> str:
        """Render horizontal rules as dashes."""
        return '\n---\n'

    def block_quote(self, text: str) -> str:
        """Render blockquotes with > prefix."""
        # Add > prefix to each line
        lines = text.strip().split('\n')
        quoted = '\n'.join(f'> {line}' for line in lines)
        return f'{quoted}\n\n'

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        """Render lists."""
        return f'{text}\n'

    def list_item(self, text: str) -> str:
        """Render list items."""
        return f'• {text.strip()}\n'

    def image(self, text: str, url: str, title: Optional[str] = None) -> str:
        """Render images as links (Telegram doesn't support inline images in text)."""
        return f'[{text}]({escape_html(url)})'

    # Table rendering - Telegram doesn't support HTML tables, so render as plain text
    def table(self, text: str) -> str:
        """Render table as plain text."""
        return f'{text}\n'

    def table_head(self, text: str) -> str:
        """Render table header."""
        return f'{text}'

    def table_body(self, text: str) -> str:
        """Render table body."""
        return text

    def table_row(self, text: str) -> str:
        """Render table row as pipe-separated values."""
        return f'{text}|\n'

    def table_cell(self, text: str, align: Optional[str] = None, head: bool = False) -> str:
        """Render table cell."""
        if head:
            return f'| <b>{text}</b> '
        return f'| {text} '


# Create a global markdown parser instance with the Telegram renderer
# Enable strikethrough plugin for ~~text~~ support
_telegram_md = mistune.create_markdown(
    renderer=TelegramHTMLRenderer(),
    plugins=['strikethrough', 'table']
)


def markdown_to_html(text: str) -> str:
    """Convert markdown to Telegram-compatible HTML using mistune.

    This properly handles all markdown edge cases including:
    - Nested formatting
    - Tables with special characters
    - Code blocks containing markdown-like syntax
    - Complex inline code
    """
    try:
        result = _telegram_md(text)
        # mistune can return str or tuple, ensure we have str
        if isinstance(result, tuple):
            result = result[0]
        # Type assertion for mypy - at this point result is always str
        result_str: str = str(result) if not isinstance(result, str) else result
        # Clean up excessive newlines
        result_str = re.sub(r'\n{3,}', '\n\n', result_str)
        return result_str.strip()
    except Exception:
        # If parsing fails, return escaped plain text
        return escape_html(text)


def format_tool_call(name: str, input_dict: dict) -> str:
    """Format a tool call for display in Telegram (HTML)."""
    # Show key args, truncate long values
    parts = []
    for k, v in input_dict.items():
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:50] + "..."
        # Escape HTML in values
        v_str = escape_html(v_str)
        parts.append(f"{k}={v_str}")

    args_str = ", ".join(parts) if parts else ""
    return f"🔧 <b>{name}</b>({args_str})"


def format_tool_calls_batch(tool_name: str, calls: list[tuple[str, dict]]) -> str:
    """Format multiple tool calls of same type as a single message (HTML)."""
    # Extract the key argument for each call (usually file_path, pattern, command, etc.)
    items = []
    for name, input_dict in calls:
        # Try to get the most relevant argument
        key_arg = None
        for key in ['file_path', 'path', 'pattern', 'command', 'query', 'prompt', 'url']:
            if key in input_dict:
                key_arg = str(input_dict[key])
                break

        if key_arg is None and input_dict:
            # Use first argument value
            key_arg = str(list(input_dict.values())[0])

        if key_arg:
            # Truncate long values
            if len(key_arg) > 60:
                key_arg = key_arg[:57] + "..."
            # Escape HTML in values
            key_arg = escape_html(key_arg)
            items.append(f"  • {key_arg}")
        else:
            items.append(f"  • (no args)")

    return f"🔧 <b>{tool_name}</b> ({len(calls)} calls)\n" + "\n".join(items)


def format_tool_output(content: Any) -> str:
    """Format tool output for display, truncating if needed."""
    if content is None:
        return ""

    text = str(content)
    if len(text) > 1000:
        return text[:1000] + "\n... (truncated)"
    return text


def split_text(text: str, max_len: int = 4000) -> list[str]:
    """Split text into chunks suitable for Telegram messages."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at newline
        split_pos = text.rfind('\n', 0, max_len)
        if split_pos == -1 or split_pos < max_len // 2:
            # Try space
            split_pos = text.rfind(' ', 0, max_len)
        if split_pos == -1 or split_pos < max_len // 2:
            # Hard split
            split_pos = max_len

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()

    return chunks


