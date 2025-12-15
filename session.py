"""
Claude Code SDK session management for Telegram bot.

Manages conversations with Claude through the Code SDK,
streaming responses to Telegram messages.
"""
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from telegram import Bot, Message
from telegram.constants import ChatAction

from claude_code_sdk import query, ClaudeCodeOptions

from config import PROJECTS_DIR
from logger import SessionLogger


@dataclass
class ClaudeSession:
    """Represents an active Claude session for a Telegram thread."""
    chat_id: int
    thread_id: int
    cwd: str
    logger: Optional[SessionLogger] = None
    last_send: float = field(default_factory=time.time)
    send_interval: float = 1.0
    last_typing_action: float = 0.0
    active: bool = True
    session_id: Optional[str] = None  # For multi-turn conversation


# Active sessions: thread_id -> ClaudeSession
sessions: dict[int, ClaudeSession] = {}

# Minimum seconds between sends to avoid flood control
MIN_SEND_INTERVAL = 1.0

# Typing action expires after ~5s, resend every 4s
TYPING_ACTION_INTERVAL = 4.0


async def start_session(chat_id: int, thread_id: int, folder_name: str, bot: Bot) -> bool:
    """Start a new Claude session for a Telegram thread."""
    cwd = PROJECTS_DIR / folder_name
    if not cwd.exists():
        return False

    # Create logger
    logger = SessionLogger(thread_id, chat_id, str(cwd))

    # Store session
    sessions[thread_id] = ClaudeSession(
        chat_id=chat_id,
        thread_id=thread_id,
        cwd=str(cwd),
        logger=logger
    )

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

    # Buffer for batching consecutive tool calls of same type
    tool_buffer: list[tuple[str, dict]] = []  # [(tool_name, input), ...]
    tool_buffer_name: Optional[str] = None  # Current tool type being buffered
    tool_buffer_msg: Optional[Message] = None  # Message being edited for batch

    def format_current_tool_buffer() -> str:
        """Format current tool buffer contents."""
        if len(tool_buffer) == 1:
            name, input_dict = tool_buffer[0]
            return format_tool_call(name, input_dict)
        else:
            return format_tool_calls_batch(tool_buffer_name, tool_buffer)

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
            except Exception:
                pass
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
        # Configure options - include resume for multi-turn
        options = ClaudeCodeOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"],
            permission_mode="acceptEdits",
            cwd=session.cwd,
            resume=session.session_id  # Resume previous conversation if exists
        )

        # Query Claude and stream response
        async for message in query(prompt=prompt, options=options):
            # Refresh typing indicator on each message
            await send_typing_action(session, bot)

            # Log SDK message
            if session.logger:
                session.logger.log_sdk_message(message)

            msg_type = type(message).__name__

            # Handle different message types
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        block_type = type(block).__name__

                        if hasattr(block, 'text'):
                            # Text content - flush tools first, then accumulate text
                            await flush_tool_buffer()

                            response_text += block.text
                            response_msg = await send_or_edit_response(
                                session, bot, response_msg, response_text
                            )

                        elif hasattr(block, 'name') and hasattr(block, 'input'):
                            # Tool use block - buffer it
                            if response_text.strip():
                                response_msg = await send_or_edit_response(
                                    session, bot, response_msg, response_text
                                )
                                response_msg = None
                                response_text = ""

                            tool_name = block.name
                            tool_input = block.input

                            # If different tool type, flush buffer first
                            if tool_buffer_name and tool_buffer_name != tool_name:
                                await flush_tool_buffer()

                            # Add to buffer and update message immediately
                            tool_buffer.append((tool_name, tool_input))
                            tool_buffer_name = tool_name
                            await update_tool_buffer_message()

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

            # Capture session_id for multi-turn conversation
            if hasattr(message, 'session_id') and message.session_id:
                session.session_id = message.session_id

            # Log stats if available (but don't show to user - it's included in subscription)
            if hasattr(message, 'cost_usd') or hasattr(message, 'total_cost_usd'):
                cost = getattr(message, 'total_cost_usd', None) or getattr(message, 'cost_usd', None)
                duration = getattr(message, 'duration_ms', 0)
                if session.logger and cost is not None:
                    session.logger.log_session_stats(cost, duration, {})

        # Flush any remaining buffers
        await flush_tool_buffer()
        if response_text.strip() and response_msg is None:
            await send_message(session, bot, response_text)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        await send_message(session, bot, error_msg)
        if session.logger:
            session.logger.log_error("send_to_claude", e)


async def send_or_edit_response(
    session: ClaudeSession,
    bot: Bot,
    existing_msg: Optional[Message],
    text: str
) -> Optional[Message]:
    """Send a new response message or edit an existing one."""
    if not text.strip():
        return existing_msg

    # Truncate if too long (before conversion to avoid cutting HTML tags)
    if len(text) > 4000:
        text = text[:3990] + "\n..."

    # Convert markdown to HTML for Telegram
    html_text = markdown_to_html(text)

    try:
        if existing_msg:
            # Edit existing message
            await bot.edit_message_text(
                chat_id=session.chat_id,
                message_id=existing_msg.message_id,
                text=html_text,
                parse_mode="HTML"
            )
            return existing_msg
        else:
            # Send new message with rate limiting
            return await send_message(session, bot, html_text, parse_mode="HTML")
    except Exception as e:
        if session.logger and "message is not modified" not in str(e).lower():
            session.logger.log_error("send_or_edit_response", e)
        # Fallback to plain text if HTML fails
        if "parse entities" in str(e).lower():
            try:
                if existing_msg:
                    await bot.edit_message_text(
                        chat_id=session.chat_id,
                        message_id=existing_msg.message_id,
                        text=text
                    )
                else:
                    return await send_message(session, bot, text)
            except Exception:
                pass
        return existing_msg


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
    except Exception:
        pass


def escape_html(text: str) -> str:
    """Escape HTML entities for Telegram."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def markdown_to_html(text: str) -> str:
    """Convert common markdown patterns to Telegram HTML."""
    # Escape HTML entities first
    text = escape_html(text)

    # Code blocks (``` ... ```) - must be done before other patterns
    text = re.sub(r'```(\w*)\n?(.*?)```', r'<pre>\2</pre>', text, flags=re.DOTALL)

    # Inline code (`code`) - before bold/italic to protect code content
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Headers (# ## ### etc.) - make bold, keep the # symbols
    text = re.sub(r'^(#{1,6})\s+(.+)$', r'<b>\1 \2</b>', text, flags=re.MULTILINE)

    # Bold (**text** or __text__) - use DOTALL for multiline
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # Italic (*text* or _text_) - be careful not to match ** or __
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)

    # Strikethrough (~~text~~)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text, flags=re.DOTALL)

    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    return text


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


