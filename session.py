"""
Claude Code SDK session management for Telegram bot.

Manages conversations with Claude through the Code SDK,
streaming responses to Telegram messages.

Supports multiple platforms via PlatformClient abstraction.
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
from typing import Optional, Any, Union, Callable, Awaitable

import mistune
from telegram import Bot, Message, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ProcessError, PermissionResultAllow, PermissionResultDeny, HookMatcher, HookContext
from claude_agent_sdk.types import (
    SystemPromptPreset, ToolPermissionContext, HookInput, HookJSONOutput,
    UserMessage, AssistantMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ThinkingBlock, ToolResultBlock
)

from config import PROJECTS_DIR, CLAUDE_MODEL
from logger import SessionLogger
from diff_image import edit_to_image
from commands import load_contextual_commands, register_commands_for_chat
from mcp_tools import create_telegram_mcp_server
from core.session_store import session_store
from core.types import make_session_key, Trigger

# Platform abstraction imports
from platforms import (
    PlatformClient,
    MessageFormatter,
    ButtonSpec,
    ButtonRow,
    MessageRef,
    TextMessage,
    ToolCallMessage,
    ThinkingMessage,
    PlatformMessage,
)
from platforms.telegram import TelegramClient, TelegramFormatter

# Check if Discord support is available
try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

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
MAX_DIFF_IMAGE_INPUT_CHARS = 20000

# Tools that are always allowed without prompting
DEFAULT_ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebSearch",
    "Skill",  # Enable skills from .claude/skills/
    "mcp__telegram-tools__send_to_telegram",  # Custom tool for sending files to chat
]

# Persistent allowlist file
ALLOWLIST_FILE = Path(__file__).parent / "tool_allowlist.json"

# Pending permission requests: request_id -> (Future, SessionLogger)
pending_permissions: dict[str, tuple[asyncio.Future, Optional["SessionLogger"]]] = {}


def md_inline_code(text: str) -> str:
    """Wrap text in markdown inline code, using a safe backtick fence."""
    safe = text.replace("\n", " ")
    fence = "`"
    while fence in safe:
        fence += "`"
    return f"{fence}{safe}{fence}"


def md_code_block(text: str, language: Optional[str] = None) -> str:
    """Wrap text in a markdown code block with a safe backtick fence."""
    fence = "```"
    while fence in text:
        fence += "`"
    lang = language or ""
    return f"{fence}{lang}\n{text}\n{fence}"


def md_blockquote(text: str) -> str:
    """Wrap text in markdown blockquote lines."""
    if not text:
        return ""
    return "\n".join(f"> {line}" for line in text.splitlines())


def md_escape(text: str) -> str:
    """Escape markdown-sensitive characters."""
    escape_map = {
        "\\": "\\\\",
        "`": "\\`",
        "*": "\\*",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "[": "\\[",
        "]": "\\]",
        "(": "\\(",
        ")": "\\)",
        "#": "\\#",
        "+": "\\+",
        "-": "\\-",
        ".": "\\.",
        "!": "\\!",
        "@": "\\@",
        "|": "\\|",
        ">": "\\>",
        "~": "\\~",
    }
    return "".join(escape_map.get(ch, ch) for ch in text)


def is_empty_message(message: PlatformMessage) -> bool:
    """Check whether a structured message has displayable content."""
    if isinstance(message, TextMessage):
        return not message.text.strip()
    if isinstance(message, ThinkingMessage):
        return not message.text.strip()
    if isinstance(message, ToolCallMessage):
        return not message.calls
    return True


def _log_session_debug(session: Optional["ClaudeSession"], context: str, message: str, **kwargs: Any) -> None:
    """Log debug details to SessionLogger if available, else module logger."""
    if session and session.logger:
        session.logger.log_debug(context, message, **kwargs)
    else:
        extra = f" {kwargs}" if kwargs else ""
        _log.debug(f"{context}: {message}{extra}")


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


def _session_platform_type(session: "ClaudeSession") -> str:
    return "discord" if session.chat_id == 0 else "telegram"


def _session_key_for_session(session: "ClaudeSession") -> str:
    if session.chat_id == 0:
        return make_session_key("discord", channel_id=session.thread_id)
    return make_session_key("telegram", chat_id=session.chat_id, thread_id=session.thread_id)


def _persist_session_id(session: "ClaudeSession", claude_session_id: str) -> None:
    if not claude_session_id:
        return
    session_store.update_session_id(
        session_key=_session_key_for_session(session),
        claude_session_id=claude_session_id,
        cwd=session.cwd,
        platform=_session_platform_type(session),
    )


def _clear_persisted_session(session: "ClaudeSession") -> None:
    session_store.remove(_session_key_for_session(session))


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
    """Represents an active Claude session for a chat thread.

    Supports both legacy Telegram-specific mode (via bot field) and
    platform-agnostic mode (via platform field).
    """
    chat_id: int
    thread_id: int
    cwd: str
    # Platform abstraction (preferred)
    platform: Optional[PlatformClient] = None
    formatter: Optional[MessageFormatter] = None
    # Legacy Telegram support (deprecated - use platform instead)
    bot: Optional[Bot] = None  # Will be removed in future version
    # Session state
    logger: Optional[SessionLogger] = None
    last_send: float = field(default_factory=time.time)
    send_interval: float = 1.0
    last_typing_action: float = 0.0
    active: bool = True
    session_id: Optional[str] = None  # For multi-turn conversation
    client: Optional[ClaudeSDKClient] = None  # Active SDK client for interrupt support
    current_task: Optional[asyncio.Task] = None  # Active send_to_claude task for cancellation
    interrupt_event: Optional[asyncio.Event] = None  # Set when interrupt requested, cleared on new query
    last_context_percent: Optional[float] = None  # Last known context remaining %
    pending_image_path: Optional[str] = None  # Buffered image waiting for prompt
    contextual_commands: list = field(default_factory=list)  # Project-specific slash commands
    sandboxed: bool = False  # If True, only load project settings (no ~/.claude/)
    model_override: Optional[str] = None  # Per-session model choice (overrides CLAUDE_MODEL)
    watchdog_enabled: bool = False  # Auto-retry on token limit
    _watchdog_task: Optional[asyncio.Task] = None  # Scheduled retry task
    actor_enqueue: Optional[Callable[[Trigger], Awaitable[None]]] = None  # Actor queue hook

    def get_platform(self) -> Optional[PlatformClient]:
        """Get platform client, creating from bot if needed (backwards compat)."""
        if self.platform:
            return self.platform
        if self.bot:
            # Create TelegramClient wrapper for legacy bot
            self.platform = TelegramClient(
                bot=self.bot,
                chat_id=self.chat_id,
                thread_id=self.thread_id,
                logger=self.logger,
            )
            return self.platform
        return None

    def get_formatter(self) -> MessageFormatter:
        """Get message formatter, defaulting to Telegram."""
        if self.formatter:
            return self.formatter
        # Default to Telegram formatter for backwards compat
        self.formatter = TelegramFormatter()
        return self.formatter


async def interrupt_session(thread_id: int) -> bool:
    """Interrupt the active Claude response for a session.

    Uses two-phase soft/hard interrupt:
    1. Set interrupt flag + call client.interrupt()
    2. Wait for reader to drain and receive interrupt ack
    3. Hard cancel only as fallback after timeout

    Returns True if an active query was interrupted, False otherwise.
    """
    session = sessions.get(thread_id)
    if not session:
        return False

    # Check if there's an active task to interrupt
    if not session.current_task or session.current_task.done():
        _log_session_debug(
            session,
            "interrupt",
            "No active task to interrupt",
            thread_id=thread_id,
            has_task=bool(session.current_task),
            task_done=bool(session.current_task.done()) if session.current_task else None,
        )
        return False

    # Phase 1: Soft interrupt - set flag and signal SDK
    if session.interrupt_event:
        session.interrupt_event.set()
        _log_session_debug(session, "interrupt", "Interrupt flag set", thread_id=thread_id)

    if session.client:
        t0 = time.perf_counter()
        _log_session_debug(session, "interrupt", "Sending client.interrupt()", thread_id=thread_id)
        try:
            await session.client.interrupt()
            _log_session_debug(
                session,
                "interrupt",
                "client.interrupt() completed",
                thread_id=thread_id,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
            )
        except Exception as e:
            if session.logger:
                session.logger.log_error("interrupt_session.client_interrupt", e)
            _log_session_debug(
                session,
                "interrupt",
                "client.interrupt() failed",
                thread_id=thread_id,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                error=str(e),
            )
    else:
        _log_session_debug(session, "interrupt", "No client available for interrupt", thread_id=thread_id)

    # Phase 2: Wait for task to drain and complete gracefully
    try:
        t0 = time.perf_counter()
        _log_session_debug(session, "interrupt", "Waiting for current_task to finish", thread_id=thread_id)
        await asyncio.wait_for(session.current_task, timeout=2.0)
        _log_session_debug(
            session,
            "interrupt",
            "current_task finished",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
    except asyncio.TimeoutError:
        # Phase 3: Hard cancel as fallback
        _log_session_debug(
            session,
            "interrupt",
            "Timeout waiting for task; hard cancelling",
            thread_id=thread_id,
        )
        session.current_task.cancel()
        try:
            t1 = time.perf_counter()
            await asyncio.wait_for(session.current_task, timeout=2.0)
            _log_session_debug(
                session,
                "interrupt",
                "Hard cancel completed",
                thread_id=thread_id,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
            )
        except asyncio.TimeoutError:
            _log_session_debug(session, "interrupt", "Hard cancel timed out", thread_id=thread_id)
        except asyncio.CancelledError:
            _log_session_debug(session, "interrupt", "Task cancelled (CancelledError)", thread_id=thread_id)
            pass

    return True


async def _format_plan_approval_message(
    platform: PlatformClient,
    plan_dir: Path | None = None
) -> str:
    """Read the most recent plan file and format it for approval.

    Looks in ~/.claude/plans/ for the most recently modified .md file.

    Args:
        platform: Platform client (used for max_message_length).
        plan_dir: Optional directory to search for plans. Defaults to ~/.claude/plans/.
    """
    if plan_dir is None:
        plan_dir = Path.home() / ".claude" / "plans"
    plan_content = None
    plan_file_name = None

    if plan_dir.exists():
        plan_files = sorted(
            plan_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if plan_files:
            plan_file = plan_files[0]
            plan_file_name = plan_file.name
            try:
                plan_content = plan_file.read_text()
            except Exception:
                pass

    if plan_content:
        # Truncate to fit platform limits (leave room for header/buttons)
        max_len = platform.max_message_length - 200
        if len(plan_content) > max_len:
            plan_content = plan_content[:max_len] + "\n\n... (truncated)"

        return (
            f"📋 **Plan Approval**\n"
            f"_{plan_file_name}_\n\n"
            f"{plan_content}"
        )
    else:
        return (
            "📋 **Plan Approval**\n\n"
            "_Could not find plan file in ~/.claude/plans/_"
        )


async def request_tool_permission(
    session: ClaudeSession,
    tool_name: str,
    tool_input: dict
) -> bool:
    """Send permission request and wait for user response.

    Uses platform abstraction for cross-platform support.
    """
    platform = session.get_platform()
    if platform is None:
        if session.logger:
            session.logger.log_error("request_tool_permission", Exception("No platform client available"))
        return False

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]

    # Special handling for ExitPlanMode - show the plan content
    if tool_name == "ExitPlanMode":
        message_text = await _format_plan_approval_message(platform)
    else:
        # Format tool input for display
        input_preview = []
        for k, v in list(tool_input.items())[:3]:  # Show first 3 args
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + "..."
            input_preview.append(f"{k}: {v_str}")

        if input_preview:
            input_text = md_code_block("\n".join(input_preview))
        else:
            input_text = "(no arguments)"

        message_text = (
            f"🔐 **Permission Request**\n\n"
            f"Tool: {md_inline_code(tool_name)}\n"
            f"Arguments:\n{input_text}"
        )

    # Build platform-agnostic keyboard
    buttons = [
        ButtonRow([
            ButtonSpec("✅ Allow", f"perm:allow:{request_id}:{tool_name}"),
            ButtonSpec("❌ Deny", f"perm:deny:{request_id}:{tool_name}"),
        ]),
        ButtonRow([
            ButtonSpec("✅ Always Allow", f"perm:always:{request_id}:{tool_name}"),
        ])
    ]

    # Send permission request message
    await platform.send_message(TextMessage(message_text), buttons=buttons)

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

    # No timeout - this is async chat, user responds when they respond
    allowed = await future
    if session.logger:
        session.logger.log_debug("permission", f"Got user response: {allowed}", request_id=request_id)
    return allowed


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

            # Notify user in chat using platform abstraction
            platform = session.get_platform()
            if platform:
                message_text = (
                    "📦 **Context compacting...**\n"
                    "Conversation history is being summarized to free up space."
                )
                await platform.send_message(TextMessage(message_text))
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

    return await _start_session_impl(chat_id, thread_id, str(cwd), folder_name, bot)


async def start_session_local(chat_id: int, thread_id: int, cwd: str, bot: Bot) -> bool:
    """Start a Claude session with an absolute path (for local project bot).

    Logs are stored in <cwd>/.bot-logs/ instead of global logs dir.
    Sandboxed mode: only loads project settings, not ~/.claude/ context.
    """
    cwd_path = Path(cwd)
    if not cwd_path.exists():
        return False

    # Use project-local logs directory
    logs_dir = cwd_path / ".bot-logs"
    return await _start_session_impl(chat_id, thread_id, cwd, cwd_path.name, bot, logs_dir, sandboxed=True)


async def _start_session_impl(
    chat_id: int,
    thread_id: int,
    cwd: str,
    display_name: str,
    bot: Bot,
    logs_dir: Optional[Path] = None,
    sandboxed: bool = False
) -> bool:
    """Internal: start Telegram session with given cwd path."""
    # Create logger (use custom logs_dir if provided)
    logger = SessionLogger(thread_id, chat_id, cwd, logs_dir)

    # Load contextual commands from project's commands/ directory
    contextual_commands = load_contextual_commands(cwd)

    # Create platform client and formatter
    platform = TelegramClient(
        bot=bot,
        chat_id=chat_id,
        thread_id=thread_id,
        logger=logger,
    )
    formatter = TelegramFormatter()

    # Store session with platform abstraction
    sessions[thread_id] = ClaudeSession(
        chat_id=chat_id,
        thread_id=thread_id,
        cwd=cwd,
        platform=platform,
        formatter=formatter,
        bot=bot,  # Keep for backwards compat
        logger=logger,
        contextual_commands=contextual_commands,
        sandboxed=sandboxed,
    )

    # Register commands with Telegram for autocompletion
    await register_commands_for_chat(bot, chat_id, contextual_commands)

    # Send welcome message using platform
    await platform.send_message(
        TextMessage(f"Claude session started in {md_inline_code(display_name)}")
    )

    return True


async def start_session_ambient(chat_id: int, thread_id: int, bot: Bot) -> bool:
    """Start an ambient Claude session with home folder as cwd.

    Used for #general or other permanent ambient sessions that don't
    need a specific project folder.

    Args:
        chat_id: Telegram chat ID
        thread_id: Telegram thread ID (e.g., GENERAL_TOPIC_ID)
        bot: Telegram Bot instance

    Returns:
        True if session started successfully
    """
    home_dir = str(Path.home())
    return await _start_session_impl(chat_id, thread_id, home_dir, "~", bot)


async def start_session_ambient_discord(channel_id: int, channel: Any) -> bool:
    """Start an ambient Claude session for Discord with home folder as cwd.

    Used for #general or other ambient channels that don't need a specific project.

    Args:
        channel_id: Discord channel ID (used as session key)
        channel: Discord channel object (TextChannel, Thread, or similar)

    Returns:
        True if session started successfully
    """
    home_dir = str(Path.home())
    return await start_session_discord(channel_id, home_dir, channel, display_name="~")


async def start_session_discord(channel_id: int, project_path: str, channel: Any, display_name: Optional[str] = None) -> bool:
    """Start a new Claude session for a Discord channel.

    Args:
        channel_id: Discord channel ID (used as session key)
        project_path: Absolute path to project directory
        channel: Discord channel object (TextChannel or similar)
        display_name: Optional display name (defaults to folder name)

    Returns:
        True if session started successfully
    """
    if not DISCORD_AVAILABLE:
        _log.error("Discord support not available - discord.py not installed")
        return False

    cwd_path = Path(project_path)
    if not cwd_path.exists():
        return False

    display_name = display_name or cwd_path.name

    # Create logger - use project-local logs
    logs_dir = cwd_path / ".bot-logs"
    logger = SessionLogger(channel_id, 0, str(cwd_path), logs_dir)  # chat_id=0 for Discord

    # Load contextual commands
    contextual_commands = load_contextual_commands(str(cwd_path))

    # Create Discord platform client and formatter
    # Import here to avoid issues when discord.py not installed
    from platforms.discord import DiscordClient as DC, DiscordFormatter as DF
    platform = DC(channel=channel, logger=logger)
    formatter = DF()

    # Store session (keyed by channel_id for Discord)
    sessions[channel_id] = ClaudeSession(
        chat_id=0,  # Not used for Discord
        thread_id=channel_id,  # Use channel_id as session key
        cwd=str(cwd_path),
        platform=platform,
        formatter=formatter,
        bot=None,  # No Telegram bot
        logger=logger,
        contextual_commands=contextual_commands,
    )

    # Send welcome message
    await platform.send_message(
        TextMessage(f"Claude session started in {md_inline_code(display_name)}")
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


def start_claude_task(thread_id: int, prompt: str, bot: Optional[Bot] = None) -> Optional[asyncio.Task]:
    """Start send_to_claude as a background task and store reference for cancellation.

    Use this instead of asyncio.create_task(send_to_claude(...)) directly.
    The task reference is stored in session.current_task for interrupt support.

    Returns:
        The created task, or None if session doesn't exist.
    """
    session = sessions.get(thread_id)
    if not session:
        _log.debug(f"start_claude_task: no session for thread_id={thread_id}")
        return None

    task = asyncio.create_task(send_to_claude(thread_id, prompt, bot))
    session.current_task = task
    _log_session_debug(
        session,
        "task",
        "Started send_to_claude task",
        thread_id=thread_id,
        task_id=id(task),
        prompt_len=len(prompt),
    )
    return task


# --- /model command ---

_cached_models: Optional[list[dict]] = None


async def _get_available_models(session: ClaudeSession) -> list[dict]:
    """Fetch available models from SDK via get_server_info(), cached globally.

    Only caches non-empty results so transient failures are retried.
    """
    global _cached_models
    if _cached_models is not None:
        return _cached_models

    models: list[dict] = []
    try:
        options = ClaudeAgentOptions(cwd=session.cwd)
        async with ClaudeSDKClient(options=options) as client:
            info = await client.get_server_info()
            models = info.get("models", []) if info else []
    except Exception as exc:
        _log.warning("Failed to fetch available models: %s", exc)

    if models:
        _cached_models = models
    return models


def _parse_limit_reset(text: str) -> Optional[float]:
    """Parse reset time from 'resets 3pm (UTC)' pattern.

    Returns delay in seconds until reset, or None if pattern not found.
    """
    from datetime import datetime, timezone

    match = re.search(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(UTC\)", text, re.IGNORECASE)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    ampm = match.group(3).lower()

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    now = datetime.now(timezone.utc)
    reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # If reset time is in the past, it must be tomorrow
    if reset_time <= now:
        from datetime import timedelta
        reset_time += timedelta(days=1)

    delay = (reset_time - now).total_seconds()
    return delay


async def _watchdog_retry(session: ClaudeSession, thread_id: int, delay: float) -> None:
    """Wait for delay, then send 'Continue' to Claude."""
    watchdog_task = asyncio.current_task()
    try:
        await asyncio.sleep(delay)
        if not session.watchdog_enabled:
            return  # watchdog was turned off while waiting
        if session.current_task and not session.current_task.done():
            return  # user already sent something, skip

        enqueue = session.actor_enqueue
        if enqueue is None:
            _log.warning("Watchdog retry skipped: actor enqueue unavailable for thread_id=%s", thread_id)
            return

        await enqueue(
            Trigger(
                platform=_session_platform_type(session),
                session_key=_session_key_for_session(session),
                prompt="Continue",
                reply_context={"cwd": session.cwd},
                source="watchdog",
            )
        )
        await send_message(session, message=TextMessage("🐕 Watchdog: sending Continue"))
    except asyncio.CancelledError:
        pass  # watchdog turned off or session closed
    finally:
        if session._watchdog_task is watchdog_task:
            session._watchdog_task = None


async def handle_watchdog_command(thread_id: int, args: str) -> None:
    """Handle /watchdog command — toggle auto-retry on token limit."""
    session = sessions.get(thread_id)
    if not session:
        return

    args = args.strip().lower()

    if args == "on":
        session.watchdog_enabled = True
        await send_message(session, message=TextMessage("🐕 Watchdog: **ON**"))
    elif args == "off":
        session.watchdog_enabled = False
        if session._watchdog_task and not session._watchdog_task.done():
            session._watchdog_task.cancel()
            session._watchdog_task = None
        await send_message(session, message=TextMessage("🐕 Watchdog: **OFF**"))
    else:
        state = "ON" if session.watchdog_enabled else "OFF"
        pending = ""
        if session._watchdog_task and not session._watchdog_task.done():
            pending = " (retry pending)"
        await send_message(session, message=TextMessage(f"🐕 Watchdog: **{state}**{pending}"))


async def handle_model_command(thread_id: int, args: str) -> None:
    """Handle /model command — list available models or switch model."""
    session = sessions.get(thread_id)
    if not session:
        return

    if not args.strip():
        # List available models from SDK + known aliases not in get_server_info()
        models = await _get_available_models(session)
        current = session.model_override or CLAUDE_MODEL or "default"
        if not models:
            await send_message(session, message=TextMessage("Could not fetch available models"))
            return

        lines = ["**Available models:**\n"]
        for m in models:
            value = m.get("value", "")
            desc = m.get("description", m.get("displayName", ""))
            marker = "  **← current**" if value == current else ""
            lines.append(f"`{value}` — {desc}{marker}")
        lines.append(f"\nAlso accepts: `opus`, `opusplan`, or any full model ID")
        lines.append(f"Use `/model <name>` to switch")
        await send_message(session, message=TextMessage("\n".join(lines)))
    else:
        # Accept any model name — let the SDK validate on next query
        model_name = args.strip()
        session.model_override = model_name
        await send_message(session, message=TextMessage(f"Model set to `{model_name}` for this session"))


def _extract_tool_result_text(block: ToolResultBlock) -> str:
    """Extract displayable text from a ToolResultBlock.

    ToolResultBlock.content is str | list[dict] | None.
    For list[dict], only extracts text-type items (skips images/base64).
    Prefixes with ERROR: if block.is_error.
    """
    block_content = block.content
    if block_content is None:
        text = ""
    elif isinstance(block_content, str):
        text = block_content
    else:
        # list[dict] — only extract text items, skip images/binary
        parts = []
        for item in block_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "\n".join(parts)

    if block.is_error and text:
        text = f"ERROR: {text}"
    elif block.is_error:
        text = "ERROR"

    return text


async def send_to_claude(thread_id: int, prompt: str, bot: Optional[Bot] = None) -> None:
    """Send a message to Claude and stream the response.

    Uses platform abstraction for cross-platform support.
    Bot parameter is deprecated and ignored.
    """
    session = sessions.get(thread_id)
    if not session or not session.active:
        return

    start_t = time.perf_counter()
    _log_session_debug(
        session,
        "send_to_claude",
        "Start",
        thread_id=thread_id,
        prompt_len=len(prompt),
        session_id=session.session_id,
    )

    platform = session.get_platform()
    if not platform:
        return

    # Initialize interrupt event for this query (clear any previous state)
    session.interrupt_event = asyncio.Event()
    _log_session_debug(session, "send_to_claude", "Interrupt event created", thread_id=thread_id)

    # Log user input
    if session.logger:
        session.logger.log_user_input(prompt)

    # Send typing indicator
    await send_typing_action(session)

    # Track current response message for streaming edits
    response_ref: Optional[MessageRef] = None
    response_text = ""
    response_msg_text_len = 0  # Length of text in current response_ref

    # Buffer for batching consecutive tool calls of same type
    tool_buffer: list[tuple[str, dict]] = []  # [(tool_name, input), ...]
    tool_buffer_name: Optional[str] = None  # Current tool type being buffered
    tool_buffer_ref: Optional[MessageRef] = None  # Message being edited for batch

    # Buffer for diff images to send as media group at end
    diff_images: list[tuple[BytesIO, str]] = []  # [(image_buffer, filename), ...]

    # Track model and usage for context calculation
    current_model: Optional[str] = None
    last_usage: Optional[dict] = None

    def build_tool_buffer_message() -> ToolCallMessage:
        """Build a structured tool call message for the current buffer."""
        tool_name = tool_buffer_name or (tool_buffer[0][0] if tool_buffer else "Tool")
        return ToolCallMessage(tool_name=tool_name, calls=list(tool_buffer))

    async def update_tool_buffer_message():
        """Send or edit the tool buffer message."""
        nonlocal tool_buffer_ref
        message = build_tool_buffer_message()

        if tool_buffer_ref and tool_buffer_ref.platform_data:
            # Edit existing message
            try:
                await platform.edit_message(tool_buffer_ref, message)
            except Exception as e:
                if session.logger and "message is not modified" not in str(e).lower():
                    session.logger.log_error("update_tool_buffer_message", e)
        else:
            # Send new message
            tool_buffer_ref = await send_message(session, message=message)

    async def flush_tool_buffer():
        """Clear tool buffer state (message already sent/edited)."""
        nonlocal tool_buffer, tool_buffer_name, tool_buffer_ref
        tool_buffer = []
        tool_buffer_name = None
        tool_buffer_ref = None

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

        # Create MCP servers bound to this session
        telegram_mcp = create_telegram_mcp_server(session)

        # Configure options - use permission handler for interactive tool approval
        def build_options(resume_session_id: Optional[str]) -> ClaudeAgentOptions:
            return ClaudeAgentOptions(
                model=session.model_override or CLAUDE_MODEL,  # Per-session override > env > SDK default
                allowed_tools=[],  # Empty - let can_use_tool handle all permissions
                can_use_tool=create_permission_handler(session),
                permission_mode="acceptEdits",
                cwd=session.cwd,
                resume=resume_session_id,  # Resume previous conversation if exists
                system_prompt=system_prompt,
                setting_sources=["project"] if session.sandboxed else ["user", "project"],  # Sandboxed: project only
                max_thinking_tokens=10000,  # Enable interleaved thinking between tool calls
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

        def _looks_like_resume_error(error: Exception) -> bool:
            message = str(error).lower()
            return "resume" in message or "session" in message

        # Watchdog: track limit-hit detection
        limit_hit = False
        limit_reset_delay: Optional[float] = None

        async def run_stream(options: ClaudeAgentOptions) -> None:
            nonlocal response_text, response_ref, response_msg_text_len, tool_buffer_name, last_usage, current_model, limit_hit, limit_reset_delay
            async with ClaudeSDKClient(options=options) as client:
                # Store client reference for interrupt support
                session.client = client
                _log_session_debug(session, "send_to_claude", "Client ready", thread_id=thread_id)
    
                await client.query(prompt_stream())
                _log_session_debug(session, "send_to_claude", "Query sent", thread_id=thread_id)
                async for message in client.receive_response():
                    # Refresh typing indicator on each message
                    await send_typing_action(session)
    
                    # Log SDK message
                    if session.logger:
                        session.logger.log_sdk_message(message)
    
                    # Check for interrupt - drain until we get ack or ResultMessage
                    if session.interrupt_event and session.interrupt_event.is_set():
                        _log_session_debug(
                            session,
                            "interrupt",
                            "Interrupt flag detected while streaming",
                            thread_id=thread_id,
                            msg_type=type(message).__name__,
                        )
                        # Check for interrupt acknowledgment from SDK
                        if isinstance(message, UserMessage):
                            content = message.content
                            if isinstance(content, str) and "interrupted" in content.lower():
                                # Display the interrupt message and exit
                                _log_session_debug(session, "interrupt", "SDK interrupt ack (UserMessage)", thread_id=thread_id)
                                await send_message(session, message=TextMessage(f"_{content}_"))
                                break
                        elif isinstance(message, ResultMessage):
                            # Capture session metadata before exiting
                            _log_session_debug(session, "interrupt", "ResultMessage while interrupted", thread_id=thread_id)
                            if message.session_id:
                                session.session_id = message.session_id
                                _persist_session_id(session, message.session_id)
                            if message.usage:
                                last_usage = message.usage
                            cost = message.total_cost_usd
                            duration = message.duration_ms
                            if session.logger and cost is not None:
                                session.logger.log_session_stats(cost, duration, last_usage or {})
                            break
                        # Skip handling other messages while draining
                        continue
    
                    # === Message dispatch — type-correct if/elif chain ===
    
                    # Branch 1: SystemMessage
                    if isinstance(message, SystemMessage):
                        if message.subtype == "init":
                            sid = message.data.get("session_id")
                            if sid and not session.session_id:
                                session.session_id = sid
                                _persist_session_id(session, sid)
                        if session.logger:
                            session.logger.log_debug("send_to_claude", f"SystemMessage: {message.subtype}")
                        continue
    
                    # Branch 2: AssistantMessage (content is ALWAYS list[ContentBlock])
                    elif isinstance(message, AssistantMessage):
                        current_model = message.model
    
                        # API errors — display before subagent check (user should see rate limits from subagents)
                        if message.error:
                            error_label = message.error.replace("_", " ").title()
                            await send_message(session, message=TextMessage(f"⚠️ API error: {error_label}"))
    
                        is_subagent = message.parent_tool_use_id is not None
    
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                if not block.text.strip() or block.text.strip() == "(no content)":
                                    continue
    
                                if is_subagent:
                                    await flush_tool_buffer()
                                    await send_message(session, message=TextMessage(md_blockquote(block.text)))
                                else:
                                    await flush_tool_buffer()
                                    response_text += block.text
                                    response_ref, response_msg_text_len = await send_or_edit_response(
                                        session, existing_ref=response_ref, text=response_text,
                                        msg_text_len=response_msg_text_len
                                    )
    
                            elif isinstance(block, ToolUseBlock):
                                if is_subagent:
                                    if session.logger:
                                        session.logger.log_debug(
                                            "send_to_claude",
                                            f"Subagent tool use: {block.name} (parent={message.parent_tool_use_id})"
                                        )
                                else:
                                    if response_text.strip():
                                        response_ref, response_msg_text_len = await send_or_edit_response(
                                            session, existing_ref=response_ref, text=response_text,
                                            msg_text_len=response_msg_text_len
                                        )
                                        response_ref = None
                                        response_text = ""
                                        response_msg_text_len = 0
    
                                    tool_name = block.name
                                    tool_input = block.input
    
                                    if tool_buffer_name and tool_buffer_name != tool_name:
                                        await flush_tool_buffer()
    
                                    tool_buffer.append((tool_name, tool_input))
                                    tool_buffer_name = tool_name
                                    await update_tool_buffer_message()
    
                                    if tool_name == "Edit" and "old_string" in tool_input and "new_string" in tool_input:
                                        file_path = tool_input.get("file_path", "file")
                                        old_string = tool_input["old_string"]
                                        new_string = tool_input["new_string"]
                                        if len(old_string) + len(new_string) <= MAX_DIFF_IMAGE_INPUT_CHARS:
                                            img_buffer = await asyncio.to_thread(
                                                edit_to_image,
                                                file_path=file_path,
                                                old_string=old_string,
                                                new_string=new_string
                                            )
                                        else:
                                            img_buffer = None
                                        if img_buffer:
                                            filename = file_path.split("/")[-1] if "/" in file_path else file_path
                                            diff_images.append((img_buffer, filename))
    
                            elif isinstance(block, ThinkingBlock):
                                if not is_subagent:
                                    await flush_tool_buffer()
                                    if response_text.strip():
                                        response_ref, response_msg_text_len = await send_or_edit_response(
                                            session, existing_ref=response_ref, text=response_text,
                                            msg_text_len=response_msg_text_len
                                        )
                                        response_ref = None
                                        response_text = ""
                                        response_msg_text_len = 0
    
                                    thinking_text = block.thinking
                                    if thinking_text:
                                        await send_message(session, message=ThinkingMessage(thinking_text))
    
                            elif isinstance(block, ToolResultBlock):
                                # Only display error results — successful tool output is noise
                                # (Claude incorporates relevant info in its text response)
                                if block.is_error:
                                    text = _extract_tool_result_text(block)
                                    if text:
                                        await flush_tool_buffer()
                                        output = format_tool_output(text)
                                        if output:
                                            if is_subagent:
                                                await send_message(session, message=TextMessage(md_blockquote(md_escape(output))))
                                            else:
                                                await send_message(session, message=TextMessage(md_code_block(output)))
    
                    # Branch 3: UserMessage (content is str | list[ContentBlock])
                    elif isinstance(message, UserMessage):
                        is_subagent = message.parent_tool_use_id is not None
                        content = message.content
    
                        if isinstance(content, str):
                            # Tool result strings are noisy — only refresh typing indicator
                            if content:
                                await send_typing_action(session)
    
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, ToolResultBlock):
                                    # Only display error results
                                    if block.is_error:
                                        await flush_tool_buffer()
                                        text = _extract_tool_result_text(block)
                                        output = format_tool_output(text) if text else None
                                        if output:
                                            if is_subagent:
                                                await send_message(session, message=TextMessage(md_blockquote(md_escape(output))))
                                            else:
                                                await send_message(session, message=TextMessage(md_code_block(output)))
                                    await send_typing_action(session)
    
                                elif isinstance(block, TextBlock):
                                    # Suppress tool result text output (noisy, non-actionable)
                                    await send_typing_action(session)
    
                    # Branch 4: ResultMessage
                    elif isinstance(message, ResultMessage):
                        if message.session_id:
                            session.session_id = message.session_id
                            _persist_session_id(session, message.session_id)
                        if message.usage:
                            last_usage = message.usage
    
                        # Surface session-level errors
                        if message.is_error:
                            error_text = message.result or "Session ended with an error"
                            await send_message(session, message=TextMessage(f"⚠️ {error_text}"))
                            # Watchdog: detect limit-hit pattern
                            if error_text and "hit your limit" in error_text.lower():
                                limit_hit = True
                                limit_reset_delay = _parse_limit_reset(error_text)
    
                        cost = message.total_cost_usd
                        duration = message.duration_ms
                        if session.logger and cost is not None:
                            session.logger.log_session_stats(cost, duration, last_usage or {})
    
                    # Branch 5: Catch-all (StreamEvent or unknown)
                    else:
                        if session.logger:
                            session.logger.log_debug(
                                "send_to_claude",
                                f"Unhandled message type: {type(message).__name__}"
                            )
    
                # Clear client reference IMMEDIATELY after loop exits.
                # This prevents interrupt_session from trying to interrupt a finished SDK,
                # and ensures the task won't hang on SDK cleanup if a new message arrives.
                session.client = None
                _log_session_debug(session, "send_to_claude", "Stream loop ended", thread_id=thread_id)
    
                # Flush any remaining buffers (after loop)
                await flush_tool_buffer()
                if response_text.strip() and response_ref is None:
                    await send_message(session, message=TextMessage(response_text))
    
                # Send diff images as media group (gallery)
                if diff_images:
                    await send_diff_images_gallery(session, images=diff_images)
    
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
                            max_len = platform.max_message_length
    
                            # Only append to text response message (not tool messages)
                            if response_ref and response_ref.platform_data and response_msg_text_len > 0:
                                # Check if warning fits in current message
                                if response_msg_text_len + len(warning) <= max_len:
                                    # Get the text currently in the message and append warning
                                    current_msg_text = response_text[-response_msg_text_len:] if len(response_text) > response_msg_text_len else response_text
                                    warning_text = current_msg_text + warning
                                    try:
                                        await platform.edit_message(response_ref, TextMessage(warning_text))
                                    except Exception:
                                        # Edit failed, send as separate message
                                        await send_message(session, message=TextMessage(f"⚠️ {context_remaining:.0f}% context remaining"))
                                else:
                                    # Warning doesn't fit, send separately
                                    await send_message(session, message=TextMessage(f"⚠️ {context_remaining:.0f}% context remaining"))
    
        try:
            resume_id = session.session_id
            _log.info("Starting stream with resume=%s for thread_id=%s", resume_id[:8] + "..." if resume_id else None, thread_id)
            await run_stream(build_options(resume_id))
        except Exception as e:
            if session.session_id and _looks_like_resume_error(e):
                session_key = _session_key_for_session(session)
                _log.warning(
                    "Resume failed for %s; clearing stored session_id and retrying without resume: %s",
                    session_key,
                    e,
                )
                session.session_id = None
                _clear_persisted_session(session)
                session.client = None
                await run_stream(build_options(None))
            else:
                raise

        # Watchdog: schedule retry if limit was hit
        if limit_hit and session.watchdog_enabled:
            delay = limit_reset_delay if limit_reset_delay else 3600.0
            delay += 60  # buffer after reset
            minutes = int(delay // 60)
            await send_message(session, message=TextMessage(f"🐕 Watchdog: limit hit, retry in {minutes} min"))
            if session._watchdog_task and not session._watchdog_task.done():
                session._watchdog_task.cancel()
            session._watchdog_task = asyncio.create_task(_watchdog_retry(session, thread_id, delay))

        _log_session_debug(
            session,
            "send_to_claude",
            "Completed",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - start_t) * 1000),
        )

    except asyncio.CancelledError:
        # Hard cancel fallback - only happens if soft interrupt timed out
        if session.logger:
            session.logger._write_log("Task hard-cancelled after timeout")
        _log_session_debug(
            session,
            "send_to_claude",
            "CancelledError",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - start_t) * 1000),
        )
        # Notify user (SDK ack message wasn't received in time)
        await send_message(session, message=TextMessage("_[interrupted by user]_"))
        # Don't re-raise - we want clean termination
    except ProcessError as e:
        # Log stderr from CLI process
        if session.logger:
            session.logger.log_error("send_to_claude", e)
            if e.stderr:
                session.logger.log_stderr(e.stderr)
        error_msg = f"❌ Error: {str(e)}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr[:500]}"
        await send_message(session, message=TextMessage(error_msg))
        _log_session_debug(
            session,
            "send_to_claude",
            "ProcessError",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - start_t) * 1000),
            error=str(e),
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        await send_message(session, message=TextMessage(error_msg))
        if session.logger:
            session.logger.log_error("send_to_claude", e)
        _log_session_debug(
            session,
            "send_to_claude",
            "Exception",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - start_t) * 1000),
            error=str(e),
        )
    finally:
        # Always clear references
        session.client = None
        session.current_task = None
        _log_session_debug(
            session,
            "send_to_claude",
            "End (cleanup)",
            thread_id=thread_id,
            elapsed_ms=int((time.perf_counter() - start_t) * 1000),
        )


async def send_or_edit_response(
    session: ClaudeSession,
    bot: Optional[Bot] = None,
    existing_ref: Optional[MessageRef] = None,
    text: str = "",
    msg_text_len: int = 0
) -> tuple[Optional[MessageRef], int]:
    """Send a new response message or edit an existing one.

    Handles overflow by starting new messages when text exceeds platform limit.
    Splits long responses into multiple messages to avoid truncation.

    Args:
        session: The Claude session
        bot: Deprecated, ignored - uses session.get_platform()
        existing_ref: Existing message ref to edit, or None to send new
        text: Full accumulated text to display
        msg_text_len: Length of text already in existing_ref (for overflow detection)

    Returns:
        Tuple of (last message ref, length of text in that message)
    """
    if not text.strip():
        return existing_ref, msg_text_len

    platform = session.get_platform()
    max_len = platform.max_message_length if platform else 4000

    # Check if we need to overflow to new messages
    if len(text) > max_len and existing_ref and msg_text_len > 0:
        # Current message is full, send overflow text as new message(s)
        overflow_text = text[msg_text_len:]

        # Split overflow into chunks and send each as a new message
        chunks = split_text(overflow_text, max_len)
        last_ref: Optional[MessageRef] = None
        last_len = 0

        for chunk in chunks:
            new_ref = await _send_with_fallback(
                session,
                message=TextMessage(chunk),
                existing_ref=None,
            )
            if new_ref:
                last_ref = new_ref
                last_len = len(chunk)

        return last_ref if last_ref else existing_ref, last_len if last_ref else msg_text_len

    # For edits, truncate if too long (can't split an edit into multiple messages)
    # For new messages, send_message will handle splitting via split_text
    display_text = text
    if existing_ref and existing_ref.platform_data and len(display_text) > max_len:
        display_text = display_text[:max_len - 10] + "\n..."

    result_ref = await _send_with_fallback(
        session,
        message=TextMessage(display_text),
        existing_ref=existing_ref,
    )
    if result_ref:
        return result_ref, len(display_text) if len(display_text) <= max_len else max_len
    return existing_ref, msg_text_len


async def _send_with_fallback(
    session: ClaudeSession,
    bot: Optional[Bot] = None,
    message: Optional[PlatformMessage] = None,
    existing_ref: Optional[MessageRef] = None,
    max_retries: int = 3
) -> Optional[MessageRef]:
    """Send or edit a message using platform abstraction.

    Platform client handles retries and fallbacks internally.

    Args:
        session: The Claude session
        bot: Deprecated, ignored - uses session.get_platform()
        message: Structured message payload (platform formats content)
        existing_ref: Existing message ref to edit, or None to send new
        max_retries: Retry attempts (handled by platform)

    Returns:
        MessageRef if successful, None otherwise
    """
    if message is None:
        return None

    platform = session.get_platform()
    if not platform:
        return None

    try:
        if existing_ref and existing_ref.platform_data:
            await platform.edit_message(existing_ref, message)
            return existing_ref
        else:
            return await platform.send_message(message)
    except Exception as e:
        error_str = str(e).lower()
        # "message is not modified" is not an error
        if "message is not modified" in error_str:
            return existing_ref
        if session.logger:
            session.logger.log_error("_send_with_fallback", e)
        return None


async def send_message(
    session: ClaudeSession,
    bot: Optional[Bot] = None,
    message: Optional[PlatformMessage] = None,
    parse_mode: Optional[str] = None
) -> Optional[MessageRef]:
    """Send a new message using platform abstraction.

    Args:
        session: The Claude session
        bot: Deprecated, ignored - uses session.get_platform()
        message: Structured message payload (platform formats content)
        parse_mode: Deprecated, ignored - platform handles formatting

    Returns:
        MessageRef for later editing, or None if send failed
    """
    if message is None or is_empty_message(message):
        return None

    platform = session.get_platform()
    if not platform:
        return None

    # Platform client handles rate limiting internally
    return await platform.send_message(message)


async def send_typing_action(session: ClaudeSession, bot: Optional[Bot] = None) -> None:
    """Send typing indicator if enough time has passed.

    Uses platform abstraction. Bot parameter is deprecated and ignored.
    """
    now = time.time()
    if now - session.last_typing_action < TYPING_ACTION_INTERVAL:
        return

    platform = session.get_platform()
    if platform:
        try:
            await platform.send_typing()
            session.last_typing_action = now
        except Exception as e:
            if session.logger and "flood" not in str(e).lower():
                session.logger.log_debug("send_typing_action", f"Failed: {e}")


async def send_diff_images_gallery(
    session: ClaudeSession,
    bot: Optional[Bot] = None,
    images: Optional[list[tuple[BytesIO, str]]] = None
) -> None:
    """Send diff images as a media group (gallery).

    Uses platform abstraction. Bot parameter is deprecated and ignored.
    """
    if not images:
        return

    platform = session.get_platform()
    if not platform:
        return

    try:
        # Convert to format expected by platform.send_photos
        # Platform expects (BytesIO, caption) tuples
        formatted_images = [(img_buffer, f"📝 {filename}") for img_buffer, filename in images]
        await platform.send_photos(formatted_images)
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
