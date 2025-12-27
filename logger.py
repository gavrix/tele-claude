"""
Session debug logging for Telegram bot.

Logs Claude Agent SDK events and Telegram messages for debugging.

Two logging systems:
1. Global app log (logs/app.log) - all non-session events, third-party lib logs
2. Per-session logs (logs/session_*/session.log) - detailed session events
"""
import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Logs directory
LOGS_DIR = Path(__file__).parent / "logs"

# Content preview length
PREVIEW_LENGTH = 60

# Global app logger
_app_logger: Optional[logging.Logger] = None


def setup_logging() -> None:
    """Configure logging: silent console, full file logging.

    - All third-party libs (httpx, telegram, claude_agent_sdk) -> logs/app.log
    - Session-specific events -> logs/session_*/session.log (via SessionLogger)
    """
    global _app_logger

    LOGS_DIR.mkdir(exist_ok=True)

    # Root logger config - capture everything but don't output to console
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers (prevents console output)
    root_logger.handlers.clear()

    # File handler for all logs - rotating to prevent huge files
    app_log_path = LOGS_DIR / "app.log"
    file_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    # App logger for our own code
    _app_logger = logging.getLogger("tele-claude")
    _app_logger.setLevel(logging.DEBUG)

    # Log startup
    _app_logger.info("="*60)
    _app_logger.info("Bot starting up")
    _app_logger.info("="*60)


def get_app_logger() -> logging.Logger:
    """Get the global app logger for non-session logging."""
    global _app_logger
    if _app_logger is None:
        # Fallback if setup_logging() wasn't called
        setup_logging()
    return _app_logger  # type: ignore


def _preview(text: str, length: int = PREVIEW_LENGTH) -> str:
    """Truncate text to preview length with ellipsis."""
    if not text:
        return ""
    # Replace newlines with spaces for single-line preview
    text = text.replace('\n', ' ').replace('\r', '')
    if len(text) <= length:
        return text
    return text[:length] + "..."


def _timestamp() -> float:
    """Get current Unix timestamp with milliseconds."""
    return time.time()


def _format_time(ts: float) -> str:
    """Format timestamp for human-readable log."""
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int((ts % 1) * 1000):03d}"


class SessionLogger:
    """Logger for a single Claude session.

    Creates a session directory with:
    - session.log: Human-readable timeline
    - session.jsonl: Machine-readable JSON lines
    """

    def __init__(self, thread_id: int, chat_id: int, cwd: str, logs_dir: Optional[Path] = None):
        self.thread_id = thread_id
        self.chat_id = chat_id
        self.cwd = cwd
        self.logs_dir = logs_dir or LOGS_DIR

        # Create session directory
        self.session_dir = self._create_session_dir()

        # Open log files
        self.jsonl_path = self.session_dir / "session.jsonl"
        self.log_path = self.session_dir / "session.log"
        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")
        self.log_file = open(self.log_path, "a", encoding="utf-8")

        # Log session start
        self.log_session_start()

    def _create_session_dir(self) -> Path:
        """Create session directory with timestamp."""
        self.logs_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.logs_dir / f"session_{self.thread_id}_{ts}"
        session_dir.mkdir(exist_ok=True)

        return session_dir

    def _write_jsonl(self, entry: dict) -> None:
        """Write a JSON line entry."""
        entry["ts"] = _timestamp()
        self.jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

    def _write_log(self, message: str) -> None:
        """Write a human-readable log line."""
        ts = _format_time(_timestamp())
        self.log_file.write(f"[{ts}] {message}\n")
        self.log_file.flush()

    def log_session_start(self) -> None:
        """Log session start."""
        self._write_jsonl({
            "type": "session_start",
            "thread_id": self.thread_id,
            "chat_id": self.chat_id,
            "cwd": self.cwd
        })
        self._write_log(f"SESSION START thread={self.thread_id} chat={self.chat_id} cwd={self.cwd}")

    def log_session_end(self, reason: str) -> None:
        """Log session end."""
        self._write_jsonl({
            "type": "session_end",
            "reason": reason
        })
        self._write_log(f"SESSION END ({reason})")

    def log_user_input(self, text: str) -> None:
        """Log user input from Telegram."""
        self._write_jsonl({
            "type": "user_input",
            "content": text
        })
        self._write_log(f"USER INPUT \"{_preview(text)}\"")

    def log_sdk_message(self, message: Any) -> None:
        """Log any Agent SDK message."""
        msg_type = type(message).__name__

        # Extract content preview based on message type
        content_preview = ""
        try:
            if hasattr(message, 'content'):
                if isinstance(message.content, str):
                    content_preview = _preview(message.content)
                elif isinstance(message.content, list):
                    # Handle content blocks
                    for block in message.content[:2]:  # First 2 blocks
                        if hasattr(block, 'text'):
                            content_preview = _preview(block.text)
                            break
                        elif hasattr(block, 'name'):
                            content_preview = f"{block.name}(...)"
                            break
            elif hasattr(message, 'result'):
                content_preview = _preview(str(message.result or ""))
        except Exception:
            pass

        self._write_jsonl({
            "type": "sdk_message",
            "msg_type": msg_type,
            "content_preview": content_preview
        })
        self._write_log(f"SDK {msg_type} \"{content_preview}\"")

    def log_tool_call(self, name: str, input_data: dict) -> None:
        """Log tool invocation."""
        # Truncate input for logging
        input_preview = {}
        for k, v in input_data.items():
            v_str = str(v)
            input_preview[k] = v_str[:100] + "..." if len(v_str) > 100 else v_str

        self._write_jsonl({
            "type": "tool_call",
            "tool": name,
            "input": input_preview
        })
        self._write_log(f"TOOL CALL {name}({json.dumps(input_preview)})")

    def log_tool_result(self, name: str, output: str, success: bool = True) -> None:
        """Log tool result."""
        self._write_jsonl({
            "type": "tool_result",
            "tool": name,
            "output_preview": _preview(output, 200),
            "success": success
        })
        status = "OK" if success else "FAIL"
        self._write_log(f"TOOL RESULT {name} [{status}] \"{_preview(output)}\"")

    def log_session_stats(self, cost: float, duration_ms: int, tokens: dict) -> None:
        """Log final session stats."""
        self._write_jsonl({
            "type": "session_stats",
            "cost_usd": cost,
            "duration_ms": duration_ms,
            "tokens": tokens
        })
        self._write_log(f"STATS cost=${cost:.4f} duration={duration_ms}ms tokens={tokens}")

    def log_telegram_send(self, content: str, message_id: int) -> None:
        """Log a Telegram message send."""
        self._write_jsonl({
            "type": "telegram_send",
            "message_id": message_id,
            "content_preview": _preview(content)
        })
        self._write_log(f"TELEGRAM SEND msg={message_id} \"{_preview(content)}\"")

    def log_telegram_edit(self, message_id: int, old_content: str, new_content: str) -> None:
        """Log a Telegram message edit."""
        self._write_jsonl({
            "type": "telegram_edit",
            "message_id": message_id,
            "old_preview": _preview(old_content),
            "new_preview": _preview(new_content)
        })
        self._write_log(f"TELEGRAM EDIT msg={message_id} \"{_preview(old_content)}\" -> \"{_preview(new_content)}\"")

    def log_error(self, context: str, error: Exception) -> None:
        """Log an error."""
        self._write_jsonl({
            "type": "error",
            "context": context,
            "error": str(error),
            "error_type": type(error).__name__
        })
        self._write_log(f"ERROR [{context}] {type(error).__name__}: {error}")

    def log_permission_request(self, request_id: str, tool_name: str, tool_input: dict) -> None:
        """Log a permission request sent to user."""
        self._write_jsonl({
            "type": "permission_request",
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_input": {k: str(v)[:100] for k, v in tool_input.items()}
        })
        self._write_log(f"PERMISSION REQUEST id={request_id} tool={tool_name}")

    def log_permission_callback(self, request_id: str, action: str, tool_name: str) -> None:
        """Log a permission callback received from user."""
        self._write_jsonl({
            "type": "permission_callback",
            "request_id": request_id,
            "action": action,
            "tool_name": tool_name
        })
        self._write_log(f"PERMISSION CALLBACK id={request_id} action={action} tool={tool_name}")

    def log_permission_resolved(self, request_id: str, allowed: bool, found: bool) -> None:
        """Log permission resolution result."""
        self._write_jsonl({
            "type": "permission_resolved",
            "request_id": request_id,
            "allowed": allowed,
            "future_found": found
        })
        status = "ALLOWED" if allowed else "DENIED"
        found_str = "found" if found else "NOT FOUND"
        self._write_log(f"PERMISSION RESOLVED id={request_id} {status} (future {found_str})")

    def log_permission_check(self, tool_name: str, in_allowlist: bool) -> None:
        """Log permission check for a tool."""
        self._write_jsonl({
            "type": "permission_check",
            "tool_name": tool_name,
            "in_allowlist": in_allowlist
        })
        status = "ALLOWLISTED" if in_allowlist else "NEEDS APPROVAL"
        self._write_log(f"PERMISSION CHECK tool={tool_name} {status}")

    def log_stderr(self, message: str) -> None:
        """Log stderr output from Claude CLI."""
        self._write_jsonl({
            "type": "stderr",
            "message": message
        })
        self._write_log(f"STDERR {message}")

    def log_debug(self, context: str, message: str, **kwargs) -> None:
        """Log debug information."""
        self._write_jsonl({
            "type": "debug",
            "context": context,
            "message": message,
            **kwargs
        })
        self._write_log(f"DEBUG [{context}] {message}")

    def log_compact_event(self, input_data: dict) -> None:
        """Log a PreCompact hook event.

        This captures the full input_data for later analysis to understand
        what token counts look like when compaction is actually triggered.
        """
        self._write_jsonl({
            "type": "compact_event",
            "input_data": input_data
        })
        # Log summary for human-readable log
        summary = json.dumps(input_data)[:200] if input_data else "{}"
        self._write_log(f"COMPACT EVENT {summary}")

    def close(self) -> None:
        """Close log files."""
        try:
            self.jsonl_file.close()
            self.log_file.close()
        except Exception:
            pass
