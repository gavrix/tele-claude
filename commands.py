"""
Slash command management for Telegram-Claude sessions.

Handles hard-coded universal commands and project-specific
contextual commands loaded from commands/*.md files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

from telegram import Bot, BotCommand, BotCommandScopeChat

_log = logging.getLogger("tele-claude.commands")


@dataclass
class SlashCommand:
    """Represents a slash command."""

    name: str  # Command name without slash (e.g., "plan")
    description: str  # Short description for Telegram autocompletion
    prompt: str  # Prompt text to send to Claude
    is_contextual: bool  # True if loaded from project, False if hard-coded


# Hard-coded universal commands - available in all sessions
HARDCODED_COMMANDS: list[SlashCommand] = [
    SlashCommand(
        name="plan",
        description="Enter plan mode - explore and design before implementing",
        prompt="/plan",
        is_contextual=False,
    ),
    SlashCommand(
        name="compact",
        description="Summarize conversation to free up context window",
        prompt="/compact",
        is_contextual=False,
    ),
]


def load_contextual_commands(cwd: str) -> list[SlashCommand]:
    """Load project-specific commands from commands/*.md files.

    Args:
        cwd: Project working directory path

    Returns:
        List of SlashCommand objects from .md files in commands/ dir
    """
    commands_dir = Path(cwd) / "commands"
    commands: list[SlashCommand] = []

    if not commands_dir.exists() or not commands_dir.is_dir():
        return commands

    for md_file in commands_dir.glob("*.md"):
        try:
            name = md_file.stem  # filename without .md
            content = md_file.read_text(encoding="utf-8").strip()

            if not content:
                _log.warning(f"Empty command file: {md_file}")
                continue

            # Extract description from first line if it's an HTML comment
            lines = content.split("\n", 1)
            first_line = lines[0].strip()

            if first_line.startswith("<!--") and "-->" in first_line:
                # HTML comment format: <!-- Description -->
                description = (
                    first_line.replace("<!--", "").replace("-->", "").strip()
                )
                prompt = lines[1].strip() if len(lines) > 1 else ""
            else:
                # No description comment - use filename as description
                description = f"Run {name} command"
                prompt = content

            if not prompt:
                _log.warning(f"No prompt content in command file: {md_file}")
                continue

            commands.append(
                SlashCommand(
                    name=name,
                    description=description[:50],  # Telegram limit
                    prompt=prompt,
                    is_contextual=True,
                )
            )
            _log.debug(f"Loaded contextual command: /{name}")

        except Exception as e:
            _log.warning(f"Failed to load command from {md_file}: {e}")
            continue

    return commands


async def register_commands_for_chat(
    bot: Bot, chat_id: int, contextual_commands: list[SlashCommand]
) -> None:
    """Register commands with Telegram for autocompletion.

    Combines hard-coded and contextual commands, registers with
    BotCommandScopeChat for this specific chat.

    Args:
        bot: Telegram bot instance
        chat_id: Chat ID to register commands for
        contextual_commands: Project-specific commands to include
    """
    all_commands = HARDCODED_COMMANDS + contextual_commands

    telegram_commands = [
        BotCommand(command=cmd.name, description=cmd.description)
        for cmd in all_commands
    ]

    try:
        await bot.set_my_commands(
            commands=telegram_commands, scope=BotCommandScopeChat(chat_id=chat_id)
        )
        _log.info(
            f"Registered {len(telegram_commands)} commands for chat {chat_id}"
        )
    except Exception as e:
        _log.error(f"Failed to register commands for chat {chat_id}: {e}")


def get_command_prompt(
    command_name: str, contextual_commands: list[SlashCommand]
) -> Optional[str]:
    """Look up prompt for a command by name.

    Args:
        command_name: Command name without slash
        contextual_commands: Project-specific commands to search

    Returns:
        Prompt string if found, None if unknown command
    """
    # Check hard-coded first (they take precedence)
    for cmd in HARDCODED_COMMANDS:
        if cmd.name == command_name:
            return cmd.prompt

    # Check contextual
    for cmd in contextual_commands:
        if cmd.name == command_name:
            return cmd.prompt

    return None
