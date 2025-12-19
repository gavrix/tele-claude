"""Tests for commands.py functionality."""
import pytest
import tempfile
from pathlib import Path

from commands import (
    SlashCommand,
    HARDCODED_COMMANDS,
    load_contextual_commands,
    get_command_prompt,
    get_help_message,
)


class TestLoadContextualCommands:
    """Tests for load_contextual_commands()."""

    def test_missing_commands_dir_returns_empty(self, tmp_path: Path):
        """Missing commands/ directory should return empty list."""
        result = load_contextual_commands(str(tmp_path))
        assert result == []

    def test_empty_commands_dir_returns_empty(self, tmp_path: Path):
        """Empty commands/ directory should return empty list."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        result = load_contextual_commands(str(tmp_path))
        assert result == []

    def test_loads_simple_command(self, tmp_path: Path):
        """Simple .md file should be loaded as command."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "test.md").write_text("Run pytest")

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].prompt == "Run pytest"
        assert result[0].is_contextual is True
        assert result[0].description == "Run test command"

    def test_loads_command_with_description_comment(self, tmp_path: Path):
        """Command with HTML comment should use it as description."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "build.md").write_text(
            "<!-- Build the project -->\nRun npm build"
        )

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert result[0].name == "build"
        assert result[0].description == "Build the project"
        assert result[0].prompt == "Run npm build"

    def test_loads_multiple_commands(self, tmp_path: Path):
        """Multiple .md files should all be loaded."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "test.md").write_text("Run tests")
        (commands_dir / "lint.md").write_text("Run linter")
        (commands_dir / "build.md").write_text("Run build")

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 3
        names = {cmd.name for cmd in result}
        assert names == {"test", "lint", "build"}

    def test_skips_empty_files(self, tmp_path: Path):
        """Empty .md files should be skipped."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "empty.md").write_text("")
        (commands_dir / "valid.md").write_text("Do something")

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert result[0].name == "valid"

    def test_skips_non_md_files(self, tmp_path: Path):
        """Non-.md files should be ignored."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "test.md").write_text("Run tests")
        (commands_dir / "readme.txt").write_text("Not a command")
        (commands_dir / "script.py").write_text("print('hello')")

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert result[0].name == "test"

    def test_description_truncated_to_50_chars(self, tmp_path: Path):
        """Description should be truncated to 50 characters (Telegram limit)."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        long_desc = "A" * 100
        (commands_dir / "long.md").write_text(f"<!-- {long_desc} -->\nDo stuff")

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert len(result[0].description) == 50

    def test_multiline_prompt_preserved(self, tmp_path: Path):
        """Multiline prompts should be preserved."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        prompt = "Line 1\nLine 2\nLine 3"
        (commands_dir / "multi.md").write_text(prompt)

        result = load_contextual_commands(str(tmp_path))

        assert len(result) == 1
        assert result[0].prompt == prompt


class TestGetCommandPrompt:
    """Tests for get_command_prompt()."""

    def test_finds_hardcoded_command(self):
        """Hard-coded commands should be found."""
        result = get_command_prompt("plan", [])
        assert result == "/plan"

        result = get_command_prompt("compact", [])
        assert result == "/compact"

    def test_finds_contextual_command(self):
        """Contextual commands should be found."""
        contextual = [
            SlashCommand("test", "Run tests", "pytest -v", True),
            SlashCommand("lint", "Run linter", "ruff check .", True),
        ]

        result = get_command_prompt("test", contextual)
        assert result == "pytest -v"

        result = get_command_prompt("lint", contextual)
        assert result == "ruff check ."

    def test_hardcoded_takes_precedence(self):
        """Hard-coded commands should take precedence over contextual."""
        # Create contextual command with same name as hard-coded
        contextual = [
            SlashCommand("plan", "Custom plan", "My custom plan prompt", True),
        ]

        result = get_command_prompt("plan", contextual)
        # Should return hard-coded prompt, not contextual
        assert result == "/plan"

    def test_unknown_command_returns_none(self):
        """Unknown command should return None."""
        contextual = [
            SlashCommand("test", "Run tests", "pytest", True),
        ]

        result = get_command_prompt("unknown", contextual)
        assert result is None

    def test_empty_contextual_list(self):
        """Empty contextual list should still find hard-coded commands."""
        result = get_command_prompt("plan", [])
        assert result == "/plan"

        result = get_command_prompt("unknown", [])
        assert result is None


class TestHardcodedCommands:
    """Tests for HARDCODED_COMMANDS constant."""

    def test_plan_command_exists(self):
        """Plan command should exist in hard-coded commands."""
        names = {cmd.name for cmd in HARDCODED_COMMANDS}
        assert "plan" in names

    def test_compact_command_exists(self):
        """Compact command should exist in hard-coded commands."""
        names = {cmd.name for cmd in HARDCODED_COMMANDS}
        assert "compact" in names

    def test_help_command_exists(self):
        """Help command should exist in hard-coded commands."""
        names = {cmd.name for cmd in HARDCODED_COMMANDS}
        assert "help" in names

    def test_all_hardcoded_are_not_contextual(self):
        """All hard-coded commands should have is_contextual=False."""
        for cmd in HARDCODED_COMMANDS:
            assert cmd.is_contextual is False


class TestGetHelpMessage:
    """Tests for get_help_message()."""

    def test_shows_global_commands(self):
        """Help message should include global commands."""
        result = get_help_message([])
        assert "/help" in result
        assert "/plan" in result
        assert "/compact" in result
        assert "<b>Global:</b>" in result

    def test_shows_contextual_commands(self):
        """Help message should include contextual commands when provided."""
        contextual = [
            SlashCommand("test", "Run tests", "pytest", True),
            SlashCommand("lint", "Run linter", "ruff check", True),
        ]
        result = get_help_message(contextual)
        assert "/test" in result
        assert "/lint" in result
        assert "<b>Project:</b>" in result

    def test_no_project_section_when_empty(self):
        """Help message should not show Project section when no contextual commands."""
        result = get_help_message([])
        assert "<b>Project:</b>" not in result

    def test_returns_html_format(self):
        """Help message should use HTML formatting."""
        result = get_help_message([])
        assert "<b>Available Commands</b>" in result
