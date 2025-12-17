"""Tests for session.py functionality."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Message

from session import (
    calculate_context_remaining,
    send_or_edit_response,
    ClaudeSession,
    MODEL_CONTEXT_WINDOWS,
    CONTEXT_WARNING_THRESHOLD,
)


class TestCalculateContextRemaining:
    """Tests for calculate_context_remaining()."""

    def test_empty_usage_returns_none(self):
        """Empty usage dict should return None."""
        assert calculate_context_remaining({}) is None
        assert calculate_context_remaining(None) is None

    def test_zero_tokens_returns_none(self):
        """Usage with zero tokens should return None."""
        usage = {
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }
        assert calculate_context_remaining(usage) is None

    def test_normal_usage_calculates_correctly(self):
        """Normal usage should calculate correct percentage remaining.

        Note: Only input_tokens + output_tokens are counted.
        Cache tokens are excluded due to inflated values from server-side tools.
        """
        # 7000 tokens (5000 input + 2000 output) out of 200000 = 3.5% used = 96.5% remaining
        usage = {
            "input_tokens": 5000,
            "cache_read_input_tokens": 10000,  # ignored
            "cache_creation_input_tokens": 3000,  # ignored
            "output_tokens": 2000,
        }
        result = calculate_context_remaining(usage)
        assert result == 96.5

    def test_half_context_used(self):
        """50% context used should return 50% remaining."""
        # Only input_tokens + output_tokens counted
        usage = {
            "input_tokens": 50000,
            "cache_read_input_tokens": 50000,  # ignored
            "cache_creation_input_tokens": 0,
            "output_tokens": 50000,
        }
        result = calculate_context_remaining(usage)
        assert result == 50.0

    def test_context_nearly_full(self):
        """Nearly full context should return small percentage."""
        # 190000 tokens = 95% used = 5% remaining
        usage = {
            "input_tokens": 190000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }
        result = calculate_context_remaining(usage)
        assert result == 5.0

    def test_context_exceeded_returns_zero(self):
        """Context exceeded should return 0, not negative."""
        usage = {
            "input_tokens": 250000,  # More than 200000 context window
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }
        result = calculate_context_remaining(usage)
        assert result == 0

    def test_missing_token_fields_treated_as_zero(self):
        """Missing token fields should be treated as 0."""
        usage = {"input_tokens": 20000}  # Only one field
        result = calculate_context_remaining(usage)
        assert result == 90.0

    def test_known_model_uses_correct_window(self):
        """Known model should use its specific context window."""
        usage = {"input_tokens": 100000}
        result = calculate_context_remaining(usage, "claude-opus-4-5-20251101")
        assert result == 50.0  # 100k out of 200k

    def test_unknown_model_uses_default_window(self):
        """Unknown model should use default context window."""
        usage = {"input_tokens": 100000}
        result = calculate_context_remaining(usage, "unknown-model")
        assert result == 50.0  # Uses default 200k

    def test_threshold_boundary(self):
        """Test values around the warning threshold."""
        # Exactly at threshold (15% remaining = 85% used = 170000 tokens)
        usage = {"input_tokens": 170000}
        result = calculate_context_remaining(usage)
        assert result == 15.0

        # Just below threshold
        usage = {"input_tokens": 171000}
        result = calculate_context_remaining(usage)
        assert result < CONTEXT_WARNING_THRESHOLD

        # Just above threshold
        usage = {"input_tokens": 169000}
        result = calculate_context_remaining(usage)
        assert result > CONTEXT_WARNING_THRESHOLD


class TestSendOrEditResponse:
    """Tests for send_or_edit_response() overflow handling."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock ClaudeSession."""
        session = MagicMock(spec=ClaudeSession)
        session.chat_id = 123
        session.logger = None
        return session

    @pytest.fixture
    def mock_bot(self):
        """Create a mock Telegram Bot."""
        bot = AsyncMock()
        return bot

    @pytest.fixture
    def mock_message(self):
        """Create a mock Telegram Message."""
        msg = MagicMock(spec=Message)
        msg.message_id = 456
        return msg

    @pytest.mark.asyncio
    async def test_empty_text_returns_existing(self, mock_session, mock_bot, mock_message):
        """Empty text should return existing message unchanged."""
        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, mock_message, "   ", 100
        )
        assert result_msg == mock_message
        assert result_len == 100
        mock_bot.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_text_edits_existing(self, mock_session, mock_bot, mock_message):
        """Short text should edit existing message."""
        text = "Hello world"
        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, mock_message, text, 0
        )
        assert result_msg == mock_message
        assert result_len == len(text)
        mock_bot.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_text_sends_new_when_no_existing(self, mock_session, mock_bot):
        """Short text with no existing message should send new."""
        text = "Hello world"
        new_msg = MagicMock(spec=Message)
        mock_bot.send_message.return_value = new_msg
        mock_session.thread_id = 1

        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, None, text, 0
        )

        assert result_msg == new_msg
        assert result_len == len(text)
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_overflow_creates_new_message(self, mock_session, mock_bot, mock_message):
        """Text exceeding 4000 chars with existing message should overflow."""
        # First 3000 chars already in message, now adding 2000 more = 5000 total
        existing_text = "a" * 3000
        new_text = "b" * 2000
        full_text = existing_text + new_text

        new_msg = MagicMock(spec=Message)
        new_msg.message_id = 789
        mock_bot.send_message.return_value = new_msg
        mock_session.thread_id = 1

        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, mock_message, full_text, 3000
        )

        assert result_msg == new_msg
        assert result_len == 2000  # Length of overflow text
        # Should send the overflow portion only
        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        # Check that the text contains the overflow content (b's)
        assert "b" in str(call_args)

    @pytest.mark.asyncio
    async def test_first_message_over_4000_truncates(self, mock_session, mock_bot):
        """First message (no existing) over 4000 should truncate as safety net."""
        text = "a" * 5000
        new_msg = MagicMock(spec=Message)
        mock_bot.send_message.return_value = new_msg
        mock_session.thread_id = 1

        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, None, text, 0
        )

        # Should truncate to ~3990 + "..."
        assert result_len <= 4000

    @pytest.mark.asyncio
    async def test_edit_failure_falls_back_to_plain_text(self, mock_session, mock_bot, mock_message):
        """HTML parse error should fallback to plain text."""
        text = "Hello **world**"
        mock_bot.edit_message_text.side_effect = [
            Exception("Can't parse entities"),
            None  # Second call succeeds
        ]

        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, mock_message, text, 0
        )

        assert result_msg == mock_message
        # Should have tried twice - once with HTML, once plain
        assert mock_bot.edit_message_text.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_correct_length_after_edit(self, mock_session, mock_bot, mock_message):
        """Should return correct text length after successful edit."""
        text = "Test message with some content"
        result_msg, result_len = await send_or_edit_response(
            mock_session, mock_bot, mock_message, text, 0
        )
        assert result_len == len(text)


class TestWarningAppendLogic:
    """Tests for context warning append behavior."""

    def test_warning_fits_in_message(self):
        """Warning should fit when message has room."""
        response_msg_text_len = 3900
        warning = "\n\n⚠️ 12% context remaining"
        # 3900 + ~35 = 3935 < 4000
        assert response_msg_text_len + len(warning) <= 4000

    def test_warning_does_not_fit(self):
        """Warning should not fit when message is near limit."""
        response_msg_text_len = 3980
        warning = "\n\n⚠️ 12% context remaining"
        # 3980 + ~35 = 4015 > 4000
        assert response_msg_text_len + len(warning) > 4000

    def test_warning_format(self):
        """Warning message format should be correct."""
        context_remaining = 12.4
        warning = f"\n\n⚠️ {context_remaining:.0f}% context remaining"
        assert warning == "\n\n⚠️ 12% context remaining"

    def test_warning_threshold_value(self):
        """Threshold should be set to 15%."""
        assert CONTEXT_WARNING_THRESHOLD == 15


class TestModelContextWindows:
    """Tests for model context window configuration."""

    def test_opus_context_window(self):
        """Opus model should have 200k context window."""
        assert MODEL_CONTEXT_WINDOWS["claude-opus-4-5-20251101"] == 200000

    def test_sonnet_context_window(self):
        """Sonnet models should have 200k context window."""
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-5-20251101"] == 200000
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-20250514"] == 200000

    def test_default_context_window(self):
        """Default context window should be 200k."""
        assert MODEL_CONTEXT_WINDOWS["default"] == 200000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
