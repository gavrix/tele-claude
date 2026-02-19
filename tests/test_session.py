"""Tests for session.py functionality."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Message

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from session import (
    calculate_context_remaining,
    send_or_edit_response,
    ClaudeSession,
    MODEL_CONTEXT_WINDOWS,
    CONTEXT_WARNING_THRESHOLD,
)
from platforms import MessageRef, TextMessage


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
        """Normal usage should calculate percentage correctly."""
        # 50k tokens used out of 200k = 25% used = 75% remaining
        usage = {
            "input_tokens": 40000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 5000,
            "output_tokens": 5000,
        }
        result = calculate_context_remaining(usage)
        assert result is not None
        assert 74 <= result <= 78  # ~75% remaining

    def test_half_context_used(self):
        """Half context used should return ~50% remaining."""
        usage = {
            "input_tokens": 90000,
            "output_tokens": 10000,
        }
        result = calculate_context_remaining(usage)
        assert result is not None
        assert 49 <= result <= 51

    def test_context_nearly_full(self):
        """Nearly full context should return low percentage."""
        usage = {
            "input_tokens": 180000,
            "output_tokens": 10000,
        }
        result = calculate_context_remaining(usage)
        assert result is not None
        assert result < 10

    def test_context_exceeded_returns_zero(self):
        """Context exceeded should return 0."""
        usage = {
            "input_tokens": 200000,
            "output_tokens": 50000,
        }
        result = calculate_context_remaining(usage)
        assert result == 0

    def test_missing_token_fields_treated_as_zero(self):
        """Missing token fields should be treated as 0."""
        usage = {"input_tokens": 50000}
        result = calculate_context_remaining(usage)
        assert result is not None
        assert result > 0

    def test_known_model_uses_correct_window(self):
        """Known model should use its specific context window."""
        usage = {"input_tokens": 100000, "output_tokens": 0}
        result = calculate_context_remaining(usage, "claude-opus-4-5-20251101")
        assert result is not None
        assert 49 <= result <= 51  # 100k/200k = 50% used

    def test_unknown_model_uses_default_window(self):
        """Unknown model should use default context window."""
        usage = {"input_tokens": 100000, "output_tokens": 0}
        result = calculate_context_remaining(usage, "unknown-model")
        assert result is not None
        assert 49 <= result <= 51

    def test_threshold_boundary(self):
        """Test behavior around warning threshold."""
        # Just above threshold (should not trigger warning)
        usage = {"input_tokens": 169000}
        result = calculate_context_remaining(usage)
        assert result is not None and result > CONTEXT_WARNING_THRESHOLD


class TestSendOrEditResponse:
    """Tests for send_or_edit_response() overflow handling."""

    @pytest.fixture
    def mock_platform(self):
        """Create a mock PlatformClient."""
        platform = AsyncMock()
        platform.max_message_length = 4000
        # send_message returns a MessageRef
        platform.send_message.return_value = MessageRef(platform_data=MagicMock())
        return platform

    @pytest.fixture
    def mock_session(self, mock_platform):
        """Create a mock ClaudeSession with platform."""
        session = MagicMock(spec=ClaudeSession)
        session.chat_id = 123
        session.logger = None
        session.get_platform.return_value = mock_platform
        return session

    @pytest.fixture
    def mock_message_ref(self):
        """Create a mock MessageRef."""
        msg = MagicMock(spec=Message)
        msg.message_id = 456
        return MessageRef(platform_data=msg)

    @pytest.mark.asyncio
    async def test_empty_text_returns_existing(self, mock_session, mock_platform, mock_message_ref):
        """Empty text should return existing message unchanged."""
        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text="   ", msg_text_len=100
        )
        assert result_ref == mock_message_ref
        assert result_len == 100
        mock_platform.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_text_edits_existing(self, mock_session, mock_platform, mock_message_ref):
        """Short text should edit existing message."""
        text = "Hello world"
        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=text, msg_text_len=0
        )
        assert result_ref == mock_message_ref
        assert result_len == len(text)
        mock_platform.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_text_sends_new_when_no_existing(self, mock_session, mock_platform):
        """Short text with no existing message should send new."""
        text = "Hello world"
        new_ref = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.return_value = new_ref

        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=None, text=text, msg_text_len=0
        )

        assert result_ref == new_ref
        assert result_len == len(text)
        mock_platform.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_overflow_creates_new_message(self, mock_session, mock_platform, mock_message_ref):
        """Text exceeding 4000 chars with existing message should overflow."""
        # First 3000 chars already in message, now adding 2000 more = 5000 total
        existing_text = "a" * 3000
        new_text = "b" * 2000
        full_text = existing_text + new_text

        new_ref = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.return_value = new_ref

        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=full_text, msg_text_len=3000
        )

        assert result_ref == new_ref
        assert result_len == 2000  # Length of overflow text
        # Should send the overflow portion only
        mock_platform.send_message.assert_called_once()
        call_args = mock_platform.send_message.call_args
        sent_message = call_args[0][0] if call_args[0] else call_args[1].get('message')
        assert isinstance(sent_message, TextMessage)
        # Check that the text contains the overflow content (b's)
        assert "b" in sent_message.text

    @pytest.mark.asyncio
    async def test_first_message_over_4000_truncates(self, mock_session, mock_platform):
        """First message (no existing) over 4000 should truncate as safety net."""
        text = "a" * 5000
        new_ref = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.return_value = new_ref

        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=None, text=text, msg_text_len=0
        )

        # Should truncate to ~3990 + "..."
        assert result_len <= 4000

    @pytest.mark.asyncio
    async def test_overflow_splits_into_multiple_messages(self, mock_session, mock_platform, mock_message_ref):
        """Very long overflow text should be split into multiple messages."""
        # First 3000 chars already in message, adding 10000 more = 13000 total
        existing_text = "a" * 3000
        overflow_text = "b" * 10000  # Should split into 3 messages
        full_text = existing_text + overflow_text

        ref1 = MessageRef(platform_data=MagicMock())
        ref2 = MessageRef(platform_data=MagicMock())
        ref3 = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.side_effect = [ref1, ref2, ref3]

        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=full_text, msg_text_len=3000
        )

        # Should have sent multiple messages
        assert mock_platform.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_edit_failure_returns_existing_ref(self, mock_session, mock_platform, mock_message_ref):
        """If edit fails with real error, should return existing ref as fallback."""
        mock_platform.edit_message.side_effect = Exception("Network error")

        text = "Hello world"
        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=text, msg_text_len=0
        )

        # Should return existing ref as fallback on error
        assert result_ref == mock_message_ref

    @pytest.mark.asyncio
    async def test_returns_correct_length_after_edit(self, mock_session, mock_platform, mock_message_ref):
        """Should return correct text length after successful edit."""
        text = "Test message with some content"
        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=text, msg_text_len=0
        )
        assert result_len == len(text)

    @pytest.mark.asyncio
    async def test_new_message_over_max_sends_full_text(self, mock_session, mock_platform):
        """New message over max_len should send full text to send_message (not pre-truncate)."""
        text = "a" * 5000
        new_ref = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.return_value = new_ref

        await send_or_edit_response(
            mock_session, existing_ref=None, text=text, msg_text_len=0
        )

        # send_message should receive full text (it handles splitting via split_text)
        mock_platform.send_message.assert_called_once()
        call_args = mock_platform.send_message.call_args
        sent_message = call_args[0][0] if call_args[0] else call_args[1].get('message')
        assert isinstance(sent_message, TextMessage)
        assert len(sent_message.text) == 5000  # Full text, not truncated

    @pytest.mark.asyncio
    async def test_edit_existing_over_max_truncates(self, mock_session, mock_platform, mock_message_ref):
        """Editing existing message over max_len should truncate (can't split an edit)."""
        text = "a" * 5000

        result_ref, result_len = await send_or_edit_response(
            mock_session, existing_ref=mock_message_ref, text=text, msg_text_len=0
        )

        # edit_message should receive truncated text
        mock_platform.edit_message.assert_called_once()
        call_args = mock_platform.edit_message.call_args
        sent_message = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('message')
        assert isinstance(sent_message, TextMessage)
        assert len(sent_message.text) < 5000  # Truncated
        assert sent_message.text.endswith("...")  # Has truncation marker

    @pytest.mark.asyncio
    async def test_discord_2000_char_limit(self, mock_session):
        """Test with Discord's 2000 char limit."""
        # Create mock platform with Discord's limit
        mock_platform = AsyncMock()
        mock_platform.max_message_length = 2000
        mock_session.get_platform.return_value = mock_platform

        text = "a" * 2500
        existing_ref = MessageRef(platform_data=MagicMock())

        await send_or_edit_response(
            mock_session, existing_ref=existing_ref, text=text, msg_text_len=0
        )

        # Should truncate to ~1990 chars
        call_args = mock_platform.edit_message.call_args
        sent_message = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('message')
        assert isinstance(sent_message, TextMessage)
        assert len(sent_message.text) <= 2000
        assert sent_message.text.endswith("...")

    @pytest.mark.asyncio
    async def test_message_ref_without_platform_data_sends_new(self, mock_session, mock_platform):
        """MessageRef with platform_data=None should send new message, not edit."""
        text = "a" * 5000
        empty_ref = MessageRef(platform_data=None)
        new_ref = MessageRef(platform_data=MagicMock())
        mock_platform.send_message.return_value = new_ref

        await send_or_edit_response(
            mock_session, existing_ref=empty_ref, text=text, msg_text_len=0
        )

        # Should call send_message (not edit_message) since platform_data is None
        mock_platform.send_message.assert_called_once()
        mock_platform.edit_message.assert_not_called()


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

    def test_opus_45_context_window(self):
        """Opus 4.5 model should have 200k context window."""
        assert MODEL_CONTEXT_WINDOWS["claude-opus-4-5-20251101"] == 200000

    def test_sonnet_context_window(self):
        """Sonnet models should have 200k context window."""
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-5-20251101"] == 200000
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-20250514"] == 200000

    def test_default_context_window(self):
        """Default context window should be 200k."""
        assert MODEL_CONTEXT_WINDOWS["default"] == 200000


class TestWatchdogRetry:
    """Tests for watchdog retry path."""

    @pytest.mark.asyncio
    async def test_watchdog_retry_enqueues_continue_via_actor(self, monkeypatch):
        """Watchdog retry should enqueue through SessionActor, not start task directly."""
        import session as session_module

        sent_messages = AsyncMock()
        start_task = MagicMock()
        enqueued = []

        async def fake_enqueue(trigger):
            enqueued.append(trigger)

        monkeypatch.setattr(session_module, "send_message", sent_messages)
        monkeypatch.setattr(session_module, "start_claude_task", start_task)

        session = ClaudeSession(chat_id=123, thread_id=456, cwd="/tmp")
        session.watchdog_enabled = True
        session.actor_enqueue = fake_enqueue

        await session_module._watchdog_retry(session, thread_id=456, delay=0)

        assert len(enqueued) == 1
        assert enqueued[0].prompt == "Continue"
        assert enqueued[0].source == "watchdog"
        assert enqueued[0].session_key == "telegram:123:456"
        sent_messages.assert_called_once()
        start_task.assert_not_called()
