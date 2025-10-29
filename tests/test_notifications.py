"""
Tests for notification system.

Tests for Telegram and Email alerts.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.risk.emergency_stop import EmergencyEvent, EmergencyTrigger
from bot.risk.notifications import (
    EmailConfig,
    EmailNotifier,
    NotificationManager,
    TelegramConfig,
    TelegramNotifier,
)


@pytest.fixture
def emergency_event():
    """Create test emergency event."""
    return EmergencyEvent(
        trigger=EmergencyTrigger.FLASH_CRASH,
        timestamp=datetime.now(UTC),
        message="Flash crash detected: 12% price drop in 5 minutes",
        details={
            "symbol": "BTCUSDT",
            "price_change_pct": 0.12,
            "min_price": "44000",
            "max_price": "50000",
        },
        severity=10,
        correlation_id="test-correlation-123",
    )


class TestTelegramNotifier:
    """Tests for Telegram notifications."""

    @pytest.fixture
    def telegram_config(self):
        """Create test Telegram configuration."""
        return TelegramConfig(
            bot_token="test_bot_token",
            chat_ids=["123456789", "987654321"],
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_send_telegram_alert(self, telegram_config, emergency_event):
        """Test sending alert via Telegram."""
        # Setup mock bot
        mock_bot = AsyncMock()

        # Mock Bot class directly
        with patch("bot.risk.notifications.TELEGRAM_AVAILABLE", True):
            with patch("bot.risk.notifications.TelegramBot", return_value=mock_bot):
                # Create notifier
                notifier = TelegramNotifier(telegram_config)

                # Send alert
                await notifier.send_alert(emergency_event)

                # Should send to all chat IDs
                assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_telegram_message_format(self, telegram_config, emergency_event):
        """Test Telegram message formatting."""
        mock_bot = AsyncMock()

        with patch("bot.risk.notifications.TELEGRAM_AVAILABLE", True):
            with patch("bot.risk.notifications.TelegramBot", return_value=mock_bot):
                notifier = TelegramNotifier(telegram_config)

                await notifier.send_alert(emergency_event)

                # Check message was called
                assert mock_bot.send_message.called

                # Get the message that was sent
                call_args = mock_bot.send_message.call_args
                message = call_args.kwargs["text"]

                # Verify message contains key information
                assert "EMERGENCY ALERT" in message
                assert "flash_crash" in message
                assert "10/10" in message
                assert "test-correlation-123" in message

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.TELEGRAM_AVAILABLE", False)
    async def test_telegram_library_not_available(self, telegram_config, emergency_event):
        """Test behavior when Telegram library is not available."""
        notifier = TelegramNotifier(telegram_config)

        # Should be disabled
        assert not notifier.enabled

        # Should not raise error when sending
        await notifier.send_alert(emergency_event)

    @pytest.mark.asyncio
    async def test_telegram_send_error_handling(self, telegram_config, emergency_event):
        """Test error handling during Telegram send."""

        # Create mock TelegramError
        class MockTelegramError(Exception):
            pass

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = MockTelegramError("Network error")

        with patch("bot.risk.notifications.TELEGRAM_AVAILABLE", True):
            with patch("bot.risk.notifications.TelegramBot", return_value=mock_bot):
                with patch("bot.risk.notifications.TelegramErrorType", MockTelegramError):
                    notifier = TelegramNotifier(telegram_config)

                    # Should not raise exception
                    await notifier.send_alert(emergency_event)


class TestEmailNotifier:
    """Tests for Email notifications."""

    @pytest.fixture
    def email_config(self):
        """Create test Email configuration."""
        return EmailConfig(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="bot@example.com",
            to_emails=["admin@example.com", "trader@example.com"],
            use_tls=True,
            enabled=True,
        )

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.smtplib.SMTP")
    async def test_send_email_alert(self, mock_smtp_class, email_config, emergency_event):
        """Test sending alert via Email."""
        # Setup mock SMTP
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        # Create notifier
        notifier = EmailNotifier(email_config)

        # Send alert
        await notifier.send_alert(emergency_event)

        # Wait for async execution
        await asyncio.sleep(0.2)

        # Should send to all recipients
        assert mock_smtp.send_message.call_count == 2

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.smtplib.SMTP")
    async def test_email_message_format(self, mock_smtp_class, email_config, emergency_event):
        """Test Email message formatting."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        notifier = EmailNotifier(email_config)

        # Capture the message
        messages = []

        def capture_message(msg):
            messages.append(msg)

        mock_smtp.send_message.side_effect = capture_message

        await notifier.send_alert(emergency_event)
        await asyncio.sleep(0.2)

        # Check at least one message was sent
        assert len(messages) > 0

        # Verify message structure
        msg = messages[0]
        assert "Subject" in msg
        assert "EMERGENCY" in msg["Subject"]

    @pytest.mark.asyncio
    async def test_email_disabled(self, emergency_event):
        """Test behavior when Email is disabled."""
        config = EmailConfig(
            smtp_host="",
            smtp_port=587,
            smtp_username="",
            smtp_password="",
            from_email="",
            to_emails=[],
            enabled=False,
        )

        notifier = EmailNotifier(config)

        # Should be disabled
        assert not notifier.enabled

        # Should not raise error when sending
        await notifier.send_alert(emergency_event)

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.smtplib.SMTP")
    async def test_email_send_error_handling(self, mock_smtp_class, email_config, emergency_event):
        """Test error handling during Email send."""
        mock_smtp = MagicMock()
        mock_smtp.send_message.side_effect = Exception("SMTP connection failed")
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        notifier = EmailNotifier(email_config)

        # Should not raise exception
        await notifier.send_alert(emergency_event)
        await asyncio.sleep(0.2)


class TestNotificationManager:
    """Tests for unified notification manager."""

    @pytest.fixture
    def telegram_config(self):
        """Create test Telegram configuration."""
        return TelegramConfig(
            bot_token="test_bot_token",
            chat_ids=["123456789"],
            enabled=True,
        )

    @pytest.fixture
    def email_config(self):
        """Create test Email configuration."""
        return EmailConfig(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="bot@example.com",
            to_emails=["admin@example.com"],
            use_tls=True,
            enabled=True,
        )

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.smtplib.SMTP")
    async def test_send_alert_all_channels(
        self,
        mock_smtp_class,
        telegram_config,
        email_config,
        emergency_event,
    ):
        """Test sending alert via all configured channels."""
        # Setup mocks
        mock_bot = AsyncMock()

        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        with patch("bot.risk.notifications.TELEGRAM_AVAILABLE", True):
            with patch("bot.risk.notifications.TelegramBot", return_value=mock_bot):
                # Create manager with both notifiers
                manager = NotificationManager(
                    telegram_config=telegram_config,
                    email_config=email_config,
                )

                # Send alert
                await manager.send_alert(emergency_event)
                await asyncio.sleep(0.2)

                # Should send via both channels
                assert mock_bot.send_message.called
                assert mock_smtp.send_message.called

    @pytest.mark.asyncio
    async def test_no_notifiers_configured(self, emergency_event):
        """Test behavior when no notifiers are configured."""
        manager = NotificationManager()

        # Should not raise error
        await manager.send_alert(emergency_event)

    @pytest.mark.asyncio
    async def test_telegram_only(self, telegram_config, emergency_event):
        """Test with only Telegram configured."""
        mock_bot = AsyncMock()

        with patch("bot.risk.notifications.TELEGRAM_AVAILABLE", True):
            with patch("bot.risk.notifications.TelegramBot", return_value=mock_bot):
                manager = NotificationManager(telegram_config=telegram_config)

                await manager.send_alert(emergency_event)

                assert mock_bot.send_message.called

    @pytest.mark.asyncio
    @patch("bot.risk.notifications.smtplib.SMTP")
    async def test_email_only(self, mock_smtp_class, email_config, emergency_event):
        """Test with only Email configured."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        manager = NotificationManager(email_config=email_config)

        await manager.send_alert(emergency_event)
        await asyncio.sleep(0.2)

        assert mock_smtp.send_message.called
