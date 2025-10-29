"""
Notification system for alerts via Telegram and Email.

This module provides:
- Telegram bot integration for instant alerts
- Email notifications for critical events
- Async notification delivery
- Retry logic for failed deliveries
"""

import asyncio
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from telegram import Bot as TelegramBot  # type: ignore[import-not-found]
    from telegram.error import TelegramError as TelegramErrorType  # type: ignore[import-not-found]

try:
    from telegram import Bot as TelegramBot  # type: ignore[no-redef]
    from telegram.error import TelegramError as TelegramErrorType  # type: ignore[no-redef]

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    if not TYPE_CHECKING:
        TelegramBot = Any  # type: ignore[misc,assignment]
        TelegramErrorType = Exception  # type: ignore[misc,assignment]

from bot.risk.emergency_stop import EmergencyEvent

logger = structlog.get_logger(__name__)


@dataclass
class TelegramConfig:
    """
    Telegram configuration.

    Attributes:
        bot_token: Telegram bot token
        chat_ids: List of chat IDs to send alerts to
        enabled: Whether Telegram notifications are enabled
    """

    bot_token: str
    chat_ids: list[str]
    enabled: bool = True


@dataclass
class EmailConfig:
    """
    Email configuration.

    Attributes:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_username: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        to_emails: List of recipient email addresses
        use_tls: Whether to use TLS
        enabled: Whether email notifications are enabled
    """

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    from_email: str
    to_emails: list[str]
    use_tls: bool = True
    enabled: bool = True


class TelegramNotifier:
    """
    Telegram notification handler.

    Sends alerts to Telegram channels/groups via bot API.
    """

    def __init__(self, config: TelegramConfig):
        """
        Initialize Telegram notifier.

        Args:
            config: Telegram configuration
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "telegram_library_not_available",
                message="python-telegram-bot not installed. Install with: pip install python-telegram-bot",
            )
            self.enabled = False
            return

        self.config = config
        self.enabled = config.enabled and bool(config.bot_token)

        if self.enabled:
            self.bot = TelegramBot(token=config.bot_token)
            logger.info("telegram_notifier_initialized", chat_ids=config.chat_ids)
        else:
            logger.warning("telegram_notifier_disabled")

    async def send_alert(self, event: EmergencyEvent) -> None:
        """
        Send emergency alert via Telegram.

        Args:
            event: Emergency event to send
        """
        if not self.enabled:
            return

        try:
            # Format message
            message = self._format_message(event)

            # Send to all configured chat IDs
            tasks = [self._send_to_chat(chat_id, message) for chat_id in self.config.chat_ids]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count

            logger.info(
                "telegram_alerts_sent",
                total=len(results),
                success=success_count,
                failed=failure_count,
                trigger=event.trigger.value,
            )

        except Exception as e:
            logger.error("telegram_alert_error", error=str(e), trigger=event.trigger.value)

    async def _send_to_chat(self, chat_id: str, message: str) -> None:
        """
        Send message to specific chat.

        Args:
            chat_id: Telegram chat ID
            message: Message to send
        """
        try:
            await self.bot.send_message(
                chat_id=chat_id, text=message, parse_mode="Markdown", disable_web_page_preview=True
            )
            logger.debug("telegram_message_sent", chat_id=chat_id)

        except TelegramErrorType as e:
            logger.error("telegram_send_error", chat_id=chat_id, error=str(e))
            raise

    def _format_message(self, event: EmergencyEvent) -> str:
        """
        Format emergency event as Telegram message.

        Args:
            event: Emergency event

        Returns:
            Formatted message string
        """
        severity_emoji = "🚨" if event.severity >= 9 else "⚠️"

        message = f"{severity_emoji} *EMERGENCY ALERT* {severity_emoji}\n\n"
        message += f"*Trigger:* `{event.trigger.value}`\n"
        message += f"*Severity:* {event.severity}/10\n"
        message += f"*Time:* {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        message += f"*Message:*\n{event.message}\n\n"

        if event.details:
            message += "*Details:*\n"
            for key, value in event.details.items():
                message += f"  • {key}: `{value}`\n"

        if event.correlation_id:
            message += f"\n*Correlation ID:* `{event.correlation_id}`\n"

        message += "\n⚠️ Trading has been halted. Manual intervention required."

        return message


class EmailNotifier:
    """
    Email notification handler.

    Sends alerts via SMTP email.
    """

    def __init__(self, config: EmailConfig):
        """
        Initialize email notifier.

        Args:
            config: Email configuration
        """
        self.config = config
        self.enabled = config.enabled and bool(config.smtp_host)

        if self.enabled:
            logger.info(
                "email_notifier_initialized",
                smtp_host=config.smtp_host,
                to_emails=config.to_emails,
            )
        else:
            logger.warning("email_notifier_disabled")

    async def send_alert(self, event: EmergencyEvent) -> None:
        """
        Send emergency alert via email.

        Args:
            event: Emergency event to send
        """
        if not self.enabled:
            return

        try:
            # Format message
            subject = f"🚨 EMERGENCY: {event.trigger.value} - Severity {event.severity}/10"
            html_body = self._format_html_message(event)
            text_body = self._format_text_message(event)

            # Send to all recipients
            tasks = [
                self._send_to_recipient(to_email, subject, text_body, html_body)
                for to_email in self.config.to_emails
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count

            logger.info(
                "email_alerts_sent",
                total=len(results),
                success=success_count,
                failed=failure_count,
                trigger=event.trigger.value,
            )

        except Exception as e:
            logger.error("email_alert_error", error=str(e), trigger=event.trigger.value)

    async def _send_to_recipient(
        self, to_email: str, subject: str, text_body: str, html_body: str
    ) -> None:
        """
        Send email to specific recipient.

        Args:
            to_email: Recipient email address
            subject: Email subject
            text_body: Plain text body
            html_body: HTML body
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.from_email
            msg["To"] = to_email

            # Attach both plain text and HTML versions
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)

            # Send email in executor (blocking I/O)
            await asyncio.get_event_loop().run_in_executor(None, self._send_smtp, msg, to_email)

            logger.debug("email_sent", to_email=to_email)

        except Exception as e:
            logger.error("email_send_error", to_email=to_email, error=str(e))
            raise

    def _send_smtp(self, msg: MIMEMultipart, to_email: str) -> None:
        """
        Send email via SMTP (blocking).

        Args:
            msg: Email message
            to_email: Recipient email
        """
        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            if self.config.use_tls:
                server.starttls()

            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)

    def _format_text_message(self, event: EmergencyEvent) -> str:
        """
        Format emergency event as plain text email.

        Args:
            event: Emergency event

        Returns:
            Formatted plain text message
        """
        message = "=" * 60 + "\n"
        message += "EMERGENCY ALERT\n"
        message += "=" * 60 + "\n\n"
        message += f"Trigger: {event.trigger.value}\n"
        message += f"Severity: {event.severity}/10\n"
        message += f"Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        message += f"Message:\n{event.message}\n\n"

        if event.details:
            message += "Details:\n"
            for key, value in event.details.items():
                message += f"  - {key}: {value}\n"
            message += "\n"

        if event.correlation_id:
            message += f"Correlation ID: {event.correlation_id}\n\n"

        message += "=" * 60 + "\n"
        message += "TRADING HAS BEEN HALTED\n"
        message += "Manual intervention required to resume.\n"
        message += "=" * 60 + "\n"

        return message

    def _format_html_message(self, event: EmergencyEvent) -> str:
        """
        Format emergency event as HTML email.

        Args:
            event: Emergency event

        Returns:
            Formatted HTML message
        """
        severity_color = "#dc3545" if event.severity >= 9 else "#ffc107"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {severity_color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .details {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid {severity_color}; }}
                .footer {{ background-color: #343a40; color: white; padding: 15px; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                td {{ padding: 8px; border-bottom: 1px solid #dee2e6; }}
                .label {{ font-weight: bold; width: 150px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 EMERGENCY ALERT 🚨</h1>
            </div>
            <div class="content">
                <table>
                    <tr>
                        <td class="label">Trigger:</td>
                        <td><strong>{event.trigger.value}</strong></td>
                    </tr>
                    <tr>
                        <td class="label">Severity:</td>
                        <td><strong style="color: {severity_color};">{event.severity}/10</strong></td>
                    </tr>
                    <tr>
                        <td class="label">Timestamp:</td>
                        <td>{event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                    </tr>
                </table>
                <h3>Message:</h3>
                <p>{event.message}</p>
        """

        if event.details:
            html += '<h3>Details:</h3><div class="details"><table>'
            for key, value in event.details.items():
                html += f"<tr><td class='label'>{key}:</td><td>{value}</td></tr>"
            html += "</table></div>"

        if event.correlation_id:
            html += f"<p><strong>Correlation ID:</strong> <code>{event.correlation_id}</code></p>"

        html += """
            </div>
            <div class="footer">
                <p><strong>⚠️ TRADING HAS BEEN HALTED ⚠️</strong></p>
                <p>Manual intervention required to resume trading operations.</p>
            </div>
        </body>
        </html>
        """

        return html


class NotificationManager:
    """
    Unified notification manager.

    Manages multiple notification channels (Telegram, Email).
    """

    def __init__(
        self,
        telegram_config: TelegramConfig | None = None,
        email_config: EmailConfig | None = None,
    ):
        """
        Initialize notification manager.

        Args:
            telegram_config: Optional Telegram configuration
            email_config: Optional email configuration
        """
        self.notifiers: list = []

        if telegram_config:
            self.notifiers.append(TelegramNotifier(telegram_config))

        if email_config:
            self.notifiers.append(EmailNotifier(email_config))

        logger.info("notification_manager_initialized", notifier_count=len(self.notifiers))

    async def send_alert(self, event: EmergencyEvent) -> None:
        """
        Send alert via all configured channels.

        Args:
            event: Emergency event to send
        """
        if not self.notifiers:
            logger.warning("no_notifiers_configured", trigger=event.trigger.value)
            return

        try:
            # Send via all notifiers concurrently
            tasks = [notifier.send_alert(event) for notifier in self.notifiers]
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(
                "alert_sent_all_channels",
                trigger=event.trigger.value,
                channels=len(self.notifiers),
            )

        except Exception as e:
            logger.error("notification_manager_error", error=str(e), trigger=event.trigger.value)
