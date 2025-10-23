"""Risk management module."""

from bot.risk.risk_manager import RiskManager, RiskCheckResult, TradeValidationResult
from bot.risk.emergency_stop import (
    EmergencyStopManager,
    EmergencyConfig,
    EmergencyTrigger,
    EmergencyState,
    EmergencyEvent,
)
from bot.risk.notifications import (
    TelegramConfig,
    EmailConfig,
    TelegramNotifier,
    EmailNotifier,
    NotificationManager,
)

__all__ = [
    "RiskManager",
    "RiskCheckResult",
    "TradeValidationResult",
    "EmergencyStopManager",
    "EmergencyConfig",
    "EmergencyTrigger",
    "EmergencyState",
    "EmergencyEvent",
    "TelegramConfig",
    "EmailConfig",
    "TelegramNotifier",
    "EmailNotifier",
    "NotificationManager",
]
