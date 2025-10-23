"""
Pydantic configuration models for the trading bot.

This module defines type-safe configuration models using Pydantic for:
- Logging configuration
- Exchange configuration
- Risk management settings
- Strategy parameters
- LLM provider settings
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class LogLevel(str, Enum):
    """Valid log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseModel):
    """
    Logging configuration.

    Attributes:
        log_dir: Directory for log files
        log_level: Global log level
        console_level: Console output log level
        enable_json: Enable JSON formatting for file logs
        enable_colors: Enable colored console output
        rotation_when: When to rotate logs ("midnight", "H", "D", "W")
        rotation_interval: Rotation interval
        backup_count: Number of backup files to keep (30 = 30 days for daily rotation)
    """

    log_dir: Path = Field(default=Path("logs"), description="Directory for log files")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Global log level")
    console_level: LogLevel = Field(default=LogLevel.INFO, description="Console log level")
    enable_json: bool = Field(default=True, description="Enable JSON formatting for files")
    enable_colors: bool = Field(default=True, description="Enable colored console output")
    rotation_when: str = Field(default="midnight", description="When to rotate logs")
    rotation_interval: int = Field(default=1, description="Rotation interval")
    backup_count: int = Field(default=30, description="Number of backup files to keep", ge=0)

    @field_validator("log_dir")
    @classmethod
    def validate_log_dir(cls, v: Path) -> Path:
        """Ensure log directory is absolute."""
        if not v.is_absolute():
            # Make it relative to current working directory
            v = Path.cwd() / v
        return v

    class Config:
        """Pydantic config."""

        use_enum_values = True


class ExchangeConfig(BaseModel):
    """
    Exchange configuration.

    Attributes:
        name: Exchange name (e.g., "binance")
        api_key: API key for exchange
        api_secret: API secret for exchange
        testnet: Whether to use testnet
        timeout: API timeout in seconds
        rate_limit_per_second: Maximum API calls per second
    """

    name: str = Field(default="binance", description="Exchange name")
    api_key: str = Field(description="API key")
    api_secret: str = Field(description="API secret")
    testnet: bool = Field(default=True, description="Use testnet")
    timeout: int = Field(default=30, description="API timeout in seconds", ge=1)
    rate_limit_per_second: int = Field(default=10, description="Max API calls per second", ge=1)

    class Config:
        """Pydantic config."""

        # Don't expose secrets in repr
        json_encoders = {
            str: lambda v: "***" if "secret" in str(v).lower() or "key" in str(v).lower() else v
        }


class RiskConfig(BaseModel):
    """
    Risk management configuration.

    Attributes:
        max_position_size_pct: Maximum position size as % of portfolio
        max_total_exposure_pct: Maximum total exposure as % of portfolio
        max_daily_loss_pct: Maximum daily loss as % of portfolio
        max_slippage_pct: Maximum acceptable slippage %
        min_liquidity_ratio: Minimum position to daily volume ratio
        var_confidence: VaR confidence level (0-1)
        enable_emergency_stop: Enable emergency stop system
    """

    max_position_size_pct: float = Field(
        default=0.03,
        description="Max position size as % of portfolio",
        ge=0.001,
        le=1.0,
    )
    max_total_exposure_pct: float = Field(
        default=0.25,
        description="Max total exposure as % of portfolio",
        ge=0.001,
        le=1.0,
    )
    max_daily_loss_pct: float = Field(
        default=0.02,
        description="Max daily loss as % of portfolio",
        ge=0.001,
        le=0.5,
    )
    max_slippage_pct: float = Field(
        default=0.005, description="Max acceptable slippage %", ge=0.0, le=0.1
    )
    min_liquidity_ratio: float = Field(
        default=0.01,
        description="Min position to daily volume ratio",
        ge=0.001,
        le=1.0,
    )
    var_confidence: float = Field(default=0.95, description="VaR confidence level", ge=0.5, le=0.99)
    enable_emergency_stop: bool = Field(default=True, description="Enable emergency stop system")


class BotConfig(BaseModel):
    """
    Main bot configuration.

    Attributes:
        name: Bot name
        version: Bot version
        environment: Environment (dev, paper, prod)
        logging: Logging configuration
        exchange: Exchange configuration
        risk: Risk management configuration
    """

    name: str = Field(default="Potato Trading Bot", description="Bot name")
    version: str = Field(default="0.1.0", description="Bot version")
    environment: str = Field(default="dev", description="Environment", pattern="^(dev|paper|prod)$")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    exchange: ExchangeConfig | None = None
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        if v not in ["dev", "paper", "prod"]:
            raise ValueError("Environment must be one of: dev, paper, prod")
        return v
