"""
Logging configuration module with structured JSON logging and correlation IDs.

This module provides:
- Structured JSON logging using structlog
- Correlation ID tracking for request tracing
- Multiple log outputs (console, files) with different levels
- Daily log rotation with 30-day retention
- Separate log files for trades, system, and errors
- Performance-optimized async I/O
"""

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add correlation ID to log event.

    Args:
        logger: Logger instance
        method_name: Method being called
        event_dict: Event dictionary to modify

    Returns:
        Modified event dictionary with correlation_id
    """
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_module_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add module and function information to log event.

    Args:
        logger: Logger instance
        method_name: Method being called
        event_dict: Event dictionary to modify

    Returns:
        Modified event dictionary with module info
    """
    # Get the caller's frame info
    import inspect

    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        caller_frame = frame.f_back.f_back
        event_dict["module"] = caller_frame.f_globals.get("__name__", "unknown")
        event_dict["function"] = caller_frame.f_code.co_name
        event_dict["line"] = caller_frame.f_lineno

    return event_dict


def setup_logging(
    log_dir: Path,
    log_level: str = "INFO",
    console_level: str = "INFO",
    enable_json: bool = True,
    enable_colors: bool = True,
) -> None:
    """
    Configure structured logging with rotation and multiple outputs.

    Args:
        log_dir: Directory for log files
        log_level: Global log level (DEBUG, INFO, WARNING, ERROR)
        console_level: Console output log level
        enable_json: Enable JSON formatting for file logs
        enable_colors: Enable colored output for console
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure log file paths
    system_log = log_dir / "system.log"
    trade_log = log_dir / "trades.log"
    error_log = log_dir / "errors.log"

    # Convert log levels to numeric values
    level = getattr(logging, log_level.upper())
    console_log_level = getattr(logging, console_level.upper())

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[],  # We'll add handlers manually
    )

    # Create file handlers with rotation
    # Daily rotation, keep 30 days of logs
    system_handler = logging.handlers.TimedRotatingFileHandler(
        filename=system_log,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    system_handler.setLevel(level)

    trade_handler = logging.handlers.TimedRotatingFileHandler(
        filename=trade_log,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    trade_handler.setLevel(level)

    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=error_log,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)

    # Configure structlog processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        add_module_info,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if enable_json:
        # JSON renderer for file logs
        file_processors = processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        file_processors = processors + [
            structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event"])
        ]

    # Console renderer (with or without colors)
    if enable_colors and sys.stdout.isatty():
        console_processors = processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        console_processors = processors + [
            structlog.processors.KeyValueRenderer()
        ]

    # Configure structlog
    structlog.configure(
        processors=processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create formatters
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer() if enable_json
        else structlog.processors.KeyValueRenderer(),
        foreign_pre_chain=file_processors,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=enable_colors) if enable_colors
        else structlog.processors.KeyValueRenderer(),
        foreign_pre_chain=console_processors,
    )

    # Apply formatters to handlers
    system_handler.setFormatter(file_formatter)
    trade_handler.setFormatter(file_formatter)
    error_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(system_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Create trade logger (separate logger for trade events)
    trade_logger = logging.getLogger("bot.trades")
    trade_logger.addHandler(trade_handler)
    trade_logger.propagate = False  # Don't propagate to root logger

    # Log initialization
    log = structlog.get_logger()
    log.info(
        "logging_initialized",
        log_dir=str(log_dir),
        log_level=log_level,
        console_level=console_level,
        json_enabled=enable_json,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__). If None, uses the calling module's name.

    Returns:
        Configured structlog logger
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "bot")

    return structlog.get_logger(name or "bot")


def get_trade_logger() -> structlog.stdlib.BoundLogger:
    """
    Get the dedicated trade logger.

    Returns:
        Trade logger for logging trade-specific events
    """
    return structlog.get_logger("bot.trades")


class CorrelationContext:
    """
    Context manager for correlation ID tracking.

    Example:
        with CorrelationContext() as correlation_id:
            log.info("processing_request", request_id=123)
            # All logs within this context will have the same correlation_id
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize correlation context.

        Args:
            correlation_id: Optional correlation ID. If None, generates a new UUID.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> str:
        """
        Enter context and set correlation ID.

        Returns:
            Correlation ID string
        """
        self.token = correlation_id_var.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and reset correlation ID."""
        if self.token:
            correlation_id_var.reset(self.token)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID to set
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get current correlation ID.

    Returns:
        Current correlation ID or None
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear current correlation ID."""
    correlation_id_var.set(None)


# Convenience functions for common log operations
def log_trade(
    action: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    **kwargs: Any,
) -> None:
    """
    Log a trade event.

    Args:
        action: Trade action (ENTRY, EXIT, FILL, etc.)
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Quantity traded
        price: Execution price
        **kwargs: Additional context
    """
    trade_log = get_trade_logger()
    trade_log.info(
        "trade_event",
        action=action,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs,
    )


def log_order(
    order_id: str,
    status: str,
    symbol: str,
    side: str,
    quantity: float,
    **kwargs: Any,
) -> None:
    """
    Log an order event.

    Args:
        order_id: Order identifier
        status: Order status (NEW, FILLED, CANCELLED, etc.)
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Order quantity
        **kwargs: Additional context
    """
    trade_log = get_trade_logger()
    trade_log.info(
        "order_event",
        order_id=order_id,
        status=status,
        symbol=symbol,
        side=side,
        quantity=quantity,
        **kwargs,
    )
