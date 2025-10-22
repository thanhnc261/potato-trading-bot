"""
Unit tests for logging configuration module.

Tests:
- Logging setup and configuration
- Correlation ID tracking
- Multiple log outputs (console, files)
- Log rotation
- Structured logging format
- Trade logging
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest
import structlog

from bot.core.logging_config import (
    CorrelationContext,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    get_trade_logger,
    log_order,
    log_trade,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    yield
    # Clear all handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    # Clear structlog configuration
    structlog.reset_defaults()
    clear_correlation_id()


class TestLoggingSetup:
    """Test logging setup and configuration."""

    def test_setup_logging_creates_log_files(self, temp_log_dir):
        """Test that setup_logging creates expected log files."""
        setup_logging(temp_log_dir, log_level="DEBUG")

        # Check that log files are created
        assert (temp_log_dir / "system.log").exists()
        assert (temp_log_dir / "trades.log").exists()
        assert (temp_log_dir / "errors.log").exists()

    def test_setup_logging_with_different_levels(self, temp_log_dir):
        """Test logging with different log levels."""
        setup_logging(temp_log_dir, log_level="WARNING", console_level="ERROR")

        log = get_logger()

        # Write to system log
        log.debug("debug_message")  # Should not appear
        log.info("info_message")  # Should not appear
        log.warning("warning_message")  # Should appear
        log.error("error_message")  # Should appear

        # Read system log
        with open(temp_log_dir / "system.log") as f:
            content = f.read()

        assert "warning_message" in content
        assert "error_message" in content
        assert "debug_message" not in content
        assert "info_message" not in content

    def test_json_format_enabled(self, temp_log_dir):
        """Test JSON formatting for log files."""
        setup_logging(temp_log_dir, enable_json=True)

        log = get_logger()
        log.info("test_event", key1="value1", key2=42)

        # Read and parse JSON log
        with open(temp_log_dir / "system.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert data["event"] == "test_event"
        assert data["key1"] == "value1"
        assert data["key2"] == 42
        assert "timestamp" in data
        assert "level" in data

    def test_error_logs_to_error_file(self, temp_log_dir):
        """Test that ERROR and above logs go to error.log."""
        setup_logging(temp_log_dir)

        log = get_logger()
        log.info("info_message")
        log.error("error_message")
        log.critical("critical_message")

        # Read error log
        with open(temp_log_dir / "errors.log") as f:
            content = f.read()

        assert "error_message" in content
        assert "critical_message" in content
        assert "info_message" not in content


class TestCorrelationID:
    """Test correlation ID tracking."""

    def test_correlation_context_sets_id(self, temp_log_dir):
        """Test that CorrelationContext sets correlation ID."""
        setup_logging(temp_log_dir, enable_json=True)
        log = get_logger()

        with CorrelationContext() as correlation_id:
            assert get_correlation_id() == correlation_id
            log.info("test_event")

        # Correlation ID should be cleared after context
        assert get_correlation_id() is None

        # Read and verify correlation ID in log
        with open(temp_log_dir / "system.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert data["correlation_id"] == correlation_id

    def test_custom_correlation_id(self, temp_log_dir):
        """Test setting custom correlation ID."""
        setup_logging(temp_log_dir, enable_json=True)
        log = get_logger()

        custom_id = "custom-correlation-123"
        with CorrelationContext(custom_id):
            assert get_correlation_id() == custom_id
            log.info("test_event")

        with open(temp_log_dir / "system.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert data["correlation_id"] == custom_id

    def test_set_and_clear_correlation_id(self):
        """Test manual set/clear of correlation ID."""
        test_id = "test-123"

        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

        clear_correlation_id()
        assert get_correlation_id() is None

    def test_nested_correlation_contexts(self, temp_log_dir):
        """Test nested correlation contexts."""
        setup_logging(temp_log_dir, enable_json=True)
        log = get_logger()

        with CorrelationContext("outer") as outer_id:
            log.info("outer_event")

            with CorrelationContext("inner") as inner_id:
                log.info("inner_event")
                assert get_correlation_id() == inner_id

            # Should restore outer correlation ID
            assert get_correlation_id() == outer_id
            log.info("outer_event_2")

        # Read log lines
        with open(temp_log_dir / "system.log") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        assert lines[0]["correlation_id"] == outer_id
        assert lines[1]["correlation_id"] == inner_id
        assert lines[2]["correlation_id"] == outer_id


class TestTradeLogging:
    """Test trade-specific logging."""

    def test_trade_logger_separate_file(self, temp_log_dir):
        """Test that trade logs go to separate file."""
        setup_logging(temp_log_dir)

        log_trade(
            action="ENTRY",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            price=50000.0,
            order_id="test-order-123",
        )

        # Trade log should exist
        assert (temp_log_dir / "trades.log").exists()

        # Read trade log
        with open(temp_log_dir / "trades.log") as f:
            content = f.read()

        assert "BTCUSDT" in content
        assert "BUY" in content
        assert "ENTRY" in content

        # Trade event should NOT propagate to system log
        with open(temp_log_dir / "system.log") as f:
            system_content = f.read()

        # System log should have initialization but not trade events
        assert "logging_initialized" in system_content

    def test_log_order_function(self, temp_log_dir):
        """Test log_order convenience function."""
        setup_logging(temp_log_dir, enable_json=True)

        log_order(
            order_id="order-456",
            status="FILLED",
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.5,
            price=3000.0,
        )

        with open(temp_log_dir / "trades.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert data["event"] == "order_event"
        assert data["order_id"] == "order-456"
        assert data["status"] == "FILLED"
        assert data["symbol"] == "ETHUSDT"

    def test_trade_logging_with_correlation_id(self, temp_log_dir):
        """Test that trade logging includes correlation ID."""
        setup_logging(temp_log_dir, enable_json=True)

        with CorrelationContext() as correlation_id:
            log_trade(
                action="EXIT",
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.01,
                price=51000.0,
            )

        with open(temp_log_dir / "trades.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert data["correlation_id"] == correlation_id


class TestLoggerRetrieval:
    """Test logger retrieval functions."""

    def test_get_logger_returns_structlog_logger(self, temp_log_dir):
        """Test that get_logger returns a structlog logger."""
        setup_logging(temp_log_dir)

        log = get_logger("test.module")
        assert isinstance(log, structlog.stdlib.BoundLogger)

    def test_get_logger_auto_name(self, temp_log_dir):
        """Test that get_logger auto-detects module name."""
        setup_logging(temp_log_dir, enable_json=True)

        log = get_logger()  # Should use calling module's __name__
        log.info("test_event")

        with open(temp_log_dir / "system.log") as f:
            line = f.readline().strip()

        data = json.loads(line)
        assert "module" in data

    def test_get_trade_logger(self, temp_log_dir):
        """Test get_trade_logger returns dedicated trade logger."""
        setup_logging(temp_log_dir)

        trade_log = get_trade_logger()
        assert isinstance(trade_log, structlog.stdlib.BoundLogger)


class TestModuleInfo:
    """Test module information in logs."""

    def test_module_info_included(self, temp_log_dir):
        """Test that module, function, and line info are included."""
        setup_logging(temp_log_dir, enable_json=True)

        log = get_logger()
        log.info("test_event")

        with open(temp_log_dir / "system.log") as f:
            # Skip initialization log
            f.readline()
            line = f.readline().strip()

        data = json.loads(line)
        assert "module" in data
        assert "function" in data
        assert "line" in data
        assert data["function"] == "test_module_info_included"


class TestLogRotation:
    """Test log file rotation configuration."""

    def test_log_handlers_configured(self, temp_log_dir):
        """Test that log handlers are properly configured."""
        setup_logging(temp_log_dir, log_level="INFO")

        root_logger = logging.getLogger()

        # Should have system, error, and console handlers
        assert len(root_logger.handlers) >= 3

        # Check for TimedRotatingFileHandler
        from logging.handlers import TimedRotatingFileHandler

        file_handlers = [h for h in root_logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) >= 2  # system and error handlers
