"""
Phase 2 smoke tests to verify critical subsystems work end-to-end.

These tests ensure that backtesting, strategy context, and paper trading
can run without crashing, even if they don't produce specific results.
"""

from decimal import Decimal

import pandas as pd
import pytest

from bot.core.strategy import Signal
from bot.core.strategy_context import SignalConflictResolution, StrategyContext
from bot.data.backtesting import BacktestConfig, BacktestEngine
from bot.execution.adapters.simulated import SimulatedExchange
from bot.execution.orchestrator import ExecutionOrchestrator
from bot.risk.risk_manager import RiskConfig, RiskManager


@pytest.fixture
def test_data_file(tmp_path):
    """Create a minimal test data file."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": [50000 + i * 10 for i in range(50)],
            "high": [50100 + i * 10 for i in range(50)],
            "low": [49900 + i * 10 for i in range(50)],
            "close": [50000 + i * 10 for i in range(50)],
            "volume": [1000] * 50,
        }
    )
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


class TestBacktestingSmoke:
    """Smoke tests for backtesting engine."""

    def test_backtest_completes_without_crash(self, test_data_file):
        """Test that a basic backtest can run to completion."""
        config = BacktestConfig(
            data_file=test_data_file,
            strategy_name="rsi",
            symbol="BTCUSDT",
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage_factor=0.001,
        )

        engine = BacktestEngine(config)
        results = engine.run()

        # Verify backtest completed
        assert results is not None
        assert results.metrics is not None
        assert "final_capital" in results.metrics or len(results.trades) >= 0

    def test_backtest_multi_symbol_support(self, test_data_file):
        """Test backtest works with different symbols."""
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            config = BacktestConfig(
                data_file=test_data_file,
                strategy_name="rsi",
                symbol=symbol,
                initial_capital=5000.0,
            )

            engine = BacktestEngine(config)
            results = engine.run()

            assert results.metrics is not None, f"Failed for symbol {symbol}"
            assert len(results.trades) >= 0, f"Failed for symbol {symbol}"


class TestSimulatedExchangeSmoke:
    """Smoke tests for simulated exchange."""

    def test_execute_order_with_custom_symbol(self):
        """Test execute_order accepts and uses custom symbols."""
        exchange = SimulatedExchange()

        # Test with ETHUSDT
        exchange.update_market_price("ETHUSDT", Decimal("3000"))
        price, commission, slippage = exchange.execute_order(
            Signal.BUY, 0.1, 0.001, symbol="ETHUSDT"
        )

        assert price > Decimal("2999")  # With slippage
        assert commission > 0
        assert slippage >= 0

    def test_execute_order_converts_float_to_decimal(self):
        """Test execute_order handles float quantity."""
        exchange = SimulatedExchange()
        exchange.update_market_price("BTCUSDT", Decimal("50000"))

        # Pass float quantity (not Decimal)
        price, commission, slippage = exchange.execute_order(
            Signal.BUY, 0.02, 0.001  # float quantity
        )

        assert isinstance(price, Decimal)
        assert commission > 0


class TestStrategyContextSmoke:
    """Smoke tests for strategy context conflict resolution."""

    @pytest.mark.asyncio
    async def test_conflict_resolution_modes(self):
        """Test each conflict resolution mode can be initialized."""
        exchange = SimulatedExchange()
        exchange.update_market_price("BTCUSDT", Decimal("50000"))

        risk_config = RiskConfig(
            max_position_size_pct=0.1,
            max_total_exposure_pct=0.5,
            max_daily_loss_pct=0.05,
        )
        risk_manager = RiskManager(
            exchange=exchange, config=risk_config, initial_portfolio_value=Decimal("10000")
        )
        orchestrator = ExecutionOrchestrator(exchange=exchange, risk_manager=risk_manager)

        # Test each conflict resolution mode
        modes = [
            SignalConflictResolution.FIRST_WINS,
            SignalConflictResolution.HIGHEST_CONFIDENCE,
            SignalConflictResolution.WEIGHTED_AVERAGE,
            SignalConflictResolution.VETO,
            SignalConflictResolution.UNANIMOUS,
        ]

        for mode in modes:
            context = StrategyContext(
                risk_manager=risk_manager,
                orchestrator=orchestrator,
                initial_capital=Decimal("10000"),
                conflict_resolution=mode,
            )
            assert context.conflict_resolution == mode

    @pytest.mark.asyncio
    async def test_resolve_signal_conflicts_no_crash(self):
        """Test resolve_signal_conflicts doesn't crash with multiple signals."""
        from bot.core.strategy import StrategySignal

        exchange = SimulatedExchange()
        exchange.update_market_price("BTCUSDT", Decimal("50000"))

        risk_config = RiskConfig()
        risk_manager = RiskManager(
            exchange=exchange, config=risk_config, initial_portfolio_value=Decimal("10000")
        )
        orchestrator = ExecutionOrchestrator(exchange=exchange, risk_manager=risk_manager)

        context = StrategyContext(
            risk_manager=risk_manager,
            orchestrator=orchestrator,
            initial_capital=Decimal("10000"),
            conflict_resolution=SignalConflictResolution.HIGHEST_CONFIDENCE,
        )

        # Create mock signals
        signals = {
            "strategy1": StrategySignal(
                signal=Signal.BUY,
                timestamp=0,
                price=50000.0,
                confidence=0.8,
                metadata={},
                reason="test",
            ),
            "strategy2": StrategySignal(
                signal=Signal.SELL,
                timestamp=0,
                price=50000.0,
                confidence=0.6,
                metadata={},
                reason="test",
            ),
        }

        # Add mock strategy instances with symbols
        from bot.core.strategy import RSIStrategy
        from bot.core.strategy_context import StrategyInstance

        context._strategies["strategy1"] = StrategyInstance(
            name="strategy1",
            strategy=RSIStrategy(),
            symbol="BTCUSDT",
            allocation_pct=0.5,
            allocated_capital=Decimal("5000"),
        )
        context._strategies["strategy2"] = StrategyInstance(
            name="strategy2",
            strategy=RSIStrategy(),
            symbol="BTCUSDT",
            allocation_pct=0.5,
            allocated_capital=Decimal("5000"),
        )

        # Should not crash
        resolved = context.resolve_signal_conflicts(signals)
        assert isinstance(resolved, dict)
