"""
Paper trading runner and orchestrator.

This module provides the main paper trading execution engine that:
- Connects to live market data streams
- Executes strategies using simulated exchange
- Tracks virtual portfolio and P/L
- Provides real-time performance metrics
- Simulates realistic order execution without real money

The paper trading system is designed to test strategies in real market
conditions without financial risk.
"""

import asyncio
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from bot.config.models import BotConfig
from bot.core.strategy import BaseStrategy, RSIStrategy, Signal
from bot.data.market_data import MarketDataStream, MarketTick
from bot.execution.adapters.simulated import SimulatedExchange
from bot.execution.orchestrator import ExecutionOrchestrator
from bot.interfaces.exchange import OrderSide
from bot.risk.risk_manager import RiskManager

logger = structlog.get_logger(__name__)


@dataclass
class PaperTradingConfig:
    """
    Configuration for paper trading session.

    Attributes:
        bot_config: Main bot configuration
        symbols: List of trading symbols to monitor
        initial_capital: Initial virtual capital in quote currency
        update_interval: Market data update interval in seconds
        performance_report_interval: Performance report interval in seconds
        max_runtime_seconds: Maximum runtime (None for unlimited)
        enable_risk_management: Enable risk management checks
    """

    bot_config: BotConfig
    symbols: list[str]
    initial_capital: float = 10000.0
    update_interval: float = 1.0
    performance_report_interval: float = 60.0
    max_runtime_seconds: float | None = None
    enable_risk_management: bool = True


class PaperTradingRunner:
    """
    Main paper trading execution engine.

    This orchestrates the entire paper trading session:
    1. Connects to live market data streams
    2. Initializes simulated exchange with virtual capital
    3. Runs strategy signal generation
    4. Executes orders through simulated exchange
    5. Tracks performance metrics
    6. Provides real-time reporting

    The runner handles graceful shutdown and comprehensive error handling.
    """

    def __init__(self, config: PaperTradingConfig):
        """
        Initialize paper trading runner.

        Args:
            config: Paper trading configuration
        """
        self.config = config
        self.bot_config = config.bot_config

        # Initialize simulated exchange
        initial_balances = self._get_initial_balances()
        self.exchange = SimulatedExchange(
            initial_balances=initial_balances,
            commission_rate=0.001,  # 0.1% default
            slippage_factor=0.001,  # 0.1% slippage
            execution_delay_ms=100,
        )

        # Initialize market data stream
        exchange_config = self.bot_config.exchange
        self.market_stream = MarketDataStream(
            exchange_id=exchange_config.name if exchange_config else "binance",
            testnet=exchange_config.testnet if exchange_config else True,
            api_key=exchange_config.api_key if exchange_config else None,
            api_secret=exchange_config.api_secret if exchange_config else None,
        )

        # Initialize risk manager (always enabled for proper portfolio tracking)
        self.risk_manager = RiskManager(
            exchange=self.exchange,
            config=self.bot_config.risk,
            initial_portfolio_value=Decimal(str(config.initial_capital)),
        )

        # Initialize execution orchestrator
        self.orchestrator = ExecutionOrchestrator(
            exchange=self.exchange,
            risk_manager=self.risk_manager,
        )

        # Initialize strategy
        self.strategy = self._create_strategy()

        # Runtime state
        self.running = False
        self.start_time: datetime | None = None
        self.last_performance_report = time.time()
        self.signal_count = 0
        self.order_count = 0

        # Market data buffer for strategy
        self.market_data_buffer: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in config.symbols
        }

        strategy_type = (
            self.bot_config.strategy.type
            if isinstance(self.bot_config.strategy.type, str)
            else self.bot_config.strategy.type.value
        )
        logger.info(
            "paper_trading_runner_initialized",
            symbols=config.symbols,
            initial_capital=config.initial_capital,
            strategy=strategy_type,
            risk_management_enabled=config.enable_risk_management,
        )

    def _get_initial_balances(self) -> dict[str, Decimal]:
        """
        Get initial virtual balances for simulated exchange.

        Returns:
            Dictionary of asset balances
        """
        # Start with quote currency (USDT) only
        balances: dict[str, Decimal] = {"USDT": Decimal(str(self.config.initial_capital))}

        # Add zero balances for base currencies of all symbols
        for symbol in self.config.symbols:
            # Parse symbol to get base asset
            for quote in ["USDT", "BUSD", "USD"]:
                if symbol.endswith(quote):
                    base = symbol[: -len(quote)]
                    balances[base] = Decimal("0")
                    break

        return balances

    def _create_strategy(self) -> BaseStrategy:
        """
        Create strategy instance based on configuration.

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type is not recognized
        """
        strategy_config = self.bot_config.strategy
        strategy_type = (
            strategy_config.type
            if isinstance(strategy_config.type, str)
            else strategy_config.type.value
        )

        if strategy_type == "rsi":
            return RSIStrategy(config=strategy_config.parameters)
        elif strategy_type == "ma_crossover":
            from bot.core.strategy import MovingAverageCrossoverStrategy

            return MovingAverageCrossoverStrategy(config=strategy_config.parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    async def start(self) -> None:
        """
        Start paper trading session.

        This connects to market data, initializes components,
        and begins the trading loop.

        Raises:
            RuntimeError: If already running
        """
        if self.running:
            raise RuntimeError("Paper trading already running")

        self.running = True
        self.start_time = datetime.now()

        logger.info("paper_trading_session_starting", start_time=self.start_time)

        try:
            # Connect to simulated exchange
            await self.exchange.connect()

            # Connect to market data stream
            await self.market_stream.connect()
            await self.market_stream.subscribe(self.config.symbols)

            # Register market data callback
            self.market_stream.add_callback(self._on_market_tick)

            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

            logger.info("paper_trading_session_started", symbols=self.config.symbols)

            # Run main trading loop
            await self._trading_loop()

        except Exception as e:
            logger.error("paper_trading_session_error", error=str(e), exc_info=True)
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """
        Stop paper trading session gracefully.

        This cancels all open orders, generates final report,
        and disconnects from services.
        """
        if not self.running:
            return

        logger.info("paper_trading_session_stopping")

        self.running = False

        # Cancel all open orders
        try:
            await self.orchestrator.cancel_all_orders()
        except Exception as e:
            logger.error("error_cancelling_orders", error=str(e))

        # Generate final performance report
        await self._generate_performance_report()

    async def _cleanup(self) -> None:
        """Cleanup resources and disconnect from services."""
        logger.info("cleaning_up_resources")

        try:
            # Disconnect from market data
            await self.market_stream.disconnect()

            # Disconnect from simulated exchange
            await self.exchange.disconnect()

        except Exception as e:
            logger.error("cleanup_error", error=str(e))

        logger.info("cleanup_complete")

    def _normalize_symbol_to_exchange_format(self, symbol: str) -> str:
        """
        Convert symbol from normalized format (BTC/USDT) to exchange format (BTCUSDT).

        Args:
            symbol: Symbol in normalized format (e.g., 'BTC/USDT')

        Returns:
            Symbol in exchange format (e.g., 'BTCUSDT')
        """
        return symbol.replace("/", "")

    def _on_market_tick(self, tick: MarketTick) -> None:
        """
        Callback for market data updates.

        Updates simulated exchange with current prices and
        buffers data for strategy analysis.

        Args:
            tick: Market tick data
        """
        # Convert symbol to exchange format (BTC/USDT -> BTCUSDT)
        exchange_symbol = self._normalize_symbol_to_exchange_format(tick.symbol)

        # Update exchange with current price (using exchange format)
        self.exchange.update_market_price(exchange_symbol, Decimal(str(tick.price)))

        # Add to market data buffer (using exchange format)
        if exchange_symbol in self.market_data_buffer:
            self.market_data_buffer[exchange_symbol].append(
                {
                    "timestamp": tick.timestamp,
                    "open": tick.open or tick.price,
                    "high": tick.high or tick.price,
                    "low": tick.low or tick.price,
                    "close": tick.close or tick.price,
                    "volume": tick.volume,
                }
            )

            # Limit buffer size to last 1000 ticks
            if len(self.market_data_buffer[exchange_symbol]) > 1000:
                self.market_data_buffer[exchange_symbol] = self.market_data_buffer[exchange_symbol][
                    -1000:
                ]

    async def _trading_loop(self) -> None:
        """
        Main trading loop.

        Continuously:
        1. Checks for strategy signals
        2. Executes orders
        3. Updates portfolio
        4. Reports performance
        """
        logger.info("trading_loop_started")

        while self.running:
            try:
                # Check runtime limit
                if self.config.max_runtime_seconds:
                    if self.start_time:
                        elapsed = (datetime.now() - self.start_time).total_seconds()
                        if elapsed >= self.config.max_runtime_seconds:
                            logger.info(
                                "max_runtime_reached",
                                elapsed_seconds=elapsed,
                                max_seconds=self.config.max_runtime_seconds,
                            )
                            await self.stop()
                            break

                # Process each symbol
                for symbol in self.config.symbols:
                    await self._process_symbol(symbol)

                # Generate periodic performance report
                if (
                    time.time() - self.last_performance_report
                    >= self.config.performance_report_interval
                ):
                    await self._generate_performance_report()
                    self.last_performance_report = time.time()

                # Sleep for update interval
                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                logger.info("trading_loop_cancelled")
                break
            except Exception as e:
                logger.error("trading_loop_error", error=str(e), exc_info=True)
                await asyncio.sleep(5)  # Back off on error

        logger.info("trading_loop_stopped")

    async def _process_symbol(self, symbol: str) -> None:
        """
        Process trading logic for a single symbol.

        Args:
            symbol: Trading symbol to process
        """
        # Check if we have enough data
        if len(self.market_data_buffer[symbol]) < 50:
            # Need at least 50 bars for most indicators
            return

        # Convert buffer to DataFrame for strategy
        import pandas as pd

        df = pd.DataFrame(self.market_data_buffer[symbol])

        # Generate trading signal
        signal = self.strategy.generate_signal(df)

        if signal.signal == Signal.HOLD:
            # No action needed
            return

        self.signal_count += 1

        # Check if we have an open position
        current_position = self.strategy.current_position

        if signal.signal == Signal.BUY and not current_position:
            # Enter long position
            await self._enter_position(symbol, signal)

        elif signal.signal == Signal.SELL and current_position:
            # Exit position
            await self._exit_position(symbol, signal)

    async def _enter_position(self, symbol: str, signal: Any) -> None:
        """
        Enter a new position based on signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal
        """
        try:
            # Calculate position size
            position_size = self.strategy.get_position_size(
                signal,
                float(self.config.initial_capital),
            )

            logger.info(
                "entering_position",
                symbol=symbol,
                side="LONG",
                size=position_size,
                price=signal.price,
                reason=signal.reason,
            )

            # Execute market order via orchestrator
            result = await self.orchestrator.execute_market_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal(str(position_size)),
                metadata={"signal": signal.reason},
            )

            if result.success:
                self.order_count += 1

                # Update strategy position
                self.strategy.enter_position(
                    signal=signal,
                    size=position_size,
                    stop_loss=None,  # Strategy calculates internally
                    take_profit=None,  # Strategy calculates internally
                )

                logger.info(
                    "position_entered_successfully",
                    symbol=symbol,
                    order_id=result.order.id if result.order else None,
                )
            else:
                logger.warning(
                    "failed_to_enter_position",
                    symbol=symbol,
                    error=result.error_message,
                )

        except Exception as e:
            logger.error(
                "error_entering_position",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

    async def _exit_position(self, symbol: str, signal: Any) -> None:
        """
        Exit current position based on signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal
        """
        try:
            position = self.strategy.current_position
            if not position:
                return

            logger.info(
                "exiting_position",
                symbol=symbol,
                side=position.side.value,
                size=position.size,
                entry_price=position.entry_price,
                current_price=signal.price,
                reason=signal.reason,
            )

            # Execute market order to close position
            result = await self.orchestrator.execute_market_order(
                symbol=symbol,
                side=OrderSide.SELL,  # Assuming long positions for now
                quantity=Decimal(str(position.size)),
                metadata={"signal": signal.reason, "exit": True},
            )

            if result.success:
                self.order_count += 1

                # Calculate P/L
                pnl = (signal.price - position.entry_price) * position.size
                pnl_pct = (signal.price - position.entry_price) / position.entry_price * 100

                # Reset strategy position
                self.strategy.reset()

                logger.info(
                    "position_exited_successfully",
                    symbol=symbol,
                    order_id=result.order.id if result.order else None,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            else:
                logger.warning(
                    "failed_to_exit_position",
                    symbol=symbol,
                    error=result.error_message,
                )

        except Exception as e:
            logger.error(
                "error_exiting_position",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

    async def _generate_performance_report(self) -> None:
        """Generate and log performance metrics report."""
        metrics = self.exchange.get_performance_metrics()

        # Calculate session duration
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            elapsed_hours = elapsed / 3600
        else:
            elapsed_hours = 0

        logger.info(
            "performance_report",
            session_duration_hours=elapsed_hours,
            initial_value=metrics["initial_value"],
            current_value=metrics["current_value"],
            pnl=metrics["pnl"],
            pnl_pct=metrics["pnl_pct"],
            total_trades=metrics["total_trades"],
            total_commission=metrics["total_commission"],
            total_slippage=metrics["total_slippage"],
            signals_generated=self.signal_count,
            orders_executed=self.order_count,
            open_orders=metrics["open_orders"],
            balances=metrics["balances"],
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current session metrics.

        Returns:
            Dictionary with comprehensive session metrics
        """
        exchange_metrics = self.exchange.get_performance_metrics()

        session_data = {
            "session": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "duration_seconds": (
                    (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                ),
                "running": self.running,
                "signals_generated": self.signal_count,
                "orders_executed": self.order_count,
            },
            "performance": exchange_metrics,
            "orchestrator": self.orchestrator.get_order_metrics(),
        }

        return session_data

    async def save_session_report(self, output_path: Path) -> None:
        """
        Save comprehensive session report to file.

        Args:
            output_path: Path to save report JSON
        """
        import json

        metrics = self.get_metrics()

        # Get trade history
        trade_list: list[dict[str, Any]] = []
        for symbol in self.config.symbols:
            trades = await self.exchange.get_trades(symbol, limit=1000)
            for trade in trades:
                trade_list.append(
                    {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "price": float(trade.price),
                        "quantity": float(trade.quantity),
                        "commission": float(trade.commission),
                        "timestamp": trade.timestamp.isoformat(),
                    }
                )

        # Add additional metadata
        strategy_type = (
            self.bot_config.strategy.type
            if isinstance(self.bot_config.strategy.type, str)
            else self.bot_config.strategy.type.value
        )
        report: dict[str, Any] = {
            "config": {
                "symbols": self.config.symbols,
                "initial_capital": self.config.initial_capital,
                "strategy": strategy_type,
                "strategy_params": self.bot_config.strategy.parameters,
                "risk_management_enabled": self.config.enable_risk_management,
            },
            "metrics": metrics,
            "trades": trade_list,
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("session_report_saved", output_path=str(output_path))


async def run_paper_trading(config: PaperTradingConfig) -> None:
    """
    Main entry point for paper trading session.

    Args:
        config: Paper trading configuration

    Raises:
        Exception: If paper trading session fails
    """
    runner = PaperTradingRunner(config)

    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        await runner.stop()
    except Exception as e:
        logger.error("paper_trading_failed", error=str(e), exc_info=True)
        raise
    finally:
        # Save final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("paper_trading_results") / f"session_{timestamp}.json"
        await runner.save_session_report(report_path)
        logger.info("paper_trading_session_ended", report_path=str(report_path))
