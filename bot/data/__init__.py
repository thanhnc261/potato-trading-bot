"""Data module for market data handling and backtesting"""

# Import backtesting components (no external dependencies)
from bot.data.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResults,
    ReplayMode,
    SimulatedExchange,
    TradeRecord,
)

# Lazy import for market data (requires ccxt)
try:
    from bot.data.market_data import (
        MarketDataBuffer,
        MarketDataManager,
        MarketDataStream,
        MarketTick,
    )

    __all__ = [
        # Market data
        "MarketTick",
        "MarketDataBuffer",
        "MarketDataStream",
        "MarketDataManager",
        # Backtesting
        "BacktestEngine",
        "BacktestConfig",
        "BacktestResults",
        "TradeRecord",
        "SimulatedExchange",
        "ReplayMode",
    ]
except ImportError:
    # ccxt not installed - only backtesting components available
    __all__ = [
        # Backtesting
        "BacktestEngine",
        "BacktestConfig",
        "BacktestResults",
        "TradeRecord",
        "SimulatedExchange",
        "ReplayMode",
    ]
