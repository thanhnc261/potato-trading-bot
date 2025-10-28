"""Exchange adapters for different trading platforms."""

from bot.execution.adapters.binance import BinanceAdapter
from bot.execution.adapters.simulated import SimulatedExchange

__all__ = ["BinanceAdapter", "SimulatedExchange"]
