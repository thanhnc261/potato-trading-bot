"""
Real-time market data streaming module with WebSocket and REST polling support.

This module provides:
- WebSocket connections for real-time price updates
- Multi-symbol subscription management
- PyArrow-based efficient in-memory buffering
- Connection health monitoring and auto-reconnection
- Data normalization to standard format
- Support for multiple exchanges via CCXT

Architecture:
- Asynchronous design using asyncio for concurrent streams
- Heartbeat monitoring for connection health
- Automatic reconnection with exponential backoff
- Thread-safe data access
"""

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
import json

import ccxt.async_support as ccxt
import pyarrow as pa
import pandas as pd
from structlog import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class MarketTick:
    """Normalized market tick data structure"""

    symbol: str
    timestamp: int  # Unix timestamp in milliseconds
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    exchange: str = "binance"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PyArrow serialization"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "close": self.close,
            "exchange": self.exchange,
        }


class MarketDataBuffer:
    """
    In-memory buffer for market data using PyArrow Tables.

    Features:
    - Efficient columnar storage
    - Zero-copy data access
    - Fast aggregation and filtering
    - Memory-efficient storage
    """

    # PyArrow schema for market tick data
    SCHEMA = pa.schema(
        [
            ("symbol", pa.string()),
            ("timestamp", pa.int64()),
            ("price", pa.float64()),
            ("volume", pa.float64()),
            ("bid", pa.float64()),
            ("ask", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("open", pa.float64()),
            ("close", pa.float64()),
            ("exchange", pa.string()),
        ]
    )

    def __init__(self, max_size: int = 10000):
        """
        Initialize market data buffer.

        Args:
            max_size: Maximum number of ticks to store per symbol
        """
        self.max_size = max_size
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._tables: Dict[str, pa.Table] = {}
        self._lock = asyncio.Lock()

        logger.info("market_data_buffer_initialized", max_size=max_size)

    async def append(self, tick: MarketTick) -> None:
        """
        Append a market tick to the buffer.

        Args:
            tick: Market tick data to append
        """
        async with self._lock:
            symbol = tick.symbol

            # Initialize buffer for new symbol
            if symbol not in self._buffers:
                self._buffers[symbol] = []

            # Append tick data
            self._buffers[symbol].append(tick.to_dict())

            # Enforce max size with ring buffer behavior
            if len(self._buffers[symbol]) > self.max_size:
                self._buffers[symbol] = self._buffers[symbol][-self.max_size :]

            # Rebuild PyArrow table every 100 ticks for efficiency
            if len(self._buffers[symbol]) % 100 == 0:
                self._rebuild_table(symbol)

    def _rebuild_table(self, symbol: str) -> None:
        """
        Rebuild PyArrow table from buffer data.

        Args:
            symbol: Trading pair symbol
        """
        if symbol in self._buffers and self._buffers[symbol]:
            self._tables[symbol] = pa.Table.from_pylist(self._buffers[symbol], schema=self.SCHEMA)

    async def get_latest(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get latest ticks for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Maximum number of ticks to return

        Returns:
            DataFrame with latest ticks or None if no data
        """
        async with self._lock:
            if symbol not in self._buffers or not self._buffers[symbol]:
                return None

            # Rebuild table if needed
            if symbol not in self._tables or len(self._buffers[symbol]) % 100 != 0:
                self._rebuild_table(symbol)

            # Return as DataFrame
            table = self._tables[symbol]
            df = table.to_pandas()
            return df.tail(limit)

    async def get_range(
        self, symbol: str, start_time: int, end_time: int
    ) -> Optional[pd.DataFrame]:
        """
        Get ticks within a time range.

        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)

        Returns:
            DataFrame with ticks in range or None if no data
        """
        async with self._lock:
            if symbol not in self._tables:
                return None

            table = self._tables[symbol]

            # Filter by timestamp
            mask = (table["timestamp"] >= start_time) & (table["timestamp"] <= end_time)
            filtered = table.filter(mask)

            return filtered.to_pandas() if len(filtered) > 0 else None

    async def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear buffer data.

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        async with self._lock:
            if symbol:
                self._buffers.pop(symbol, None)
                self._tables.pop(symbol, None)
                logger.info("buffer_cleared", symbol=symbol)
            else:
                self._buffers.clear()
                self._tables.clear()
                logger.info("all_buffers_cleared")


class MarketDataStream:
    """
    Real-time market data streaming client with WebSocket support.

    Features:
    - Asynchronous WebSocket connections
    - Multi-symbol subscription management
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring
    - Data normalization
    - In-memory buffering with PyArrow
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        buffer_size: int = 10000,
        heartbeat_interval: int = 30,
        reconnect_delay: int = 5,
        max_reconnect_delay: int = 300,
    ):
        """
        Initialize market data stream.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            testnet: Use testnet if available
            api_key: API key for private endpoints
            api_secret: API secret for private endpoints
            buffer_size: Maximum buffer size per symbol
            heartbeat_interval: Heartbeat check interval in seconds
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
        """
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.buffer = MarketDataBuffer(max_size=buffer_size)

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self._subscribed_symbols: Set[str] = set()
        self._callbacks: List[Callable[[MarketTick], None]] = []

        # Reconnection settings
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self._current_reconnect_delay = reconnect_delay

        # Exchange instance
        self._exchange: Optional[ccxt.Exchange] = None
        self._api_key = api_key
        self._api_secret = api_secret

        # Background tasks
        self._stream_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_message_time = 0

        # Shutdown flag
        self._shutdown = False

        logger.info(
            "market_data_stream_initialized",
            exchange=exchange_id,
            testnet=testnet,
            buffer_size=buffer_size,
        )

    def _create_exchange(self) -> ccxt.Exchange:
        """Create exchange instance with configuration"""
        exchange_class = getattr(ccxt, self.exchange_id)

        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        if self._api_key and self._api_secret:
            config["apiKey"] = self._api_key
            config["secret"] = self._api_secret

        # Enable testnet if supported
        if self.testnet and hasattr(exchange_class, "set_sandbox_mode"):
            config["options"]["sandboxMode"] = True

        return exchange_class(config)

    async def connect(self) -> None:
        """
        Establish connection to the exchange.

        Raises:
            ConnectionError: If connection fails
        """
        if self.state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("already_connected_or_connecting", state=self.state.value)
            return

        try:
            self.state = ConnectionState.CONNECTING
            logger.info("connecting_to_exchange", exchange=self.exchange_id)

            # Create exchange instance
            self._exchange = self._create_exchange()

            # Load markets
            await self._exchange.load_markets()

            self.state = ConnectionState.CONNECTED
            self._last_message_time = time.time()
            self._current_reconnect_delay = self.reconnect_delay

            # Start heartbeat monitor
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            logger.info("connected_to_exchange", exchange=self.exchange_id)

        except Exception as e:
            self.state = ConnectionState.FAILED
            logger.error("connection_failed", exchange=self.exchange_id, error=str(e))
            raise ConnectionError(f"Failed to connect to {self.exchange_id}: {e}")

    async def disconnect(self) -> None:
        """Close connection and cleanup resources"""
        logger.info("disconnecting", exchange=self.exchange_id)

        self._shutdown = True

        # Cancel background tasks
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close exchange connection
        if self._exchange:
            await self._exchange.close()

        self.state = ConnectionState.DISCONNECTED
        logger.info("disconnected", exchange=self.exchange_id)

    async def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to market data for symbols.

        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])

        Raises:
            ConnectionError: If not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to exchange")

        # Normalize symbols
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]

        # Add to subscribed symbols
        new_symbols = set(normalized_symbols) - self._subscribed_symbols
        if new_symbols:
            self._subscribed_symbols.update(new_symbols)
            logger.info("subscribed_to_symbols", symbols=list(new_symbols))

            # Start streaming if not already running
            if not self._stream_task or self._stream_task.done():
                self._stream_task = asyncio.create_task(self._stream_data())

    async def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from market data for symbols.

        Args:
            symbols: List of trading pair symbols to unsubscribe
        """
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]
        self._subscribed_symbols -= set(normalized_symbols)
        logger.info("unsubscribed_from_symbols", symbols=normalized_symbols)

    def add_callback(self, callback: Callable[[MarketTick], None]) -> None:
        """
        Add callback for real-time tick data.

        Args:
            callback: Function to call with each market tick
        """
        self._callbacks.append(callback)
        logger.debug("callback_added", total_callbacks=len(self._callbacks))

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format.

        Args:
            symbol: Symbol in any format

        Returns:
            Normalized symbol (e.g., 'BTC/USDT')
        """
        # Remove spaces and convert to uppercase
        symbol = symbol.replace(" ", "").upper()

        # Ensure slash format
        if "/" not in symbol:
            # Try to split common pairs
            for quote in ["USDT", "BUSD", "USD", "BTC", "ETH"]:
                if symbol.endswith(quote):
                    base = symbol[: -len(quote)]
                    return f"{base}/{quote}"

        return symbol

    def _normalize_tick(self, raw_data: Dict[str, Any], symbol: str) -> MarketTick:
        """
        Normalize raw tick data to standard format.

        Args:
            raw_data: Raw tick data from exchange
            symbol: Trading pair symbol

        Returns:
            Normalized MarketTick object
        """
        # Extract timestamp
        timestamp = raw_data.get("timestamp", int(time.time() * 1000))

        # Extract price data
        last_price = raw_data.get("last") or raw_data.get("close") or 0.0

        return MarketTick(
            symbol=symbol,
            timestamp=timestamp,
            price=last_price,
            volume=raw_data.get("volume", 0.0) or raw_data.get("baseVolume", 0.0),
            bid=raw_data.get("bid"),
            ask=raw_data.get("ask"),
            high=raw_data.get("high"),
            low=raw_data.get("low"),
            open=raw_data.get("open"),
            close=raw_data.get("close"),
            exchange=self.exchange_id,
        )

    async def _stream_data(self) -> None:
        """
        Main streaming loop for market data.

        Continuously fetches ticker data for subscribed symbols.
        """
        logger.info("streaming_started", symbols=list(self._subscribed_symbols))

        while not self._shutdown and self._subscribed_symbols:
            try:
                if self.state != ConnectionState.CONNECTED:
                    await asyncio.sleep(1)
                    continue

                # Fetch tickers for all subscribed symbols
                symbols_list = list(self._subscribed_symbols)

                # Use watch_ticker if available (WebSocket), otherwise fetch_ticker (REST)
                for symbol in symbols_list:
                    try:
                        # Check if exchange supports WebSocket
                        if hasattr(self._exchange, "watch_ticker"):
                            ticker = await self._exchange.watch_ticker(symbol)
                        else:
                            ticker = await self._exchange.fetch_ticker(symbol)

                        # Normalize and process tick
                        tick = self._normalize_tick(ticker, symbol)

                        # Update buffer
                        await self.buffer.append(tick)

                        # Call registered callbacks
                        for callback in self._callbacks:
                            try:
                                callback(tick)
                            except Exception as e:
                                logger.error(
                                    "callback_error",
                                    error=str(e),
                                    symbol=symbol,
                                )

                        self._last_message_time = time.time()

                    except Exception as e:
                        logger.error(
                            "ticker_fetch_error",
                            symbol=symbol,
                            error=str(e),
                        )

                # Small delay between iterations for REST polling
                if not hasattr(self._exchange, "watch_ticker"):
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("streaming_cancelled")
                break
            except Exception as e:
                logger.error("streaming_error", error=str(e))
                await self._handle_reconnection()

    async def _heartbeat_monitor(self) -> None:
        """
        Monitor connection health and trigger reconnection if needed.

        Checks for stale connections and initiates reconnection.
        """
        logger.info("heartbeat_monitor_started", interval=self.heartbeat_interval)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Check if we've received data recently
                time_since_last_message = time.time() - self._last_message_time

                if time_since_last_message > self.heartbeat_interval * 3:
                    logger.warning(
                        "connection_stale",
                        seconds_since_last_message=time_since_last_message,
                    )
                    await self._handle_reconnection()

            except asyncio.CancelledError:
                logger.info("heartbeat_monitor_cancelled")
                break
            except Exception as e:
                logger.error("heartbeat_monitor_error", error=str(e))

    async def _handle_reconnection(self) -> None:
        """
        Handle connection loss and reconnection with exponential backoff.
        """
        if self.state == ConnectionState.RECONNECTING:
            return

        self.state = ConnectionState.RECONNECTING
        logger.warning(
            "reconnecting",
            delay=self._current_reconnect_delay,
        )

        # Close existing connection
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.error("error_closing_connection", error=str(e))

        # Wait before reconnecting
        await asyncio.sleep(self._current_reconnect_delay)

        # Exponential backoff
        self._current_reconnect_delay = min(
            self._current_reconnect_delay * 2,
            self.max_reconnect_delay,
        )

        # Attempt to reconnect
        try:
            await self.connect()

            # Re-subscribe to symbols
            if self._subscribed_symbols:
                await self.subscribe(list(self._subscribed_symbols))

        except Exception as e:
            logger.error("reconnection_failed", error=str(e))
            self.state = ConnectionState.FAILED


class MarketDataManager:
    """
    High-level manager for market data streams across multiple exchanges.

    Features:
    - Manage multiple exchange connections
    - Unified interface for multi-exchange data
    - Aggregated data access
    """

    def __init__(self):
        """Initialize market data manager"""
        self._streams: Dict[str, MarketDataStream] = {}
        logger.info("market_data_manager_initialized")

    async def add_stream(
        self,
        exchange_id: str,
        testnet: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> MarketDataStream:
        """
        Add a new market data stream.

        Args:
            exchange_id: Exchange identifier
            testnet: Use testnet if available
            api_key: API key
            api_secret: API secret

        Returns:
            MarketDataStream instance
        """
        if exchange_id in self._streams:
            logger.warning("stream_already_exists", exchange=exchange_id)
            return self._streams[exchange_id]

        stream = MarketDataStream(
            exchange_id=exchange_id,
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )

        await stream.connect()
        self._streams[exchange_id] = stream

        logger.info("stream_added", exchange=exchange_id)
        return stream

    async def remove_stream(self, exchange_id: str) -> None:
        """
        Remove and disconnect a market data stream.

        Args:
            exchange_id: Exchange identifier
        """
        if exchange_id in self._streams:
            await self._streams[exchange_id].disconnect()
            del self._streams[exchange_id]
            logger.info("stream_removed", exchange=exchange_id)

    def get_stream(self, exchange_id: str) -> Optional[MarketDataStream]:
        """
        Get a market data stream by exchange ID.

        Args:
            exchange_id: Exchange identifier

        Returns:
            MarketDataStream instance or None
        """
        return self._streams.get(exchange_id)

    async def shutdown(self) -> None:
        """Shutdown all streams"""
        logger.info("shutting_down_all_streams", count=len(self._streams))

        for stream in self._streams.values():
            await stream.disconnect()

        self._streams.clear()
        logger.info("all_streams_shutdown")
