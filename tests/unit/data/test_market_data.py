"""
Comprehensive unit tests for market data streaming module.

Tests cover:
- MarketTick data structure
- MarketDataBuffer with PyArrow tables
- MarketDataStream connection and subscriptions
- WebSocket connection handling
- Reconnection logic with exponential backoff
- Multi-symbol management
- Data normalization
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest

import pyarrow as pa
import pandas as pd

from bot.data.market_data import (
    MarketTick,
    MarketDataBuffer,
    MarketDataStream,
    MarketDataManager,
    ConnectionState,
)


class TestMarketTick:
    """Tests for MarketTick data structure"""

    def test_market_tick_creation(self):
        """Test creating a MarketTick instance"""
        tick = MarketTick(
            symbol="BTC/USDT",
            timestamp=1700000000000,
            price=45000.0,
            volume=100.5,
            bid=44999.0,
            ask=45001.0,
            exchange="binance",
        )

        assert tick.symbol == "BTC/USDT"
        assert tick.timestamp == 1700000000000
        assert tick.price == 45000.0
        assert tick.volume == 100.5
        assert tick.bid == 44999.0
        assert tick.ask == 45001.0
        assert tick.exchange == "binance"

    def test_market_tick_to_dict(self):
        """Test converting MarketTick to dictionary"""
        tick = MarketTick(
            symbol="ETH/USDT",
            timestamp=1700000000000,
            price=3000.0,
            volume=50.0,
        )

        tick_dict = tick.to_dict()

        assert isinstance(tick_dict, dict)
        assert tick_dict["symbol"] == "ETH/USDT"
        assert tick_dict["price"] == 3000.0
        assert tick_dict["volume"] == 50.0
        assert "timestamp" in tick_dict
        assert "exchange" in tick_dict

    def test_market_tick_optional_fields(self):
        """Test MarketTick with optional fields as None"""
        tick = MarketTick(
            symbol="BTC/USDT",
            timestamp=1700000000000,
            price=45000.0,
            volume=100.5,
        )

        assert tick.bid is None
        assert tick.ask is None
        assert tick.high is None
        assert tick.low is None
        assert tick.open is None
        assert tick.close is None


class TestMarketDataBuffer:
    """Tests for MarketDataBuffer with PyArrow"""

    @pytest.mark.asyncio
    async def test_buffer_initialization(self):
        """Test buffer initialization"""
        buffer = MarketDataBuffer(max_size=1000)

        assert buffer.max_size == 1000
        assert len(buffer._buffers) == 0
        assert len(buffer._tables) == 0

    @pytest.mark.asyncio
    async def test_append_single_tick(self):
        """Test appending a single tick to buffer"""
        buffer = MarketDataBuffer()
        tick = MarketTick(
            symbol="BTC/USDT",
            timestamp=1700000000000,
            price=45000.0,
            volume=100.0,
        )

        await buffer.append(tick)

        assert "BTC/USDT" in buffer._buffers
        assert len(buffer._buffers["BTC/USDT"]) == 1

    @pytest.mark.asyncio
    async def test_append_multiple_ticks(self):
        """Test appending multiple ticks"""
        buffer = MarketDataBuffer()

        for i in range(10):
            tick = MarketTick(
                symbol="BTC/USDT",
                timestamp=1700000000000 + i,
                price=45000.0 + i,
                volume=100.0,
            )
            await buffer.append(tick)

        assert len(buffer._buffers["BTC/USDT"]) == 10

    @pytest.mark.asyncio
    async def test_buffer_max_size_enforcement(self):
        """Test that buffer enforces max size with ring buffer behavior"""
        buffer = MarketDataBuffer(max_size=100)

        # Append more ticks than max size
        for i in range(150):
            tick = MarketTick(
                symbol="BTC/USDT",
                timestamp=1700000000000 + i,
                price=45000.0 + i,
                volume=100.0,
            )
            await buffer.append(tick)

        # Should only keep last 100 ticks
        assert len(buffer._buffers["BTC/USDT"]) == 100

        # Verify it's the most recent ticks
        last_tick = buffer._buffers["BTC/USDT"][-1]
        assert last_tick["timestamp"] == 1700000000000 + 149

    @pytest.mark.asyncio
    async def test_get_latest(self):
        """Test getting latest ticks from buffer"""
        buffer = MarketDataBuffer()

        # Add 50 ticks
        for i in range(50):
            tick = MarketTick(
                symbol="BTC/USDT",
                timestamp=1700000000000 + i,
                price=45000.0 + i,
                volume=100.0,
            )
            await buffer.append(tick)

        # Get latest 10
        df = await buffer.get_latest("BTC/USDT", limit=10)

        assert df is not None
        assert len(df) == 10
        assert df.iloc[-1]["timestamp"] == 1700000000000 + 49

    @pytest.mark.asyncio
    async def test_get_latest_nonexistent_symbol(self):
        """Test getting latest for a symbol with no data"""
        buffer = MarketDataBuffer()

        df = await buffer.get_latest("NONEXISTENT/USDT")

        assert df is None

    @pytest.mark.asyncio
    async def test_get_range(self):
        """Test getting ticks within a time range"""
        buffer = MarketDataBuffer()

        # Add 100 ticks with incrementing timestamps
        for i in range(100):
            tick = MarketTick(
                symbol="BTC/USDT",
                timestamp=1700000000000 + i * 1000,
                price=45000.0 + i,
                volume=100.0,
            )
            await buffer.append(tick)

        # Force table rebuild
        buffer._rebuild_table("BTC/USDT")

        # Get range
        start_time = 1700000000000 + 20 * 1000
        end_time = 1700000000000 + 30 * 1000

        df = await buffer.get_range("BTC/USDT", start_time, end_time)

        assert df is not None
        assert len(df) == 11  # Inclusive range: 20 to 30

    @pytest.mark.asyncio
    async def test_clear_single_symbol(self):
        """Test clearing buffer for a single symbol"""
        buffer = MarketDataBuffer()

        # Add data for two symbols
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            tick = MarketTick(
                symbol=symbol,
                timestamp=1700000000000,
                price=45000.0,
                volume=100.0,
            )
            await buffer.append(tick)

        # Clear one symbol
        await buffer.clear("BTC/USDT")

        assert "BTC/USDT" not in buffer._buffers
        assert "ETH/USDT" in buffer._buffers

    @pytest.mark.asyncio
    async def test_clear_all_symbols(self):
        """Test clearing entire buffer"""
        buffer = MarketDataBuffer()

        # Add data for multiple symbols
        for symbol in ["BTC/USDT", "ETH/USDT", "XRP/USDT"]:
            tick = MarketTick(
                symbol=symbol,
                timestamp=1700000000000,
                price=45000.0,
                volume=100.0,
            )
            await buffer.append(tick)

        # Clear all
        await buffer.clear()

        assert len(buffer._buffers) == 0
        assert len(buffer._tables) == 0

    @pytest.mark.asyncio
    async def test_pyarrow_table_creation(self):
        """Test that PyArrow tables are created correctly"""
        buffer = MarketDataBuffer()

        # Add exactly 100 ticks to trigger table rebuild
        for i in range(100):
            tick = MarketTick(
                symbol="BTC/USDT",
                timestamp=1700000000000 + i,
                price=45000.0 + i,
                volume=100.0,
            )
            await buffer.append(tick)

        # Table should be created
        assert "BTC/USDT" in buffer._tables
        table = buffer._tables["BTC/USDT"]

        assert isinstance(table, pa.Table)
        assert len(table) == 100
        assert "symbol" in table.column_names
        assert "price" in table.column_names


class TestMarketDataStream:
    """Tests for MarketDataStream"""

    @pytest.mark.asyncio
    async def test_stream_initialization(self):
        """Test stream initialization"""
        stream = MarketDataStream(
            exchange_id="binance",
            testnet=True,
        )

        assert stream.exchange_id == "binance"
        assert stream.testnet is True
        assert stream.state == ConnectionState.DISCONNECTED
        assert len(stream._subscribed_symbols) == 0

    @pytest.mark.asyncio
    async def test_normalize_symbol(self):
        """Test symbol normalization"""
        stream = MarketDataStream()

        # Test various symbol formats
        assert stream._normalize_symbol("BTCUSDT") == "BTC/USDT"
        assert stream._normalize_symbol("ETHUSDT") == "ETH/USDT"
        assert stream._normalize_symbol("btc/usdt") == "BTC/USDT"
        assert stream._normalize_symbol("BTC/USDT") == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_normalize_tick(self):
        """Test tick data normalization"""
        stream = MarketDataStream(exchange_id="binance")

        raw_data = {
            "timestamp": 1700000000000,
            "last": 45000.0,
            "volume": 100.0,
            "bid": 44999.0,
            "ask": 45001.0,
            "high": 45500.0,
            "low": 44500.0,
        }

        tick = stream._normalize_tick(raw_data, "BTC/USDT")

        assert tick.symbol == "BTC/USDT"
        assert tick.price == 45000.0
        assert tick.volume == 100.0
        assert tick.bid == 44999.0
        assert tick.ask == 45001.0
        assert tick.exchange == "binance"

    @pytest.mark.asyncio
    async def test_add_callback(self):
        """Test adding data callbacks"""
        stream = MarketDataStream()

        callback = Mock()
        stream.add_callback(callback)

        assert len(stream._callbacks) == 1

    @pytest.mark.asyncio
    async def test_callback_invocation(self):
        """Test that callbacks are invoked with tick data"""
        stream = MarketDataStream()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ticker = AsyncMock(
            return_value={
                "timestamp": 1700000000000,
                "last": 45000.0,
                "volume": 100.0,
            }
        )

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()

            # Add callback
            callback = Mock()
            stream.add_callback(callback)

            # Subscribe to symbol
            await stream.subscribe(["BTC/USDT"])

            # Wait a bit for streaming to process
            await asyncio.sleep(2)

            # Callback should have been called
            assert callback.called

            await stream.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_single_symbol(self):
        """Test subscribing to a single symbol"""
        stream = MarketDataStream()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            assert "BTC/USDT" in stream._subscribed_symbols
            await stream.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_multiple_symbols(self):
        """Test subscribing to multiple symbols simultaneously"""
        stream = MarketDataStream()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()
            await stream.subscribe(["BTC/USDT", "ETH/USDT", "XRP/USDT"])

            assert "BTC/USDT" in stream._subscribed_symbols
            assert "ETH/USDT" in stream._subscribed_symbols
            assert "XRP/USDT" in stream._subscribed_symbols
            await stream.disconnect()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from symbols"""
        stream = MarketDataStream()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()
            await stream.subscribe(["BTC/USDT", "ETH/USDT"])

            # Unsubscribe from one
            await stream.unsubscribe(["BTC/USDT"])

            assert "BTC/USDT" not in stream._subscribed_symbols
            assert "ETH/USDT" in stream._subscribed_symbols
            await stream.disconnect()

    @pytest.mark.asyncio
    async def test_connection_state_transitions(self):
        """Test connection state transitions"""
        stream = MarketDataStream()

        assert stream.state == ConnectionState.DISCONNECTED

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            # Connect
            await stream.connect()
            assert stream.state == ConnectionState.CONNECTED

            # Disconnect
            await stream.disconnect()
            assert stream.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnection_exponential_backoff(self):
        """Test reconnection with exponential backoff"""
        stream = MarketDataStream(
            reconnect_delay=1,
            max_reconnect_delay=10,
        )

        # Mock exchange that fails first time
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.close = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()

            initial_delay = stream._current_reconnect_delay

            # Trigger reconnection
            stream.state = ConnectionState.CONNECTED
            await stream._handle_reconnection()

            # Delay should have increased
            assert stream._current_reconnect_delay == initial_delay * 2

            await stream.disconnect()

    @pytest.mark.asyncio
    async def test_buffer_data_storage(self):
        """Test that tick data is stored in buffer"""
        stream = MarketDataStream()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ticker = AsyncMock(
            return_value={
                "timestamp": 1700000000000,
                "last": 45000.0,
                "volume": 100.0,
            }
        )

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            # Wait for some data to be collected
            await asyncio.sleep(2)

            # Check buffer has data
            df = await stream.buffer.get_latest("BTC/USDT")
            assert df is not None
            assert len(df) > 0

            await stream.disconnect()


class TestMarketDataManager:
    """Tests for MarketDataManager"""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization"""
        manager = MarketDataManager()

        assert len(manager._streams) == 0

    @pytest.mark.asyncio
    async def test_add_stream(self):
        """Test adding a market data stream"""
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            stream = await manager.add_stream("binance", testnet=True)

            assert stream is not None
            assert "binance" in manager._streams
            assert stream.state == ConnectionState.CONNECTED

            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_stream(self):
        """Test retrieving a stream by exchange ID"""
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            await manager.add_stream("binance", testnet=True)

            stream = manager.get_stream("binance")
            assert stream is not None
            assert stream.exchange_id == "binance"

            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_remove_stream(self):
        """Test removing a stream"""
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.close = AsyncMock()

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            await manager.add_stream("binance", testnet=True)
            await manager.remove_stream("binance")

            assert "binance" not in manager._streams

    @pytest.mark.asyncio
    async def test_shutdown_all_streams(self):
        """Test shutting down all streams"""
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.close = AsyncMock()

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            # Add multiple streams
            await manager.add_stream("binance", testnet=True)
            await manager.add_stream("coinbase", testnet=True)

            assert len(manager._streams) == 2

            # Shutdown all
            await manager.shutdown()

            assert len(manager._streams) == 0

    @pytest.mark.asyncio
    async def test_multiple_exchange_streams(self):
        """Test managing streams from multiple exchanges"""
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            # Add streams for different exchanges
            binance_stream = await manager.add_stream("binance", testnet=True)
            coinbase_stream = await manager.add_stream("coinbase", testnet=True)

            assert binance_stream.exchange_id == "binance"
            assert coinbase_stream.exchange_id == "coinbase"
            assert len(manager._streams) == 2

            await manager.shutdown()


class TestIntegration:
    """Integration tests for complete workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self):
        """Test complete end-to-end streaming workflow"""
        # Create manager
        manager = MarketDataManager()

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ticker = AsyncMock(
            return_value={
                "timestamp": int(time.time() * 1000),
                "last": 45000.0,
                "volume": 100.0,
                "bid": 44999.0,
                "ask": 45001.0,
            }
        )

        with patch(
            "bot.data.market_data.MarketDataStream._create_exchange", return_value=mock_exchange
        ):
            # Add stream
            stream = await manager.add_stream("binance", testnet=True)

            # Subscribe to symbols
            await stream.subscribe(["BTC/USDT", "ETH/USDT"])

            # Collect data for a short period
            await asyncio.sleep(3)

            # Verify data was collected
            btc_data = await stream.buffer.get_latest("BTC/USDT")
            assert btc_data is not None
            assert len(btc_data) > 0

            eth_data = await stream.buffer.get_latest("ETH/USDT")
            assert eth_data is not None
            assert len(eth_data) > 0

            # Cleanup
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_reconnection_recovery(self):
        """Test that stream recovers from connection loss"""
        stream = MarketDataStream(reconnect_delay=1)

        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.close = AsyncMock()

        with patch.object(stream, "_create_exchange", return_value=mock_exchange):
            # Connect
            await stream.connect()
            assert stream.state == ConnectionState.CONNECTED

            # Subscribe
            await stream.subscribe(["BTC/USDT"])

            # Simulate connection loss
            stream.state = ConnectionState.CONNECTED
            await stream._handle_reconnection()

            # Should reconnect
            assert stream.state == ConnectionState.CONNECTED

            await stream.disconnect()
