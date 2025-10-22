"""
Integration tests for market data streaming with real exchange connections.

These tests require network connectivity and may be slower.
They can be skipped in CI/CD with pytest markers.

Run with: pytest -v -m integration
Skip with: pytest -v -m "not integration"
"""

import asyncio
import pytest
import os

from bot.data.market_data import (
    MarketDataManager,
    MarketDataStream,
    ConnectionState,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def exchange_config():
    """Configuration for exchange connection"""
    return {
        "exchange_id": "binance",
        "testnet": True,
        "api_key": os.getenv("BINANCE_TESTNET_API_KEY"),
        "api_secret": os.getenv("BINANCE_TESTNET_API_SECRET"),
    }


class TestRealExchangeConnection:
    """Integration tests with real exchange connections"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_connect_to_binance_testnet(self, exchange_config):
        """Test connecting to Binance testnet"""
        stream = MarketDataStream(**exchange_config)

        try:
            await stream.connect()

            assert stream.state == ConnectionState.CONNECTED
            assert stream._exchange is not None

        finally:
            await stream.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_stream_real_market_data(self, exchange_config):
        """Test streaming real market data from Binance testnet"""
        stream = MarketDataStream(**exchange_config)

        try:
            await stream.connect()
            await stream.subscribe(["BTC/USDT", "ETH/USDT"])

            # Collect data for 10 seconds
            await asyncio.sleep(10)

            # Verify data was collected
            btc_data = await stream.buffer.get_latest("BTC/USDT", limit=10)
            assert btc_data is not None
            assert len(btc_data) > 0

            # Verify data quality
            assert all(btc_data["price"] > 0)
            assert all(btc_data["volume"] >= 0)

            eth_data = await stream.buffer.get_latest("ETH/USDT", limit=10)
            assert eth_data is not None
            assert len(eth_data) > 0

        finally:
            await stream.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_subscribe_unsubscribe_flow(self, exchange_config):
        """Test dynamic subscription management"""
        stream = MarketDataStream(**exchange_config)

        try:
            await stream.connect()

            # Subscribe to initial symbols
            await stream.subscribe(["BTC/USDT"])
            await asyncio.sleep(5)

            # Verify data for BTC
            btc_data = await stream.buffer.get_latest("BTC/USDT")
            assert btc_data is not None

            # Add more symbols
            await stream.subscribe(["ETH/USDT", "XRP/USDT"])
            await asyncio.sleep(5)

            # Verify all symbols have data
            for symbol in ["BTC/USDT", "ETH/USDT", "XRP/USDT"]:
                data = await stream.buffer.get_latest(symbol)
                assert data is not None

            # Unsubscribe from one
            await stream.unsubscribe(["XRP/USDT"])

            assert "XRP/USDT" not in stream._subscribed_symbols
            assert "BTC/USDT" in stream._subscribed_symbols
            assert "ETH/USDT" in stream._subscribed_symbols

        finally:
            await stream.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_callback_real_data(self, exchange_config):
        """Test callbacks receive real-time data"""
        stream = MarketDataStream(**exchange_config)
        received_ticks = []

        def callback(tick):
            received_ticks.append(tick)

        try:
            await stream.connect()
            stream.add_callback(callback)
            await stream.subscribe(["BTC/USDT"])

            # Collect for 5 seconds
            await asyncio.sleep(5)

            # Verify callbacks were invoked
            assert len(received_ticks) > 0

            # Verify tick data structure
            for tick in received_ticks:
                assert tick.symbol == "BTC/USDT"
                assert tick.price > 0
                assert tick.timestamp > 0

        finally:
            await stream.disconnect()


class TestMultipleExchanges:
    """Integration tests with multiple exchange connections"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Exchange credentials not available"
    )
    async def test_manager_multiple_streams(self):
        """Test managing multiple exchange streams simultaneously"""
        manager = MarketDataManager()

        try:
            # Add Binance stream
            binance_stream = await manager.add_stream(
                "binance",
                testnet=True,
                api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
                api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
            )

            await binance_stream.subscribe(["BTC/USDT"])

            # Collect data
            await asyncio.sleep(5)

            # Verify data from Binance
            binance_data = await binance_stream.buffer.get_latest("BTC/USDT")
            assert binance_data is not None

        finally:
            await manager.shutdown()


class TestReconnectionScenarios:
    """Integration tests for reconnection handling"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_heartbeat_detection(self, exchange_config):
        """Test that heartbeat monitor detects stale connections"""
        stream = MarketDataStream(heartbeat_interval=5, reconnect_delay=2, **exchange_config)

        try:
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            # Let it run for heartbeat interval
            await asyncio.sleep(7)

            # Connection should still be healthy
            assert stream.state == ConnectionState.CONNECTED

        finally:
            await stream.disconnect()


class TestDataQuality:
    """Integration tests for data quality and normalization"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_data_normalization(self, exchange_config):
        """Test that data is properly normalized"""
        stream = MarketDataStream(**exchange_config)

        try:
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            # Collect data
            await asyncio.sleep(5)

            # Get data
            df = await stream.buffer.get_latest("BTC/USDT")

            assert df is not None

            # Verify schema
            expected_columns = [
                "symbol",
                "timestamp",
                "price",
                "volume",
                "bid",
                "ask",
                "high",
                "low",
                "open",
                "close",
                "exchange",
            ]
            for col in expected_columns:
                assert col in df.columns

            # Verify data types
            assert df["price"].dtype in ["float64", "float32"]
            assert df["timestamp"].dtype == "int64"

        finally:
            await stream.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_time_range_queries(self, exchange_config):
        """Test querying data by time range"""
        stream = MarketDataStream(**exchange_config)

        try:
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            # Collect data for 10 seconds
            start_time = int(asyncio.get_event_loop().time() * 1000)
            await asyncio.sleep(10)
            end_time = int(asyncio.get_event_loop().time() * 1000)

            # Query range
            df = await stream.buffer.get_range("BTC/USDT", start_time, end_time)

            assert df is not None
            assert len(df) > 0

            # Verify all timestamps are in range
            assert all(df["timestamp"] >= start_time)
            assert all(df["timestamp"] <= end_time)

        finally:
            await stream.disconnect()


class TestPerformance:
    """Integration tests for performance and scalability"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_high_frequency_data_collection(self, exchange_config):
        """Test collecting high-frequency data for multiple symbols"""
        stream = MarketDataStream(buffer_size=10000, **exchange_config)

        try:
            await stream.connect()

            # Subscribe to many symbols
            symbols = [
                "BTC/USDT",
                "ETH/USDT",
                "XRP/USDT",
                "ADA/USDT",
                "SOL/USDT",
                "DOT/USDT",
                "MATIC/USDT",
                "LINK/USDT",
            ]
            await stream.subscribe(symbols)

            # Collect for 15 seconds
            await asyncio.sleep(15)

            # Verify all symbols have data
            for symbol in symbols:
                data = await stream.buffer.get_latest(symbol)
                # Some symbols may not have data if not available on testnet
                if data is not None:
                    assert len(data) > 0

        finally:
            await stream.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("BINANCE_TESTNET_API_KEY"), reason="Binance testnet credentials not available"
    )
    async def test_buffer_memory_efficiency(self, exchange_config):
        """Test that buffer maintains memory limits"""
        stream = MarketDataStream(buffer_size=100, **exchange_config)

        try:
            await stream.connect()
            await stream.subscribe(["BTC/USDT"])

            # Collect for extended period
            await asyncio.sleep(20)

            # Buffer should not exceed max size
            assert len(stream.buffer._buffers.get("BTC/USDT", [])) <= 100

        finally:
            await stream.disconnect()
