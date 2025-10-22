"""
Example usage of the Market Data Streaming module.

This script demonstrates:
1. Setting up a market data stream
2. Subscribing to multiple symbols
3. Using callbacks for real-time data
4. Querying historical data from buffer
5. Managing multiple exchange connections
"""

import asyncio
import os
from datetime import datetime

from bot.data.market_data import (
    MarketDataManager,
    MarketDataStream,
    MarketTick,
)


async def basic_streaming_example():
    """Basic example: Stream market data for a single symbol"""
    print("=== Basic Streaming Example ===\n")

    # Create a market data stream
    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
        api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
        api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
    )

    try:
        # Connect to exchange
        print("Connecting to Binance testnet...")
        await stream.connect()
        print("Connected!\n")

        # Subscribe to BTC/USDT
        print("Subscribing to BTC/USDT...")
        await stream.subscribe(["BTC/USDT"])
        print("Subscribed!\n")

        # Collect data for 10 seconds
        print("Collecting data for 10 seconds...")
        await asyncio.sleep(10)

        # Get latest data
        print("\nFetching latest data...")
        df = await stream.buffer.get_latest("BTC/USDT", limit=5)

        if df is not None:
            print(f"\nLatest 5 ticks for BTC/USDT:")
            print(df[["timestamp", "price", "volume", "bid", "ask"]])
        else:
            print("No data collected yet")

    finally:
        # Clean up
        print("\nDisconnecting...")
        await stream.disconnect()
        print("Done!")


async def multi_symbol_example():
    """Example: Stream data for multiple symbols simultaneously"""
    print("\n=== Multi-Symbol Streaming Example ===\n")

    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
    )

    try:
        await stream.connect()

        # Subscribe to multiple symbols
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        print(f"Subscribing to: {', '.join(symbols)}")
        await stream.subscribe(symbols)

        # Collect data
        print("Collecting data for 15 seconds...\n")
        await asyncio.sleep(15)

        # Display data for each symbol
        for symbol in symbols:
            df = await stream.buffer.get_latest(symbol, limit=3)
            if df is not None:
                print(f"\n{symbol}:")
                print(f"  Latest price: ${df.iloc[-1]['price']:,.2f}")
                print(f"  Latest volume: {df.iloc[-1]['volume']:.4f}")
                print(f"  Data points collected: {len(df)}")

    finally:
        await stream.disconnect()


async def callback_example():
    """Example: Use callbacks for real-time price alerts"""
    print("\n=== Callback Example (Price Alerts) ===\n")

    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
    )

    # Track price changes
    price_history = {}

    def price_alert_callback(tick: MarketTick):
        """Alert when price changes significantly"""
        symbol = tick.symbol

        if symbol not in price_history:
            price_history[symbol] = tick.price
            print(f"[{symbol}] Initial price: ${tick.price:,.2f}")
            return

        old_price = price_history[symbol]
        price_change = ((tick.price - old_price) / old_price) * 100

        # Alert on 0.1% change
        if abs(price_change) >= 0.1:
            direction = "↑" if price_change > 0 else "↓"
            print(
                f"[{symbol}] {direction} ${old_price:,.2f} → ${tick.price:,.2f} "
                f"({price_change:+.2f}%)"
            )
            price_history[symbol] = tick.price

    try:
        await stream.connect()

        # Add callback
        stream.add_callback(price_alert_callback)

        # Subscribe and monitor
        await stream.subscribe(["BTC/USDT", "ETH/USDT"])
        print("Monitoring price changes for 20 seconds...\n")
        await asyncio.sleep(20)

    finally:
        await stream.disconnect()


async def time_range_query_example():
    """Example: Query data by time range"""
    print("\n=== Time Range Query Example ===\n")

    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
    )

    try:
        await stream.connect()
        await stream.subscribe(["BTC/USDT"])

        # Record start time
        start_time = int(asyncio.get_event_loop().time() * 1000)
        print(f"Start time: {datetime.fromtimestamp(start_time / 1000)}")

        # Collect data
        print("Collecting data for 10 seconds...\n")
        await asyncio.sleep(10)

        # Record end time
        end_time = int(asyncio.get_event_loop().time() * 1000)
        print(f"End time: {datetime.fromtimestamp(end_time / 1000)}")

        # Query range
        df = await stream.buffer.get_range("BTC/USDT", start_time, end_time)

        if df is not None:
            print(f"\nData points in range: {len(df)}")
            print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
            print(f"Average price: ${df['price'].mean():,.2f}")
            print(f"Total volume: {df['volume'].sum():.4f}")
        else:
            print("No data in range")

    finally:
        await stream.disconnect()


async def multi_exchange_example():
    """Example: Manage multiple exchange streams"""
    print("\n=== Multi-Exchange Example ===\n")

    # Create manager
    manager = MarketDataManager()

    try:
        # Add Binance stream
        print("Adding Binance stream...")
        binance_stream = await manager.add_stream(
            "binance",
            testnet=True,
            api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
            api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
        )
        await binance_stream.subscribe(["BTC/USDT", "ETH/USDT"])

        print("Collecting data from Binance for 10 seconds...\n")
        await asyncio.sleep(10)

        # Compare data from different exchanges
        binance_btc = await binance_stream.buffer.get_latest("BTC/USDT", limit=1)

        if binance_btc is not None:
            print(f"Binance BTC/USDT: ${binance_btc.iloc[-1]['price']:,.2f}")

    finally:
        print("\nShutting down all streams...")
        await manager.shutdown()
        print("Done!")


async def advanced_usage_example():
    """Example: Advanced usage with dynamic subscriptions"""
    print("\n=== Advanced Usage Example ===\n")

    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
        buffer_size=5000,  # Larger buffer
        heartbeat_interval=30,  # Check connection every 30s
    )

    try:
        await stream.connect()

        # Start with one symbol
        print("Phase 1: Subscribing to BTC/USDT")
        await stream.subscribe(["BTC/USDT"])
        await asyncio.sleep(5)

        # Add more symbols dynamically
        print("Phase 2: Adding ETH/USDT and XRP/USDT")
        await stream.subscribe(["ETH/USDT", "XRP/USDT"])
        await asyncio.sleep(5)

        # Remove a symbol
        print("Phase 3: Removing XRP/USDT")
        await stream.unsubscribe(["XRP/USDT"])
        await asyncio.sleep(5)

        # Display final state
        print(f"\nActive subscriptions: {stream._subscribed_symbols}")

        for symbol in stream._subscribed_symbols:
            df = await stream.buffer.get_latest(symbol, limit=1)
            if df is not None:
                print(f"  {symbol}: ${df.iloc[-1]['price']:,.2f}")

    finally:
        await stream.disconnect()


async def main():
    """Run all examples"""
    print("Market Data Streaming Examples")
    print("=" * 50)

    # Check for API keys
    if not os.getenv("BINANCE_TESTNET_API_KEY"):
        print("\nWARNING: BINANCE_TESTNET_API_KEY not set")
        print("Some examples may not work without API credentials\n")

    try:
        # Run examples sequentially
        await basic_streaming_example()
        await multi_symbol_example()
        await callback_example()
        await time_range_query_example()
        await multi_exchange_example()
        await advanced_usage_example()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
