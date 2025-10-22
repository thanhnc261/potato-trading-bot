# Market Data Module

Real-time market data streaming with WebSocket and REST polling support for cryptocurrency exchanges.

## Features

- **WebSocket Support**: Real-time price updates via WebSocket connections (when supported by exchange)
- **Multi-Symbol Streaming**: Subscribe to multiple trading pairs simultaneously
- **Efficient Buffering**: PyArrow-based in-memory columnar storage for fast access
- **Auto-Reconnection**: Automatic reconnection with exponential backoff
- **Health Monitoring**: Heartbeat monitoring for connection health
- **Data Normalization**: Standardized data format across exchanges
- **Multi-Exchange**: Support for multiple exchanges via CCXT

## Quick Start

### Basic Usage

```python
import asyncio
from bot.data.market_data import MarketDataStream

async def main():
    # Create stream
    stream = MarketDataStream(
        exchange_id="binance",
        testnet=True,
    )

    # Connect
    await stream.connect()

    # Subscribe to symbols
    await stream.subscribe(["BTC/USDT", "ETH/USDT"])

    # Collect data
    await asyncio.sleep(10)

    # Get latest data
    df = await stream.buffer.get_latest("BTC/USDT", limit=5)
    print(df)

    # Disconnect
    await stream.disconnect()

asyncio.run(main())
```

### Using Callbacks for Real-Time Data

```python
from bot.data.market_data import MarketDataStream, MarketTick

def price_callback(tick: MarketTick):
    print(f"{tick.symbol}: ${tick.price:,.2f}")

stream = MarketDataStream(exchange_id="binance", testnet=True)
stream.add_callback(price_callback)

await stream.connect()
await stream.subscribe(["BTC/USDT"])
await asyncio.sleep(30)  # Monitor for 30 seconds
await stream.disconnect()
```

### Managing Multiple Exchanges

```python
from bot.data.market_data import MarketDataManager

manager = MarketDataManager()

# Add streams for different exchanges
binance = await manager.add_stream("binance", testnet=True)
await binance.subscribe(["BTC/USDT"])

# Get stream
stream = manager.get_stream("binance")

# Shutdown all
await manager.shutdown()
```

## Architecture

### Components

1. **MarketTick**: Normalized data structure for market ticks
2. **MarketDataBuffer**: PyArrow-based in-memory buffer with fast queries
3. **MarketDataStream**: WebSocket/REST streaming client with auto-reconnection
4. **MarketDataManager**: Multi-exchange stream manager

### Data Flow

```
Exchange (WebSocket/REST)
    ↓
MarketDataStream
    ├── Normalization
    ├── Callbacks (real-time)
    └── MarketDataBuffer (PyArrow)
         ├── In-memory storage
         └── Fast queries
```

### Connection States

- `DISCONNECTED`: Not connected
- `CONNECTING`: Establishing connection
- `CONNECTED`: Active connection
- `RECONNECTING`: Attempting to reconnect
- `FAILED`: Connection failed

## Configuration

### Stream Parameters

```python
stream = MarketDataStream(
    exchange_id="binance",           # Exchange identifier
    testnet=True,                    # Use testnet
    api_key="your_key",              # API key (optional)
    api_secret="your_secret",        # API secret (optional)
    buffer_size=10000,               # Max buffer size per symbol
    heartbeat_interval=30,           # Heartbeat check interval (seconds)
    reconnect_delay=5,               # Initial reconnect delay (seconds)
    max_reconnect_delay=300,         # Max reconnect delay (seconds)
)
```

### Supported Exchanges

All exchanges supported by CCXT can be used. Popular options:
- Binance (`binance`)
- Coinbase (`coinbase`)
- Kraken (`kraken`)
- Bybit (`bybit`)
- And many more...

## Data Schema

### MarketTick Structure

```python
@dataclass
class MarketTick:
    symbol: str              # Trading pair (e.g., "BTC/USDT")
    timestamp: int           # Unix timestamp in milliseconds
    price: float             # Last traded price
    volume: float            # Trading volume
    bid: Optional[float]     # Best bid price
    ask: Optional[float]     # Best ask price
    high: Optional[float]    # High price
    low: Optional[float]     # Low price
    open: Optional[float]    # Open price
    close: Optional[float]   # Close price
    exchange: str            # Exchange identifier
```

### PyArrow Schema

```python
schema = pa.schema([
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
])
```

## Buffer Operations

### Get Latest Data

```python
# Get latest N ticks
df = await stream.buffer.get_latest("BTC/USDT", limit=100)
```

### Query Time Range

```python
# Get ticks in time range (milliseconds)
start_time = 1700000000000
end_time = 1700001000000
df = await stream.buffer.get_range("BTC/USDT", start_time, end_time)
```

### Clear Buffer

```python
# Clear specific symbol
await stream.buffer.clear("BTC/USDT")

# Clear all symbols
await stream.buffer.clear()
```

## Subscription Management

### Subscribe

```python
# Subscribe to single symbol
await stream.subscribe(["BTC/USDT"])

# Subscribe to multiple symbols
await stream.subscribe(["BTC/USDT", "ETH/USDT", "XRP/USDT"])
```

### Unsubscribe

```python
# Unsubscribe from symbols
await stream.unsubscribe(["XRP/USDT"])
```

### Dynamic Subscriptions

```python
# Start with one symbol
await stream.subscribe(["BTC/USDT"])

# Add more later
await stream.subscribe(["ETH/USDT"])

# Remove some
await stream.unsubscribe(["BTC/USDT"])
```

## Error Handling

### Reconnection

The stream automatically handles connection failures:

1. Detects stale connections via heartbeat
2. Attempts reconnection with exponential backoff
3. Re-subscribes to all symbols after reconnection
4. Continues data collection seamlessly

```python
stream = MarketDataStream(
    reconnect_delay=5,        # Start with 5s delay
    max_reconnect_delay=300,  # Max 5 minutes
)
```

### Connection Monitoring

```python
# Check connection state
if stream.state == ConnectionState.CONNECTED:
    print("Stream is connected")
elif stream.state == ConnectionState.RECONNECTING:
    print("Stream is reconnecting...")
```

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/unit/data/test_market_data.py -v

# Specific test
pytest tests/unit/data/test_market_data.py::TestMarketDataBuffer -v

# With coverage
pytest tests/unit/data/test_market_data.py --cov=bot.data.market_data
```

### Run Integration Tests

```bash
# Requires exchange credentials
export BINANCE_TESTNET_API_KEY="your_key"
export BINANCE_TESTNET_API_SECRET="your_secret"

# Run integration tests
pytest tests/integration/data/test_market_data_integration.py -v -m integration

# Skip integration tests
pytest -v -m "not integration"
```

## Examples

See `bot/data/examples/market_data_example.py` for comprehensive examples:

```bash
# Run all examples
python -m bot.data.examples.market_data_example

# Run specific example
python -c "from bot.data.examples.market_data_example import basic_streaming_example; import asyncio; asyncio.run(basic_streaming_example())"
```

## Performance

### Benchmarks

- **Latency**: <100ms for WebSocket updates, ~1s for REST polling
- **Throughput**: Handles 10+ symbols simultaneously
- **Memory**: ~10MB per symbol for 10,000 ticks (with PyArrow compression)
- **Buffer Operations**: Sub-millisecond queries with PyArrow

### Optimization Tips

1. **Use WebSocket**: Prefer exchanges with WebSocket support for lower latency
2. **Adjust Buffer Size**: Balance between memory and data retention
3. **Batch Subscriptions**: Subscribe to multiple symbols at once
4. **Limit Callbacks**: Keep callback logic lightweight

## Troubleshooting

### Connection Issues

**Problem**: Stream fails to connect

**Solutions**:
- Check API credentials
- Verify network connectivity
- Enable testnet for development
- Check exchange status

### No Data Collected

**Problem**: Buffer is empty after subscription

**Solutions**:
- Wait longer for initial data
- Verify symbol format (use `BTC/USDT` not `BTCUSDT`)
- Check if symbol is traded on exchange
- Review logs for errors

### High Memory Usage

**Problem**: Memory consumption growing

**Solutions**:
- Reduce `buffer_size` parameter
- Clear old data periodically
- Limit number of subscribed symbols
- Use time range queries instead of full buffer

## Best Practices

1. **Always use testnet** for development and testing
2. **Handle callbacks gracefully** - exceptions in callbacks are logged but don't stop streaming
3. **Monitor connection state** - check state before operations
4. **Clear buffers** when switching symbols or strategies
5. **Use time range queries** for historical analysis
6. **Implement proper shutdown** - always call `disconnect()` or use context managers

## Integration with Trading Bot

### Example Integration

```python
from bot.data.market_data import MarketDataStream
from bot.core.strategy import TradingStrategy

class MarketDataStrategy(TradingStrategy):
    def __init__(self):
        self.stream = MarketDataStream(exchange_id="binance", testnet=True)
        self.stream.add_callback(self.on_tick)

    async def start(self):
        await self.stream.connect()
        await self.stream.subscribe(["BTC/USDT"])

    def on_tick(self, tick):
        # Process tick data
        if tick.price > 45000:
            self.signal_buy(tick.symbol)

    async def stop(self):
        await self.stream.disconnect()
```

## API Reference

Full API documentation is available in the module docstrings:

```python
help(MarketDataStream)
help(MarketDataBuffer)
help(MarketDataManager)
```

## License

Proprietary - All rights reserved
