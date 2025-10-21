# Exchange Adapters

This directory contains exchange adapter implementations that provide a unified interface for interacting with different cryptocurrency exchanges.

## Architecture

All exchange adapters implement the `ExchangeInterface` base class defined in `bot/interfaces/exchange.py`. This ensures consistent behavior across different exchanges.

### Interface Features

- Account information and balance queries
- Order creation, cancellation, and retrieval
- Trade history
- Real-time price data
- Comprehensive error handling
- Rate limiting
- Request/response logging

## Binance Adapter

The Binance adapter (`binance.py`) provides robust integration with Binance exchange.

### Features

- **Dual Environment Support**: Seamless switching between testnet and mainnet
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Rate Limiting**: Token bucket algorithm to respect API limits
- **Request Logging**: Comprehensive logging with sensitive data sanitization
- **Type Safety**: Full Pydantic models for all data structures
- **Async/Await**: Non-blocking async operations for high performance

### Configuration

See `.env.binance.example` for all configuration options.

**Minimum required:**
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true  # Use testnet for development
```

### Usage Example

```python
from decimal import Decimal
from bot.execution.adapters.binance import BinanceAdapter
from bot.interfaces.exchange import OrderSide, OrderType, TimeInForce

# Initialize adapter
adapter = BinanceAdapter(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True  # Use testnet for development
)

# Connect to exchange
await adapter.connect()

try:
    # Get account information
    account_info = await adapter.get_account_info()
    print(f"Can trade: {account_info.can_trade}")

    # Get balance for specific asset
    btc_balance = await adapter.get_balance("BTC")
    if btc_balance:
        print(f"BTC Balance: {btc_balance[0].free}")

    # Get current price
    price = await adapter.get_ticker_price("BTCUSDT")
    print(f"BTC/USDT: {price}")

    # Create a limit order
    order = await adapter.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000.00"),
        time_in_force=TimeInForce.GTC
    )
    print(f"Order created: {order.id}")

    # Get open orders
    open_orders = await adapter.get_open_orders("BTCUSDT")
    print(f"Open orders: {len(open_orders)}")

    # Cancel order
    if open_orders:
        canceled = await adapter.cancel_order("BTCUSDT", open_orders[0].id)
        print(f"Order {canceled.id} canceled")

finally:
    # Always disconnect
    await adapter.disconnect()
```

### Error Handling

The adapter maps Binance error codes to specific exception types:

```python
from bot.interfaces.exchange import (
    ExchangeAPIError,
    ExchangeAuthenticationError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    RateLimitExceededError,
)

try:
    order = await adapter.create_order(...)
except InvalidOrderError as e:
    print(f"Invalid order parameters: {e}")
except InsufficientBalanceError as e:
    print(f"Not enough balance: {e}")
except RateLimitExceededError as e:
    print(f"Rate limit exceeded: {e}")
except ExchangeAPIError as e:
    print(f"API error: {e.message}, Code: {e.status_code}")
```

### Rate Limiting

The adapter automatically manages rate limits using a token bucket algorithm:

- **General endpoints**: 1200 requests/minute (configurable)
- **Order endpoints**: 100 requests/minute (configurable)

Rate limiting is transparent - requests are automatically delayed when limits are approached.

### Retry Logic

Failed requests are automatically retried with exponential backoff:

- **Max retries**: 3 (configurable)
- **Initial delay**: 1 second (configurable)
- **Max delay**: 30 seconds (configurable)
- **Backoff factor**: 2.0 (configurable)

Retries are applied for:
- Network errors
- Timeout errors
- Rate limit errors (429 status)
- Temporary server errors (5xx status)

### Logging

All requests and responses are logged with structured logging:

```python
# Request log
{
    "event": "binance_request",
    "method": "POST",
    "endpoint": "/api/v3/order",
    "attempt": 1,
    "params": {"symbol": "BTCUSDT", "signature": "***"}  # Sensitive data sanitized
}

# Response log
{
    "event": "binance_response",
    "status": 200,
    "endpoint": "/api/v3/order",
    "response_length": 342
}

# Order created log
{
    "event": "order_created",
    "symbol": "BTCUSDT",
    "side": "buy",
    "order_type": "limit",
    "order_id": "12345"
}
```

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/execution/adapters/test_binance.py -v

# Run specific test
pytest tests/unit/execution/adapters/test_binance.py::TestBinanceAdapter::test_create_market_order -v

# With coverage
pytest tests/unit/execution/adapters/test_binance.py --cov=bot.execution.adapters.binance --cov-report=html
```

### Integration Tests

Integration tests require Binance testnet credentials:

```bash
# Set up testnet credentials
export BINANCE_TESTNET_API_KEY="your_testnet_key"
export BINANCE_TESTNET_API_SECRET="your_testnet_secret"

# Run integration tests
pytest tests/integration/execution/adapters/test_binance_integration.py -v

# Run specific test
pytest tests/integration/execution/adapters/test_binance_integration.py::TestBinanceIntegration::test_create_and_cancel_limit_order -v
```

**Note**: Integration tests make real API calls to Binance testnet and may be slow due to rate limiting.

## Security Best Practices

### API Key Configuration

1. **Create API keys with minimal permissions**:
   - Enable: Spot Trading, Read
   - Disable: Withdrawals, Futures, Margin

2. **Use IP whitelist**: Restrict API key to specific IPs

3. **Separate keys for environments**:
   - Testnet keys for development
   - Mainnet keys for production

4. **Never commit credentials**: Always use environment variables

5. **Rotate keys regularly**: Change API keys every 3-6 months

### Development Workflow

1. **Always start with testnet**:
   ```bash
   BINANCE_TESTNET=true
   ```

2. **Test thoroughly before mainnet**:
   - Run full test suite
   - Paper trade for at least 2-4 weeks
   - Verify all edge cases

3. **Gradual mainnet rollout**:
   - Start with minimum amounts
   - Monitor closely for first week
   - Gradually increase position sizes

### Monitoring

- Monitor API usage in Binance dashboard
- Set up alerts for unusual activity
- Review logs regularly for errors
- Track order execution quality

## Adding New Exchanges

To add support for a new exchange:

1. Create new adapter file: `bot/execution/adapters/your_exchange.py`

2. Implement `ExchangeInterface`:
   ```python
   from bot.interfaces.exchange import ExchangeInterface

   class YourExchangeAdapter(ExchangeInterface):
       async def connect(self) -> None:
           # Implementation

       async def get_account_info(self) -> AccountInfo:
           # Implementation

       # ... implement all abstract methods
   ```

3. Add configuration to `.env.example`

4. Create unit tests: `tests/unit/execution/adapters/test_your_exchange.py`

5. Create integration tests: `tests/integration/execution/adapters/test_your_exchange_integration.py`

6. Update this README with exchange-specific documentation

## Troubleshooting

### Common Issues

**Authentication Errors**:
- Verify API key and secret are correct
- Check API key permissions in exchange dashboard
- Ensure IP whitelist includes your IP (if enabled)
- Verify system time is synchronized (important for signatures)

**Rate Limit Errors**:
- Reduce request frequency
- Increase rate limit buffer (lower configured limits)
- Check for other processes using same API keys

**Timeout Errors**:
- Increase `BINANCE_TIMEOUT` value
- Check network connectivity
- Verify exchange is not experiencing downtime

**Order Errors**:
- Verify symbol is correct and actively traded
- Check minimum order size requirements
- Ensure sufficient balance
- Validate price/quantity precision

### Debug Mode

Enable detailed logging:

```bash
# In .env
LOG_LEVEL=DEBUG
BINANCE_LOG_REQUESTS=true

# Or programmatically
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)
```

## Performance Considerations

### Connection Pooling

The adapter uses aiohttp's built-in connection pooling for efficiency.

### Rate Limit Optimization

- Batch requests when possible
- Cache frequently accessed data (prices, balances)
- Use websockets for real-time data (future enhancement)

### Memory Usage

- Order history queries are limited to prevent memory issues
- Use pagination for large result sets

## Future Enhancements

- [ ] WebSocket support for real-time data
- [ ] Futures trading support
- [ ] Margin trading support
- [ ] Advanced order types (OCO, trailing stop)
- [ ] Market data streaming
- [ ] Connection pooling optimization
- [ ] Circuit breaker pattern for resilience
- [ ] Metrics export (Prometheus)

## References

- [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Binance Testnet](https://testnet.binance.vision/)
- [CCXT Library](https://github.com/ccxt/ccxt) (alternative implementation)
