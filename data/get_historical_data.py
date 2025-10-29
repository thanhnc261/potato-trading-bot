# Create a script to download data
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Initialize Binance
exchange = ccxt.binance()

# Download 1 month of 1-hour data for BTCUSDT
symbol = 'BTC/USDT'
timeframe = '1h'
since = exchange.parse8601('2024-09-01T00:00:00Z')

print(f'Downloading {symbol} data...')
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save to CSV
df.to_csv('data/historical/BTCUSDT_1h.csv', index=False)
print(f'Saved {len(df)} rows to data/historical/BTCUSDT_1h.csv')