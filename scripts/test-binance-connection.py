#!/usr/bin/env python3
"""
Quick test script to verify Binance testnet connectivity.
This script tests the API connection without running the full test suite.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.execution.adapters.binance import BinanceAdapter


def load_env_file():
    """Load credentials from .env.binance file."""
    env_file = project_root / ".env"

    if not env_file.exists():
        print("❌ ERROR: .env.binance file not found!")
        print(f"Expected location: {env_file}")
        return None, None

    credentials = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    credentials[key] = value

    api_key = credentials.get("BINANCE_TESTNET_API_KEY")
    api_secret = credentials.get("BINANCE_TESTNET_API_SECRET")

    return api_key, api_secret


async def test_connection():
    """Test Binance testnet connection and basic operations."""
    print("=" * 60)
    print("  Binance Testnet Connection Test")
    print("=" * 60)
    print()

    # Load credentials
    print("📋 Loading credentials from .env.binance...")
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

    # If not in environment, load from file
    if not api_key or not api_secret:
        print("   (Not found in environment, loading from file)")
        api_key, api_secret = load_env_file()

    if not api_key or not api_secret:
        print("❌ ERROR: Could not load testnet credentials")
        return False

    # Mask credentials for display
    key_masked = f"{api_key[:4]}...{api_key[-4:]}"
    secret_masked = f"{api_secret[:4]}...{api_secret[-4:]}"
    print(f"✓ API Key: {key_masked}")
    print(f"✓ API Secret: {secret_masked}")
    print()

    # Create adapter
    print("🔌 Creating Binance adapter (testnet mode)...")
    adapter = BinanceAdapter(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )
    print("✓ Adapter created")
    print()

    try:
        # Test 1: Connection
        print("🔗 Test 1: Connecting to Binance testnet...")
        await adapter.connect()
        print("✅ Connection successful!")
        print()

        # Test 2: Ping
        print("🏓 Test 2: Testing connectivity (ping)...")
        ping_result = await adapter.test_connectivity()
        if ping_result:
            print("✅ Ping successful!")
        else:
            print("❌ Ping failed!")
            return False
        print()

        # Test 3: Account info
        print("👤 Test 3: Fetching account information...")
        account_info = await adapter.get_account_info()
        print(f"✅ Account retrieved:")
        print(f"   - Account Type: {account_info.account_type}")
        print(f"   - Can Trade: {account_info.can_trade}")
        print(f"   - Can Withdraw: {account_info.can_withdraw}")
        print(f"   - Can Deposit: {account_info.can_deposit}")
        print(f"   - Maker Commission: {account_info.maker_commission}")
        print(f"   - Taker Commission: {account_info.taker_commission}")
        print()

        # Test 4: Balance
        print("💰 Test 4: Fetching account balances...")
        balances = await adapter.get_balance()
        print(f"✅ Retrieved {len(balances)} assets with balance:")
        for balance in balances[:5]:  # Show first 5
            print(f"   - {balance.asset}: {balance.free} (free), {balance.locked} (locked)")
        if len(balances) > 5:
            print(f"   ... and {len(balances) - 5} more")
        print()

        # Test 5: Ticker price
        print("📊 Test 5: Fetching ticker price (BTCUSDT)...")
        price = await adapter.get_ticker_price("BTCUSDT")
        print(f"✅ Current BTC price: ${price:,.2f}")
        print()

        # Summary
        print("=" * 60)
        print("✅ All tests passed! Binance testnet is ready to use.")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run full integration tests:")
        print("   ./scripts/test-with-testnet.sh")
        print()
        print("2. Run specific test file:")
        print("   ./scripts/test-with-testnet.sh tests/integration/execution/adapters/test_binance_integration.py")
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        print()
        print("Troubleshooting:")
        print("- Check your API key and secret are correct")
        print("- Verify system time is synchronized")
        print("- Ensure you have internet connection")
        print("- Try regenerating testnet API keys at https://testnet.binance.vision/")
        return False

    finally:
        # Cleanup
        await adapter.disconnect()


def main():
    """Run the connection test."""
    try:
        success = asyncio.run(test_connection())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
