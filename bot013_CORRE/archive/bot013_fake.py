#!/usr/bin/env python3
"""
15-Minute Binary Options Pair Trading Bot

Strategy:
- Trade COIN1 PUT + COIN2 CALL pairs when:
  1. More than 10 minutes until period end
  2. Both asks between $0.24-0.47
  3. Sum of asks < $0.77
- Opens positions sequentially (first then second mandatory)
- Scales position size based on available balance
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.append('/home/ubuntu/013_2025_polymarket')
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env

class PairTrader:
    """15-minute binary options pair trading bot"""

    # File paths for market data
    DATA_FILES = {
        'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
        'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
        'ETH_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_ETH.json',
        'ETH_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_ETH.json',
        'SOL_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_SOL.json',
        'SOL_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_SOL.json',
        'XRP_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_XRP.json',
        'XRP_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_XRP.json',
    }

    COINS = ['BTC', 'ETH', 'SOL', 'XRP']

    # Trading parameters
    MIN_ASK_PRICE = 0.24
    MAX_ASK_PRICE = 0.77
    MAX_SUM_PRICE = 1.97
    MIN_TIME_TO_EXPIRY_MINUTES = 10

    # Position sizing
    BASE_POSITION_SIZE = 5 # Base size per leg
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC ERC-20 address on Polygon

    def __init__(self, credentials: Dict[str, str]):
        """Initialize the pair trader"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        self.traded_pairs = set()  # Track pairs already traded this period

        print("=" * 70)
        print("🎯 15-MINUTE PAIR TRADING BOT INITIALIZED")
        print("=" * 70)

    def read_market_data(self, coin: str, option_type: str) -> Optional[Dict]:
        """Read market data from JSON file"""
        try:
            file_key = f"{coin}_{option_type}"
            file_path = self.DATA_FILES.get(file_key)

            if not file_path or not os.path.exists(file_path):
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            return {
                'asset_id': data.get('asset_id'),
                'asset_name': data.get('asset_name'),
                'best_bid': data.get('best_bid', {}),
                'best_ask': data.get('best_ask', {}),
                'timestamp': data.get('timestamp'),
                'timestamp_readable': data.get('timestamp_readable')
            }

        except Exception as e:
            print(f"❌ Error reading {coin} {option_type}: {e}")
            return None

    def get_next_expiry_time(self) -> datetime:
        """Calculate the next 15-minute expiry time (00, 15, 30, 45)"""
        now = datetime.now()
        minute = now.minute

        # Find next 15-minute mark
        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0
            now += timedelta(hours=1)

        next_expiry = now.replace(minute=next_minute, second=0, microsecond=0)
        return next_expiry

    def minutes_to_expiry(self) -> float:
        """Calculate minutes until next expiry"""
        now = datetime.now()
        next_expiry = self.get_next_expiry_time()
        diff = (next_expiry - now).total_seconds() / 60
        return diff

    def get_usdc_balance(self) -> float:
        """Get USDC balance (ERC-20 token)"""
        try:
            # For ERC-20 COLLATERAL tokens, token_id should be empty
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                    # No token_id for COLLATERAL - it's the base USDC balance
                )
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6  # USDC has 6 decimals

            return balance_usdc

        except Exception as e:
            print(f"❌ Error getting USDC balance: {e}")
            return 0.0

    def calculate_max_pairs(self, usdc_balance: float, pair_cost: float) -> int:
        """Calculate how many pairs we can afford"""
        if pair_cost <= 0:
            return 0

        max_pairs = int(usdc_balance / pair_cost)
        return max(0, max_pairs)

    def check_pair_conditions(self, coin1: str, coin2: str) -> Optional[Dict]:
        """
        Check if pair trading conditions are met

        Returns:
            Dict with trade details if conditions met, None otherwise
        """
        # Read market data
        put_data = self.read_market_data(coin1, 'PUT')
        call_data = self.read_market_data(coin2, 'CALL')

        if not put_data or not call_data:
            return None

        # Extract ask prices
        put_ask = put_data['best_ask'].get('price', 0)
        call_ask = call_data['best_ask'].get('price', 0)

        if put_ask == 0 or call_ask == 0:
            return None

        # Check conditions
        put_in_range = self.MIN_ASK_PRICE <= put_ask <= self.MAX_ASK_PRICE
        call_in_range = self.MIN_ASK_PRICE <= call_ask <= self.MAX_ASK_PRICE
        sum_under_max = (put_ask + call_ask) < self.MAX_SUM_PRICE

        if put_in_range and call_in_range and sum_under_max:
            return {
                'coin1': coin1,
                'coin2': coin2,
                'put_ask': put_ask,
                'call_ask': call_ask,
                'total_cost': put_ask + call_ask,
                'put_token_id': put_data['asset_id'],
                'call_token_id': call_data['asset_id'],
                'put_asset_name': put_data['asset_name'],
                'call_asset_name': call_data['asset_name']
            }

        return None

    def execute_pair_trade_test(self, pair_info: Dict, num_pairs: int) -> bool:
        """
        Execute the pair trade (PUT + CALL) and immediately place sell orders at $0.99

        Args:
            pair_info: Pair trading information
            num_pairs: Number of pairs to trade

        Returns:
            True if both legs executed successfully
        """
        position_size = self.BASE_POSITION_SIZE
        sell_price = 0.99

        print("\n" + "=" * 70)
        print(f"🎯 EXECUTING PAIR TRADE x{num_pairs}")
        print("=" * 70)
        print(f"📊 Pair: {pair_info['coin1']} PUT + {pair_info['coin2']} CALL")
        print(f"💰 PUT ask: ${pair_info['put_ask']:.4f}")
        print(f"💰 CALL ask: ${pair_info['call_ask']:.4f}")
        print(f"💵 Total cost: ${pair_info['total_cost']:.4f} x {num_pairs} = ${pair_info['total_cost'] * num_pairs:.2f}")
        print(f"📦 Position size: {position_size:.2f} per leg")
        print(f"🎯 Auto-sell price: ${sell_price}")
        print("=" * 70)

        # LEG 1: Buy PUT
        print("\n🔵 LEG 1: Buying PUT...")
        put_order_id = self.trader.place_buy_order(
            token_id=pair_info['put_token_id'],
            price=pair_info['put_ask']-0.3,
            quantity=position_size
        )

        cancelled_count = self.trader.cancel_all_orders()

        if not put_order_id:
            print("❌ Failed to execute LEG 1 (PUT) - ABORTING")
            return False

        print(f"✅ LEG 1 executed: {pair_info['coin1']} PUT")

        cancelled_count = self.trader.cancel_all_orders()

    def execute_pair_trade(self, pair_info: Dict, num_pairs: int) -> bool:
        """
        Execute the pair trade (PUT + CALL) and immediately place sell orders at $0.99

        Args:
            pair_info: Pair trading information
            num_pairs: Number of pairs to trade

        Returns:
            True if both legs executed successfully
        """
        position_size = self.BASE_POSITION_SIZE
        sell_price = 0.99

        print("\n" + "=" * 70)
        print(f"🎯 EXECUTING PAIR TRADE x{num_pairs}")
        print("=" * 70)
        print(f"📊 Pair: {pair_info['coin1']} PUT + {pair_info['coin2']} CALL")
        print(f"💰 PUT ask: ${pair_info['put_ask']:.4f}")
        print(f"💰 CALL ask: ${pair_info['call_ask']:.4f}")
        print(f"💵 Total cost: ${pair_info['total_cost']:.4f} x {num_pairs} = ${pair_info['total_cost'] * num_pairs:.2f}")
        print(f"📦 Position size: {position_size:.2f} per leg")
        print(f"🎯 Auto-sell price: ${sell_price}")
        print("=" * 70)

        # LEG 1: Buy PUT
        print("\n🔵 LEG 1: Buying PUT...")
        put_order_id = self.trader.place_buy_order(
            token_id=pair_info['put_token_id'],
            price=pair_info['put_ask']-0.4,
            quantity=position_size
        )

        if not put_order_id:
            print("❌ Failed to execute LEG 1 (PUT) - ABORTING")
            return False

        print(f"✅ LEG 1 executed: {pair_info['coin1']} PUT")
        cancelled_count = self.trader.cancel_all_orders()


        # Immediately place sell order for PUT at $0.99
        time.sleep(1)  # Brief pause to let buy order settle

        return True

    def scan_all_pairs(self) -> List[Dict]:
        """Scan all possible coin pairs for trading opportunities"""
        opportunities = []

        for coin1 in self.COINS:
            for coin2 in self.COINS:
                if coin1 == coin2:
                    continue  # Skip same coin pairs

                pair_key = f"{coin1}_PUT_{coin2}_CALL"

                # Skip if already traded this period
                if pair_key in self.traded_pairs:
                    continue

                pair_info = self.check_pair_conditions(coin1, coin2)
                if pair_info:
                    opportunities.append((pair_key, pair_info))

        return opportunities

    def run_trading_cycle(self) -> bool:
        """
        Execute one trading cycle

        Returns:
            True if we should check again in 1 second, False if we should sleep 30 seconds
        """
        try:
            # Check time to expiry
            time_left = self.minutes_to_expiry()

            print(f"\n⏰ Time to expiry: {time_left:.1f} minutes")

            # Check if we're in a new period (reset traded pairs)
            if time_left > 14.5:  # New period started
                if self.traded_pairs:
                    print("🔄 New period detected - resetting traded pairs")
                    self.traded_pairs.clear()

            # Must have at least 10 minutes left
            if time_left < self.MIN_TIME_TO_EXPIRY_MINUTES:
                print(f"⏸️  Less than {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes left - waiting for next period")
                return False  # Sleep 30 seconds when waiting for next period

            # Check USDC balance
            usdc_balance = self.get_usdc_balance()
            print(f"💰 USDC Balance: ${usdc_balance:.2f}")

            if usdc_balance < (self.MIN_ASK_PRICE * 2 * self.BASE_POSITION_SIZE):
                print("⚠️  Insufficient USDC balance for even 1 pair")
                return True  # Keep checking every second - balance might change

            # Scan for opportunities
            print("\n🔍 Scanning for pair trading opportunities...")
            opportunities = self.scan_all_pairs()

            if not opportunities:
                print("📊 No opportunities found matching criteria")
                return True  # Keep checking every second - opportunities can appear

            print(f"\n✨ Found {len(opportunities)} opportunities:")
            for pair_key, pair_info in opportunities:
                print(f"   • {pair_key}: ${pair_info['total_cost']:.4f}")

            # Execute trades
            for pair_key, pair_info in opportunities:
                # Calculate how many pairs we can afford
                pair_cost = pair_info['total_cost'] * self.BASE_POSITION_SIZE
                num_pairs = self.calculate_max_pairs(usdc_balance, pair_cost)

                if num_pairs < 1:
                    print(f"\n⚠️  Insufficient balance for {pair_key}")
                    continue

                # Execute the trade
                success = self.execute_pair_trade(pair_info, num_pairs)

                if success:
                    # Mark as traded
                    self.traded_pairs.add(pair_key)

                    # Update balance
                    usdc_balance -= (pair_cost * num_pairs)
                    print(f"\n💰 Remaining USDC: ${usdc_balance:.2f}")

                    # Check if we can afford more pairs
                    if usdc_balance < (self.MIN_ASK_PRICE * 2 * self.BASE_POSITION_SIZE):
                        print("💼 Insufficient balance for additional pairs")
                        break

            return True  # Keep checking every second

        except Exception as e:
            print(f"\n❌ Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            return True  # Keep checking even after errors

    def run(self):
        """Main trading loop - checks every second when trading window is open"""
        print("\n🚀 Starting pair trading bot...")
        print(f"📊 Strategy: {self.COINS} coins, PUT+CALL pairs")
        print(f"💵 Price range: ${self.MIN_ASK_PRICE} - ${self.MAX_ASK_PRICE}")
        print(f"💰 Max sum: ${self.MAX_SUM_PRICE}")
        print(f"⏰ Min time to expiry: {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes")
        print(f"⚡ Check interval: 1 second when trading window open, 30 seconds when waiting")
        print("\n" + "=" * 70 + "\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n{'='*70}")
                print(f"🔄 CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                # Run trading cycle and get back whether to check frequently
                check_frequently = self.run_trading_cycle()

                if check_frequently:
                    # Active trading window - check every second
                    print(f"\n⚡ Checking again in 1 second...")
                    time.sleep(1)
                else:
                    # Waiting for next period - check every 30 seconds
                    print(f"\n😴 Waiting for next period - sleeping 12 seconds...")
                    time.sleep(12)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎯 15-MINUTE BINARY OPTIONS PAIR TRADER")
    print("="*70)

    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        creds = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return

    # Initialize and run bot
    try:
        bot = PairTrader(creds)
        bot.run()  # Dynamic checking: 1s when trading, 30s when waiting
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
