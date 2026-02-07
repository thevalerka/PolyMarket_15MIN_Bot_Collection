#!/usr/bin/env python3
"""
15-Minute Binary Options Pair Trading Bot - With Safety Nets

Strategy:
- Trade COIN1 PUT + COIN2 CALL pairs when:
  1. More than 10 minutes until period end
  2. Both asks between $0.24-0.47
  3. Sum of asks < $0.77
- Opens positions sequentially (first then second mandatory)
- Scales position size based on available balance
- Safety checks: verifies order execution, cancels unfilled orders, checks position before retry
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.insert(0, '/home/claude')  # Add current directory first for debug version
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

class PairTrader:
    """15-minute binary options pair trading bot with safety checks"""

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
    MAX_ASK_PRICE = 0.47
    MAX_SUM_PRICE = 0.77
    MIN_TIME_TO_EXPIRY_MINUTES = 10

    # Position sizing
    BASE_POSITION_SIZE = 5  # Base size per leg
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
        self.open_positions = {}  # Track open positions: {token_id: {'size': float, 'coin': str, 'type': str, 'sell_order_placed': bool}}

        print("=" * 70)
        print("🎯 15-MINUTE PAIR TRADING BOT INITIALIZED (WITH SAFETY NETS)")
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

    def verify_position_opened(self, token_id: str, expected_size: float) -> bool:
        """
        Verify if a position was actually opened by checking token balance

        Args:
            token_id: Token to check
            expected_size: Expected position size

        Returns:
            True if position is open with expected size
        """
        try:
            print(f"   🔍 DEBUG: Checking balance for token {token_id[:16]}...")
            
            balance_raw, balance_tokens = self.trader.get_token_balance(token_id)
            
            print(f"   🔍 DEBUG: Raw balance: {balance_raw}, Tokens: {balance_tokens:.6f}")
            print(f"   🔍 DEBUG: Expected: {expected_size:.6f}, Threshold (95%): {expected_size * 0.95:.6f}")

            # Allow small tolerance for rounding
            if balance_tokens >= (expected_size * 0.95):
                print(f"   ✅ Position verified: {balance_tokens:.2f} tokens (expected {expected_size:.2f})")
                return True
            else:
                print(f"   ℹ️  Position check: {balance_tokens:.2f} tokens (expected {expected_size:.2f})")
                return False

        except Exception as e:
            print(f"   ❌ Error verifying position: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_token_balance(self, token_id: str) -> float:
        """Check current balance of a specific token"""
        try:
            print(f"   🔍 DEBUG check_token_balance: token_id={token_id[:16]}...")
            balance_raw, balance = self.trader.get_token_balance(token_id)
            print(f"   🔍 DEBUG check_token_balance: raw={balance_raw}, tokens={balance:.6f}")
            return balance
        except Exception as e:
            print(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def verify_sell_order_exists(self, token_id: str) -> bool:
        """
        Check if there's an active sell order for a token
        
        Returns:
            True if sell order exists, False otherwise
        """
        try:
            open_orders = self.trader.get_open_orders()
            
            for order in open_orders:
                if (order.get('asset_id') == token_id and 
                    order.get('side') == 'SELL' and
                    order.get('status') in ['LIVE', 'OPEN']):
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error checking sell orders: {e}")
            return False

    def place_sell_order_safely(self, token_id: str, coin: str, option_type: str, position_size: float) -> bool:
        """
        Safely place a sell order at $0.99 with verification
        
        Returns:
            True if sell order successfully placed and verified
        """
        sell_price = 0.99
        
        print(f"📤 Placing sell order for {coin} {option_type} @ ${sell_price}...")
        
        # Check if sell order already exists
        if self.verify_sell_order_exists(token_id):
            print(f"   ℹ️  Sell order already exists")
            return True
        
        # Place sell order
        sell_order_id = self.trader.place_sell_order(
            token_id=token_id,
            price=sell_price,
            quantity=position_size
        )
        
        if not sell_order_id:
            print(f"   ❌ Sell order failed")
            return False
        
        # Wait and verify
        time.sleep(1)
        
        if self.verify_sell_order_exists(token_id):
            print(f"   ✅ Sell order verified @ ${sell_price}")
            return True
        else:
            print(f"   ⚠️  Sell order not found in open orders")
            return False

    def execute_pair_trade(self, pair_info: Dict, num_pairs: int) -> bool:
        """
        Execute the pair trade (PUT + CALL) with safety checks

        Args:
            pair_info: Pair trading information
            num_pairs: Number of pairs to trade

        Returns:
            True if both legs executed successfully
        """
        position_size = self.BASE_POSITION_SIZE * num_pairs
        sell_price = 0.99
        max_retries = 3

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

        # ============================================================
        # LEG 1: Buy PUT with safety checks
        # ============================================================
        print("\n🔵 LEG 1: Buying PUT...")
        put_opened = False

        for attempt in range(1, max_retries + 1):
            print(f"   Attempt {attempt}/{max_retries}")

            # Check if position already exists before placing order
            if self.verify_position_opened(pair_info['put_token_id'], position_size):
                print("   ℹ️  Position already exists - skipping buy")
                put_opened = True
                break

            # Place buy order
            put_order_id = self.trader.place_buy_order(
                token_id=pair_info['put_token_id'],
                price=pair_info['put_ask'],
                quantity=position_size
            )

            if not put_order_id:
                print(f"   ❌ Buy order failed")
                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)
                continue

            # Wait briefly for order to execute
            time.sleep(2)

            # Check order status
            order_status = self.trader.get_order_status(put_order_id)

            if order_status and order_status.get('status') in ['LIVE', 'OPEN']:
                # Order not filled immediately - cancel it
                print(f"   ⚠️  Order not filled immediately (status: {order_status.get('status')})")
                print("   🔄 Cancelling order...")
                self.trader.cancel_order(put_order_id)
                time.sleep(1)

                # Verify position wasn't partially filled
                if self.verify_position_opened(pair_info['put_token_id'], position_size):
                    put_opened = True
                    break

                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)
                continue

            elif order_status and order_status.get('status') in ['FILLED', 'MATCHED']:
                # Order filled successfully
                print(f"   ✅ Order filled (status: {order_status.get('status')})")
                put_opened = True
                break
            else:
                # Unknown status - verify position
                print(f"   ⚠️  Unknown order status: {order_status.get('status') if order_status else 'None'}")
                if self.verify_position_opened(pair_info['put_token_id'], position_size):
                    put_opened = True
                    break

                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)

        if not put_opened:
            print("❌ Failed to open LEG 1 (PUT) after all retries - ABORTING")
            return False

        print(f"✅ LEG 1 confirmed: {pair_info['coin1']} PUT")

        # Track open position
        self.open_positions[pair_info['put_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin1'],
            'type': 'PUT',
            'sell_order_placed': False
        }

        # Place sell order for PUT at $0.99
        time.sleep(1)
        if self.place_sell_order_safely(pair_info['put_token_id'], pair_info['coin1'], 'PUT', position_size):
            self.open_positions[pair_info['put_token_id']]['sell_order_placed'] = True

        time.sleep(0.5)

        # ============================================================
        # LEG 2: Buy CALL (MANDATORY) with safety checks
        # ============================================================
        print("\n🔵 LEG 2: Buying CALL (MANDATORY)...")
        call_opened = False

        for attempt in range(1, max_retries + 1):
            print(f"   Attempt {attempt}/{max_retries}")

            # Check if position already exists before placing order
            if self.verify_position_opened(pair_info['call_token_id'], position_size):
                print("   ℹ️  Position already exists - skipping buy")
                call_opened = True
                break

            # Place buy order
            call_order_id = self.trader.place_buy_order(
                token_id=pair_info['call_token_id'],
                price=pair_info['call_ask'],
                quantity=position_size
            )

            if not call_order_id:
                print(f"   ❌ Buy order failed")
                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)
                continue

            # Wait briefly for order to execute
            time.sleep(2)

            # Check order status
            order_status = self.trader.get_order_status(call_order_id)

            if order_status and order_status.get('status') in ['LIVE', 'OPEN']:
                # Order not filled immediately - cancel it
                print(f"   ⚠️  Order not filled immediately (status: {order_status.get('status')})")
                print("   🔄 Cancelling order...")
                self.trader.cancel_order(call_order_id)
                time.sleep(1)

                # Verify position wasn't partially filled
                if self.verify_position_opened(pair_info['call_token_id'], position_size):
                    call_opened = True
                    break

                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)
                continue

            elif order_status and order_status.get('status') in ['FILLED', 'MATCHED']:
                # Order filled successfully
                print(f"   ✅ Order filled (status: {order_status.get('status')})")
                call_opened = True
                break
            else:
                # Unknown status - verify position
                print(f"   ⚠️  Unknown order status: {order_status.get('status') if order_status else 'None'}")
                if self.verify_position_opened(pair_info['call_token_id'], position_size):
                    call_opened = True
                    break

                if attempt < max_retries:
                    print("   🔄 Retrying...")
                    time.sleep(1)

        if not call_opened:
            print("❌ WARNING: Failed to open LEG 2 (CALL) but LEG 1 is open!")
            print("⚠️  Manual intervention may be needed")
            return False

        print(f"✅ LEG 2 confirmed: {pair_info['coin2']} CALL")

        # Track open position
        self.open_positions[pair_info['call_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin2'],
            'type': 'CALL',
            'sell_order_placed': False
        }

        # Place sell order for CALL at $0.99
        time.sleep(1)
        if self.place_sell_order_safely(pair_info['call_token_id'], pair_info['coin2'], 'CALL', position_size):
            self.open_positions[pair_info['call_token_id']]['sell_order_placed'] = True

        print("\n" + "=" * 70)
        print("🎉 PAIR TRADE COMPLETED WITH AUTO-SELL ORDERS")
        print("=" * 70)

        return True

    def scan_all_pairs(self) -> List[Dict]:
        """Scan all possible coin pairs for trading opportunities"""
        opportunities = []

        print(f"\n   🔍 DEBUG: Currently traded pairs: {self.traded_pairs}")
        print(f"   🔍 DEBUG: Currently open positions: {len(self.open_positions)}")

        for coin1 in self.COINS:
            for coin2 in self.COINS:
                if coin1 == coin2:
                    continue  # Skip same coin pairs

                pair_key = f"{coin1}_PUT_{coin2}_CALL"

                # Skip if already traded this period
                if pair_key in self.traded_pairs:
                    print(f"   ⏭️  Skipping {pair_key} (already traded)")
                    continue

                pair_info = self.check_pair_conditions(coin1, coin2)
                if pair_info:
                    print(f"   ✅ Found opportunity: {pair_key}")
                    opportunities.append((pair_key, pair_info))

        return opportunities

    def verify_and_fix_sell_orders(self):
        """
        Verify all open positions have sell orders when trading window closes
        Place missing sell orders
        """
        if not self.open_positions:
            return
        
        print("\n" + "=" * 70)
        print("🔍 VERIFYING POSITIONS AND SELL ORDERS")
        print("=" * 70)
        
        positions_to_remove = []
        
        for token_id, position_info in self.open_positions.items():
            coin = position_info['coin']
            option_type = position_info['type']
            expected_size = position_info['size']
            sell_order_placed = position_info['sell_order_placed']
            
            print(f"\n📊 Checking {coin} {option_type}:")
            
            # Check current balance
            current_balance = self.check_token_balance(token_id)
            print(f"   Balance: {current_balance:.2f} tokens")
            
            if current_balance < 0.01:
                # Position already sold/closed
                print(f"   ✅ Position closed (no tokens remaining)")
                positions_to_remove.append(token_id)
                continue
            
            # Check if sell order exists
            has_sell_order = self.verify_sell_order_exists(token_id)
            
            if not has_sell_order:
                print(f"   ⚠️  No sell order found - placing now...")
                if self.place_sell_order_safely(token_id, coin, option_type, current_balance):
                    self.open_positions[token_id]['sell_order_placed'] = True
                    print(f"   ✅ Sell order placed for {current_balance:.2f} tokens")
                else:
                    print(f"   ❌ Failed to place sell order")
            else:
                print(f"   ✅ Sell order verified")
                self.open_positions[token_id]['sell_order_placed'] = True
        
        # Remove closed positions
        for token_id in positions_to_remove:
            del self.open_positions[token_id]
        
        print("\n" + "=" * 70)

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
                    # Clear open positions tracking for new period
                    self.open_positions.clear()

            # Must have at least 10 minutes left
            if time_left < self.MIN_TIME_TO_EXPIRY_MINUTES:
                print(f"⏸️  Less than {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes left - waiting for next period")
                
                # Verify all positions have sell orders
                if self.open_positions:
                    self.verify_and_fix_sell_orders()
                
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
                    print(f"\n   🔍 DEBUG: Marking {pair_key} as traded")
                    self.traded_pairs.add(pair_key)
                    print(f"   🔍 DEBUG: Traded pairs now: {self.traded_pairs}")

                    # Update balance
                    usdc_balance -= (pair_cost * num_pairs)
                    print(f"\n💰 Remaining USDC: ${usdc_balance:.2f}")

                    # Check if we can afford more pairs
                    if usdc_balance < (self.MIN_ASK_PRICE * 2 * self.BASE_POSITION_SIZE):
                        print("💼 Insufficient balance for additional pairs")
                        break
                else:
                    print(f"\n   ⚠️  DEBUG: Trade failed for {pair_key}, NOT marking as traded")

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
        print(f"⚡ Check interval: 1 second when trading window open, 12 seconds when waiting")
        print(f"🛡️  Safety: Order verification, auto-cancel unfilled, position checks")
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
                    # Waiting for next period - check every 12 seconds
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
    print("🎯 15-MINUTE BINARY OPTIONS PAIR TRADER (WITH SAFETY NETS)")
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
        bot.run()  # Dynamic checking: 1s when trading, 12s when waiting
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
